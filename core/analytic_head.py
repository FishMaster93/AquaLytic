"""
Analytic CIL 分类头：把CORAL的3个阈值 Linear(D,3) 换成闭式解Ridge回归。

前提：backbone(conv+fc1)+frontend.bn0 在Phase0之后永久冻结，embedding从那时起
对任何输入都是固定值。在这个前提下：

    A = sum_i x_i x_i^T   (对所有见过的真实样本累加，x_i已augment常数1做bias)
    B = sum_i x_i y_i^T   (y_i是3个阈值目标，按MARGIN缩放)
    beta = (A + lambda*I)^-1 B

每个phase只需要用当前phase自己的真实数据算出这一phase的(A,B)贡献，累加到全局
A/B里，重新解一次beta——这个解在数学上等价于用"迄今为止所有真实样本"联合训练
出的解，不需要重放、不需要蒸馏、不需要猜lambda权重。旧phase的数据不需要重新
访问，只要它们当时贡献的(A,B)被留了下来。
"""
import torch

MARGIN = 4.0  # 阈值目标缩放到±MARGIN，配合coral_predict_intensity里现成的sigmoid


def fit_whitening_stats(raw_embeddings, eps=1e-3):
    """PCA白化：返回(mu, W)，W=Sigma^{-1/2}。跟PANNS_Cnn10.fit_whitening同样
    的数学，单独抽出来一份不依赖模型/不修改模型buffer的版本，给"每个物种一份
    独立白化"这种要动态选择/混合多份白化矩阵的场景用。"""
    mu = raw_embeddings.mean(dim=0)
    centered = raw_embeddings - mu
    cov = (centered.T @ centered) / max(raw_embeddings.shape[0] - 1, 1)
    eigval, eigvec = torch.linalg.eigh(cov)
    eigval = eigval.clamp(min=eps)
    w = eigvec @ torch.diag(eigval.rsqrt()) @ eigvec.T
    return mu, w


def predictive_uncertainty(embeddings, A_inv):
    """
    Ridge回归自带的贝叶斯不确定性，不需要额外训练：
    假设 beta | data ~ N(beta_hat, sigma^2 (A+lambda*I)^-1)，
    对新输入x，预测方差 Var(x^T beta) = sigma^2 * x^T(A+lambda*I)^-1 x。
    这里返回不带sigma^2的相对不确定性 q(x)=x^T A_inv x：
    q(x)小 => x所在区域被大量已见过的真实数据"覆盖"过，模型有把握；
    q(x)大 => x离所有累积数据都很远(比如一个从没见过的新物种)，模型没把握。
    """
    n = embeddings.shape[0]
    ones = torch.ones(n, 1, device=embeddings.device, dtype=embeddings.dtype)
    x = torch.cat([embeddings, ones], dim=1)
    q = (x @ A_inv * x).sum(dim=1)
    return q


def ridge_stats(x_aug, y_scaled):
    """
    通用版本：给定已经拼好特征的x_aug和已经按MARGIN缩放好的目标y_scaled，
    直接算A=x^Tx, B=x^Ty。compute_batch_stats是这个的特化版本(固定只用
    embedding+bias)；物种分层模型需要更灵活的特征拼接(embedding+bias+
    物种独热)，所以单独留一个通用入口，避免在多处重复"augment+算AB"的逻辑。
    """
    return x_aug.T @ x_aug, x_aug.T @ y_scaled


def compute_batch_stats(embeddings, targets, num_thresholds=3, margin=MARGIN):
    """
    embeddings: (N, D) 冻结backbone算出来的真实embedding
    targets: (N,) 0-3的强度整数标签
    返回: A (D+1,D+1), B (D+1,num_thresholds)
    """
    device = embeddings.device
    n = embeddings.shape[0]
    ones = torch.ones(n, 1, device=device, dtype=embeddings.dtype)
    x = torch.cat([embeddings, ones], dim=1)  # 最后一列是bias的常数特征

    y = torch.stack([
        torch.where(targets > k,
                    torch.full_like(targets, margin, dtype=embeddings.dtype),
                    torch.full_like(targets, -margin, dtype=embeddings.dtype))
        for k in range(num_thresholds)
    ], dim=1)  # (N, num_thresholds)

    A = x.T @ x
    B = x.T @ y
    return A, B


def compute_batch_stats_weighted(embeddings, targets, weights, num_thresholds=3, margin=MARGIN):
    """
    跟compute_batch_stats一样，但每个样本按weights加权：A=X^T diag(w) X，
    B=X^T diag(w) Y。原型库场景下要用这个：一个类别原来有n_true个真实样本，
    只留了n_kept个代表性embedding，如果不加权直接跟别的类别的全量真实数据
    拼在一起算最小二乘，这个类别在目标函数里的权重会被稀释成n_kept/n_true，
    实测比例能到1/500，等于最小二乘几乎完全不管这个类别拟合得好不好——
    每个保留的样本按 weight=n_true/n_kept 加权，才能让这个类别在目标函数里
    的总权重跟"当年真的有n_true个样本参与训练"时一致。
    """
    device = embeddings.device
    n = embeddings.shape[0]
    ones = torch.ones(n, 1, device=device, dtype=embeddings.dtype)
    x = torch.cat([embeddings, ones], dim=1)

    y = torch.stack([
        torch.where(targets > k,
                    torch.full_like(targets, margin, dtype=embeddings.dtype),
                    torch.full_like(targets, -margin, dtype=embeddings.dtype))
        for k in range(num_thresholds)
    ], dim=1)

    xw = x * weights.to(device=device, dtype=embeddings.dtype).unsqueeze(1)
    A = xw.T @ x
    B = xw.T @ y
    return A, B


def solve_ridge(A, B, ridge_lambda=1.0):
    """beta = (A + lambda*I)^-1 B，最后一行是bias。"""
    d = A.shape[0]
    reg = ridge_lambda * torch.eye(d, device=A.device, dtype=A.dtype)
    beta = torch.linalg.solve(A + reg, B)
    return beta


def solve_ridge_ordinal_coupled(A, B, ridge_lambda=1.0, coupling_mu=0.0):
    """
    CORAL的3个阈值(k=0,1,2)不再独立求解，加一个耦合正则项让相邻阈值的权重
    向量互相靠近：

        min  sum_k ||X beta_k - y_k||^2 + lambda||beta_k||^2
             + mu*(||beta_0-beta_1||^2 + ||beta_1-beta_2||^2)

    coupling_mu=0时退化成跟solve_ridge完全一样的独立解。mu越大，3个阈值的
    决策方向越平滑/一致，直觉上对应"强度是连续递进的"这个序数先验。

    推导：对beta_0/beta_1/beta_2求梯度=0，得到一个块三对角线性系统：
        (A+λI+μI)  -μI        0     | beta_0 |   | B_0 |
        -μI    (A+λI+2μI)  -μI     | beta_1 | = | B_1 |
        0          -μI   (A+λI+μI) | beta_2 |   | B_2 |
    用块消元(而不是显式拼出3D×3D矩阵)求解，只需要D×D规模的矩阵运算——
    这一点在random_proj_dim很大(比如8192)时尤其重要，3D×3D在那个维度下
    完全解不动。整个求解仍然是A/B的线性函数，累加(A,B)=联合训练解的等价性
    不受影响。
    """
    if coupling_mu == 0.0:
        return solve_ridge(A, B, ridge_lambda)

    d = A.shape[0]
    identity = torch.eye(d, device=A.device, dtype=A.dtype)
    c1 = A + ridge_lambda * identity + coupling_mu * identity       # beta_0, beta_2 共用
    c2 = A + ridge_lambda * identity + 2 * coupling_mu * identity   # beta_1 两侧都耦合
    b0, b1, b2 = B[:, 0:1], B[:, 1:2], B[:, 2:3]

    c1_inv = torch.linalg.inv(c1)
    rhs1 = b1 + coupling_mu * (c1_inv @ (b0 + b2))
    m1 = c2 - 2 * (coupling_mu ** 2) * c1_inv
    beta1 = torch.linalg.solve(m1, rhs1)

    beta0 = c1_inv @ (b0 + coupling_mu * beta1)
    beta2 = c1_inv @ (b2 + coupling_mu * beta1)

    return torch.cat([beta0, beta1, beta2], dim=1)


@torch.no_grad()
def ordinal_violation_rate(model, data_loader, device, tol=0.0):
    """
    统计有多少测试样本出现CORAL阈值分数不单调(s_0>=s_1>=s_2应该成立，
    对应P(强度>None)>=P(强度>Weak)>=P(强度>Medium))的"序数违反"情况。
    """
    model.eval()
    n_violations, n_total = 0, 0
    for batch in data_loader:
        waveform = batch['waveform'].to(device)
        logits = model(waveform)['clipwise_output']
        s0, s1, s2 = logits[:, 0], logits[:, 1], logits[:, 2]
        violated = (s0 < s1 - tol) | (s1 < s2 - tol)
        n_violations += violated.sum().item()
        n_total += logits.shape[0]
    return n_violations / max(n_total, 1)


def set_fc_audioset_from_beta(fc_audioset, beta):
    """把闭式解直接写进nn.Linear(D,3)的weight/bias，不走梯度下降。"""
    weight = beta[:-1].T.to(fc_audioset.weight.device, fc_audioset.weight.dtype)
    bias = beta[-1].to(fc_audioset.bias.device, fc_audioset.bias.dtype)
    with torch.no_grad():
        fc_audioset.weight.copy_(weight)
        fc_audioset.bias.copy_(bias)
    fc_audioset.weight.requires_grad = False
    fc_audioset.bias.requires_grad = False
