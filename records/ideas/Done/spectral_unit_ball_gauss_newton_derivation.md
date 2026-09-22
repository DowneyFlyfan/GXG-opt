# 谱范数单位球约束下的全网络广义高斯–牛顿二阶更新

## 核心结果

保留损失的一阶项和二阶项，将海森矩阵替换为 GGN（Generalized Gauss–Newton，广义高斯–牛顿）近似，保留该近似中的全部跨层块；通过投影梯度求解联合更新，每轮解析消去矩阵拉格朗日乘子。

$$
\boxed{
\begin{aligned}
B&=J^\top C J,\\
\delta^\star
&\in\underset{\|X_\ell\|_2\le 1,\;\ell=1,\ldots,K}
{\operatorname{argmin}}
\left\{g^\top\delta+\frac12\delta^\top B\delta\right\},\\
\delta&=
\begin{bmatrix}
\operatorname{vec}(X_1)\\ \vdots\\ \operatorname{vec}(X_K)
\end{bmatrix},\\
X_{\ell,t+1}
&=\Pi_1\!\left(
X_{\ell,t}-\frac1\beta
\left[G_\ell+\sum_{k=1}^K\mathcal B_{\ell k}[X_{k,t}]\right]
\right).
\end{aligned}
}
$$

其中，$\Pi_1$ 是到谱范数单位球的 Frobenius（弗罗贝尼乌斯）投影。下面给出符号、近似依据、乘子闭式、无需显式矩阵分解的投影表达式及收敛条件。一般曲率下，这是一种迭代解法，不是一步求得原问题最优更新的闭式公式。

## 符号、形状与约束

考虑一次固定训练批次上的网络损失。将该批次的全部网络输出拼成一个向量，将批次平均等归一化系数包含在输出损失函数中：

$$
\begin{aligned}
W_\ell,X_\ell,G_\ell&\in\mathbb R^{m_\ell\times n_\ell},
&&\ell=1,\ldots,K,\\
p_\ell&=m_\ell n_\ell,
&p&=\sum_{\ell=1}^Kp_\ell,\\
\theta_\ell&=\operatorname{vec}(W_\ell),
&\delta_\ell&=\operatorname{vec}(X_\ell),\\
\theta&=
\begin{bmatrix}\theta_1\\\vdots\\\theta_K\end{bmatrix}
\in\mathbb R^p,
&\delta&=
\begin{bmatrix}\delta_1\\\vdots\\\delta_K\end{bmatrix}
\in\mathbb R^p,\\
z&=f(\theta)\in\mathbb R^d,
&\mathcal L(\theta)&=\ell(z;y),\\
g&=\nabla_\theta\mathcal L(\theta),
&G_\ell&=\nabla_{W_\ell}\mathcal L(\theta).
\end{aligned}
$$

$K$ 是参数矩阵块数；无参数共享时可直接对应网络层。有共享参数时，每个独立参数只放入 $\theta$ 一次。$y$ 是固定目标，$d$ 是拼接后的输出维数。

$\operatorname{vec}$ 表示 Vectorization（按列向量化）；$\operatorname{mat}_\ell$ 表示 Matricization（将长度为 $p_\ell$ 的向量按列还原为第 $\ell$ 层矩阵）。$X_\ell$ 是实际加到该层参数上的候选更新。

$$
\begin{aligned}
\langle A,D\rangle_F&=\operatorname{tr}(A^\top D),\\
\|A\|_F^2&=\operatorname{tr}(A^\top A),\\
\|A\|_2&=\sigma_{\max}(A),\\
\|\delta\|_2^2&=\sum_{\ell=1}^K\|X_\ell\|_F^2,\\
\mathcal C
&=\{\delta:\|X_\ell\|_2\le1,\ \ell=1,\ldots,K\}.
\end{aligned}
$$

$\operatorname{tr}$ 是 Trace（迹），$I_n$ 是 $n$ 阶单位矩阵，$\preceq$ 表示对称矩阵的半正定序。这里约束的是每个更新矩阵的谱范数，不是整个拼接向量的欧氏范数。单矩阵问题取 $K=1$。

## 完整海森矩阵的链式分解

假设当前位置附近所需导数存在。定义输出梯度、输出海森矩阵和网络雅可比矩阵：

$$
\begin{aligned}
a&=\nabla_z\ell(z;y)\in\mathbb R^d,\\
C&=\nabla_z^2\ell(z;y)\in\mathbb R^{d\times d},\\
J&=\frac{\partial f(\theta)}{\partial\theta^\top}
\in\mathbb R^{d\times p},\\
g&=J^\top a.
\end{aligned}
$$

逐元素求导：

$$
\begin{aligned}
g_u
&=\sum_{i=1}^dJ_{iu}a_i,\\
H_{uv}
&=\frac{\partial g_u}{\partial\theta_v}\\
&=\sum_{i=1}^d\sum_{j=1}^d
J_{iu}C_{ij}J_{jv}
+\sum_{i=1}^d
 a_i\frac{\partial^2 f_i}{\partial\theta_u\partial\theta_v}.
\end{aligned}
$$

因此：

$$
\boxed{
\begin{aligned}
H&=\nabla_\theta^2\mathcal L(\theta)=B+E,\\
B&=\underbrace{J^\top C J}_{\text{保留的第一项}},\\
E&=\underbrace{\sum_{i=1}^d a_i\nabla_\theta^2f_i(\theta)}
_{\text{舍弃的网络输出二阶导数项}}.
\end{aligned}
}
$$

仅保留第一项，就是广义高斯–牛顿近似；并非只保留损失的一阶 Taylor（泰勒）项。这个分解及近似可参见 Martens 的讨论。[^martens]

$$
\boxed{H\approx B=J^\top C J.}
$$

如果输出损失在 $z$ 上凸，则：

$$
\begin{aligned}
C&\succeq0,\\
v^\top Bv
&=(Jv)^\top C(Jv)\ge0,
\qquad v\in\mathbb R^p,\\
B&\succeq0.
\end{aligned}
$$

这是后面二次子问题具有全局最优性保证的关键假设。仅仅写出 $J^\top C J$ 并不会使任意非凸输出损失的 $C$ 自动变成半正定矩阵。

## 二阶目标究竟近似了什么

完整 Taylor 展开是：

$$
\begin{aligned}
\mathcal L(\theta+\delta)-\mathcal L(\theta)
&=g^\top\delta+\frac12\delta^\top H\delta
+o(\|\delta\|_2^2)\\
&=g^\top\delta+\frac12\delta^\top B\delta
+\frac12\delta^\top E\delta
+o(\|\delta\|_2^2).
\end{aligned}
$$

海森矩阵局部 Lipschitz（利普希茨）连续时，最后的余项可加强为 $O(\|\delta\|_2^3)$。因此舍弃 $E$ 本身会引入二阶误差，不能把它归入三阶余项。

另一个等价的构造视角，是先线性化网络，再保留输出损失的二阶项：

$$
\begin{aligned}
f(\theta+\delta)&\approx z+J\delta,\\
\ell(z+J\delta;y)-\ell(z;y)
&\approx a^\top J\delta
+\frac12(J\delta)^\top C(J\delta).
\end{aligned}
$$

最终求解的局部二次模型为：

$$
\boxed{
\begin{aligned}
q(\delta)
&=g^\top\delta+\frac12\delta^\top B\delta\\
&=a^\top J\delta+\frac12(J\delta)^\top C(J\delta),\\
\delta^\star&\in\underset{\delta\in\mathcal C}
{\operatorname{argmin}}\ q(\delta).
\end{aligned}
}
$$

这不等于精确最小化原神经网络损失，也不保证单位半径内的真实损失变化始终被该模型准确预测。[^martens]

## 跨层关系没有被改成块对角近似

将雅可比矩阵按参数层分块：

$$
\begin{aligned}
J&=\begin{bmatrix}J_1&\cdots&J_K\end{bmatrix},\\
J_\ell&\in\mathbb R^{d\times p_\ell},\\
B_{\ell k}&=J_\ell^\top C J_k
\in\mathbb R^{p_\ell\times p_k},\\
B_{k\ell}&=B_{\ell k}^\top.
\end{aligned}
$$

于是：

$$
\boxed{
\begin{aligned}
q(\delta)
&=\sum_{\ell=1}^K\langle G_\ell,X_\ell\rangle_F\\
&\quad+\frac12\sum_{\ell=1}^K
\delta_\ell^\top B_{\ell\ell}\delta_\ell\\
&\quad+\sum_{1\le\ell<k\le K}
\delta_\ell^\top B_{\ell k}\delta_k.
\end{aligned}
}
$$

最后一行仍然属于二阶项。其中每对层的耦合为：

$$
\delta_\ell^\top B_{\ell k}\delta_k
=(J_\ell\delta_\ell)^\top C(J_k\delta_k).
$$

本文保留全部 $B_{\ell k}$，包括非对角块；不会再将 $B$ 替换为由对角块构成的矩阵。但原海森矩阵中属于 $E_{\ell k}$ 的跨层曲率已经随 $E$ 一起舍弃，因此不能说保留了原海森矩阵的所有信息。

## 不构造海森矩阵或跨层块的计算方式

对任意固定候选更新 $\delta$，按下列顺序计算：

$$
\boxed{
\begin{aligned}
u&=J\delta=\sum_{k=1}^KJ_k\delta_k\in\mathbb R^d,\\
v&=Cu\in\mathbb R^d,\\
w&=J^\top v=B\delta\in\mathbb R^p.
\end{aligned}
}
$$

第一个运算是 JVP（Jacobian–Vector Product，雅可比矩阵–向量积）；最后一个是 VJP（Vector–Jacobian Product，向量–雅可比矩阵积），采用列向量记法时写成 $J^\top v$。组合这两个自动微分原语即可计算广义高斯–牛顿矩阵与向量的乘积。[^martens] [^pytorch]

二次目标梯度可以合并为：

$$
\boxed{
\begin{aligned}
r(\delta)&=\nabla_\delta q(\delta)=g+B\delta\\
&=J^\top(a+CJ\delta),\\
R_\ell(X)
&=\operatorname{mat}_\ell(r_\ell(\delta))\\
&=G_\ell+\sum_{k=1}^K\mathcal B_{\ell k}[X_k],\\
\mathcal B_{\ell k}[X_k]
&=\operatorname{mat}_\ell
\left(B_{\ell k}\operatorname{vec}(X_k)\right).
\end{aligned}
}
$$

计算最后的 VJP 时，输入的 $a+Cu$ 应作为固定余切向量，而不是把整个表达式再次对当前网络参数求全导数。否则可能重新引入本来要舍弃的网络二阶导数。

整个内层求解期间，训练批次、基础参数 $\theta$、$z,a,C,J,g,B$ 均固定。改变的是候选更新 $\delta_t$，不是先真的更新网络再重新定义同一个二次模型。

### 平方损失示例

$$
\begin{aligned}
\ell(z;y)&=\frac12\|z-y\|_2^2,\\
a&=z-y,\\
C&=I_d,\\
B\delta&=J^\top(J\delta),\\
q(\delta)
&=\frac12\|z-y+J\delta\|_2^2
-\frac12\|z-y\|_2^2.
\end{aligned}
$$

### Softmax 交叉熵示例

对于单个分类输出，取 $z$ 为未归一化分数、$y$ 为总和为一的标签分布：

$$
\begin{aligned}
\pi_i&=\frac{e^{z_i}}{\sum_j e^{z_j}},\\
\ell(z;y)&=-y^\top z+\log\sum_j e^{z_j},\\
a&=\pi-y,\\
C&=\operatorname{diag}(\pi)-\pi\pi^\top,\\
Cu&=\pi\odot u-\pi(\pi^\top u).
\end{aligned}
$$

$\operatorname{diag}$ 表示 Diagonal Matrix（由向量构造的对角矩阵），$\odot$ 表示逐元素乘法。最后一式不需要构造输出空间中的稠密矩阵 $C$。其半正定性也可直接验证：

$$
\begin{aligned}
\bar u&=\sum_i\pi_i u_i,\\
u^\top Cu
&=\sum_i\pi_i u_i^2-\bar u^2\\
&=\sum_i\pi_i(u_i-\bar u)^2\ge0.
\end{aligned}
$$

多样本或多位置损失将这些输出曲率块按损失的实际归一化权重组合。

## 原约束问题的矩阵乘子

谱约束的等价形式为：

$$
\|X_\ell\|_2\le1
\quad\Longleftrightarrow\quad
X_\ell^\top X_\ell\preceq I_{n_\ell}.
$$

引入对称半正定矩阵乘子，构造拉格朗日函数：

$$
\begin{aligned}
\Lambda_\ell&\in\mathbb R^{n_\ell\times n_\ell},
\qquad\Lambda_\ell\succeq0,\\
\mathscr L(X,\Lambda)
&=q(\delta)+\frac12\sum_{\ell=1}^K
\operatorname{tr}\!\left[
\Lambda_\ell(X_\ell^\top X_\ell-I_{n_\ell})
\right],\\
\nabla_{X_\ell}\mathscr L
&=R_\ell(X)+X_\ell\Lambda_\ell.
\end{aligned}
$$

由于 $B\succeq0$，目标凸；约束集凸，且 $X_\ell=0$ 严格可行。Karush–Kuhn–Tucker（卡鲁什–库恩–塔克，KKT）条件在这里刻画全局最优解。凸约束最优性与投影固定点的关系可参见 Parikh 与 Boyd。[^proximal]

$$
\boxed{
\begin{aligned}
R_\ell(X^\star)+X_\ell^\star\Lambda_\ell^\star&=0,\\
X_\ell^{\star\top}X_\ell^\star&\preceq I_{n_\ell},\\
\Lambda_\ell^\star&\succeq0,\\
\Lambda_\ell^\star
(I_{n_\ell}-X_\ell^{\star\top}X_\ell^\star)&=0.
\end{aligned}
}
$$

矩阵互补条件与零迹互补条件等价。对半正定矩阵 $\Lambda,S$：

$$
\begin{aligned}
\operatorname{tr}(\Lambda S)
&=\|\Lambda^{1/2}S^{1/2}\|_F^2,\\
\operatorname{tr}(\Lambda S)=0
&\Longleftrightarrow\Lambda S=S\Lambda=0.
\end{aligned}
$$

单矩阵情况下，驻点条件也可写为：

$$
\left(B+\Lambda\otimes I_m\right)\operatorname{vec}(X)=-g.
$$

$\otimes$ 是 Kronecker Product（克罗内克积）。该系数矩阵可能奇异，因此不应无条件写成普通逆矩阵公式。下面不通过构造或求逆这个矩阵来求解。

## 用二次上界构造可解析的乘子子问题

令 $\beta>0$ 且 $\beta\ge\|B\|_2$。在固定候选更新 $\delta_t$ 处，记 $r_t=g+B\delta_t$。由于 $q$ 是二次函数，有精确恒等式：

$$
\begin{aligned}
q(\delta)
&=q(\delta_t)+r_t^\top(\delta-\delta_t)\\
&\quad+\frac12(\delta-\delta_t)^\top
B(\delta-\delta_t)\\
&\le q(\delta_t)+r_t^\top(\delta-\delta_t)
+\frac\beta2\|\delta-\delta_t\|_2^2.
\end{aligned}
$$

最小化这个与原目标在当前位置相切的上界：

$$
\begin{aligned}
\delta_{t+1}
&=\underset{\delta\in\mathcal C}{\operatorname{argmin}}
\left\{r_t^\top(\delta-\delta_t)
+\frac\beta2\|\delta-\delta_t\|_2^2\right\}\\
&=\underset{\delta\in\mathcal C}{\operatorname{argmin}}
\frac\beta2\left\|\delta-
\left(\delta_t-\frac1\beta r_t\right)\right\|_2^2.
\end{aligned}
$$

因此各层投影子问题可以分别计算：

$$
\boxed{
\begin{aligned}
Y_{\ell,t}&=X_{\ell,t}-\frac1\beta R_\ell(X_t),\\
X_{\ell,t+1}
&=\underset{X^\top X\preceq I_{n_\ell}}
{\operatorname{argmin}}
\frac\beta2\|X-Y_{\ell,t}\|_F^2.
\end{aligned}
}
$$

这正是投影梯度方法；上界中的各向同性二次项是求解器的构造，不是把原来的 $B$ 永久替换为 $\beta I_p$。完整的 $B\delta_t$ 仍在每轮梯度中。[^proximal]

## 谱范数球投影与乘子的闭式解

暂时省略层和迭代下标，考虑：

$$
\min_{X^\top X\preceq I_n}\frac\beta2\|X-Y\|_F^2,
\qquad Y\in\mathbb R^{m\times n}.
$$

令 $r=\min(m,n)$，将 $Y$ 的 SVD（Singular Value Decomposition，奇异值分解）写成：

$$
Y=U\Sigma V^\top,
\qquad
U\in\mathbb R^{m\times m},\quad
V\in\mathbb R^{n\times n},\quad
\Sigma\in\mathbb R^{m\times n}.
$$

$\Sigma$ 的对角元素为 $s_1,\ldots,s_r\ge0$。Frobenius 投影到这一谱约束集等价于投影奇异值；这里可以直接构造候选解并验证最优性。[^proximal]

令 $X$ 共享 $Y$ 的左右奇异向量，奇异值为 $x_i$，令子问题乘子 $\widehat\Lambda$ 在右奇异向量基下的特征值为 $\lambda_i$。标量条件为：

$$
\begin{aligned}
\beta(x_i-s_i)+x_i\lambda_i&=0,\\
0\le x_i\le1,\qquad\lambda_i&\ge0,\\
\lambda_i(1-x_i^2)&=0.
\end{aligned}
$$

分情况求解：

$$
\begin{aligned}
s_i\le1
&\Longrightarrow x_i=s_i,\quad\lambda_i=0,\\
s_i>1
&\Longrightarrow x_i=1,\quad\lambda_i=\beta(s_i-1).
\end{aligned}
$$

因此：

$$
\boxed{
\begin{aligned}
x_i&=\min(s_i,1),\\
\lambda_i&=\beta(s_i-1)_+,\\
\Pi_1(Y)&=\sum_{i=1}^r\min(s_i,1)u_i v_i^\top,\\
\widehat\Lambda
&=\beta\sum_{i=1}^r(s_i-1)_+v_i v_i^\top.
\end{aligned}
}
$$

$(b)_+=\max(b,0)$。未由这 $r$ 个方向覆盖的右零空间上，乘子为零。上述候选解满足矩阵驻点、可行性和互补条件；目标严格凸，故它就是唯一投影。

对称矩阵的正部定义为对其特征值取正部，而不是逐元素截断。于是乘子和投影可以写成：

$$
\boxed{
\begin{aligned}
\widehat\Lambda
&=\beta\left[(Y^\top Y)^{1/2}-I_n\right]_+,\\
\beta(X-Y)+X\widehat\Lambda&=0,\\
X&=\beta Y(\beta I_n+\widehat\Lambda)^{-1}.
\end{aligned}
}
$$

这里 $\beta I_n+\widehat\Lambda$ 确实正定。但下一节还会消去公式中的显式平方根、正部和求逆。

$\widehat\Lambda$ 是本轮投影子问题的乘子；在内迭代尚未收敛时，不能把它当作原二次问题的最优乘子。

## 将投影写成矩阵符号运算

本文的 $\operatorname{msign}$ 表示 Matrix Sign（矩阵符号）的奇异值版本，即极分解的规范偏等距因子：

$$
\begin{aligned}
M&=\sum_{\sigma_i>0}\sigma_i u_i v_i^\top,\\
\operatorname{msign}(M)
&=\sum_{\sigma_i>0}u_i v_i^\top,\\
\operatorname{msign}(0)&=0.
\end{aligned}
$$

零奇异值保持为零。在一般非对称方阵上，不应将这个定义混同于按特征值定义的传统矩阵符号函数。对称矩阵上，两者在非零谱处一致，并在本文中约定零特征值的符号为零。

### 对称中间矩阵形式

定义：

$$
\begin{aligned}
P&=\operatorname{msign}(Y),\\
A&=P^\top Y=(Y^\top Y)^{1/2},\\
S&=\operatorname{msign}(A-I_n).
\end{aligned}
$$

由标量恒等式：

$$
\begin{aligned}
\min(s,1)
&=s-\frac12(s-1)\bigl[1+\operatorname{sign}(s-1)\bigr],\\
(s-1)_+
&=\frac12(s-1)\bigl[1+\operatorname{sign}(s-1)\bigr],
\qquad s\ge0,
\end{aligned}
$$

逐个奇异方向代入，得到：

$$
\boxed{
\begin{aligned}
\Pi_1(Y)&=Y-\frac12(Y-P)(I_n+S),\\
\widehat\Lambda&=\frac\beta2(A-I_n)(I_n+S).
\end{aligned}
}
$$

在 $Y$ 的右零空间上，$A=0$、$S=-I$，所以这两式同样成立。

### 两次矩形矩阵符号形式

为了避免对一个较大的方阵再做符号运算，可以令：

$$
\begin{aligned}
P&=\operatorname{msign}(Y)\in\mathbb R^{m\times n},\\
D&=Y-P\in\mathbb R^{m\times n},\\
Q&=\operatorname{msign}(D)\in\mathbb R^{m\times n}.
\end{aligned}
$$

对 $s_i>0$ 的方向：

$$
\begin{aligned}
Y&:\ s_i,\\
P&:\ 1,\\
D&:\ s_i-1,\\
Q&:\ \operatorname{sign}(s_i-1),\\
DQ^\top P&:\ |s_i-1|.
\end{aligned}
$$

利用：

$$
\frac{s+1-|s-1|}{2}=\min(s,1),
$$

得到投影和乘子的另一组闭式：

$$
\boxed{
\begin{aligned}
\Pi_1(Y)
&=\frac12\left(Y+P-DQ^\top P\right),\\
\widehat\Lambda
&=\frac\beta2\left(P^\top D+D^\top Q\right).
\end{aligned}
}
$$

零奇异方向上，$Y,P,D,Q$ 均为零，所以不需要补齐奇异向量。$\widehat\Lambda$ 的非零谱在逐方向上为：

$$
\frac\beta2\left[(s_i-1)+|s_i-1|\right]
=\beta(s_i-1)_+.
$$

两次矩阵符号实现谱截断的等价表达式已有相关推导；这里同时写出了本问题对应的投影乘子。[^spectral]

实际迭代不需要显式计算 $\widehat\Lambda$。计算 $DQ^\top P$ 时，应选择较小的中间方阵：

$$
DQ^\top P=
\begin{cases}
(DQ^\top)P,&m\le n,\\
D(Q^\top P),&m>n.
\end{cases}
$$

## 用 Newton–Schulz 迭代近似矩阵符号

Newton–Schulz（牛顿–舒尔茨）迭代可用矩阵乘法逼近上述极分解因子。矩阵符号与谱截断的这种实现路线见 Cesista 的推导。[^spectral]

对于非零矩阵 $M\in\mathbb R^{m\times n}$，取：

$$
\begin{aligned}
Z_0&=\frac{M}{\|M\|_F},\\
Z_{j+1}
&=\frac12Z_j(3I_n-Z_j^\top Z_j)\\
&=\frac12(3I_m-Z_jZ_j^\top)Z_j.
\end{aligned}
$$

两种乘法顺序代数等价，选用较小的方阵。初始非零奇异值处于 $(0,1]$，每个奇异值独立满足：

$$
\begin{aligned}
s_{j+1}&=\frac12s_j(3-s_j^2),\\
s_{j+1}-s_j&=\frac12s_j(1-s_j^2)\ge0,\\
1-s_{j+1}&=\frac12(1-s_j)^2(s_j+2)\ge0.
\end{aligned}
$$

因此，非零奇异值单调趋于一，零奇异值始终为零：

$$
\boxed{\lim_{j\to\infty}Z_j=\operatorname{msign}(M).}
$$

$M=0$ 时直接返回零。对称输入的正负特征值分别趋于正负一，零特征值保持零。

这里必须区分精确恒等式与有限步实现：有限轮迭代得到的是近似符号函数，代回截断公式后不自动保证严格满足单位谱范数约束。特别是输入谱很大、非零奇异值很小或接近截断阈值时，近似误差与消减误差需要检查。[^spectral]

如果已经获得严格误差上界：

$$
\|\widetilde X-\Pi_1(Y)\|_2\le\varepsilon,
\qquad\varepsilon\ge0,
$$

则可以获得严格可行的修正：

$$
\left\|\frac{\widetilde X}{1+\varepsilon}\right\|_2\le1.
$$

不能把未验证的经验误差估计当作这个证书。后面的精确收敛证明假设使用精确投影；有限步近似实现需要额外控制误差。

## 固定点为何就是原二次问题的最优解

原二次模型在 $B\succeq0$ 时凸。凸集投影的最优性条件给出：

$$
\begin{aligned}
\delta^\star
&=\Pi_{\mathcal C}
\left(\delta^\star-\frac1\beta\nabla q(\delta^\star)\right)\\
&\Longleftrightarrow
\nabla q(\delta^\star)^\top(\delta-\delta^\star)\ge0,
\qquad\forall\delta\in\mathcal C\\
&\Longleftrightarrow
\delta^\star\in\underset{\delta\in\mathcal C}
{\operatorname{argmin}}\ q(\delta).
\end{aligned}
$$

投影子问题驻点条件为：

$$
\beta(X_{\ell,t+1}-Y_{\ell,t})
+X_{\ell,t+1}\widehat\Lambda_{\ell,t}=0.
$$

到达固定点时：

$$
\begin{aligned}
X_{\ell,t+1}=X_{\ell,t}=X_\ell^\star
&\Longrightarrow
\beta(X_\ell^\star-Y_\ell^\star)=R_\ell(X^\star),\\
R_\ell(X^\star)+X_\ell^\star\widehat\Lambda_\ell^\star&=0.
\end{aligned}
$$

加上投影的可行性和互补条件，即恢复原二次问题的全部最优性条件，而非仅求得一个无关上界问题的解。

原问题的最优乘子还可以从最终更新直接恢复：

$$
\begin{aligned}
R_\ell(X^\star)&=-X_\ell^\star\Lambda_\ell^\star,\\
X_\ell^{\star\top}X_\ell^\star\Lambda_\ell^\star
&=\Lambda_\ell^\star,\\
\Lambda_\ell^\star
&=-X_\ell^{\star\top}R_\ell(X^\star).
\end{aligned}
$$

最后一个等式只在最优性条件成立时给出正确的半正定乘子，不能对任意未收敛的候选更新直接使用。

## 单调下降与收敛

设使用精确投影、固定 $\beta>0$、$\beta\ge\|B\|_2$，并从可行点开始。记：

$$
 d_t=\delta_{t+1}-\delta_t.
$$

投影最优性条件中代入可行比较点 $\delta_t$：

$$
\begin{aligned}
(r_t+\beta d_t)^\top(\delta_t-\delta_{t+1})&\ge0,\\
r_t^\top d_t&\le-\beta\|d_t\|_2^2.
\end{aligned}
$$

因此：

$$
\boxed{
\begin{aligned}
q(\delta_{t+1})-q(\delta_t)
&=r_t^\top d_t+\frac12d_t^\top B d_t\\
&\le-\frac\beta2\|d_t\|_2^2\le0.
\end{aligned}
}
$$

约束集非空且紧，目标连续，所以最优解存在。进一步，对任一最优解 $\delta^\star$，上界不等式、凸性和投影最优性共同给出：

$$
\begin{aligned}
q(\delta_{t+1})-q(\delta^\star)
&\le r_t^\top(\delta_{t+1}-\delta^\star)
+\frac\beta2\|d_t\|_2^2\\
&\le\frac\beta2\left(
\|\delta_t-\delta^\star\|_2^2
-\|\delta_{t+1}-\delta^\star\|_2^2
\right).
\end{aligned}
$$

求和并使用目标单调性，得到：

$$
\boxed{
0\le q(\delta_T)-q(\delta^\star)
\le\frac{\beta\|\delta_0-\delta^\star\|_2^2}{2T},
\qquad T\ge1.
}
$$

取 $\delta_0=0$，还可由各层谱约束得到：

$$
\|\delta^\star\|_2^2
=\sum_{\ell=1}^K\|X_\ell^\star\|_F^2
\le\sum_{\ell=1}^K\min(m_\ell,n_\ell).
$$

这与标准投影梯度方法的收敛性质一致。[^proximal] 若 $B$ 奇异，最优解可能不唯一，但不需要对 $B$ 求逆；若目标严格凸，则最优解唯一。

可以使用投影梯度映射作为驻点检查：

$$
\begin{aligned}
\mathcal M_\beta(\delta)
&=\beta\left[
\delta-\Pi_{\mathcal C}
\left(\delta-\frac1\beta\nabla q(\delta)\right)
\right],\\
\mathcal M_\beta(\delta)=0
&\Longleftrightarrow\delta\text{ 是该凸二次问题的最优解}.
\end{aligned}
$$

## 不求最大特征值的步长选择

$\beta$ 是内层求解步长的倒数，不是约束乘子。无需先精确计算 $\|B\|_2$；可以采用标准回溯上界检验。[^proximal]

对当前正数 $\beta$ 计算试探投影，并令 $d=\delta_{t+1}-\delta_t$。接受条件为：

$$
\begin{aligned}
q(\delta_{t+1})
&\le q(\delta_t)+r_t^\top d+\frac\beta2\|d\|_2^2\\
&\Longleftrightarrow d^\top B d\le\beta\|d\|_2^2\\
&\Longleftrightarrow (Jd)^\top C(Jd)\le\beta\|d\|_2^2.
\end{aligned}
$$

不满足时增大 $\beta$，例如：

$$
\beta\leftarrow2\beta,
$$

然后重新计算试探投影。检验本身仍然不需要显式 $B$，但需要额外的雅可比矩阵–向量积。固定步长的收敛界不能直接将 $\beta$ 替换成任意变化的数列；回溯方法按其相应条件分析。

## 汇总后的全网络更新公式

初始化所有候选更新为零。在内层求解期间固定基础网络和数据，使用合适的 $\beta$；以下公式使用精确矩阵符号时，每轮都是精确谱投影：

$$
\boxed{
\begin{aligned}
X_{\ell,0}&=0,\\
\delta_t
&=\begin{bmatrix}
\operatorname{vec}(X_{1,t})\\ \vdots\\
\operatorname{vec}(X_{K,t})
\end{bmatrix},\\
u_t&=J\delta_t,\\
v_t&=a+Cu_t,\\
r_t&=J^\top v_t,\\
R_{\ell,t}&=\operatorname{mat}_\ell((r_t)_\ell),\\
Y_{\ell,t}&=X_{\ell,t}-\frac1\beta R_{\ell,t},\\
P_{\ell,t}&=\operatorname{msign}(Y_{\ell,t}),\\
D_{\ell,t}&=Y_{\ell,t}-P_{\ell,t},\\
Q_{\ell,t}&=\operatorname{msign}(D_{\ell,t}),\\
X_{\ell,t+1}
&=\frac12\left(
Y_{\ell,t}+P_{\ell,t}
-D_{\ell,t}Q_{\ell,t}^\top P_{\ell,t}
\right),\\
\|X_{\ell,t+1}\|_2&\le1,
\qquad\ell=1,\ldots,K.
\end{aligned}
}
$$

需要输出本轮投影乘子时，再计算：

$$
\boxed{
\widehat\Lambda_{\ell,t}
=\frac\beta2\left(
P_{\ell,t}^\top D_{\ell,t}
+D_{\ell,t}^\top Q_{\ell,t}
\right).
}
$$

停止内层求解后，再同时应用所有层的更新：

$$
W_\ell^{\mathrm{new}}=W_\ell+X_{\ell,T}.
$$

$T$ 是内层求解迭代数，$\mathrm{new}$ 表示更新后。单矩阵形式就是以上公式取 $K=1$。不需要独立迭代矩阵乘子，不需要构造完整海森矩阵，也不需要逐对存储跨层块。

## 各向同性曲率的闭式特例与一阶极限

对于单矩阵且 $B=hI_{mn}$、$h>0$：

$$
\begin{aligned}
q(X)
&=\langle G,X\rangle_F+\frac h2\|X\|_F^2\\
&=\frac h2\left\|X+\frac Gh\right\|_F^2
-\frac{\|G\|_F^2}{2h}.
\end{aligned}
$$

因此原问题本身一步可解：

$$
\boxed{
\begin{aligned}
G&=\sum_i\sigma_i u_i v_i^\top,\\
X^\star
&=\Pi_1(-G/h)
=-\sum_i\min(\sigma_i/h,1)u_i v_i^\top,\\
\Lambda^\star
&=\left[(G^\top G)^{1/2}-hI_n\right]_+.
\end{aligned}
}
$$

在固定 $G$ 下，令 $h\downarrow0$，得到一阶谱范数约束问题的一个最优解：

$$
\begin{aligned}
X^\star&\longrightarrow-\operatorname{msign}(G),\\
\min_{\|X\|_2\le1}\langle G,X\rangle_F
&=-\|G\|_*,\\
\|G\|_*&=\sum_i\sigma_i(G).
\end{aligned}
$$

$\|\cdot\|_*$ 是核范数。$G=0$ 时，线性目标的每个可行点都是最优点，零更新是其中一个选择。

一般的 $B=J^\top C J$ 并不具有各向同性结构，不能由这个特例推出“一次裁剪无约束高斯–牛顿步就是原约束问题的精确解”。本方法在一般情况下通过前述内迭代求解。

## 单位约束与实际学习率的区别

本文主问题把 $X_\ell$ 视为实际更新。如果将其改为单位谱范数的方向 $\Phi_\ell$，并另用固定学习率 $\alpha>0$：

$$
\begin{aligned}
\Delta W_\ell&=\alpha\Phi_\ell,
\qquad\|\Phi_\ell\|_2\le1,\\
\varphi&=
\begin{bmatrix}
\operatorname{vec}(\Phi_1)\\\vdots\\
\operatorname{vec}(\Phi_K)
\end{bmatrix},\\
q_\alpha(\varphi)
&=\alpha g^\top\varphi
+\frac{\alpha^2}{2}\varphi^\top B\varphi.
\end{aligned}
$$

此时应使用相应的一阶系数和二阶系数求解，或等价地将目标除以 $\alpha$：

$$
\underset{\|\Phi_\ell\|_2\le1}{\min}
\left\{g^\top\varphi+
\frac\alpha2\varphi^\top B\varphi\right\}.
$$

不能先按未缩放目标求最优解，再任意乘学习率，并仍将它称为缩放后二阶目标的精确最优解。单位谱范数约束也不是 Stiefel（斯蒂费尔）等式约束，不要求每个奇异值都等于一。

## 计算收益与边界

若显式构造完整参数海森矩阵，需要 $O(p^2)$ 存储。本方法不显式构造 $H$、$B$ 或 $J$，每轮曲率计算由一次网络雅可比矩阵–向量积、一次输出曲率乘积和一次向量–雅可比矩阵积组成；基础前向计算、缓存或重计算还会有额外开销。

投影若对每次矩阵符号执行 $s$ 轮上述迭代，各层的主要矩阵运算量级为：

$$
O\!\left(
\sum_{\ell=1}^K
s\,m_\ell n_\ell\min(m_\ell,n_\ell)
\right).
$$

两次矩阵符号不是两次矩阵乘法。舍弃 $E$ 避免了网络输出对参数的二阶微分，并使凸输出损失下的曲率半正定；但完整海森矩阵–向量积同样可以无矩阵实现，所以不能仅凭舍弃 $E$ 就断言相对于高效海森矩阵–向量积必然获得显著加速。实际收益取决于自动微分实现、网络结构、缓存、精度和内层迭代次数。[^pytorch]

上述全局最优性针对固定的广义高斯–牛顿二次子问题；它不等价于原非凸网络训练的全局最优性，也不构成未经实验验证的训练加速结论。

## 参考来源

[^martens]: James Martens. *New Insights and Perspectives on the Natural Gradient Method*. 2020，尤其是关于广义高斯–牛顿矩阵与曲率矩阵–向量积的章节。`https://arxiv.org/html/1412.1193`

[^proximal]: Neal Parikh and Stephen Boyd. *Proximal Algorithms*. 2014，关于投影梯度、固定点与谱函数近端运算的章节。`https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf`

[^pytorch]: PyTorch 官方教程，关于组合自动微分计算雅可比矩阵、海森矩阵及其向量积的说明。`https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html`

[^spectral]: Franz Louis Cesista. *Fast, Numerically Stable, and Auto-Differentiable Spectral Clipping via Newton-Schulz Iteration*. 2025，关于矩阵符号实现谱截断及近似实现的数值局限。`https://leloykun.github.io/ponder/spectral-clipping/`
