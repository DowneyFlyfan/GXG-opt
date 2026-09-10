# Stiefel 约束下的最速特征下降

本文件的主公式适用于：单个线性层 Y=WX，输入 X 固定，
权重 W 满足列正交约束，输入二阶矩正定，特征变化使用 Frobenius 范数。
它解的是给定有限特征变化预算下的线性化损失最小化问题，
不是任意非线性损失的有限步全局最优化。

在极分解输入满列秩且标量预算方程可达的分支上，公式同时保证：
参数严格列正交、实际特征预算满足、线性化目标全局最优。
非驻点时，充分小的预算可达。

一般网络特征的公式在末尾；它提供切空间内的一阶最速方向，
再由精确回缩保持 Stiefel 约束，不享有单层公式的有限预算全局最优结论。
数值近似的精度应由正交误差、预算误差与最优性残差核验。

$$
\begin{equation}
\begin{aligned}
&W,G,Z\in\mathbb R^{n\times m},n\ge m,X\in\mathbb R^{m\times b},Y=WX\in\mathbb R^{n\times b},\\
&W^\top W=I_m,G=\nabla_W\mathcal L(W),C=XX^\top/b\in\mathbb R^{m\times m},C\succ0,\\
&E_Y=\nabla_Y\mathcal L\in\mathbb R^{n\times b},d\mathcal L=\langle E_Y,dW X\rangle_F,\\
&G=E_YX^\top,\widetilde Y_\alpha=WX-\alpha bE_Y\in\mathbb R^{n\times b},\alpha>0,\\
&d_X(Z,W)=\|(Z-W)X\|_F/\sqrt b,\mathcal S=\{Z\in\mathbb R^{n\times m}:Z^\top Z=I_m\},\\
&Z_\eta^\star\in\arg\min_{Z\in\mathcal S,\ d_X(Z,W)\le\eta}\langle G,Z-W\rangle_F,\eta>0,\\
&\mathcal L(Z)=\mathcal L(W)+\langle G,Z-W\rangle_F+O(\|Z-W\|_F^2),\\
&\textbf{关键恒等式：}d_X(Z,W)^2=\operatorname{tr}((Z-W)C(Z-W)^\top),Z\in\mathcal S,\\
&d_X(Z,W)^2=\operatorname{tr}(CZ^\top Z)+\operatorname{tr}(CW^\top W)-2\langle WC,Z\rangle_F,\\
&\boxed{d_X(Z,W)^2=2\operatorname{tr}(C)-2\langle WC,Z\rangle_F},Z,W\in\mathcal S,\\
&\textbf{引入标量：}\alpha>0,\Psi_\alpha(Z)=\langle G,Z-W\rangle_F+d_X(Z,W)^2/(2\alpha),\\
&\Psi_\alpha(Z)=\|ZX-\widetilde Y_\alpha\|_F^2/(2\alpha b)-\alpha b\|E_Y\|_F^2/2,\\
&\alpha\Psi_\alpha(Z)=\operatorname{tr}(C)-\alpha\langle G,W\rangle_F-\langle A_\alpha,Z\rangle_F,\\
&A_\alpha=WC-\alpha G,Z_\alpha\in\arg\max_{Z\in\mathcal S}\langle A_\alpha,Z\rangle_F,\\
&A_\alpha=U\Sigma V^\top,U\in\mathbb R^{n\times m},V\in\mathbb R^{m\times m},\Sigma\succ0,\\
&U^\top U=V^\top V=I_m,\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_m),\sigma_i>0,\\
&\langle A_\alpha,Z\rangle_F=\sum_{i=1}^m\sigma_i u_i^\top Zv_i\le\sum_{i=1}^m\sigma_i,\\
&u_i^\top Zv_i\le\|u_i\|_2\|Zv_i\|_2=1,Z=UV^\top\Rightarrow u_i^\top Zv_i=1,\\
&B_\alpha=(A_\alpha^\top A_\alpha)^{-1/2}\in\mathbb R^{m\times m},B_\alpha^\top=B_\alpha,\\
&\boxed{Z_\alpha=A_\alpha B_\alpha=\operatorname{polar}(WC-\alpha G)},Z_\alpha^\top Z_\alpha=I_m,\\
&Z_\alpha^\top Z_\alpha=B_\alpha A_\alpha^\top A_\alpha B_\alpha=I_m,\alpha>0,\\
&\textbf{确定预算：}d_X(Z_{\alpha^\star},W)=\eta,\alpha^\star>0,Z_\eta^\star=Z_{\alpha^\star},\\
&\boxed{\Phi_\eta^\star=\frac{W-(WC-\alpha^\star G)B_{\alpha^\star}}{\eta}},\eta>0,\\
&W^+=W-\eta\Phi_\eta^\star=Z_{\alpha^\star},\Phi_\eta^\star\in\mathbb R^{n\times m},\\
&\boxed{(W^+)^\top W^+=I_m,\|\Phi_\eta^\star X\|_F/\sqrt b=1},\eta=d_X(W^+,W),\\
&\textbf{全局最优证书：}\langle A_{\alpha^\star},Z_{\alpha^\star}-Z\rangle_F\ge0,Z\in\mathcal S,\\
&\alpha^\star\langle G,Z_{\alpha^\star}-Z\rangle_F\le\langle WC,Z_{\alpha^\star}-Z\rangle_F,\\
&\langle WC,Z_{\alpha^\star}-Z\rangle_F=(d_X(Z,W)^2-\eta^2)/2\le0,d_X(Z,W)\le\eta,\\
&\boxed{\langle G,Z_\eta^\star-W\rangle_F\le\langle G,Z-W\rangle_F},d_X(Z,W)\le\eta,\\
&0<\alpha_1<\alpha_2,d_i=d_X(Z_{\alpha_i},W),\ell_i=\langle G,Z_{\alpha_i}-W\rangle_F,\\
&\ell_1+d_1^2/(2\alpha_1)\le\ell_2+d_2^2/(2\alpha_1),i\in\{1,2\},\\
&\ell_2+d_2^2/(2\alpha_2)\le\ell_1+d_1^2/(2\alpha_2),0<\alpha_1<\alpha_2,\\
&(\alpha_1^{-1}-\alpha_2^{-1})(d_1^2-d_2^2)\le0\Rightarrow d_1\le d_2,\\
&A_0=WC,\operatorname{polar}(WC)=W,d_X(Z_0,W)=0,C\succ0,W^\top W=I_m,\\
&\sigma_m(A_\alpha)\ge\lambda_{\min}(C)-\alpha\|G\|_2>0,\alpha\|G\|_2<\lambda_{\min}(C),\\
&A_\alpha=Z_\alpha P_\alpha,P_\alpha=(A_\alpha^\top A_\alpha)^{1/2},Z_0=W,P_0=C,\\
&-G=\dot Z_0C+W\dot P_0,\dot P_0^\top=\dot P_0,W^\top\dot Z_0+\dot Z_0^\top W=0,\\
&\dot Z_0=0\Rightarrow G=W\Lambda,\Lambda^\top=\Lambda,\Lambda=-\dot P_0,\\
&G\notin\{W\Lambda:\Lambda^\top=\Lambda\}\Rightarrow s_0=\|\dot Z_0X\|_F/\sqrt b>0,\\
&d_X(Z_\alpha,W)=\alpha s_0+O(\alpha^2),\alpha^\star=\eta/s_0+O(\eta^2),\eta\downarrow0,\\
&\textbf{避免重复大矩阵乘法：}E=W^\top G,T=G^\top G\in\mathbb R^{m\times m},\\
&A_\alpha^\top A_\alpha=C^2-\alpha(CE+E^\top C)+\alpha^2T\in\mathbb R^{m\times m},\\
&d_X(Z_\alpha,W)^2=2\operatorname{tr}(C)-2\operatorname{tr}(C(C-\alpha E)B_\alpha),\\
&\textbf{阻尼会改变度量：}C_\delta=C+\delta I_m,\delta>0,d_\delta^2=d_X^2+\delta\|Z-W\|_F^2,\\
&\textbf{近似最优性证书：}\widehat Z^\top\widehat Z=I_m,d_X(\widehat Z,W)=\eta,\alpha>0,\\
&\epsilon=\|A_\alpha\|_*-\langle A_\alpha,\widehat Z\rangle_F\ge0,\widehat Z\in\mathcal S,\\
&0\le\langle G,\widehat Z-Z_\eta^\star\rangle_F\le\epsilon/\alpha,\eta=d_X(\widehat Z,W),\\
&K=(\widehat Z^\top A_\alpha+A_\alpha^\top\widehat Z)/2\succeq0,R=A_\alpha-\widehat ZK,\\
&\|A_\alpha\|_*\le\|\widehat ZK\|_*+\|R\|_*\le\operatorname{tr}(K)+\sqrt m\|R\|_F,\\
&\operatorname{tr}(K)=\langle A_\alpha,\widehat Z\rangle_F\Rightarrow\epsilon\le\sqrt m\|R\|_F,\\
&\textbf{一般网络特征：}\mathscr J(D)=DF(W)[D]\in\mathbb R^p,D\in\mathbb R^{n\times m},\\
&\mathcal T_W=\{D:W^\top D+D^\top W=0\},\Pi_W(Z)=Z-W(W^\top Z+Z^\top W)/2,\\
&\mathscr A(D)=\Pi_W(\mathscr J^*\mathscr J(D)),\mathscr A:\mathcal T_W\to\mathcal T_W,\\
&\mathscr A\succ0,\mathscr A(H)=\Pi_W(G),H\in\mathcal T_W,\Phi=H/\|\mathscr J(H)\|_2,\\
&\langle G,D\rangle_F=\langle\mathscr J(H),\mathscr J(D)\rangle,D\in\mathcal T_W,\\
&\langle G,D\rangle_F\le\|\mathscr J(H)\|_2\|\mathscr J(D)\|_2,D\in\mathcal T_W,\\
&\boxed{\Phi\in\arg\max_{D\in\mathcal T_W,\ \|\mathscr J(D)\|_2\le1}\langle G,D\rangle_F},H\ne0,\\
&\widehat W=W-\tau\Phi,\widehat W^\top\widehat W=I_m+\tau^2\Phi^\top\Phi,\tau>0,\\
&W^+=\widehat W(I_m+\tau^2\Phi^\top\Phi)^{-1/2},(W^+)^\top W^+=I_m,\dot W^+(0)=-\Phi.
\end{aligned}
\end{equation}
$$

## 计算与适用范围

- polar 表示列正交极分解因子。证明中用到奇异值分解，不代表实现必须求奇异值分解。
- 标量 alpha 固定时只需求一次极分解；输出对本次实际产生的特征预算仍是线性化最优。
- 预先指定特征预算 eta 时，在连续满秩分支上求单调标量方程。
- 不可把任意正交—上三角分解的正交因子当作极分解因子：前者可保正交，但未必最优。
- 输入二阶矩加入阻尼或采用对角近似，会改变最速下降所依据的度量。
- 列正交且 Y=WX 是有限步化简的必要适用条件之一。行正交层不能不加分析地照搬。
- 一般特征雅可比系统可通过自动微分的雅可比—向量及向量—雅可比乘积，
  在切空间上使用共轭梯度求解，无须构造完整雅可比。
- 所有严格代数保证均指精确运算；实现中应注明求解容差。
