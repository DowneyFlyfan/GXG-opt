# 谱带约束下的一阶最速、有限步可行更新

这里“最速”采用两篇参考文章的定义：在原始权重空间内，以谱范数作为单位速度约束，使起点的一阶损失下降率最大。有限更新后的权重严格满足谱带约束，但不声称它是任意指定有限步长下、原非线性损失的全局最优终点。

参考文章：《流形上的最速下降：2. Muon + 正交》先求切向方向，再通过极分解回缩恢复严格正交；《流形上的最速下降：7. Stiefel的解析解》在方程（1）到（2）之间显式舍去二阶项。以下谱带构造是本文推导，不是这两篇的现成结论。

约定：
- 参数当前可行，矩阵满列秩，损失局部二阶连续可微，并且存在严格下降方向。
- 边界子空间包括该边界奇异值的全部重数。不存在的边界对应空项，直接删除。
- F 下标表示 Frobenius（弗罗贝尼乌斯）范数或内积；星号范数是核范数。
- msign 是 Matrix Sign，这里采用极分解部分等距因子的意义。修正梯度秩亏时，必须选择满足原约束的核范数次梯度，不能任意取一个极分解补全。
- 边界变化率矩阵 H 不是海森矩阵。
- 没有内部子空间时，删除包含 V_perp、ell_0、u_0 的条件。
- 下面所有量在单次更新开始时计算；时间下标在大部分公式中省略。

$$
\begin{equation}
\begin{aligned}
&W,G,\Phi,D\in\mathbb R^{n\times m},n\ge m,0<\rho<1,G=\nabla_W\mathcal L(W),\\
&a=\sqrt{1-\rho},b=\sqrt{1+\rho},C=W^\top W,\mathcal B=\{W:a^2I_m\preceq C\preceq b^2I_m\},\\
&W\in\mathcal B\iff\|W^\top W-I_m\|_2\le\rho,\|W\|_2\le b,\sigma_m(W)\ge a,\\
&\textbf{边界子空间：}C V_s=s^2V_s,V_s^\top V_s=I_{k_s},s\in\{a,b\},\\
&V_s\in\mathbb R^{m\times k_s},U_s=WV_s/s\in\mathbb R^{n\times k_s},U_s^\top U_s=I_{k_s},\\
&S_\Phi=W^\top\Phi+\Phi^\top W,S_\Phi^\top=S_\Phi,S_\Phi\in\mathbb R^{m\times m},\\
&(W-h\Phi)^\top(W-h\Phi)=C-hS_\Phi+h^2\Phi^\top\Phi,h\ge0,\\
&\mathcal D_W=\{\Phi:\|\Phi\|_2\le1,V_b^\top S_\Phi V_b\succeq0,V_a^\top S_\Phi V_a\preceq0\},\\
&\Phi^\star\in\arg\max_{\Phi\in\mathcal D_W}\langle G,\Phi\rangle_F,D=-\Phi^\star,\\
&v^\star=\langle G,\Phi^\star\rangle_F>0\Rightarrow\|\Phi^\star\|_2=\|D\|_2=1,\\
&\textbf{对偶求解：}\Lambda_s\in\mathbb R^{k_s\times k_s},\Lambda_s=\Lambda_s^\top\succeq0,\\
&N=V_b\Lambda_bV_b^\top-V_a\Lambda_aV_a^\top,Z=G+2WN\in\mathbb R^{n\times m},\\
&q_s=\operatorname{tr}(\Lambda_sV_s^\top S_\Phi V_s),q_b\ge0,q_a\le0,\Phi\in\mathcal D_W,\\
&\langle G,\Phi\rangle_F=\langle Z,\Phi\rangle_F-q_b+q_a\le\|Z\|_*,\Phi\in\mathcal D_W,\\
&\Phi_0=\epsilon W(V_bV_b^\top-V_aV_a^\top),0<\epsilon<b^{-1},\|\Phi_0\|_2<1,\\
&V_b^\top S_{\Phi_0}V_b=2\epsilon b^2I_{k_b},V_a^\top S_{\Phi_0}V_a=-2\epsilon a^2I_{k_a},\\
&(\Lambda_a^\star,\Lambda_b^\star)\in\arg\min_{\Lambda_a,\Lambda_b\succeq0}\|G+2WN\|_*,\\
&Z^\star=G+2W(V_b\Lambda_b^\star V_b^\top-V_a\Lambda_a^\star V_a^\top),\\
&\Phi^\star\in\partial\|Z^\star\|_*,\Phi^\star\in\mathcal D_W,q_a^\star=q_b^\star=0,\\
&\langle G,\Phi^\star\rangle_F=\langle Z^\star,\Phi^\star\rangle_F=\|Z^\star\|_*=v^\star,\\
&\operatorname{rank}(Z^\star)=m\Rightarrow\boxed{\Phi^\star=\operatorname{msign}(Z^\star)},\\
&\textbf{反解边界转动：}H_s=(U_s^\top DV_s+V_s^\top D^\top U_s)/2\in\mathbb R^{k_s\times k_s},\\
&2sH_s=V_s^\top(W^\top D+D^\top W)V_s=-V_s^\top S_{\Phi^\star}V_s,\\
&H_a\succeq0,H_b\preceq0,r_s=sD^\top U_s+W^\top DV_s-2sV_sH_s,\\
&(s^2I_m-C+V_sV_s^\top)B_s=r_s,B_s\in\mathbb R^{m\times k_s},V_s^\top B_s=0,\\
&(s^2I_m-C+V_sV_s^\top)x=s^2x-W^\top(Wx)+V_s(V_s^\top x),x\in\mathbb R^m,\\
&A_s=(DV_s+WB_s-U_sH_s)/s\in\mathbb R^{n\times k_s},U_s^\top A_s+A_s^\top U_s=0,\\
&DV_s+WB_s=sA_s+U_sH_s,D^\top U_s+W^\top A_s=sB_s+V_sH_s,\\
&U=[U_a,U_b],A=[A_a,A_b]\in\mathbb R^{n\times k},k=k_a+k_b,\\
&V=[V_a,V_b],B=[B_a,B_b]\in\mathbb R^{m\times k},U^\top U=V^\top V=I_k,\\
&F_{ts}=U_t^\top DV_s,V_t^\top B_s=(tF_{ts}+sF_{st}^\top)/(s^2-t^2),t\ne s,\\
&U_t^\top A_s=(sF_{ts}+tF_{st}^\top)/(s^2-t^2),U_s^\top A_s=(F_{ss}-F_{ss}^\top)/(2s),\\
&C_U=U^\top A=-A^\top U,C_V=V^\top B=-B^\top V,C_U,C_V\in\mathbb R^{k\times k},\\
&L=AU^\top-UA^\top-UC_UU^\top\in\mathbb R^{n\times n},L^\top=-L,\\
&R=BV^\top-VB^\top-VC_VV^\top\in\mathbb R^{m\times m},R^\top=-R,\\
&LU=A,RV=B,E=D-LW+WR\in\mathbb R^{n\times m},D=LW+E-WR,\\
&EV_s=U_sH_s,E^\top U_s=V_sH_s,W_0=W-\sum_{s\in\{a,b\}}sU_sV_s^\top,\\
&E_0=E-\sum_{s\in\{a,b\}}U_sH_sV_s^\top,W_0V=E_0V=0,W_0^\top U=E_0^\top U=0,\\
&W+hE=\sum_{s\in\{a,b\}}U_s(sI_{k_s}+hH_s)V_s^\top+W_0+hE_0,\\
&\textbf{可行步长：}V_\perp\in\mathbb R^{m\times(m-k)},V^\top V_\perp=0,\\
&V_\perp^\top V_\perp=I_{m-k},[V,V_\perp]^\top[V,V_\perp]=I_m,k<m,\\
&\ell_0=\sigma_{\min}(W_0V_\perp),u_0=\sigma_{\max}(W_0V_\perp),a<\ell_0\le u_0<b,\\
&h\|H_s\|_2\le b-a,h\|E_0\|_F\le\min(\ell_0-a,b-u_0),h>0,\\
&a\le\lambda_i(sI_{k_s}+hH_s)\le b,s\in\{a,b\},i=1,\ldots,k_s,\\
&\ell_0-h\|E_0\|_F\le\sigma_i((W_0+hE_0)V_\perp)\le u_0+h\|E_0\|_F,\\
&\textbf{可行曲线：}\mathcal C_h(K)=(I-hK/2)^{-1}(I+hK/2),K^\top=-K,\\
&x^\top(I-hK/2)x=\|x\|_2^2>0,x\ne0,\det(I-hK/2)\ne0,h\in\mathbb R,\\
&(I-hK/2)^\top=I+hK/2,(I-hK/2)(I+hK/2)=I-h^2K^2/4,\\
&\mathcal C_h(K)^\top\mathcal C_h(K)=I,\mathcal C_0(K)=I,\mathcal C'_0(K)=K,\\
&\boxed{W(h)=\mathcal C_h(L)(W+hE)\mathcal C_h(R)^\top},W(0)=W,\\
&\sigma_i(W(h))=\sigma_i(W+hE)\in[a,b],\|W(h)^\top W(h)-I_m\|_2\le\rho,\\
&W'(0)=LW+E-WR=D=-\Phi^\star,\|W'(0)\|_2=1,W(0)=W,\\
&\textbf{显式更新量：}T_h=\Phi^\star-h(LE-ER)/2+h^2LER/4\in\mathbb R^{n\times m},\\
&\boxed{Z_h=(I_n-hL/2)^{-1}T_h(I_m+hR/2)^{-1}},Z_0=\Phi^\star,\\
&(I_n-hL/2)(W-W(h))(I_m+hR/2)=hT_h,W(h)=W-hZ_h,h>0,\\
&\boxed{\Phi_h=Z_h/\|Z_h\|_2,\eta_h=h\|Z_h\|_2,W^+=W-\eta_h\Phi_h},\\
&\boxed{\|\Phi_h\|_2=1,\|W^+\|_2\le b,\|(W^+)^\top W^+-I_m\|_2\le\rho},\\
&\eta_h=h+O(h^2),\Phi_h=\Phi^\star+O(h),\mathcal L(W(h))=\mathcal L(W)-hv^\star+O(h^2),\\
&h_t\|Z_{h_t}\|_2=\eta_t,h_t=\eta_t+O(\eta_t^2),W_{t+1}=W_t-\eta_t\Phi_{h_t},\\
&\widetilde W(0)=W,\widetilde W(h)\in\mathcal B,\|\widetilde W'(0)\|_2\le1,\\
&-\widetilde W'(0)\in\mathcal D_W\Rightarrow\langle G,\widetilde W'(0)\rangle_F\ge-v^\star,\\
&\textbf{低秩化：}T_U=A-UC_U/2,F=[T_U,-U],J=[U,T_U]\in\mathbb R^{n\times2k},\\
&L=FJ^\top,\operatorname{rank}(L),\operatorname{rank}(R)\le2k,\\
&(I_n-hL/2)^{-1}=I_n+\frac h2F(I_{2k}-\frac h2J^\top F)^{-1}J^\top,\\
&\textbf{数值最速证书：}\Phi\in\mathcal D_W,\delta=\|Z\|_*-\langle G,\Phi\rangle_F\ge0,\\
&0\le v^\star-\langle G,\Phi\rangle_F\le\delta,\Lambda_a,\Lambda_b\succeq0,\\
&\epsilon>0,f_\epsilon(Z)=\operatorname{tr}((Z^\top Z+\epsilon^2I_m)^{1/2}),\\
&f_\epsilon(Z)=\|[Z^\top,\epsilon I_m]^\top\|_*,\epsilon>0,Z\in\mathbb R^{n\times m},\\
&P_\epsilon=Z(Z^\top Z+\epsilon^2I_m)^{-1/2},\|P_\epsilon\|_2\le1,\\
&\Lambda^\epsilon\in\arg\min_{\Lambda_a,\Lambda_b\succeq0}f_\epsilon(G+2WN),\epsilon>0,\\
&J_s=V_s^\top(W^\top P_\epsilon+P_\epsilon^\top W)V_s,\nabla_{\Lambda_b}f_\epsilon=J_b,\\
&\nabla_{\Lambda_a}f_\epsilon=-J_a,\langle\Lambda_s^\epsilon,J_s\rangle_F=0,s\in\{a,b\},\\
&J_b\succeq0,J_a\preceq0,Z=G+2WN^\epsilon,P_\epsilon\in\mathcal D_W,\\
&\langle G,P_\epsilon\rangle_F=\langle Z,P_\epsilon\rangle_F,f_\epsilon(Z)\ge\|Z\|_*,\\
&z_i=\sqrt{\sigma_i(Z)^2+\epsilon^2},z_i\ge\epsilon>0,i=1,\ldots,m,\\
&f_\epsilon(Z)-\langle Z,P_\epsilon\rangle_F=\sum_i\epsilon^2/z_i\le m\epsilon,\\
&0\le v^\star-\langle G,P_\epsilon\rangle_F\le m\epsilon,\|P_\epsilon\|_2\le1,\\
&\Phi_\epsilon=P_\epsilon/\|P_\epsilon\|_2\in\mathcal D_W,\|\Phi_\epsilon\|_2=1,\\
&0\le v^\star-\langle G,\Phi_\epsilon\rangle_F\le m\epsilon,\epsilon\downarrow0,\\
&A_\epsilon=Z^\top Z+\epsilon^2I_m,\nu\ge\|A_\epsilon\|_2,B_0=I_m/\sqrt\nu,\\
&B_{j+1}=B_j(3I_m-A_\epsilon B_j^2)/2,\lim_{j\to\infty}ZB_j=P_\epsilon.
\end{aligned}
\end{equation}
$$

## 最速性与严格可行性

线性化的边界条件先用于原权重空间的凸最速方向问题。严格可行点保证强对偶，互补条件使方向的下降值达到对偶上界。若最优下降值正，则齐次性保证最优方向的谱范数为一。

然后从该方向反解边界子空间的转动速度，而不是分别优化辅助变量。两组线性方程使剩余更新在边界子空间和内部子空间之间完全解耦；左右凯莱变换保持奇异值，显式步长界确保中间矩阵的每个奇异值仍在允许区间。整个公式不对权重做奇异值裁剪或事后归一化。

更新曲线的起始导数恰好等于已求出的最速方向的负值。因此它在全部单位初速度的可行曲线中达到最大的起始损失下降率。实际有限位移方向与切向方向只是一阶一致；这和参考文章的最速性定义相同。

若需要预先指定实际位移长度，只在允许的小步长范围内求解给出的一维长度方程。不能任意指定超过局部可行范围的大步长。

## 计算与近似

边界特征子空间可以用矩阵向量乘法和块特征值迭代跟踪，无须完整奇异值分解。边界内有重数时必须跟踪整个子空间。线性系统在相应边界的正交补上，经适当整体变号后为正定，可使用预条件共轭梯度法；收敛速度依赖边界与内部特征值之间的间隔。

左右旋转生成元的秩最多是边界总维数的两倍。伍德伯里恒等式把大矩阵求逆缩成该阶数的小矩阵求逆，不必显式形成大旋转矩阵。

矩阵平方根逆可用牛顿–舒尔茨迭代等数值方法计算。平滑核范数方案在其对偶问题精确求解时给出原始可行方向，并有明确的一阶最优值误差上界。再进行方向归一化不会破坏齐次的切锥约束，也不会降低正的下降值。

精确公式的硬约束保证属于实数精确运算。有限精度中，特征子空间误差、线性方程残差和凯莱小系统误差都必须控制；不能把固定少量迭代直接当作严格最优或零约束误差的证明。严格机器可验证的界需要保守谱界及误差证书。

当容差参数为零时，上下谱边界重合，应改用参考文章的 Stiefel（斯蒂费尔）专用公式，不能把两个不同边界的计算机械地重复使用。

这里给出的是数学构造与误差控制思路，不是已证明优于 Muon 的训练结果。用动量替代真实梯度时，“最速”相应变成对动量线性模型而言。
