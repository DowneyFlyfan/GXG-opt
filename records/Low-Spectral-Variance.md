# 带条件数约束的谱范数一阶最速下降

适用条件：权重满列秩、最大与最小奇异值单重、条件数上限大于一、损失局部二阶连续可微，并且当前点不是约束一阶驻点。对偶部分先写条件数边界的情形；内部情形删除最小奇异值的一阶约束并令对应乘子为零。

结论：起始切向方向在原权重空间中严格一阶最速；足够小的有限步通过直接可行曲线保持权重谱范数与条件数约束。实际有限位移的归一化方向与切向方向一阶一致，但不声称它是指定有限步长下的全局最优方向。

`msign` 沿用 Muon 的惯例，表示极分解的部分等距因子；零奇异值映为零。秩亏时，必须使用核范数的完整次梯度集合选取可行最优方向。

$$
\begin{equation}
\begin{aligned}
&W,G,\Phi\in\mathbb{R}^{n\times m},n\ge m\ge3,c>1,a=c^{-1},G=\nabla_W\mathcal L(W),\\
&1=\sigma_1>\sigma_2\ge\cdots\ge\sigma_{m-1}>\sigma_m=s\ge a,\mathcal I=\{1,m\},\\
&Wv_j=\sigma_j u_j,W^\top u_j=\sigma_jv_j,u_j\in\mathbb{R}^n,v_j\in\mathbb{R}^m,\\
&\|u_j\|_2=\|v_j\|_2=1,d\sigma_j=u_j^\top(dW)v_j,j\in\mathcal I,\\
&A=u_1v_1^\top,B=u_mv_m^\top\in\mathbb{R}^{n\times m},\langle A,B\rangle_F=0,\\
&\mathcal A_W=\{\Phi\in\mathbb R^{n\times m}:\|\Phi\|_2\le1,\langle A,\Phi\rangle_F=0\},\\
&\mathcal D_W=\mathcal A_W\cap\{\Phi:\langle B,\Phi\rangle_F\le0\},s=a,\\
&s>a:\mathcal D_W=\mathcal A_W,\mu^\star=0,\sigma_m(W)>c^{-1},\\
&v^\star=\max_{\Phi\in\mathcal D_W}\langle G,\Phi\rangle_F>0,D=-\Phi^\star,\|D\|_2=1,\\
&\textbf{条件数边界：}s=a,\lambda\in\mathbb{R},\mu\ge0,Z_{\lambda,\mu}=G+\lambda A-\mu B,\\
&\langle G,\Phi\rangle_F=\langle Z_{\lambda,\mu},\Phi\rangle_F+\mu\langle B,\Phi\rangle_F,\\
&\langle G,\Phi\rangle_F\le\|Z_{\lambda,\mu}\|_*,\Phi\in\mathcal D_W,\lambda\in\mathbb R,\mu\ge0,\\
&\Phi_0=-\epsilon B,0<\epsilon<1:\|\Phi_0\|_2<1,\langle A,\Phi_0\rangle_F=0,\\
&\langle B,\Phi_0\rangle_F=-\epsilon<0,\Phi_0\in\mathcal D_W,\|\Phi_0\|_2<1,\\
&\boxed{v^\star=\min_{\lambda\in\mathbb R,\mu\ge0}\|G+\lambda A-\mu B\|_*},s=a,\\
&Z^\star=G+\lambda^\star A-\mu^\star B,\Phi^\star\in\partial\|Z^\star\|_*,\\
&\langle A,\Phi^\star\rangle_F=0,\langle B,\Phi^\star\rangle_F\le0,\Phi^\star\in\mathcal D_W,\\
&\mu^\star\langle B,\Phi^\star\rangle_F=0,\langle G,\Phi^\star\rangle_F=\|Z^\star\|_*=v^\star,\\
&\operatorname{rank}(Z^\star)=m\Rightarrow\boxed{\Phi^\star=\operatorname{msign}(Z^\star)},\\
&Z^\star=U_r\Sigma_rV_r^\top,U_r\in\mathbb R^{n\times r},V_r\in\mathbb R^{m\times r},\\
&r=\operatorname{rank}(Z^\star),U_r^\top U_r=V_r^\top V_r=I_r,\Sigma_r\succ0,\\
&\Phi^\star=U_rV_r^\top+T,U_r^\top T=0,TV_r=0,\|T\|_2\le1,T\in\mathbb R^{n\times m},\\
&\mathcal G(\lambda,\mu,\Phi)=\|Z_{\lambda,\mu}\|_*-\langle G,\Phi\rangle_F,\Phi\in\mathcal D_W,\\
&0\le v^\star-\langle G,\Phi\rangle_F\le\mathcal G(\lambda,\mu,\Phi),\lambda\in\mathbb R,\mu\ge0,\\
&\textbf{乘子求解：}P_k\in\partial\|Z_{\lambda_k,\mu_k}\|_*,P_k\in\mathbb R^{n\times m},\\
&p_k=\langle A,P_k\rangle_F,q_k=\langle B,P_k\rangle_F,|p_k|\le1,|q_k|\le1,\\
&\lambda_{k+1}=\lambda_k-\rho_kp_k,\mu_{k+1}=\max(0,\mu_k+\rho_kq_k),\\
&\rho_k>0,\sum_k\rho_k=\infty,\sum_k\rho_k^2<\infty,s>a\Rightarrow\mu_k=0,\\
&\textbf{矩阵乘法求极分解：}X_0=Z/\|Z\|_F,X_{k+1}=X_k(3I_m-X_k^\top X_k)/2,\\
&f(x)=x(3-x^2)/2,f(0)=0,f(1)=1,f'(x)=3(1-x^2)/2\ge0,x\in[0,1],\\
&p_{i,0}=\sigma_i(Z)/\|Z\|_F,p_{i,k+1}=f(p_{i,k}),0\le p_{i,k}\le1,\\
&Z\ne0,X_k=U\operatorname{diag}(p_{i,k})V^\top,f(x)-x=x(1-x^2)/2\ge0,\\
&0\le\|Z\|_*-\langle Z,X_k\rangle_F=\sum_i\sigma_i(Z)(1-p_{i,k}),\\
&\sum_i\sigma_i(Z)(1-p_{i,k})\le\sum_i\sigma_i(Z)(1-p_{i,k}^2),\\
&\|Z\|_*\le\langle Z,X_k\rangle_F+\sqrt m\|Z-ZX_k^\top X_k\|_F,\\
&\textbf{为何不能直走：}\|(W+hD)v_1\|_2^2=1+h^2\|Dv_1\|_2^2,\langle A,D\rangle_F=0,\\
&\textbf{反解转动速度：}d_j=u_j^\top Dv_j,d_1=0,s=a\Rightarrow d_m\ge0,\\
&a_j=\dot u_j\in\mathbb R^n,b_j=\dot v_j\in\mathbb R^m,u_j^\top a_j=v_j^\top b_j=0,\\
&Dv_j+Wb_j=d_ju_j+\sigma_ja_j,D^\top u_j+W^\top a_j=d_jv_j+\sigma_jb_j,\\
&r_j=\sigma_jD^\top u_j+W^\top Dv_j-2\sigma_jd_jv_j\in\mathbb R^m,j\in\mathcal I,\\
&(\sigma_j^2I_m-W^\top W+v_jv_j^\top)b_j=r_j,b_j\in\mathbb R^m,v_j^\top b_j=0,\\
&a_j=(Dv_j+Wb_j-d_ju_j)/\sigma_j\in\mathbb R^n,u_j^\top a_j=0,j\in\mathcal I,\\
&U=[u_1,u_m],\dot U=[a_1,a_m]\in\mathbb R^{n\times2},U^\top U=I_2,\\
&V=[v_1,v_m],\dot V=[b_1,b_m]\in\mathbb R^{m\times2},V^\top V=I_2,\\
&C_U=U^\top\dot U=-\dot U^\top U,C_V=V^\top\dot V=-\dot V^\top V,\\
&L=\dot U U^\top-U\dot U^\top-UC_UU^\top\in\mathbb R^{n\times n},L^\top=-L,\\
&R=\dot V V^\top-V\dot V^\top-VC_VV^\top\in\mathbb R^{m\times m},R^\top=-R,\\
&LU=\dot U,RV=\dot V,E=D-LW+WR\in\mathbb R^{n\times m},\\
&Ev_j=d_ju_j,E^\top u_j=d_jv_j,E_0=E-U\operatorname{diag}(0,d_m)V^\top,\\
&E_0V=0,E_0^\top U=0,W_0=W-U\operatorname{diag}(1,s)V^\top,\\
&W+hE=U\operatorname{diag}(1,s+hd_m)V^\top+W_0+hE_0,h\ge0,\\
&\textbf{直接可行更新：}\mathcal C_h(K)=(I-hK/2)^{-1}(I+hK/2),K^\top=-K,\\
&x^\top(I-hK/2)x=\|x\|_2^2>0,x\ne0,(I-hK/2)(I+hK/2)=I-h^2K^2/4,\\
&\mathcal C_h(K)^\top=\mathcal C_h(K)^{-1},\mathcal C_0(K)=I,\dot{\mathcal C}_0(K)=K,\\
&\boxed{W(h)=\mathcal C_h(L)(W+hE)\mathcal C_h(R)^\top},W(0)=W,\\
&\dot W(0)=LW+E-WR=D=-\Phi^\star,\operatorname{rank}(L),\operatorname{rank}(R)\le4,\\
&\delta=\min(1-\sigma_2,\sigma_{m-1}-s)>0,e=\|E_0\|_F,h(e+|d_m|)<\delta/2,\\
&s+hd_m\ge a,\sigma_{m-1}-he>s+hd_m,\sigma_2+he<1,\\
&V^\top x=0,\|x\|_2=1:\sigma_{m-1}-he\le\|(W_0+hE_0)x\|_2\le\sigma_2+he,\\
&\boxed{\sigma_1(W(h))=1,\sigma_m(W(h))=s+hd_m,\kappa(W(h))\le c},\\
&\textbf{实际单位更新：}F(h)=\|W(h)-W\|_2=h+O(h^2),F(h_t)=\eta_t>0,\\
&\boxed{\Phi_t=\frac{W-\mathcal C_{h_t}(L)(W+h_tE)\mathcal C_{h_t}(R)^\top}{\eta_t}},\\
&h_t=\eta_t+O(\eta_t^2),W_{t+1}=W-\eta_t\Phi_t,\|\Phi_t\|_2=1,\\
&\boxed{\|W_{t+1}\|_2=1,\kappa(W_{t+1})\le c},\Phi_t=\Phi^\star+O(\eta_t),\\
&\mathcal L(W_{t+1})=\mathcal L(W)-\eta_tv^\star+O(\eta_t^2),v^\star>0,\\
&\mathcal C_c=\{X:\|X\|_2=1,\kappa(X)\le c\},Y(0)=W,\|Y'(0)\|_2\le1,\\
&Y(h)\in\mathcal C_c\Rightarrow-Y'(0)\in\mathcal D_W,\|Y'(0)\|_2\le1,\\
&\frac{d}{dh}\mathcal L(Y(h))|_{h=0}=\langle G,Y'(0)\rangle_F\ge-v^\star,\\
&\textbf{低秩计算：}F_U=[U,\dot U]\in\mathbb R^{n\times4},L=F_UJ_UF_U^\top,\\
&J_U=\begin{bmatrix}-C_U&-I_2\\I_2&0\end{bmatrix}\in\mathbb R^{4\times4},J_U^\top=-J_U,\\
&\mathcal C_h(L)=I_n+hF_U(I_4-hJ_UF_U^\top F_U/2)^{-1}J_UF_U^\top.
\end{aligned}
\end{equation}
$$

注意：有限轮数值迭代只给近似解。严格保证要求准确的极端奇异子空间与子问题解，或者带可靠误差界的数值计算；不能把固定少量 Newton–Schulz 迭代称为精确解。

