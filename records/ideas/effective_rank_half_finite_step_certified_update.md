$$
\begin{equation}
\begin{aligned}
&W,G,\Phi,Y\in\mathbb R^{m\times n},r=\min(m,n),G=\nabla_W\mathcal L(W),\\
&m\ge n=r\ge3,\chi=\sqrt{r/2},W\ne0,0<\eta<\|W\|_F/\sqrt r,\\
&a(Y)=\|Y\|_F^2,b(Y)=\|Y^\top Y\|_F^2,\mathcal E(Y)=a(Y)^2/(rb(Y)),\\
&a(Y)^2-b(Y)=2\sum_{i<j}\sigma_i(Y)^2\sigma_j(Y)^2\ge0,\mathcal E(Y)\ge1/r,\\
&f(Y)=\chi\sqrt{b(Y)}-a(Y),Y\ne0:\mathcal E(Y)\ge1/2\iff f(Y)\le0,\\
&\mathcal E(W)\ge1/2,\mathcal B=\{\Phi\in\mathbb R^{m\times n}:\|\Phi\|_2\le1\},\\
&Y_\Phi=W-\eta\Phi,\|Y_\Phi\|_F\ge\|W\|_F-\eta\sqrt r>0,\Phi\in\mathcal B,\\
&v^\star=\max_{\Phi\in\mathcal B,f(Y_\Phi)\le0}\langle G,\Phi\rangle_F,\Phi=0\textbf{可行},\\
&\textbf{非凸反例：}r=4,X_\pm=\operatorname{diag}(1+t,1-t,\pm2t,0),0<t<1,\\
&a(X_\pm)=2+6t^2,b(X_\pm)=2+12t^2+18t^4,\mathcal E(X_\pm)=1/2,\\
&\bar X=(X_++X_-)/2,\mathcal E(\bar X)=\frac{(2+2t^2)^2}{4(2+12t^2+2t^4)}<1/2,\\
&\textbf{凸性：}q(Y)=\sqrt{b(Y)}=\|Y^\top Y\|_F,C=Y^\top Y\succeq0,\\
&q(Y)=\max_{S\succeq0,\|S\|_F\le1}\langle S,C\rangle_F,S\in\mathbb R^{r\times r},\\
&\langle S,C\rangle_F\le\|S\|_F\|C\|_F,S=C/\|C\|_F\textbf{达到等号},\\
&\langle S,Y^\top Y\rangle_F=\|YS^{1/2}\|_F^2,S\succeq0\Rightarrow q\textbf{为凸函数},\\
&db(Y)=4\langle Y(Y^\top Y),dY\rangle_F,\nabla q(Y)=2Y(Y^\top Y)/q(Y),\\
&\textbf{保留有限步：}a(Y_\Phi)=a(W)-2\eta\langle W,\Phi\rangle_F+\eta^2\|\Phi\|_F^2,\\
&c_\eta(\Phi)=\chi q(Y_\Phi)-a(W)+2\eta\langle W,\Phi\rangle_F-\eta^2r,\\
&\boxed{f(Y_\Phi)=c_\eta(\Phi)+\eta^2(r-\|\Phi\|_F^2)},\Phi\in\mathcal B,\\
&\|\Phi\|_F^2\le r\Rightarrow c_\eta(\Phi)\le f(Y_\Phi),c_\eta\textbf{为凸函数},\\
&v_{\mathrm{up}}=\max_{\Phi\in\mathcal B,c_\eta(\Phi)\le0}\langle G,\Phi\rangle_F\ge v^\star,\\
&c_\eta(0)=f(W)-\eta^2r<0,\|0\|_2<1,\textbf{凸问题满足严格可行性},\\
&\mathscr L_\mu(\Phi)=\langle G,\Phi\rangle_F-\mu c_\eta(\Phi)/\eta,\mu\ge0,\\
&d(\mu)=\max_{\Phi\in\mathcal B}\mathscr L_\mu(\Phi),v_{\mathrm{up}}=\min_{\mu\ge0}d(\mu),\\
&B_\mu(\Phi)=G+2\mu[\chi Y_\Phi(Y_\Phi^\top Y_\Phi)/q(Y_\Phi)-W],\\
&\nabla_\Phi\mathscr L_\mu=B_\mu(\Phi),B_\mu(\Phi)\in\mathbb R^{m\times n},\\
&\Phi_\mu\in\arg\max_{\Phi\in\mathcal B}\mathscr L_\mu(\Phi),B_\mu=B_\mu(\Phi_\mu),\\
&\langle B_\mu,\Psi-\Phi_\mu\rangle_F\le0,\Psi\in\mathcal B,\Phi_\mu\in\partial\|B_\mu\|_*,\\
&B_\mu=U\Sigma V^\top,U\in\mathbb R^{m\times r},V\in\mathbb R^{r\times r},\Sigma\succ0,\\
&\langle B_\mu,\Psi\rangle_F=\sum_i\sigma_i u_i^\top\Psi v_i\le\sum_i\sigma_i,\Psi\in\mathcal B,\\
&\Psi=UV^\top\Rightarrow\langle B_\mu,\Psi\rangle_F=\sum_i\sigma_i,\Psi^\top\Psi=I_r,\\
&\boxed{\Phi_\mu=\operatorname{polar}(B_\mu)},\operatorname{rank}(B_\mu)=r,\\
&\mu^\star\ge0,c_\eta(\Phi_{\mu^\star})\le0,\mu^\star c_\eta(\Phi_{\mu^\star})=0,\\
&\boxed{W^+=W-\eta\Phi_{\mu^\star}},\Phi_{\mu^\star}=\operatorname{polar}(B_{\mu^\star}),\\
&\operatorname{rank}(B_{\mu^\star})=r\Rightarrow\Phi_{\mu^\star}^\top\Phi_{\mu^\star}=I_r,\\
&\|\Phi_{\mu^\star}\|_F^2=r,\|\Phi_{\mu^\star}\|_2=1,\mu^\star\ge0,\\
&f(W^+)=c_\eta(\Phi_{\mu^\star})\le0,\boxed{\mathcal E(W^+)\ge1/2},\\
&\boxed{\langle G,\Phi_{\mu^\star}\rangle_F=v^\star=v_{\mathrm{up}}},\mathcal E(W)\ge1/2,\\
&\textbf{任意数值解的证书：}Q\in\mathcal B,Y_Q=W-\eta Q,\mu\ge0,B=B_\mu(Q),\\
&c_\eta(\Psi)\ge c_\eta(Q)+\langle\nabla c_\eta(Q),\Psi-Q\rangle_F,\Psi\in\mathcal B,\\
&\nabla c_\eta(Q)=-2\eta[\chi Y_Q(Y_Q^\top Y_Q)/q(Y_Q)-W],B=G-\mu\nabla c_\eta(Q)/\eta,\\
&\mathcal U(Q,\mu)=\langle G,Q\rangle_F+\|B\|_*-\langle B,Q\rangle_F-\mu c_\eta(Q)/\eta,\\
&f(Y_\Psi)\le0\Rightarrow c_\eta(\Psi)\le0\Rightarrow\langle G,\Psi\rangle_F\le\mathcal U(Q,\mu),\\
&P\in\mathcal B,f(Y_P)\le0,\Delta=\mathcal U(Q,\mu)-\langle G,P\rangle_F,\\
&\boxed{0\le v^\star-\langle G,P\rangle_F\le\Delta},\Delta\ge0,\\
&\textbf{内层数值迭代：}O_j\in\arg\max_{O\in\mathcal B}\langle B_\mu(\Phi_j),O\rangle_F,\\
&O_j=\operatorname{polar}(B_\mu(\Phi_j)),\Phi_{j+1}=(1-\alpha_j)\Phi_j+\alpha_jO_j,\\
&\alpha_j\in\arg\max_{\alpha\in[0,1]}\mathscr L_\mu((1-\alpha)\Phi_j+\alpha O_j),\\
&0\le\alpha_j\le1,\|\Phi_j\|_2\le1,\|O_j\|_2\le1\Rightarrow\|\Phi_{j+1}\|_2\le1,\\
&-c_\eta(\Phi_\mu)/\eta\in\partial d(\mu),\mu\ge0,d(\mu)=\mathscr L_\mu(\Phi_\mu),\\
&\mathscr L_{\mu_1}(\Phi_{\mu_1})\ge\mathscr L_{\mu_1}(\Phi_{\mu_2}),0\le\mu_1<\mu_2,\\
&\mathscr L_{\mu_2}(\Phi_{\mu_2})\ge\mathscr L_{\mu_2}(\Phi_{\mu_1}),0\le\mu_1<\mu_2,\\
&\mu_2>\mu_1\Rightarrow c_\eta(\Phi_{\mu_2})\le c_\eta(\Phi_{\mu_1}),\mu_1\ge0,\\
&\textbf{始终可行的内逼近：}P_0=0,X_k=W-\eta P_k,f(X_k)\le0,\\
&\widehat c_k(\Phi)=\chi q(Y_\Phi)-2\langle X_k,Y_\Phi\rangle_F+\|X_k\|_F^2,\\
&\boxed{\widehat c_k(\Phi)=f(Y_\Phi)+\|Y_\Phi-X_k\|_F^2},\widehat c_k\textbf{为凸函数},\\
&P_{k+1}\in\arg\max_{\Phi\in\mathcal B,\widehat c_k(\Phi)\le0}\langle G,\Phi\rangle_F,\\
&\widehat c_k(P_k)=f(X_k)\le0\Rightarrow\langle G,P_{k+1}\rangle_F\ge\langle G,P_k\rangle_F,\\
&f(W-\eta P_{k+1})\le-\eta^2\|P_{k+1}-P_k\|_F^2\le0,\|P_{k+1}\|_2\le1,\\
&W^+=W-\eta P_K,\mathcal U(Q,\mu)-\langle G,P_K\rangle_F\le\varepsilon_*,\\
&\boxed{\mathcal E(W^+)\ge1/2,\|P_K\|_2\le1},W^+=W-\eta P_K,\\
&\boxed{\langle G,P_K\rangle_F\ge v^\star-\varepsilon_*},\varepsilon_*\ge0,\\
&\textbf{全局细化：}L,U,Y\in\mathbb R^{m\times n},L_{ij}\le Y_{ij}\le U_{ij},\\
&Y_{ij}^2\le(L_{ij}+U_{ij})Y_{ij}-L_{ij}U_{ij},(U_{ij}-Y_{ij})(Y_{ij}-L_{ij})\ge0,\\
&f_{L,U}(Y)=\chi q(Y)-\langle L+U,Y\rangle_F+\langle L,U\rangle_F\le f(Y),\\
&0\le f(Y)-f_{L,U}(Y)=\sum_{ij}(U_{ij}-Y_{ij})(Y_{ij}-L_{ij})\le\|U-L\|_F^2/4,\\
&\textbf{数值极分解：}B\ne0,T_0=B/\|B\|_F,T_{j+1}=T_j(3I_r-T_j^\top T_j)/2,\\
&p_{i,0}=\sigma_i(B)/\|B\|_F,p_{i,j+1}=p_{i,j}(3-p_{i,j}^2)/2\in[0,1],\\
&T_j=U\operatorname{diag}(p_{i,j})V^\top,\|T_j\|_2\le1,0\le p_{i,j}\le1,\\
&\|B\|_*-\langle B,T_j\rangle_F=\sum_i\sigma_i(B)(1-p_{i,j})\le\sum_i\sigma_i(B)(1-p_{i,j}^2),\\
&\boxed{\|B\|_*\le\langle B,T_j\rangle_F+\sqrt r\|B-BT_j^\top T_j\|_F},j\ge0.\\
\end{aligned}
\end{equation}
$$
