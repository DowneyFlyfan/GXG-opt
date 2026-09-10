\[
\begin{aligned}
&W_l\in\mathbb R^{m_l\times n_l},
\qquad
G_{l,t}\in\mathbb R^{m_l\times n_l},
\qquad
M_{l,t}\in\mathbb R^{m_l\times n_l},
\\
&d_l=m_ln_l,
\qquad
r_l=\min(m_l,n_l),
\qquad
p\in\mathcal P,
\\
&\frac1p+\frac1q=1,
\qquad
q=\frac{p}{p-1},
\qquad
q-1=\frac1{p-1},
\\[2mm]
&M
=
U\operatorname{Diag}(\sigma_1,\ldots,\sigma_r)V^\top,
\\
&U\in\mathbb R^{m\times r},
\qquad
V\in\mathbb R^{n\times r},
\qquad
r=\min(m,n),
\\
&\Phi_p(M)
=
U\operatorname{Diag}
\left(
(\sigma_i+\epsilon)^{q-1}
\right)V^\top,
\\
&T_p(M)
=
\sqrt r\,
\frac{\Phi_p(M)}
{\|\Phi_p(M)\|_F+\epsilon},
\qquad
\|T_p(M)\|_F\approx\sqrt r,
\\
&T_2(M)
=
\sqrt r\,
\frac{M}{\|M\|_F+\epsilon},
\qquad
T_\infty(M)\approx UV^\top,
\\[2mm]
&M_{l,t}
=
\beta M_{l,t-1}
+(1-\beta)G_{l,t},
\\
&D_{l,t}
=
s(W_l)T_{p_{l,t}}(M_{l,t}),
\\
&W_{l,t+1}
=
(1-\eta_t\lambda)W_{l,t}
-\eta_tD_{l,t},
\\[2mm]
&s_j
=
\operatorname{vec}(W_{j+1}-W_j)
\in\mathbb R^{d_l},
\\
&y_j
=
\operatorname{vec}(G_{j+1}-G_j)
\in\mathbb R^{d_l},
\\
&Q_l\in\mathbb R^{d_l\times k_l},
\qquad
Q_l^\top Q_l=I_{k_l},
\\
&S_{\mathrm{full}}
=
\begin{bmatrix}
s_1&\cdots&s_k
\end{bmatrix}
\in\mathbb R^{d_l\times k},
\\
&Y_{\mathrm{full}}
=
\begin{bmatrix}
y_1&\cdots&y_k
\end{bmatrix}
\in\mathbb R^{d_l\times k},
\\
&S
=
Q_l^\top S_{\mathrm{full}}
\in\mathbb R^{k_l\times k},
\\
&Y
=
Q_l^\top Y_{\mathrm{full}}
\in\mathbb R^{k_l\times k},
\\[2mm]
&H_{\mathrm{raw}}
=
YS^\top
\left(
SS^\top+\rho I_{k_l}
\right)^{-1}
\in\mathbb R^{k_l\times k_l},
\\
&H_{\mathrm{sym}}
=
\frac12
\left(
H_{\mathrm{raw}}
+
H_{\mathrm{raw}}^\top
\right),
\\
&H_{\mathrm{sym}}
=
V_H
\operatorname{Diag}(\lambda_1,\ldots,\lambda_{k_l})
V_H^\top,
\\
&\widetilde\lambda_i
=
\operatorname{clip}
\left(
\lambda_i,
h_{\min},
h_{\max}
\right),
\\
&H_k
=
V_H
\operatorname{Diag}(\widetilde\lambda_i)
V_H^\top
\succeq0,
\\
&\widehat H_lx
=
Q_lH_kQ_l^\top x
+
\lambda_\perp
\left(
I_{d_l}-Q_lQ_l^\top
\right)x,
\qquad
x\in\mathbb R^{d_l},
\\[2mm]
&\widehat G_{p,0}=G_t,
\qquad
\widehat M_{p,0}=M_t,
\\
&h=0,\ldots,H-1,
\\
&\widehat D_{p,h}
=
s(W_l)T_p(\widehat M_{p,h}),
\\
&\Delta\widehat W_{p,h}
=
-\eta_{t+h}\widehat D_{p,h},
\\
&\operatorname{vec}(\widehat G_{p,h+1})
=
\operatorname{vec}(\widehat G_{p,h})
+
\widehat H_l
\operatorname{vec}(\Delta\widehat W_{p,h}),
\\
&\widehat M_{p,h+1}
=
\beta\widehat M_{p,h}
+(1-\beta)\widehat G_{p,h+1},
\\[2mm]
&\langle A,B\rangle
=
\operatorname{Tr}(A^\top B)
=
\operatorname{vec}(A)^\top\operatorname{vec}(B),
\\
&\Delta\widehat{\mathcal L}_{p,h}
=
\left\langle
\widehat G_{p,h},
\Delta\widehat W_{p,h}
\right\rangle
+
\frac12
\operatorname{vec}(\Delta\widehat W_{p,h})^\top
\widehat H_l
\operatorname{vec}(\Delta\widehat W_{p,h}),
\\[2mm]
&J_{l,t}(p)
=
\sum_{h=0}^{H-1}
w_h\Delta\widehat{\mathcal L}_{p,h}
+
\lambda_p
d_{\mathcal P}(p,p_{l,t^-})^2
+
\lambda_c\operatorname{Cost}(p),
\\
&d_{\mathcal P}(p_i,p_j)
=
\left|
(q_i-1)-(q_j-1)
\right|
=
\left|
\frac1{p_i-1}
-
\frac1{p_j-1}
\right|,
\\
&p^\star_{l,t}
=
\underset{p\in\mathcal P}{\operatorname{argmin}}
\;J_{l,t}(p),
\\
&p_{l,t}
=
\begin{cases}
p^\star_{l,t},
&
J_{l,t}(p_{\mathrm{old}})
-
J_{l,t}(p^\star_{l,t})
>
\delta_{\mathrm{switch}},
\\[1mm]
p_{\mathrm{old}},
&
\text{otherwise}.
\end{cases}
\end{aligned}
\] 05_multi_step_spectral_geometry_policy.mdMD
