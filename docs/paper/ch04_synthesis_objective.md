## 장 4. 합성식, 목적함수, 자연 그래디언트

### 4.1 합성식(두 성분)
$$\hat{w}(i,j)=\alpha_g\Big(\delta+\sum_{n=1}^2\alpha_n\,g_n(i,j)\Big),\quad \hat{w}_{clip}=\mathrm{clip}(\hat{w},-w_{max},w_{max}).$$

### 4.2 목적함수(MSE + 정칙화)
$$\mathcal{L}=\frac{1}{b_hb_v}\sum_{i,j}(W_{ij}-\hat{W}_{ij})^2+\lambda_\alpha\sum_n\alpha_n^2+\lambda_{\mathrm{smooth}}\sum_n\|\nabla g_n\|_2^2.$$

### 4.3 자연(리만) 그래디언트(요지)
푸앵카레 볼 메트릭 $ds^2=\tfrac{4}{(1-r^2)^2}(dr^2+r^2 d\theta^2)$에서 역메트릭을 적용:
$$\partial^{(R)}_r=\tfrac{(1-r^2)^2}{4}\,\partial_r,\qquad \partial^{(R)}_\theta=\tfrac{(1-r^2)^2}{4r^2}\,\partial_\theta.$$


