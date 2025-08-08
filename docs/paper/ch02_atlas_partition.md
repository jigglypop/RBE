## 장 2. Atlas 기반 패치 블렌딩(Partition of Unity)

### 2.1 원리

- Partition of Unity: $\sum_p \phi_p(u,v)=1$, $\phi_p\ge 0$.
- Hann/Hamming 창을 분리형으로 구성: $\phi_p(u,v)=\phi^x_p(u)\,\phi^y_p(v)$,
  $$\phi^x_p(t) = \tfrac{1-\cos(2\pi\,w_p(t))}{2},\quad w_p:[0,1]\to[0,1].$$

### 2.2 합성식

패치 변환 $T_p$와 성분 $g_{p,k}$에 대해
$$\hat{W}(u,v)=\delta+\alpha_g\sum_p \phi_p(u,v)\sum_{k=1}^K \alpha_{p,k}\,g_{p,k}(\,T_p(u,v);\,\theta_{p,k}\,).$$

### 2.3 경계 연속성

- C0 연속: $\phi$ 합으로 값 연속 보장
- C1 근접: 경계 페널티
$$\mathcal{L}_{\mathrm{seam}}=\lambda_C\sum_{\Gamma}\Big(|\![\![\hat{W}]\!]|^2+\eta\,|\![\![\nabla\hat{W}\cdot n]|\!]|^2\Big).$$

### 2.4 piecewise 곡률(가우스의 빼어난 정리와의 관련)

패치별 $\kappa_p\ge 0$로 §4의 변환을 로컬 적용:
$$\rho'_p=\frac{\tanh(\kappa_p\rho)}{\tanh(\kappa_p)},\quad (x',y')_p=\frac{\rho'_p}{\rho}(x,y).$$
단일 전역 매핑으로는 내재 곡률을 평탄화할 수 없으므로(가우스의 정리), 조각별 $\kappa_p$가 필요합니다.


