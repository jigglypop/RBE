## 장 5. 오차 모델: 양자화·LUT·수치 근사

### 5.1 양자화(Qm.n)
균등 양자화 간격 $\Delta$에서 분산 $\mathrm{Var}\approx \Delta^2/12$. 파라미터 민감도 $S$에 대해 오차 기여 $S^2\Delta^2/12$.

### 5.2 LUT 보간
격자 간격 $\Delta=1/M$, 선형 보간 상대오차 $O(\Delta^2)$. 예: $M=4096\Rightarrow \sim 10^{-6}$ 수준.

### 5.3 합성 RMSE 상계(보수)
독립 가정 평균제곱합 근사:
$$\mathrm{RMSE}\lesssim\sqrt{\mathrm{RMSE}^2_{\mathrm{basis}}+\mathrm{RMSE}^2_{\mathrm{quant}}+\mathrm{RMSE}^2_{\mathrm{LUT}}+\mathrm{RMSE}^2_{\mathrm{num}}}.$$


