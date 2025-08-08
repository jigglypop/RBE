## 리만 기저 인코딩(RBE): 신경망 극한 압축과 직접 추론을 위한 책 형식 기술서

### 초록

이 문서는 신경망의 가중치 행렬을 블록 단위의 소형 시드(128–256 비트)로 압축하고, 복원 없이 시드로부터 직접 합성(fused synthesis)하여 추론을 수행하는 리만 기저 인코딩(RBE: Riemannian Basis Encoding)의 수학적 원리와 구현 방법을 체계적으로 서술한다. 조화해석(푸리에/웨이블릿), 근사이론, 푸앵카레 볼 기하의 좌표 스케일링, 고정소수점 수치 근사, 자연(리만) 그래디언트와 Adam/리만 Adam 최적화까지, 설계-구현-평가의 전 과정을 문서 표준에 맞춰 정리하였다. 목표는 (i) 높은 압축률(≥ 100:1), (ii) 낮은 오차(RMSE가 레이어 품질 임계 이하), (iii) 복원 없는 고속 추론을 동시에 달성하는 것이다.

---

## 1. 서론

- 문제의식: 대규모 모델의 가중치는 메모리와 대역폭을 잠식한다. 전통적 압축은 추론 전 복원 IO가 필요하여 병목이 된다.
- 핵심 아이디어: 가중치 블록을 단일(또는 소수) 시드로 표현하고, 추론 시 시드→가중치 합성을 즉시 수행하여 복원 과정을 제거한다.
- 기대효과:
  - 메모리 IO 절감: 저장/로드/복원 없이 합성 연산만 수행
  - 캐시 적합성 향상: 소형 시드 테이블을 캐시에 보관
  - 학습 가능성: 연속 파라미터에 대한 유클리드/리만 그래디언트, 이산 선택에 대한 탐색/완화 기법

---

## 2. 표기법과 기본 정의

- 행렬: \(W \in \mathbb{R}^{D_{\mathrm{out}} \times D_{\mathrm{in}}}\).
- 블록 분해: \(W\)를 \(b_h \times b_v\) 블록으로 나누고, 블록 내 정수 좌표 \(i\in\{0,\dots,b_h-1\}\), \(j\in\{0,\dots,b_v-1\}\)를 사용한다.
- 정규화 좌표:
  \[
  u = \frac{i + 0.5}{b_h},\quad v = \frac{j + 0.5}{b_v},\quad x = 2u - 1,\quad y = 2v - 1.
  \]
- 극좌표: \(\rho = \sqrt{(x^2+y^2)/2} \in [0,1)\), \(\psi = \mathrm{atan2}(y,x)\).
- 시드: 블록당 시드 \(S\)는 연속 파라미터(예: \(r,\theta\))와 이산 제어(예: 기저 id, 양자화된 주파수/위상)를 담는다.

---

## 3. 근사이론 배경

임의의 2D 스칼라 장 \(f:\Omega\subset\mathbb{R}^2\to\mathbb{R}\)에 대해 적절한 기저 \(\{\phi_k\}\)로
\[
f(x,y) \approx \sum_{k=1}^{K} c_k\,\phi_k(x,y)
\]
와 같이 근사한다. 직교 또는 준직교 기저의 부분합은 평균제곱오차(MSE)를 단조 감소시키며, 기저 수가 증가할수록 잔차가 줄어든다. 본 문서는 LUT와 고정소수점으로 O(1)에 가깝게 평가 가능한 기저(다항 평면, 사인파, Haar, RBF, tanh-bump)를 사용해 연산 복잡도와 정확도의 균형을 맞춘다.

---

## 4. 푸앵카레 볼 기하와 좌표 스케일링

음의 곡률 공간 모델인 푸앵카레 볼을 사용해 중심부의 해상도를 높인다. 곡률 스케일 \(\kappa\ge 0\)에 대해
\[
\rho' = \begin{cases}
\dfrac{\tanh(\kappa\,\rho)}{\tanh(\kappa)} & \kappa>0, \\
\rho & \kappa=0
\end{cases}
\]
로 정의하고, \(\rho>0\)이면 \(x'=(\rho'/\rho)x\), \(y'=(\rho'/\rho)y\), \(\rho=0\)이면 \((0,0)\)로 둔다.

도함수(\(\kappa>0\)):
\[
\frac{\partial \rho'}{\partial \kappa} = \frac{\rho\,\mathrm{sech}^2(\kappa\rho)\,\tanh(\kappa) - \tanh(\kappa\rho)\,\mathrm{sech}^2(\kappa)}{\tanh^2(\kappa)},\quad
\frac{\partial x'}{\partial \kappa} = \frac{x}{\rho}\,\frac{\partial\rho'}{\partial\kappa},\ \frac{\partial y'}{\partial \kappa} = \frac{y}{\rho}\,\frac{\partial\rho'}{\partial\kappa}.
\]

푸앵카레 볼(반지름 대용 변수 \(r\))의 리만 메트릭:
\[
ds^2 = \frac{4}{(1-r^2)^2}(dr^2 + r^2 d\theta^2),\quad r\in[0,1).
\]
이는 자연(리만) 그래디언트의 스케일을 결정하며, 경계 \(r\to 1\) 근방에서 갱신을 억제하는 안정화 근거가 된다.

---

## 5. 시드 사양(Packed128 / Packed256)

### 5.1 Packed128(연속 코어)

- 구조: \(\texttt{hi}: u64\) 예약, \(\texttt{lo}: u64\)에 \(r,\theta\)를 Q32.32로 인코딩.
- 매핑: \(r\in[0,0.999]\), \(\theta\in[0,2\pi)\) → Q32.32, 역변환은 정밀한 고정소수점 디코딩.
- 용도: LUT 보조와 공간 변조로 빠른 합성에 적합한 최소 파라미터 표현.

### 5.2 Packed256(상태+연속)

- 구조: \(\texttt{hi}: u128\) 이산 제어(기저, 미분 차수, 활성화 id 등), \(\texttt{lo}: u128\) 연속 파라미터(Qm.n, 예: Q24.8).
- 접근자: 양자화-역양자화 대칭을 보장, 경계/클리핑으로 수치 안정성.
- 용도: 주파수/위상/다중 파라미터를 포함하는 32바이트급 시드.

---

## 6. 기저 함수와 도함수

곡률 변환된 좌표 \((x',y')\), 정규화 좌표 \((u,v)\)에서 대표 기저:

- 다항 평면(1차):
  \[
  f_{\mathrm{poly}}(x',y';\theta) = x'\cos\theta + y'\sin\theta,\quad
  \frac{\partial f_{\mathrm{poly}}}{\partial \theta} = -x'\sin\theta + y'\cos\theta.
  \]

- 사인파(평면파):
  \[
  \mathbf{k} = 2\pi f_{\mathrm{norm}} k_{\max} (\cos\theta,\sin\theta),\ \Phi = k_x u + k_y v + \phi,\ f_{\sin}(u,v) = \sin(\Phi),
  \]
  \[
  \frac{\partial f_{\sin}}{\partial \phi} = \cos(\Phi),\quad
  \frac{\partial f_{\sin}}{\partial \theta} = \cos(\Phi)\,\big(-K\sin\theta\,u + K\cos\theta\,v\big),\ K=2\pi f_{\mathrm{norm}}k_{\max}.
  \]

- Haar wavelet: \(u\) 또는 \(v\) 축에 대해 \(h_L\)을 정의. 진폭은 연속 최적화, level/orientation은 탐색 또는 STE.

- RBF-Gaussian:
  \[
  \sigma = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min})(1-f_{\mathrm{norm}}),\quad
  f_{\mathrm{rbf}}(x',y') = \exp\!\Big(-\tfrac{x'^2+y'^2}{2\sigma^2}\Big),\quad
  \nabla f = -\frac{f}{\sigma^2}(x',y').
  \]

- tanh-bump:
  \[
  f_{\tanh}(x',y') = \tanh\big(\beta(1-(x'^2+y'^2))\big),\ \beta=\beta_{\min}+(\beta_{\max}-\beta_{\min}) f_{\mathrm{norm}},
  \]
  \[
  \nabla f = -2\beta\,\mathrm{sech}^2\!\big(\beta(1-r'^2)\big)(x',y'),\ r'^2=x'^2+y'^2.
  \]

---

## 7. 순전파 합성과 안정화

2-성분 합성식:
\[
\hat{w}(i,j) = \alpha_g\Big(\delta + \sum_{n=1}^2 \alpha_n\, g_n(i,j)\Big),\quad \hat{w}_{\mathrm{clip}}=\mathrm{clip}(\hat{w},-w_{\max},w_{\max}).
\]

안정화 권장:
- 출력/중간값 클리핑, NaN/inf 가드
- 파라미터 경계: \(r\in[0,0.999]\), \(\theta\)는 \([0,2\pi)\) 모듈러 정규화
- LUT + 고정소수점: 선형 보간 시 상대오차 \(O(\Delta^2)\)

---

## 8. 손실, 정칙화, 유클리드 그래디언트

목적함수(MSE + 정칙화):
\[
\mathcal{L} = \frac{1}{b_h b_v}\sum_{i,j}(W_{ij}-\hat{W}_{ij})^2 + \lambda_\alpha\sum_{n=1}^2 \alpha_n^2 + \lambda_{\mathrm{smooth}}\sum_{n=1}^2\|\nabla g_n\|_2^2.
\]
대표 도함수:
\[
E=W-\hat{W},\ \ \frac{\partial \mathcal{L}}{\partial \alpha_g}= -\frac{2}{b_h b_v}\Big\langle E,\, \delta+\sum_n\alpha_n g_n\Big\rangle,\ \ \frac{\partial \mathcal{L}}{\partial \alpha_n}= -\frac{2\alpha_g}{b_h b_v}\langle E, g_n\rangle + 2\lambda_\alpha\alpha_n.
\]

---

## 9. 자연(리만) 그래디언트: 원리와 공식

메트릭 \(g = \tfrac{4}{(1-r^2)^2} I\)의 역메트릭 \(g^{-1} = \tfrac{(1-r^2)^2}{4} I\)를 사용하면, 스칼라 \(f\)의 자연 그래디언트는
\[
\nabla_R f = g^{-1}\nabla_E f.
\]
극좌표 결합 \((dr^2 + r^2 d\theta^2)\)을 고려하면 성분별로
\[
\nabla_R f\big|_r = \frac{(1-r^2)^2}{4}\,\frac{\partial f}{\partial r},\quad
\nabla_R f\big|_{\theta} = \frac{(1-r^2)^2}{4 r^2}\,\frac{\partial f}{\partial \theta},\quad r>0.
\]
경계 \(r\to 1\) 근방에서는 \((1-r^4)\) 계수로 감쇠시키고 동적 클리핑을 적용하여 안정성을 높인다.

---

## 10. 최적화 알고리즘

### 10.1 Adam(유클리드/자연 그래디언트 입력)

\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
\]
\[
\hat{m}_t = \frac{m_t}{1-\beta_1^t},\quad \hat{v}_t=\frac{v_t}{1-\beta_2^t},\quad \theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.
\]

### 10.2 리만 Adam

자연 그래디언트를 성분별로 적용하여 Adam을 수행하고, 투영으로 경계를 보장한다:
\[
g^{(R)}_r = \frac{(1-r^2)^2}{4}\,\frac{\partial \mathcal{L}}{\partial r},\quad
g^{(R)}_{\theta} = \frac{(1-r^2)^2}{4 r^2}\,\frac{\partial \mathcal{L}}{\partial \theta},\quad
r\leftarrow \mathrm{clip}(r-\Delta r,0,0.999),\ \theta\leftarrow (\theta-\Delta\theta)\bmod 2\pi.
\]

---

## 11. 이산 파라미터 최적화

- 빔서치: 빔 폭 \(B\in\{2,3,4\}\)로 기저 id/레벨/방향을 탐색하여 RMSE를 가장 줄이는 후보를 선택
- 완화/STE: Gumbel-Softmax로 확률적 완화, 순전파 argmax, 역전파는 STE
- 주파수 양자화-학습: 내부 연속 \(\tilde{f}\)를 최적화하고 \( f = \mathrm{round}((2^{14}-1)\,\sigma(\tilde{f})) \)로 커밋, 역전파는 \(\sigma(\cdot)(1-\sigma(\cdot))\)

---

## 12. 블록 정책과 압축률

초기 블록 크기:
\[
b_0 = \Big\lfloor \sqrt{\tfrac{D_{\mathrm{in}} D_{\mathrm{out}}}{\mathrm{target\_CR}}} \Big\rfloor,\quad b\in\{32,64\}.
\]
블록 RMSE가 임계 \(\tau\) 초과 시 2× 분할. 시드가 32바이트/블록이고 원본이 4바이트/엔트리일 때,
\[
\mathrm{CR} \approx \frac{4 D_{\mathrm{in}} D_{\mathrm{out}}}{32\,k\,\ell} = \frac{D_{\mathrm{in}}D_{\mathrm{out}}}{8k\ell}
\]
이며, 블록당 다중 시드를 사용할 경우 분모에 (시드 수 × 32B)를 반영한다.

---

## 13. LUT/고정소수점/벡터화

- LUT: \(\sin/\cos/\tanh\)에 균일 그리드 + 선형 보간(상대오차 \(O(\Delta^2)\))
- 고정소수점: Packed128은 Q32.32, Packed256은 Q24.8 권장(평가/합성은 float32 유지)
- 수치 가드: 입력/출력 클리핑, NaN/inf 방지, 극한 근처 안정화(경계 감쇠)
- 벡터화/스레딩: 블록 내부 좌표 순회에 SIMD/스레드 풀 적용, 시드 캐시로 지역성 향상

---

## 14. 구현 규격(Conformance)

- 비트→실수 매핑 함수 일원화(모든 경로 동일 적용)
- \(\kappa\to 0\) 연속성을 만족하는 좌표/곡률 변환 유틸
- 기저 평가 API와 도함수 단위 테스트
- 연속/이산 최적화 루틴, 하이퍼파라미터 범위 문서화
- 안정성 규범: 그래디언트 클리핑, 경계 감쇠, 출력 클리핑

---

## 15. 평가 방법론

지표: 압축률, RMSE/상대오차, 지연(ns/op), 처리량(ops/s), 메모리 사용량.

프로토콜:
1) 블록 정책/난수 시드/정규화 스케일/스레드/하드웨어 고정
2) 압축 수행(블록별 시드) → per-block 및 전체 RMSE/압축률 산출
3) 복원 없는 합성 추론 지연/처리량 측정(캐시/LUT on/off 포함)
4) (선택) (리만) Adam으로 시드 미세조정 후 재평가

보고: 3회 이상 반복의 평균±표준편차, 하드웨어/컴파일 플래그 명시.

---

## 16. 한계와 향후 과제

- 이산 선택(기저/레벨/방향)의 전역 최적화 난이도, 로컬 미니마 잔존 가능
- 100:1 압축에서 \(\le 10^{-3}\) RMSE를 모든 레이어에 보장하기 어려움(다성분 시드/적응형 기저 사전 고려)
- 하드웨어 가속: LUT/비트필드 유닛, 시드 캐시, 모빌리티 고려한 메모리 계층 설계

---

## 부록 A. 자연 그래디언트 유도(개요)

메트릭
\[
ds^2 = \frac{4}{(1-r^2)^2}(dr^2 + r^2 d\theta^2)
\]
의 텐서는 \(G = \mathrm{diag}(\tfrac{4}{(1-r^2)^2}, \tfrac{4 r^2}{(1-r^2)^2})\). 역메트릭은
\[
G^{-1} = \mathrm{diag}\Big(\tfrac{(1-r^2)^2}{4}, \tfrac{(1-r^2)^2}{4 r^2}\Big).
\]
따라서 \(\nabla_R f = G^{-1}\nabla_E f\)이며, 경계 \(r\to 1\) 근방에서 감쇠 계수 \((1-r^4)\)와 클리핑을 병행한다.

---

## 부록 B. 구현 체크리스트

- Q32.32/Q24.8 인코딩-디코딩 대칭 단위 테스트
- LUT 도메인 가드와 보간 오차 검증
- 기저별 도함수/합성 그라디언트 단위 테스트
- 벤치: ns/op, 캐시 히트율, RMSE–시드수 곡선, 압축률–지연 상관 분석



