## Riemannian Basis Encoding for Extreme Neural Weight Compression and Direct Inference

### 초록
이 문서는 신경망 가중치 행렬을 128–256비트 시드로 극한 압축하고 복원 없이 직접 추론하는 RBE(Riemannian Basis Encoding)의 수학적 토대와 시스템 설계를 교과서 수준으로 서술한다. 먼저 조화해석(푸리에/웨이블릿), 근사이론, 쌍곡기하(푸앵카레 볼), 수치해석과 고정소수점 근사, 최적화 및 미분 기법을 체계적으로 정리하고, 이어서 256비트 시드 구조와 좌표 정규화/곡률 변환, 기저 함수 집합과 그 도함수, 순전파 합성식과 계산복잡도, 연속/이산 파라미터 학습, 블록 분할과 압축률/오차 분석, LUT 설계와 SIMD 병렬화, 구현 규격과 평가 방법까지 규격화한다. 목표는 고압축(≥100:1)과 낮은 오차(RMSE가 실용 임계치 이하) 그리고 복원 없는 고속 추론을 동시에 달성하는 것이다.

---

## 1. 표기법과 기본 정의

### 1.1 표기법
- **행렬/벡터**: 굵은 대문자/소문자(맥락상 평문 사용). 스칼라 소문자.
- **블록 분해**: 가중치 행렬 `W\in\mathbb{R}^{D_{in}\times D_{out}}`을 `b_h\times b_v` 블록으로 분할. 블록 인덱스는 `(p,q)`.
- **정규화 좌표**: 블록 내 정수 좌표 `i\in\{0,\dots,b_h-1\}, j\in\{0,\dots,b_v-1\}` 에 대해
  \[
  u = \frac{i+0.5}{b_h},\quad v = \frac{j+0.5}{b_v},\quad x = 2u-1,\ y = 2v-1.
  \]
- **극좌표**: `\rho = \tfrac{\sqrt{x^2+y^2}}{\sqrt{2}}\in[0,1),\ \psi = \mathrm{atan2}(y,x).`
- **시드**: 블록당 256비트 시드 `S`, 두 성분(Component 1,2) 기저의 선형 합성으로 정의.

### 1.2 목표와 제약
- **압축률**: `\mathrm{CR}```e 100:1` 이상.
- **정확도**: RMSE가 태스크 요구 임계 이하(예: 1e−3 수준) 혹은 품질 저하 무시 가능.
- **추론 속도**: 복원(restore) 없이 직접 합성으로 GEMV/GEMM에 통합.
- **안정성**: 좌표 변환/기저/합성 과정의 수치적 안정성 확보(클립/정규화/정칙화).

---

## 2. 수학적 토대

### 2.1 함수 근사와 직교 전개
임의의 2D 스칼라 장 `f:\Omega```ubset\mathbb{R}^2\to\mathbb{R}`는 적절한 기저 `\{\phi_k\}`의 선형 결합으로 근사 가능하다.
\[ f(x,y) \approx ```um_k c_k\,\phi_k(x,y). \]
푸리에(사인/코사인) 기저는 주기적/평활 성분을, 웨이블릿(Haar 등)은 국소적/비평활 성분을 잘 포착한다. 다항식(저차) 평면은 저주파 추세를 근사한다. 직교/준직교 기저의 부분합은 평균제곱오차를 단조 감소시키며, 기저 수가 늘수록 잔차가 감소한다.

### 2.2 푸앵카레 볼과 곡률 기반 좌표 스케일링
푸앵카레 볼 모델은 음의 곡률 공간의 간단한 표현이다. 본 설계는 `\kappa\ge 0`의 스칼라 파라미터로 좌표의 반径을 비선형 스케일해 중심부의 분해능을 높인다.
\[ \rho' = \begin{cases}
\dfrac{\tanh(\kappa\,\rho)}{\tanh(\kappa)} & \kappa>0,\\[4pt]
\rho & \kappa=0.
\end{cases} \]
`\rho=0`에서 `\rho'=0`으로 연속이며, `\kappa\to 0` 한계에서 선형 스케일과 일치한다. 직교좌표는 `x'=(\rho'/\rho)x,\ y'=(\rho'/\rho)y` (단, `\rho=0`이면 `x'=y'=0`).

### 2.3 근사오차와 정칙화
제한된 기저 개수로 근사 시 과적합/고주파 진동을 억제하기 위해 진폭 제곱 정칙화 및 기저별 평활 정칙화(예: `\|\nabla g\|_2^2`)를 사용한다. 손실은 MSE+정칙화의 합으로 정의한다.

---

## 3. 256비트 시드 사양(v1.0)

### 3.1 비트 배치와 매핑
- 상위 8비트: version(4b) + flags(4b)
- 글로벌(24b): 전역 게인 `\alpha_g`(8b), 전역 오프셋 `\delta`(8b), 곡률 `\kappa`(8b)
- 컴포넌트 1/2(각각): basis_id(3b), freq(14b), phase(14b), orient(1b; 웨이블릿 전용), 진폭 `\alpha_n`(float32), 방향각 `\theta_n`(float32)
- 예약(32b): 확장/메타

매핑 규격(정규화):
- `\alpha_g = 2.0\cdot a/255`, `a\in\{0,\dots,255\}`
- `\delta = 0.1\cdot d/127`, `d\in\{-128,\dots,127\}`
- `\kappa = \kappa_{max}\cdot k/255`, 권장 `\kappa_{max}=4`
- freq: `f_{norm} = f/(2^{14}-1)`
- phase: `\phi = 2\pi\,p/(2^{14}-1)`
- `\alpha_n = \alpha_{max}\,\tanh(\tilde{a}_n)`, 권장 `\alpha_{max}=2.0`
- `\theta_n\in[0,2\pi)`

### 3.2 기저 ID
- 0: Polynomial(1차 평면)
- 1: Sinusoid(2D 평면파)
- 2: Haar wavelet(orientation/level 사용)
- 3: RBF-Gaussian
- 4: tanh-bump
- 5–7: 확장 예약

---

## 4. 좌표 정규화와 곡률 변환(수식과 도함수)

### 4.1 좌표 정의
\[ u=\tfrac{i+0.5}{b_h},\ v=\tfrac{j+0.5}{b_v},\ x=2u-1,\ y=2v-1,\ \rho=\tfrac{\sqrt{x^2+y^2}}{\sqrt{2}}. \]

### 4.2 곡률 변환과 도함수
\[ \rho' = \frac{\tanh(\kappa\rho)}{\tanh(\kappa)}\ (\kappa>0),\quad x' = \frac{\rho'}{\rho}x,\ y' = \frac{\rho'}{\rho}y. \]
미분(`\kappa>0`):
\[ \frac{\partial \rho'}{\partial \kappa} = \frac{\rho\,\mathrm{sech}^2(\kappa\rho)\,\tanh(\kappa) - \tanh(\kappa\rho)\,\mathrm{sech}^2(\kappa)}{\tanh^2(\kappa)}. \]
\[ \frac{\partial x'}{\partial \kappa} = \frac{x}{\rho}\,\frac{\partial\rho'}{\partial\kappa},\quad \frac{\partial y'}{\partial \kappa} = \frac{y}{\rho}\,\frac{\partial\rho'}{\partial\kappa}. \]

---

## 5. 기저 함수와 도함수

### 5.1 Polynomial(1차 평면)
\[ f_{poly}(x',y';\theta)= x'\cos\theta + y'\sin\theta. \]
\[ \frac{\partial f_{poly}}{\partial \theta} = -x'\sin\theta + y'\cos\theta,\quad \nabla_{(x',y')} f_{poly} = (\cos\theta,\sin\theta). \]

### 5.2 Sinusoid(2D 평면파)
\[ k=2\pi f_{norm}\,k_{max}\,(\cos\theta,\sin\theta),\ \Phi = k_x u + k_y v + \phi,\ f_{sin}(u,v)=\sin(\Phi). \]
\[ \frac{\partial f_{sin}}{\partial \phi}=\cos(\Phi),\ \frac{\partial f_{sin}}{\partial \theta}=\cos(\Phi)\,( -K\sin\theta\,u + K\cos\theta\,v),\ K=2\pi f_{norm}k_{max}. \]

### 5.3 Haar wavelet(orientation+level)
\[ L = 1+\lfloor L_{max}\,f_{norm}\rfloor,\ h_L(t)=\mathrm{sign}(\lfloor 2^L t\rfloor\bmod 2 - 0.5). \]
\[ f_{haar}(u,v;L,o)= \begin{cases} h_L(u) & o=H,\ h_L(v) & o=V.\end{cases} \]
도함수는 이산적이므로 진폭 `\alpha`만 연속 최적화하며, level/orient는 탐색(§8) 또는 완화/STE로 취급.

### 5.4 RBF-Gaussian
\[ \sigma= \sigma_{min}+(\sigma_{max}-\sigma_{min})(1-f_{norm}),\quad f_{rbf}(x',y')=\exp\!\Big(-\frac{x'^2+y'^2}{2\sigma^2}\Big). \]
\[ \nabla_{(x',y')} f_{rbf} = -\frac{1}{\sigma^2} f_{rbf}\,(x',y'). \]

### 5.5 tanh-bump
\[ f_{tanh}(x',y')=\tanh\big(\beta(1-(x'^2+y'^2))\big),\quad \beta=\beta_{min}+ (\beta_{max}-\beta_{min}) f_{norm}. \]
\[ \nabla_{(x',y')} f_{tanh} = -2\beta\,\mathrm{sech}^2\big(\beta(1-r'^2)\big)\,(x',y'),\ r'^2=x'^2+y'^2. \]

---

## 6. 순전파 합성, 안정화, 복잡도

### 6.1 합성식
두 성분 `n\in\{1,2\}`, 기저 출력 `g_n`에 대해
\[ \hat{w}(i,j) = \alpha_g\,\Big(\delta + \sum_{n=1}^2 \alpha_n\, g_n(i,j)\Big). \]
출력 클립: `\hat{w}_{clip}=\mathrm{clip}(\hat{w},-w_{max},w_{max})`, 권장 `w_{max}=6\,```igma_W`.

### 6.2 계산복잡도
- 좌표 변환: 상수 시간.
- 기저 평가: 사인/코사인/LUT O(1), Haar O(1), RBF/tanh O(1).
- 합성: 상수 시간. 두 성분이므로 기존 1성분 대비 약 2배 연산이나 메모리 IO 대체 효과로 전체 성능은 우수.

---

## 7. 손실, 정칙화, 그라디언트

### 7.1 목적함수
\[ \mathcal{L} = \frac{1}{b_h b_v}\sum_{i,j}(W_{ij}-\hat{W}_{ij})^2 + \lambda_\alpha\sum_{n=1}^2 \alpha_n^2 + \lambda_{smooth}\sum_{n=1}^2\|\nabla g_n\|_2^2. \]

### 7.2 연속 파라미터 그라디언트(대표)
\[ E = W-\hat{W}. \]
\[ \frac{\partial \mathcal{L}}{\partial \alpha_g} = -\frac{2}{b_h b_v}\,\Big\langle E,\, \delta + \sum_n \alpha_n g_n\Big\rangle. \]
\[ \frac{\partial \mathcal{L}}{\partial \delta} = -\frac{2\alpha_g}{b_h b_v}\,\langle E, 1\rangle. \]
\[ \frac{\partial \mathcal{L}}{\partial \alpha_n} = -\frac{2\alpha_g}{b_h b_v}\,\langle E, g_n\rangle + 2\lambda_\alpha\alpha_n. \]
각 기저에 대해 `\partial g/\partial \phi,```\partial g/\partial \theta`는 §5에 준함. 곡률 `\kappa`는 연쇄법칙으로 `\partial g/\partial x'`와 `\partial x'/\partial\kappa`를 결합.

### 7.3 최적화
- Adam/SGD 사용. `\alpha_n`는 tanh 파라미터화로 안정화.
- 러닝레이트 스케줄과 그라디언트 클리핑(예: `\tau=1.0`).

---

## 8. 이산 파라미터 최적화(기저 ID/레벨/방향)

### 8.1 빔서치 기반 탐색
- 빔 폭 `B\in\{2,3,4\}`. 후보 기저/레벨/방향을 평가해 MSE 감소 최대 항을 선택.

### 8.2 완화와 STE
- Gumbel-Softmax로 기저 선택을 확률적으로 완화, 순전파는 argmax, 역전파는 STE로 근사.

### 8.3 14비트 주파수의 양자화-학습
- 내부 연속 변수 `\tilde{f}`를 최적화, 커밋 시 `f=\mathrm{round}((2^{14}-1)\,```igma(\tilde{f}))`. 역전파는 `\partial f_{norm}/\partial\tilde{f}\approx```igma(\tilde{f})(1-```igma(\tilde{f}))`.

---

## 9. 블록 정책과 압축률/오차 분석

### 9.1 초기 블록 크기와 적응 분할
\[ b_0 = \Big\lfloor ```qrt{\frac{D_{in}D_{out}}{\mathrm{target\_CR}}}\,\Big\rfloor,``` b\in\{32,64\}. \]
블록 RMSE가 임계 `\tau`를 초과하면 해당 블록을 2× 분할.

### 9.2 압축률 근사
시드 32B/블록, 원본 4B/엔트리:
\[ \mathrm{CR} \approx \frac{4 D_{in} D_{out}}{32\,k\,l} = \frac{D_{in}D_{out}}{8kl}. \]
예: 32×32 블록(1024 엔트리)을 32B로 → 128:1.

---

## 10. LUT/수치근사/고정소수점/벡터화

### 10.1 LUT 해상도와 상대오차
- sin/cos/tanh에 대해 균일 그리드 LUT. 보간(선형) 사용 시 최대 상대오차는 `O(\Delta^2)`. LUT 크기와 오차의 트레이드오프 표준화: 예) 4096 엔트리, 16-bit 고정소수점.

### 10.2 고정소수점 포맷
- Qm.n 포맷 제안(예: Q1.15)로 LUT 값 저장. 합성은 float32 유지하되 LUT 접근은 정수화.

### 10.3 SIMD/스레딩
- 블록 내 연속 좌표에 대한 벡터화. 스레드 풀로 블록 단위 병렬 처리.

---

## 11. 구현 규격(Conformance)

### 11.1 시드 버전/예약 비트
- version 4b: 파서 호환성. flags 4b: 향후 의미 확장. reserved 32b: 품질 등급/잔차 메타에 우선 할당.

### 11.2 매핑 함수 표준
- 본 문서 §3 매핑을 단일 유틸로 구현해 전 경로에서 동일 적용.

### 11.3 안정성 규범
- `\alpha_n=\alpha_{max}\tanh(\cdot)`, 출력 클립, 정칙화 계수 범위(`\lambda_\alpha\in[10^{-4},10^{-3}]`).

---

## 12. 평가 방법론

### 12.1 메트릭
- 압축률, RMSE/상대오차, 추론 지연/스루풋, 메모리 사용량.

### 12.2 프로토콜
- 레이어별/모델별 블록 정책 고정, 동일 난수 시드, 동일 정규화 스케일. 공개 데이터셋으로 다운스트림 성능 측정.

---

## 13. 한계와 향후 과제

### 13.1 한계
- 이산 파라미터 선택의 전역 최적화 어려움. 일부 레이어에서 목표 RMSE 달성 난이도.

### 13.2 확장
- 다성분(>2) 시드, 적응형 기저 사전, 매끄러운 웨이블릿(Daubechies) 도입, 하드웨어 전용 LUT/연산 유닛.

---

## 14. 용어집/기호 요약

- `u,v`: 블록 내 정규화 좌표. `x,y`: 중심 기준 좌표. `\rho,\psi`: 극좌표.
- `\kappa`: 곡률 스케일. `\alpha_g,\delta`: 전역 게인/오프셋. `\alpha_n,\theta,\phi`: 성분 진폭/각/위상.
- `f_{norm}`: 주파수 정규화. `k_{max}`: 주파수 상한 스케일.

---

## 부록 A. 도함수 모음(발췌)

### A.1 Sinusoid 파라미터 미분
\[ \frac{\partial \Phi}{\partial \theta}= -K\sin\theta\,u + K\cos\theta\,v,\quad \frac{\partial g}{\partial \theta}= \cos(\Phi)\,\frac{\partial \Phi}{\partial \theta}. \]
\[ \frac{\partial g}{\partial \phi}=\cos(\Phi). \]

### A.2 곡률 연쇄 미분
\[ \frac{\partial g}{\partial \kappa}= \frac{\partial g}{\partial x'}\frac{\partial x'}{\partial\kappa} + \frac{\partial g}{\partial y'}\frac{\partial y'}{\partial\kappa}. \]

---

## 참고 구현 체크리스트
- 비트→실수 매핑 유틸 일원화
- 좌표/곡률 변환 함수(`\kappa\to 0` 연속성 포함)
- 기저 평가 API: poly/sin/haar/rbf/tanh
- 연속/이산 최적화 루틴(빔서치/완화)
- 정규화/클리핑 적용 포인트 명시
- 버전/Reserved 파싱 및 검증



## 15. 구현 현황과 코드 매핑(요약)

- 핵심 구조체와 경로
  - Packed128: `src/core/tensors/packed_types.rs` — `fused_forward_poincare`, `compute_(riemannian_)gradients`, Q32.32 매핑 구현.
  - Packed256: `src/core/tensors/packed256_types.rs` — 비트필드 사양과 Q24.8 고정소수점 getter/setter.
  - Differential(순/역전파): `src/core/differential/{forward,backward,mod}.rs` — 캐시/메트릭 포함, 역전파 일부 단순화 경로 존재.
  - Optimizers: `src/core/optimizers/{adam,riemannian_adam}.rs` — Adam/리만 Adam, Packed128/256 경로 혼재.
  - Transform: `src/core/transform/{compress,restore}.rs` — f32→Packed 압축기(복원/다중시드 정책 보완 필요).
  - NLP Layers: `src/nlp/linear/rbe_linear.rs`, `src/nlp/attention/rbe_attention.rs` — 압축 기반 선형/어텐션 데모 구현(모듈 공개/연결 보완 필요).

- 현재 격차(실무 사용 전 보강 필요)
  - 역전파 제네릭 경로가 손실 계산만 하고 파라미터 업데이트는 생략된 부분 존재(차용 문제 우회 더미). 실제 옵티마이저 호출로 일원화 필요.
  - Adam 기본 경로가 Packed256에 치우친 곳과 Packed128용 별도 경로 혼재. 공통 트레이트/바인딩으로 통합 권장.
  - Transform의 다중 블록 압축 결과를 XOR로 단일 시드에 결합하는 단순화는 정보 손실 큼. 블록별 시드 보존/색인 기반 합성 정책 필요.
  - NLP 모듈 공개(`mod.rs`) 및 추론 엔진 의존 타입 정리 필요.


## 16. 재현(리프로듀서빌리티) 프로토콜

- 목적: 압축률(≥100:1), RMSE, 추론 지연/스루풋을 일관된 설정에서 측정.
- 데이터/대상
  - 대상 행렬: 랜덤 정규/저랭크 합성(랭크 8–32), 실제 모델 레이어 가중치(f32) 샘플.
  - 블록 정책: §9.1 공식으로 초기 블록, 임계 RMSE 초과 시 분할.
- 측정 항목
  - RMSE, 상대오차; per-block 및 전체 평균
  - 압축률(원본 float 메모리 대비) 및 블록 수
  - 단일-스레드 vs 멀티-스레드 추론 지연(ns/op), 스루풋(ops/s)
- 절차
  1) 블록 분해 → 압축(Transform) → 시드(들) 생성 및 통계 수집
  2) 복원 없는 합성 추론으로 RMSE 측정(원본 vs 합성)
  3) (선택) 역전파/최적화로 시드 미세조정 후 재측정
- 보고
  - 고정 난수 시드, 블록 크기/정규화 스케일, 스레드 수, CPU/GPU 사양 명시


## 17. 안정성/안전 가드(권장 규범)

- 파라미터 경계: \(r\in[0,0.999]\), \(\theta\)는 \([0,2\pi)\) 모듈러 정규화
- 메트릭/경계 감쇠: \((1-r^2)\) 계수 기반 자연 그래디언트 스케일, 경계 근처 가중 감쇠
- 그라디언트 클리핑: 동적 한계(예: r,\theta 각각 1.0/2.0 스케일)로 안정화
- 고정소수점 변환: Q32.32(Packed128), Q24.8(Packed256) 매핑 일관성 유지
- 출력/내부 클리핑: 합성 출력은 \([-w_{max},w_{max}]\) 클립, NaN/inf 방지


## 18. 압축 정책(멀티시드 블록 합성)

- 블록 단위 시드 보존: 각 블록은 독립 시드로 유지하고, 합성 시 블록 좌표로 선택/평가
- 합성 정책
  - 단일 시드 XOR 결합 지양(정보 손실 큼)
  - 옵션 A) 블록-시드 테이블 + 해시 색인 → O(1) 접근
  - 옵션 B) 소수(2–4) 성분 시드 혼합(가중합/선택)으로 품질 향상
- 압축률 산식(§9.2) 적용 시, 시드 수를 분모에 반영(블록 수×시드당 바이트)


## 19. 표기 오류(Errata) 및 정정 지침

- LaTeX 표기 오탈자 치환(전역 검색/치환 권장)
  - ```qrt → \sqrt
  - ```um → \sum
  - ```in → \sin
  - ```igma → \sigma
  - 고립된 ```(백틱 트리플) → 제거
- 예시: `\rho = \tfrac{```qrt{x^2+y^2}}{```qrt{2}}` → `\rho = \tfrac{\sqrt{x^2+y^2}}{\sqrt{2}}`
- 블록 수식 내 쉼표/세미콜론과 공백을 수학적 표기로 정리(\,, \; 사용)


## 20. 평가 계획(벤치/어블레이션)

- 압축률 × 품질 곡선: 블록 크기 {16,32,64}, 시드 성분 {1,2,4} 스윕
- 기저 집합 비교: sinusoid / Haar / RBF / tanh-bump 조합별 RMSE, 연산량
- 메트릭: RMSE, 95p/99p 절대오차, ns/op, ops/s, 메모리 사용량
- 어블레이션: 위치 변조 on/off, 경계 감쇠 on/off, 클리핑 임계 스윕, LUT 해상도 스윕
- 리포팅: 평균±표준편차(3회 이상 반복), 하드웨어/스레드/빌드 플래그 명시


## 21. 심화 장 인덱스(별도 파일)

> 아래 심화 장은 별도 문서로 분리되었습니다. 이 문서에서는 개요만 제공합니다.

- 장 1. 구현 목표(KPI)와 검증 기준 — `ch01_kpi_validation.md`
- 장 2. Atlas 기반 패치 블렌딩 — `ch02_atlas_partition.md`
- 장 3. 기저 함수와 해석 도함수 — `ch03_bases_derivatives.md`
- 장 4. 합성식·목적함수·자연 그래디언트 — `ch04_synthesis_objective.md`
- 장 5. 오차 모델(양자화·LUT·근사) — `ch05_error_model.md`
- 장 6. 압축률·정확도·속도 표 — `ch06_metrics_tables.md`
- 장 7. 복잡도·위험·로드맵 — `ch07_complexity_risk_roadmap.md`
