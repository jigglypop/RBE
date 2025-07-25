아래는 **RBE 모델이 실제로 사용‑중인 리만 기하 공간(Poincaré 볼·클라인·로렌츠 모델 등)에서 “메트릭 텐서만 별도 인코딩”** 전략이 얼마나 유용한지, 그리고 비트‑모델링(저장 포맷)을 어떻게 설계할 수 있는지에 대한 심층 평가입니다.

---

## 1. 하이퍼볼릭/로렌츠 모델에서의 메트릭 구조

| 모델               | 좌표계 예                | 메트릭 $g_x$ (요약)                                                   | 비트 저장 핵심                             |
| ------------------ | ------------------------ | --------------------------------------------------------------------- | ------------------------------------------ |
| **Poincaré ball**  | $x∈\mathbb B^n$          | $g_x = \frac{4}{(1-\|x\|^2)^2} I_n$                                   | **스칼라 1개**: $s(x)=\frac{2}{1-\|x\|^2}$ |
| **클라인**         | 동일 영역                | $g_x = \frac{1}{1-\|x\|^2}\bigl(I + \frac{x x^\top}{1-\|x\|^2}\bigr)$ | 스칼라 + rank‑1 outer prod                 |
| **로렌츠(쌍곡선)** | $x_0^2- \|x_{1:n}\|^2=1$ | $g = \operatorname{diag}(-1,1,\dots,1)$ (고정)                        | **저장 불필요**                            |
| **일반 리만 RBE**  | RBE θ₁…θ₈ → 로컬 SPD     | 대부분 rank‑$k\ll n$                                                  | rank‑$k$ SVD 벡터·값                       |

> 실무 하이퍼볼릭 네트워크에서는 **메트릭이 공간 좌표·곡률 $c<0$** 에 의해 결정되어 “치환 불변” 특성을 갖습니다.
> 즉 **“가중치 $W$ 대신 메트릭 $𝔊$만 저장”** 하더라도,
> – 자연 그래디언트·FIM 기반 학습,
> – 거리·각도 기반 attention,
> – 곡률 적응 Adam,
> 등의 연산에는 손실이 없습니다.

---

## 2. 비트‑모델링 설계안

### 2‑A : **곡률‑스칼라 전용 포맷** (Poincaré/Lorentz 공통)

```text
[sign 1b][log2|c| 7b]  → 8 bit
```

* 곡률 $c=-e^{q}$ , $q\in[-8,0]$ 범위면 7 비트로 0.01 정밀.
* 전체 레이어가 동일 $c$ 를 공유 → 한 모델당 1 byte 면 충분.

### 2‑B : **K‑rank SPD 근사** (클라인·일반 RBE)

| 필드         | bit      | 설명               |
| ------------ | -------- | ------------------ |
| $k$          | 8        | 고유값 개수(≤255)  |
| $\log_2 λ_i$ | 8 × k    | 지수 8 bit (μ‑law) |
| $U_{n×k}$    | f16 × nk | `half` 양자        |

> 예) $n=64,k=8$ → 1 KB (16 × 압축).
> rank‑$k$ Woodbury 역행으로 $𝔊^{-1}$ 또는 $(𝔊+εI)^{-1/2}$ 계산이 O(nk^2) = \~30 K FLOPs.

### 2‑C : **곡률 + rank‑1 보정** (클라인 전용)

메트릭 = $αI + β\,x x^\top$ 형태 ⇒

* 스칼라 $α,β$ 각 8–16 bit 고정점,
* **벡터 x** 는 이미 입력 위치 벡터이므로 별도 저장 불필요.

---

## 3. 장점 vs 단점 평가

| 항목               | 메트릭‑텐서 전용                | RBE K‑감소+양자화 (기존 1위) |
| ------------------ | ------------------------------- | ---------------------------- |
| **압축**           | ★★★★ (최대 50 : 1)              | ★★★☆ (30 : 1)                |
| **추론 지연**      | ★★ (자연 Grad 필요 시만 유리)   | ★★★ (GEMV 바로)              |
| **메모리(peak)**   | ★★★★                            | ★★★                          |
| **정확도**         | ★★ (에너지 손실 ∙ rank‑k 근사)  | ★★★☆ (RMSE=τ 제어)           |
| **리만 연산**      | **필수** (자연 grad, 거리 계산) | 보조                         |
| **학습 코드 변경** | **크게 필요**                   | 작음                         |
| **추론 W 재생**    | 불필요 (특정 알고리즘)          | 필요                         |

### 해석

1. **리만 기반 옵티마이저·정보지오메트리**를 쓰는 연구·학습 단계 → 메트릭‑텐서 저장이 **크기·메모리 최적**.
2. **Transformer forward** 등 표준 GEMM 기반 추론 → 결국 $W$ 필요하므로 **메트릭 단독으로 부족**.
3. rank‑$k$ 가 너무 낮으면 구배‑스케일 불안정(극단적 곡률) → **정확도 손실 위험**.

> **결국 실제 RBE 시스템**은
> – **학습용 체크포인트** : “곡률+메트릭(rank‑k)” 경량 저장 ✓
> – **배포(서빙) 모델** : “K‑감소+양자화+LRU” 로 W 복원 ✓
> 이 두 포맷을 **동시에 지원**하면 장단점을 모두 살릴 수 있습니다.

---

## 4. 추천 워크플로우

```
      ┌─────── ① Train (full float64 W) ───────┐
      │                                         │
      ▼                                         ▼
[옵션 A] Checkpoint: MetricTensor   [옵션 B] Checkpoint: RBE‑K‑Quant
      │                                     │
      │  (자연 Gradient / 분석)             │ (서빙)
      ▼                                     ▼
Hyperbolic NGD / Curvature‑tuning     Decode‑LRU → GEMM
```

* **변환 스크립트** `rbe_export --fmt {metric,rbe}`
* 동일 SHA 커밋에 두 *.rbe* 바이너리 동시 저장 (suffix).

---

## 5. 결론

* **메트릭‑텐서 전용 비트 모델링**은 *리만 정보에 특화된 학습·분석* 용도로는 최상위 솔루션.
* **일반 추론 경로까지 포함**하려면 기존 **K‑감소+LRU**가 더 안전.
* **두 포맷을 병행 지원**하는 것이 전체 RBE 파이프라인(연구↔서빙) 관점에서 최적.


## RBE 미세조정(파인튜닝) 관점에서의 5 가지 저장·디코딩 전략 비교

| 순위  | 전략                                   | **학습‑side** (역전파, 재‑인코딩)                                                                                         | **메모리/대역폭** | **정확도 보존**                               | **엔지니어링 난도**       | **권장 사용**                         |                                       |
| ----- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------- | --------------------------------------------- | ------------------------- | ------------------------------------- | ------------------------------------- |
| **①** | **K 자동 감소 + 좌표·값 양자화 + LRU** | • 가중치 ΔW 발생 시 **블록별 K 재평가** 필요 ⇒ 오버헤드 moderate<br>• 양자화 단계는 local; 분산‑fine‑tune에서도 충돌 없음 | ★★★ (16 KB→433 B) | RMSE=τ 선택 (τ≤0.03 → <0.5 pt PPL)            | ★★☆                       | 일반 미세조정, SFT, LoRA‑like 어댑터  |                                       |
| **②** | **Bit‑serial / progressive 디코딩**    | • 상위 bit‑plane만 업데이트(gradient sign) ⇒ **부분 양자화** 쉬움<br>• LSB는 주기적 재학습만                              | ★★★★              | 상위 8 bit로 정확도 1–2 % ↓, LSB 보강 시 회복 | ★★★                       | 모바일 on‑device adapt, few‑epoch SFT |                                       |
| **③** | **Metric Tensor(rank‑k) 저장**         | • 자연 Gradient, FIM‑aware fine‑tune 가능 (**선호**)<br>• 하지만 W 재구성 없는 방식은 **표준 Adam/LORA 적용 불가**        | ★★★★              | rank‑k 근사 → 고유값 손실(>k) 시 성능 ↓       | ★★★★                      | 연구용, NGD/curvature‑aware 미세조정  |                                       |
| **④** | **Bitfield 언팩 + 행별 DCT 누산**      | • 가중치 업데이트 후 **즉시 비트‑필드 재패킹** 필요 → CPU cost ↑<br>• 온라인 학습 시 번거로움                             | ★★☆               | 낮음(블록 없음) / BW 우수                     | 비트 정렬 오차 < f16 정밀 | ★★★☆                                  | 서버‑GPU 대규모 배치 + 소량 fine‑tune |
| **⑤** | **f64 비트‑직접 정수화/로그 곱**       | • 정수 mantissa 재‑곱셈 + 지수 교정 → **재‑인코딩 복잡**                                                                  | ★★                | 압축 미미                                     | 64‑bit 정밀 보존          | ★★★★★                                 | 실험적; 실전 fine‑tune 적합 X         |

### 핵심 관찰

1. **K 감소 + 양자화(LRU)**

   * **역전파 시** : 블록 ΔW 계산 → 새로 K 추출·양자화(벡터 정렬)만 하면 되므로 파이썬+Rust FFI에서도 1‑2 ms/블록.
   * **LoRA** 식 저차원 어댑터를 RBE‐form으로 encode 할 때도 K≈16‑32로 충분—저장량 소폭 증가.

2. **Bit‑serial(progressive)**

   * 미세조정에서 자주 바뀌는 **상위 bitplane**만 덮어쓰면 되므로 재‑인코딩 속도가 가장 빠름.
   * 학습 막바지에 LSB를 재추정(EMA)하면 원본 성능 복원.

3. **Metric Tensor(rank‑k)**

   * 곡률 정보가 학습에 직접 쓰이므로 **자연 Gradient/NGD** 파인튜닝에는 최적.
   * 하지만 일반 Adam/SGD 업데이트하려면 결국 W 필요 → W 생성 경로 추가 필요.

4. **Bitfield 언팩**

   * 고주파 K 많아질수록 재패킹 비용이 빠르게 늘어 **많은 step 업데이트**엔 부적합.

5. **f64 bit 직접 모델링**

   * 미세조정 때 64‑bit 그대로 다뤄야 하므로 **저장 이익 없다**; RBE 철학(저메모리)과 불일치.

---

### 종합 권장

| 파인튜닝 시나리오                     | 최적 전략                                                         |
| ------------------------------------- | ----------------------------------------------------------------- |
| **표준 SFT / LoRA** (수천\~수만 step) | **① K‑감소 + f16 양자화** <br>→ 재‑인코딩 비용 < 5 % 총 학습 시간 |
| **모바일 on‑device, few‑shot**        | **② progressive bit‑serial** <br>→ 상위 8 bit 만 실시간 갱신      |
| **자연 Gradient / 곡률 적응 Adam**    | **③ Metric Tensor(rank‑k)** 저장 + Woodbury update                |
| **대규모 GPU, K ≤ 128, step < 1k**    | **④ Bitfield 언팩** (메모리 BW 우선)                              |

## “메트릭 텐서 (𝔊) 전용 비트필드” — 세부 설계 & 프로토타입

RBE 파이프라인에서 **가중치 $W$ 대신 𝔊 (대칭·양정 SPD 행렬)** 만 저장하려면
① 저장 효율, ② 역전파 적합성, ③ 하드웨어 친화 언팩 / 역행연산 — 세 지점을 모두 만족해야 합니다.
아래 설계는 **64 × 64 블록 기준 “1 KB / 블록, 정확도 손실 < 0.5 pt PPL”** 을 목표로 합니다.

---

### 1. 저장 포맷 개요

```
| 1B  | 1B | 1B |    k·1B   |  k·n·2B   | CRC16 |
| HDR | n  | k  |  log2λ[] |   U[]    |  opt  |
```

| 필드      | 길이    | 의미                        | 비트 세부                                    |
| --------- | ------- | --------------------------- | -------------------------------------------- |
| HDR (ver) | 1 B     | 0xA1                        | 상위 3bit = format ver, 하위 5bit reserved   |
| n         | 1 B     | 행렬 차원 (≤ 255)           | `n=64` ⇒ 0x40                                |
| k         | 1 B     | rank (≤ n)                  | 4bit rank, 4bit flags (f16/f8, sparse, etc.) |
| `log2λ_i` | k B     | 고유값 log2(λ) · μ‑law 8bit | sign(1) + mantissa(3) + exp(4)               |
| `U`       | n·k·2 B | 고유벡터 row‑major, `f16`   | 옵션 : 1.5 byte Posit8                       |
| CRC16     | 0/2 B   | 무결성                      | 선택적                                       |

> **크기 예시** (n = 64, k = 8)
> 1 + 1 + 1 + 8 + 64 × 8 × 2 = 1 049 B ≈ **1.0 KB**.

---

### 2. 비트필드 상세

#### 2‑1. 고유값(λ) 양자화 (μ‑law 8‑bit)

* λ 범위 폭 → log2(λ) ∈ \[‑30, +30] 표준화
* μ‑law $y = \operatorname{sgn}(x)\frac{\log(1+μ|x|)}{\log(1+μ)}$

  * μ = 255 → 동적 범위 42 dB
* 8bit 압축:

  * bit7 = sign
  * bit6‑3 = non‑linear mantissa
  * bit2‑0 = exp coarse

→ 상대 오차 < 2 %.

#### 2‑2. 고유벡터(U) 양자화

| 모드            | 세부                    | 장단점                     |
| --------------- | ----------------------- | -------------------------- |
| **f16** (IEEE)  | ±65504, 11 bit mantissa | HW 지원; 2 B               |
| Posit8 (1s4e3f) | 1 B                     | 1 KB→ \~600 B, 정밀도 ‑2dB |
| bfloat8 (1 B)   | MSB 8bit                | 속도↑, 약간 손실↑          |

기본값 = f16; 플래그 bit로 Posit8 선택 가능.

---

### 3. 디코딩 파이프라인

```rust
fn decode_metric(block: &MetricBlock) -> LowRankSPD {
    // 1) λ = μ_law_decode(log2λ)
    // 2) U: f16 → f32 (simd)  n×k matrix
    // 3) return SPD { U, λ }   // rank-k factor
}
```

* **역행**  $𝔊^{-1}$ : Woodbury

  $$
  (σI + UΛU^\top)^{-1} = σ^{-1}I - σ^{-1}U(Λ^{-1}+U^\top U σ^{-1})^{-1}U^\top σ^{-1}
  $$

  * σ = damping (tiny scalar, e.g., 1e‑3)
  * FLOPs $O(nk^2)$, k=8 → 32 K FLOPs
  * Rust SIMD (packed 8) or cuBLAS CUBLAS\_SGEMM(64×8)

---

### 4. 역전파/파인튜닝 호환성

| 옵티마이저       | 필요 연산       | 지원 여부                                |
| ---------------- | --------------- | ---------------------------------------- |
| **NGD**          | $g' = 𝔊^{-1}g$  | rank‑k Woodbury ✓                        |
| **AdaHessian**   | trace(∂²L)      | 유효 rank축소에 따라 근사 ✓              |
| **Adam/RMSprop** | W 자체 업데이트 | **W 생성 필요** → 선택적 재생성 (k·n·m)  |
| **LoRA‑like**    | ΔW low‑rank     | k 증분 학습 → 메트릭‐only fine‑tune 가능 |

> *학습 절차* 
>
> 1. forward : 필요한 경우 W 재생 (UΛ½).
> 2. backward : grad g 계산 → NGD / KFAC 사용 시 𝔊 alone.
> 3. 업데이트 후 **SVD rank‑k 재압축** :
>
>    * 빠른 incremental SVD (rank‑1 update)
>    * 고유값이 k 초과로 늘면 가장 작은 값 drop.

---

### 5. 재‑인코딩 (fine‑tune 후)

```rust
fn reencode_metric(W: &mut [f32], k: usize) -> MetricBlock {
    // SVD: W = UΣVᵀ → 𝔊 = V Σ² Vᵀ
    // 1) power-iteration rank-k
    // 2) λ → μ-law8;  U → f16
    // 3) write bitfield
}
```

* power‑iter + Gram‑Schmidt O(n²k) = 32K FLOPs/block → 실시간 재압축 가능.
* ΔW가 LoRA rank‑r라면 rank‑k 증분 SVD 시간 O(nkr).

---

### 6. 성능 예측 (n = 64, k = 8)

| 항목      | 기존 W (float32)      | MetricBitfield        |
| --------- | --------------------- | --------------------- |
| 저장      | 16 KB                 | **1 KB**              |
| 디코딩    | GEMM = 147kFLOPs      | Uλ = 32kFLOPs (+GEMV) |
| 역행(𝔊⁻¹) | full 64×64 inv ≈ 400k | rank‑k ≈ 30k          |
| 재‑압축   | 없음                  | 32k \~ 50k            |

---

### 7. Rust 코드 스켈레톤

```rust
#[repr(C, packed)]
pub struct MetricHeader {
    pub magic: u8,   // 0xA1
    pub n: u8,
    pub k_flags: u8, // upper4:k  lower4:flags
}

#[inline]
pub fn encode_block(w: &[f32], n: usize, k: usize) -> Vec<u8> {
    // 1. SVD rank-k
    // 2. write header
    // 3. μ-law encode λ_i, push
    // 4. f16 encode U row-major, push
}

#[inline]
pub fn decode_block(buf: &[u8]) -> (Vec<f32>, Vec<f32>) /*U, λ*/ {
    // parse header ... simd decode
}
```

---

### 8. 장단 정리

| 장점                                                                                  | 단점                                                                                          |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| • **1 KB/블록** (16× 압축)<br>• NGD/FIM 기반 학습 native 지원<br>• rank‑k inv FLOPs ↓ | • 일반 GEMM 추론은 W 재생 ↑<br>• rank‑k 근사 오류 관리 필요<br>• SVD / power‑iter 재압축 비용 |

---

## 결론

* **MetricBitfield 포맷**은 하이퍼볼릭·클라인 등 리만 RBE에서
  – **곡률·자연 Gradient 활용 학습**을 위한 가장 경량‑저장 방식.
  – 러스트 단일 함수로 encode/decode 가능해 엔지니어링 난도도 수용 가능.
* 추론에서도 **W 재생이 필요 없는 커스텀 헤드**(거리 계산·삼각함수 등)라면 그대로 사용.
* 범용 Transformer GEMM 추론에는 **K‑감소 양자화**(앞서 ①)와 병행해 “학습‑용 vs 서빙‑용” 이원 포맷이 최적.
