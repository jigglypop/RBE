# 3. 디코딩: 가중치의 즉석 합성 및 융합 순전파

## 3.1. 개요: 디코딩 패러다임의 전환

전통적인 관점에서 '디코딩(Decoding)'은 압축된 데이터를 원본 형태로 완전히 복원(restore)하는 과정을 의미한다. 그러나 본 패러다임에서 디코딩은 그러한 정적(static) 복원 과정을 지칭하지 않는다. 대신, 신경망의 순전파(Forward Pass) 연산이 일어나는 바로 그 순간에, 필요한 가중치 값을 **즉석에서 합성(On-the-fly Synthesis)**하는 동적(dynamic) 과정을 의미한다.

이러한 접근법을 **융합 순전파(Fused Forward Pass)**라고 명명한다. 이는 가중치 생성(generation)과 행렬 곱셈(GEMM) 연산을 하나의 논리적, 연산적 단위로 융합하여, 대규모 가중치 행렬을 메모리에 적재(materialize)하는 과정에서 발생하는 심각한 메모리 대역폭 병목(memory bandwidth bottleneck)을 원천적으로 제거한다.

본 장에서는 이 융합 순전파의 수학적 원리와 그 구현 메커니즘을 상세히 기술한다.

## 3.2. 단일 가중치(`W_ij`)의 합성 원리

융합 순전파의 핵심은 행렬-벡터 곱 `y = Wx`를 계산할 때, `i`번째 출력 뉴런 `y_i`에 기여하는 가중치 행의 각 요소 `W_ij`를 그 자리에서 계산하는 것이다. `W_ij` 값은 `HybridEncodedBlock`에 저장된 세 가지 구성 요소의 합으로 합성된다.

**`W_ij = F_state(seed.hi, i, j) + F_continuous(seed.lo, i, j) + F_residual(residuals, i, j)`**

여기서 `i`와 `j`는 각각 행렬의 행과 열 인덱스를 나타낸다.

### 3.2.1. `F_state`: 상태-전이 코어로부터의 기저 패턴

`F_state` 함수는 `Packed128`의 `hi` 필드에 인코딩된 이산 상태 정보로부터 기저 패턴(base pattern) 값을 생성한다.

**입력**: `hi` (u64), `i` (행 인덱스), `j` (열 인덱스)
**출력**: 기저 패턴 값 (f32)

**의사 코드 (Pseudo-code):**
```
function F_state(hi, i, j):
    // 1. 비트 필드로부터 상태 파라미터 디코딩
    phase_state       = (hi >> 62) & 0b11
    frequency_quant   = (hi >> 50) & 0xFFF
    amplitude_quant   = (hi >> 38) & 0xFFF
    function_selector = (hi >> 32) & 0b111111

    // 2. 양자화된 값을 실수로 변환 (역양자화)
    frequency = dequantize(frequency_quant)
    amplitude = dequantize(amplitude_quant)
    
    // 3. 좌표 정규화
    x_norm, y_norm = normalize_coordinates(i, j)
    
    // 4. 기저 함수 계산
    base_value = select_basis_function(function_selector, frequency, x_norm, y_norm)

    // 5. 위상 상태 적용 (핵심)
    final_value = apply_phase(phase_state, base_value)
    
    return amplitude * final_value
```
이 과정에서 CORDIC과 같은 하드웨어 친화적인 알고리즘을 사용하여 `apply_phase`와 `select_basis_function`의 연산을 최적화할 수 있다.

### 3.2.2. `F_continuous`: 연속계 코어로부터의 미세 조정

`F_continuous` 함수는 `lo` 필드에 저장된 두 개의 `f32` 실수(`r_fp32`, `theta_fp32`)를 이용해 기저 패턴을 아핀 변환(affine transformation)하여 미세 조정한다.

**입력**: `lo` (u64), `i` (행 인덱스), `j` (열 인덱스)
**출력**: 연속계 보정 값 (f32)

**의사 코드:**
```
function F_continuous(lo, i, j):
    // 1. lo 필드로부터 실수 파라미터 추출
    r_fp32, theta_fp32 = decode_floats_from_lo(lo)
    
    // 2. 좌표 기반 계산
    x_norm, y_norm = normalize_coordinates(i, j)
    dist = sqrt(x_norm^2 + y_norm^2)
    
    // 3. 패턴에 스케일 및 오프셋 적용
    // (이 함수는 설계에 따라 다양하게 정의될 수 있음)
    correction = r_fp32 - dist * r_fp32 + theta_fp32

    return correction
```
이 함수는 `F_state`가 만들어낸 거시적 패턴 위에 부드러운 그라데이션(gradient)이나 전역적인 밝기 조절과 같은 효과를 추가한다.

### 3.2.3. `F_residual`: 희소 잔차로부터의 최종 보정

`F_residual` 함수는 `residuals` 벡터에 저장된 희소 계수로부터 최종 오차 보정 값을 재구성한다.

**입력**: `residuals` (벡터), `i` (행 인덱스), `j` (열 인덱스)
**출력**: 최종 잔차 보정 값 (f32)

이 함수를 효율적으로 구현하는 것은 매우 중요하다. 가장 순진한 방법은 저장된 `K`개의 계수로 전체 잔차 행렬 `R`을 역변환(IDCT/IDWT)하여 복원한 뒤, `R[i, j]` 값을 읽는 것이다. 그러나 이는 융합 연산의 목적에 반한다.

**효율적인 구현 (IDCT 예시):**
IDCT의 선형성(linearity)을 이용하면, 특정 위치 `(i,j)`의 값은 모든 계수 `C_kl`의 가중치 합으로 표현된다.
`R[i,j] = Σ_k Σ_l C_kl * cos(...) * cos(...)`
`residuals` 벡터는 단 `K`개의 0이 아닌 `C_kl`만을 포함하므로, 이 합산은 `K`번의 연산만으로 계산될 수 있다. 이는 `K`가 행렬의 전체 크기 `N*M`에 비해 매우 작으므로 엄청난 연산량 절감 효과를 가져온다.

## 3.3. 융합 순전파: 알고리즘 전체

융합 순전파 알고리즘은 상기한 가중치 합성 원리를 행렬-벡터 곱 연산 전체에 적용한 것이다.

**알고리즘: Fused-Forward-Pass(HybridEncodedBlock `B`, DVector `x`)**
1.  출력 벡터 `y`를 0으로 초기화한다.
2.  **for** `i` in `0..output_dim`:
3.      **for** `j` in `0..input_dim`:
4.          `w_ij = F_state(B.seed.hi, i, j) + F_continuous(B.seed.lo, i, j) + F_residual(B.residuals, i, j)`
5.          `y[i] += w_ij * x[j]`
6.  **return** `y`

이 알고리즘의 4, 5번 단계는 하드웨어 수준에서 단일 커널(single kernel)로 융합되어 실행될 수 있다. 이로 인해 거대한 `W` 행렬을 위한 메모리 할당, 로딩, 캐싱이 전혀 필요 없게 되어, 이론적으로 메모리 대역폭이 아닌 **산술 연산 능력(compute capability)**에 의해서만 성능이 제한된다. 이는 '원본보다 빠른 압축 모델'을 가능하게 하는 핵심 원리이다.

## 3.4. 요약

본 장에서 설명한 '디코딩'과 '융합 순전파'는 다음과 같은 혁신을 제공한다.

1.  **메모리 장벽 제거**: 가중치 행렬의 명시적 생성을 회피하여, 모델 크기가 GPU 메모리 용량을 초과하더라도 실행 가능하다.
2.  **성능 병목 전환**: 메모리 대역폭 병목에서 벗어나, 고도로 병렬화 가능한 산술 연산 성능에 의존하게 되므로 하드웨어의 잠재력을 최대한 활용할 수 있다.
3.  **동적 가중치 생성**: 모든 가중치가 연산 시점에 즉시 생성되므로, 정적인 모델보다 더 유연한 아키텍처 설계가 가능해진다.

이러한 특성은 대규모 언어 모델(LLM)과 같이 메모리 용량이 성능의 핵심 제약 조건이 되는 분야에서 특히 강력한 이점을 제공한다. 