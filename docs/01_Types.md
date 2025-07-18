# 1. 데이터 구조: 이중-코어 파라미터의 설계

## 1.1. 개요

본 압축 패러다임의 핵심은 신경망의 가중치를 표현하는 데이터 구조 그 자체에 있다. 기존의 동종(homogeneous) 실수형 파라미터 배열 방식에서 탈피하여, 단일 파라미터 내에 질적으로 다른 두 가지 학습 메커니즘을 융합하는 이종(heterogeneous) 데이터 구조를 제안한다. 본 장에서는 이 이중-코어(Dual-Core) 파라미터의 기본 단위인 `Packed128`과, 최종 오차 보정을 위한 `HybridEncodedBlock`의 설계 사상과 기술적 사양을 상세히 기술한다.

## 1.2. `Packed128`: 파라미터의 원자 단위

`Packed128`은 128비트의 공간 안에, 양자화된 이산 상태(discrete state)와 연속계 실수(continuous value)를 동시에 인코딩하는 핵심 데이터 구조이다. 이는 두 개의 64비트 정수, `hi`와 `lo`로 구성된다.

```rust
pub struct Packed128 {
    pub hi: u64,   // 상태-전이 코어 (State-Transition Core)
    pub lo: u64,   // 연속계 보정 코어 (Continuous-Refinement Core)
}
```

### 1.2.1. `hi: u64` - 상태-전이 코어 (The State-Transition Core)

`hi` 필드는 단순한 난수 시드가 아니라, 가중치 패턴을 생성하는 함수의 동작을 정의하는 고도로 구조화된 **상태 기계(State Machine)**이다. 각 비트 그룹은 기저 함수(basis function)의 특정 속성을 제어한다.

#### 제안 비트 필드 레이아웃 (Proposed Bit-Field Layout)

다음은 `hi` 필드의 64비트에 대한 구체적인 역할 할당 예시이다.

| 비트 범위 (Bits) | 크기 (Size) | 필드명 (Field Name)     | 설명 (Description)                                                                                                    |
| :--------------- | :---------- | :---------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| `[63:62]`        | 2 bits      | **`phase_state`**       | 기저 함수의 **위상 상태**를 인코딩. `sin` 계열 함수의 주기적 미분 관계를 표현하는 핵심. `00(sin)`, `01(cos)`, `10(-sin)`, `11(-cos)` |
| `[61:50]`        | 12 bits     | `frequency_quant`       | 기저 함수의 주파수(frequency)를 양자화하여 저장.                                                                        |
| `[49:38]`        | 12 bits     | `amplitude_quant`       | 기저 함수의 기본 진폭(amplitude)을 양자화하여 저장.                                                                       |
| `[37:32]`        | 6 bits      | `function_selector`     | `sin`, `tanh`, `bessel` 등 사용할 기저 함수의 종류를 선택.                                                              |
| `[31:0]`         | 32 bits     | `transform_sequence`    | CORDIC과 같은 직교 변환(orthogonal transformation)에 사용될 회전 시퀀스 또는 기타 파라미터.                                 |

**핵심 개념**: 이 구조에서 '학습'의 일부는 `phase_state` 비트의 상태 전이를 통해 이루어진다. 이는 3장에서 상세히 논한다.

### 1.2.2. `lo: u64` - 연속계 보정 코어 (The Continuous-Refinement Core)

`lo` 필드는 `hi` 코어가 생성한 기본 패턴을 미세 조정(fine-tuning)하기 위한 연속계(continuous-domain) 파라미터를 담고 있다. 이 64비트는 두 개의 32비트 부동소수점(`f32`) 값으로 해석된다.

```rust
// lo 필드의 해석
let r_fp32     = f32::from_bits((lo >> 32) as u32);
let theta_fp32 = f32::from_bits(lo as u32);
```

-   **`r_fp32`**: `hi` 코어가 생성한 패턴의 크기(magnitude) 또는 스케일(scale)을 조절하는 역할을 수행한다.
-   **`theta_fp32`**: 패턴에 위상 변이(phase shift) 또는 오프셋(offset)을 가하는 역할을 한다.

이 두 실수는 표준적인 경사 하강법(Gradient Descent)을 통해 직접 학습되며, `hi` 코어의 이산적 상태 변화로는 표현하기 어려운 부드럽고 연속적인 패턴의 변형을 담당한다.

## 1.3. `HybridEncodedBlock`: 최종 표현

`HybridEncodedBlock`은 신경망 가중치 행렬의 한 블록(예: 64x64)을 표현하는 최종 데이터 컨테이너이다. 이는 이중-코어 파라미터와 최종 오차 보정 정보를 통합한다.

```rust
pub struct HybridEncodedBlock {
    /// 이중-코어 파라미터. 블록의 기본 패턴과 연속적 변형을 정의한다.
    pub seed: Packed128,

    /// 최종 잔차(residual) 보정을 위한 희소(sparse) 계수 벡터.
    pub residuals: Vec<ResidualCoefficient>,
    
    // ... 블록의 크기 등 메타데이터
}

pub struct ResidualCoefficient {
    pub index: u32, // 1차원으로 평탄화된 계수의 인덱스
    pub value: f32, // 계수의 실수 값
}
```

-   **`seed: Packed128`**: 블록의 패턴을 생성하는 핵심 '유전자'이다. `hi`와 `lo` 코어를 통해 대부분의 정보를 표현한다.
-   **`residuals: Vec<ResidualCoefficient>`**: `seed`만으로는 완벽히 표현되지 않는 미세한 고주파성 오차(high-frequency error components)를 보정하기 위한 정보다. 전체 잔차 행렬에서, 에너지(크기)가 가장 큰 상위 K개의 DCT 또는 Wavelet 계수만을 희소하게 저장한다. 이 계수들 또한 `f32` 실수로서 연속계 학습의 대상이 된다.

## 1.4. 요약

본 장에서 제안한 데이터 구조는 다음과 같은 설계 철학을 가진다.

1.  **파라미터의 이중성**: 단일 파라미터 `Packed128` 안에 이산적 상태-전이 메커니즘(`hi`)과 연속적 미세조정 메커니즘(`lo`)을 함께 내장시켰다.
2.  **계층적 표현**: `Packed128`이 가중치의 거시적/핵심적 특징을 정의하고, `residuals`가 미시적/희소적 특징을 보정하는 계층적(hierarchical) 정보 표현 구조를 가진다.
3.  **학습 가능성 내재**: 구조의 모든 부분-`hi`의 상태, `lo`의 실수 값, `residuals`의 계수-이 각각에 맞는 방식으로 학습될 수 있도록 설계되었다.

이러한 데이터 구조는 이어지는 장에서 설명할 '융합 순전파'와 '하이브리드 역전파'의 기반이 된다. 