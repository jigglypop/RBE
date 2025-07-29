# Packed256 재설계 및 구현 계획서

> **문서 버전**: 1.0
> **작성일**: 2024-05-22
> **목표**: 기존 `Packed256` 구현의 근본적인 설계 오류를 바로잡고, 레거시 시스템의 비트 미분 철학을 1:1로 계승하는 새로운 구현을 통해 안정적인 RMSE 수렴(0.01 이하)을 달성한다.

---

## 1. 문제 진단 및 핵심 철학

### 1.1. 기존 구현의 문제점
- **설계 철학 부재**: 256비트를 단순히 필드 나열에 사용하여, 레거시의 '상태'와 '연속값' 분리 원칙을 따르지 못함.
- **수치 불안정성**: `f32`에 직접 의존하여 양자화 및 업데이트 과정에서 그래디언트 소실 또는 폭주(NaN, inf) 문제 발생.
- **비트 미분 오해**: '미리 정의된 미분 함수를 비트 패턴으로 선택'하는 핵심 로직 대신, 부정확한 수치 미분을 시도하여 오차가 누적됨.
- **복잡성**: 책임 분리가 되지 않아 `packed256_types.rs` 파일이 비대해지고 디버깅이 어려웠음.

### 1.2. 새로운 핵심 철학: 상태와 값의 완벽한 분리
새로운 `Packed256`은 레거시 시스템의 핵심 원리를 계승하여 256비트를 두 개의 u128 필드로 명확히 분리한다.

- **`hi: u128` (상태 및 제어 필드)**: 기저 함수 종류, 미분 차수, 곡률, 활성 함수 등 이산적이고 조합적인 '상태'를 제어한다. 이 값들은 주로 `if` 또는 `match` 문으로 분기하는 데 사용된다.
- **`lo: u128` (고정밀 연속 파라미터 필드)**: 좌표, 주파수, 위상 등 연속적인 '값'을 32비트 고정소수점으로 표현한다. 이 값들은 Adam과 같은 옵티마이저에 의해 점진적으로 업데이트되는 대상이다.

## 2. 신규 `Packed256` 비트 필드 설계

```mermaid
graph TD
    subgraph Packed256 (256 bits)
        direction LR
        Hi_u128(hi: u128<br/>State & Control) --> Lo_u128(lo: u128<br/>Continuous Parameters)
    end

    subgraph "hi: u128 (상태 및 제어 필드)"
        direction TB
        A[basis_id<br/>8 bits] --> B(d_r / d_theta<br/>4+4 bits)
        B --> C(log2_c<br/>8 bits)
        C --> D(activation_id<br/>8 bits)
        D --> E(q_value / k_value<br/>8+8 bits)
        E --> F(control_flags<br/>8 bits)
        F --> G(Reserved<br/>72 bits)
    end

    subgraph "lo: u128 (고정밀 연속 파라미터 필드)"
        direction TB
        H[r<br/>32-bit FP] --> I(theta<br/>32-bit FP)
        I --> J(param1<br/>32-bit FP)
        J --> K(param2<br/>32-bit FP)
    end
```

| 필드명 | 할당 위치 | 비트 수 | 타입 / 범위 | 역할 및 설명 |
| :--- | :--- | :--- | :--- | :--- |
| **lo: u128** | | **128** | | **연속 파라미터 공간 (고정밀도)** |
| `r` | `lo` | 32 | `FixedPoint32` | 쌍곡 공간의 반경(radial) 좌표. |
| `theta` | `lo` | 32 | `FixedPoint32` | 쌍곡 공간의 각도(angular) 좌표. |
| `param1` | `lo` | 32 | `FixedPoint32` | 기저 함수 고유 파라미터 1 (e.g., 주파수, 스케일). |
| `param2` | `lo` | 32 | `FixedPoint32` | 기저 함수 고유 파라미터 2 (e.g., 위상, 시프트). |
| **hi: u128** | | **128** | | **이산 상태 및 제어 공간** |
| `basis_id` | `hi` | 8 | `u8` | 사용할 핵심 기저 함수(Bessel, Morlet 등) 선택. |
| `d_r` | `hi` | 4 | `u8` (0-15) | **비트 미분:** 반경 방향 미분 차수/종류 선택. |
| `d_theta`| `hi` | 4 | `u8` (0-15) | **비트 미분:** 각도 방향 미분 차수/종류 선택. |
| `log2_c` | `hi` | 8 | `i8` | 푸앵카레 공간의 곡률 (`c = 2^log2_c`) 제어. |
| `activation_id` | `hi` | 8 | `u8` | 후처리 활성 함수(sech, tanh 등) 선택. |
| `q_value`| `hi` | 8 | `u8` | 양자화/스케일링 파라미터 1. |
| `k_value`| `hi` | 8 | `u8` | 양자화/스케일링 파라미터 2. |
| `flags` | `hi` | 8 | bitfield | 각종 제어 플래그 (`use_bias`, `is_hyperbolic` 등). |
| `reserved` | `hi` | 72 | - | 향후 확장을 위한 예약 공간. |

## 3. 구현 계획 (4-Phase)

### Phase 1: `Packed256` 타입 재설계 및 기반 구축
- **담당 파일**: `src/core/tensors/packed256_types.rs`
- **작업 내용**:
    1.  **파일 초기화**: 기존 내용 삭제 후 재시작.
    2.  **`Packed256` 구조체 정의**: `hi: u128`, `lo: u128` 필드 정의.
    3.  **비트 필드 접근자 구현**: `hi`, `lo` 내부 데이터에 안전하게 접근하는 `get_*`, `set_*` 함수 구현. (e.g., `fn get_basis_id(&self) -> u8`, `fn set_r(&mut self, r_val: f32)`)
    4.  **`decode` / `update_from_params` 재구현**: 비트 필드 접근자를 사용하여 `Packed256` ↔ `Packed256Params` (f32 값) 변환 로직 구현.

### Phase 2: 레거시 비트 미분 엔진 신규 구현
- **담당 파일**: `src/core/differential/bit_engine.rs` (신규 생성)
- **작업 내용**:
    1.  **모듈 신설**: 순수 계산 로직을 `Packed256`과 물리적으로 분리.
    2.  **레거시 수학 함수 1:1 이식**: `bessel`, `sech`, `morlet` 등 모든 레거시 수학 함수를 NaN/inf 안전 가드와 함께 그대로 이식.
    3.  **핵심 비트 미분 로직 구현**: `apply_angular_derivative`, `apply_radial_derivative` 로직을 `d_r`, `d_theta` 값에 따라 분기하는 `match` 문으로 완벽히 재현.
    4.  **통합 계산 함수 `compute_fused_output` 구현**: `Packed256Params`를 입력받아 `(predicted_value, grad_r, grad_theta)` 튜플을 반환하는 단일 진입점 함수 구현.

### Phase 3: `RBESeed` Trait 통합 및 안정화
- **담당 파일**: `src/core/tensors/packed256_types.rs`, `src/core/optimizers/adam.rs`
- **작업 내용**:
    1.  **`Packed256`과 `bit_engine` 연동**: `fused_forward_256`, `compute_gradients`가 내부적으로 `bit_engine::compute_fused_output`을 호출하도록 수정.
    2.  **안정적인 `random` 생성자**: `Packed256::random()`이 처음부터 수치적으로 안정적인 범위(`r` ∈ [0.2, 0.8], `log2_c` ∈ [-4, 4] 등)의 파라미터만 생성하도록 제한.
    3.  **`adam_update` 구현**: `RBESeed` 트레이트 요구사항에 맞춰, 계산된 그래디언트를 받아 `lo` 필드의 고정소수점 값들을 안정적으로 업데이트.

### Phase 4: 전체 정리 및 최종 수렴 테스트
- **담당 파일**: `tests/core/tensors/packed256_convergence_test.rs` (신규 생성) 및 프로젝트 전역
- **작업 내용**:
    1.  **프로젝트 클린업**: `cargo fix` 및 수동 수정을 통해 모든 `dead_code`, `unused_import` 등 컴파일 경고 제거.
    2.  **기존 테스트 삭제**: 혼란을 야기했던 모든 `packed256` 관련 테스트 파일 삭제.
    3.  **단일 최종 수렴 테스트 작성**: `packed256_최종_수렴_및_정합성_테스트` 케이스 작성.
        - **목표**: Adam 옵티마이저, 5000 epoch 이내, RMSE < `0.01` 수렴 증명.
        - **검증**: 주요 기저 함수(0~3)에 대해 각각 테스트 실행.

## 4. 자원 분석 및 결론
256비트는 본 설계안에 따라 구현 시, 목표 기능 달성에 충분하며 오히려 향후 확장을 위한 72비트의 예약 공간까지 확보할 수 있다. 문제의 원인은 비트의 양이 아닌 설계의 질이었으며, 본 계획은 이를 근본적으로 해결하는 것을 목표로 한다.

---
*본 계획서는 상호 합의 하에 수정될 수 있으며, 각 Phase 완료 시 보고를 통해 진행 상황을 공유한다.* 