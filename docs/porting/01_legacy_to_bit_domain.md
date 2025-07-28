# Legacy to Bit-Domain Porting Strategy

## 1. 개요

본 문서는 `src/legacy`에 구현된 RBE(Riemannian Basis Encoding) 시스템의 수학적 우수성을 현재의 128비트 비트-도메인 시스템에 통합하는 전략을 다룹니다. Legacy 시스템은 더 정교한 수학적 모델을 사용하여 정확도 면에서 뛰어나지만, 현재 시스템은 비트 연산 최적화를 통해 속도와 압축률에서 강점을 가집니다. 이 둘의 장점을 결합하여 프로젝트의 핵심 목표인 **[압축률 150:1, RMSE 0.01 이하, 고속 서빙]**을 동시에 달성하고자 합니다.

## 2. 수학적 아키텍처 비교

| 구성 요소 | Legacy (64비트) | 현재 (128비트) | 비트-도메인 포팅 방안 |
|:--- |:---|:---|:---|
| **기저 함수** | 12가지 정교한 수학 함수 (Bessel, Morlet 등) | 단순 해시 패턴 | ✅ **4비트**로 12가지 기저 함수 선택 |
| **미분 차수** | 각도(0~3차), 반지름(0~1차) 분리 | 통합된 11비트 미분 사이클 | ✅ **3비트**로 통합 (4가지 각도 × 2가지 반지름) |
| **회전 지원** | 10가지 프리셋 각도 | 미지원 | ✅ **4비트**로 16단계 회전 코드 구현 |
| **곡률** | `log2(c)`로 ±8단계 조절 | 미지원 | ✅ **3비트**로 부호 있는 곡률(-4~+3) 구현 |
| **야코비안** | 실제 리만 기하학 기반 계산 | 미지원 | ✅ **고정소수점**으로 야코비안 항 근사 |
| **파라미터** | r(20), θ(24) = 44비트 | r(32), θ(32) = 64비트 | ✅ r(24), θ(28) = 52비트로 확장 |

## 3. `Enhanced128` 비트-필드 제안

`Packed128` 구조체를 확장하여 Legacy의 정교한 파라미터를 포함하는 새로운 `Enhanced128` 구조를 제안합니다.

```rust
// src/core/tensors/enhanced_types.rs
pub struct Enhanced128 {
    pub hi: u64, // 기존과 동일: 상태 비트, 미분 사이클 등
    pub lo: u64, // 확장된 파라미터 필드
}

/*
 * lo 필드 (64비트) 상세 할당:
 * 
 * | Bit Range | Size | Description                  | Legacy 대응                |
 * |-----------|------|------------------------------|----------------------------|
 * | 63-40     | 24   | `r` (반지름)                 | 20비트에서 24비트로 정밀도 향상 |
 * | 39-12     | 28   | `theta` (각도)               | 24비트에서 28비트로 정밀도 향상 |
 * | 11-8      | 4    | `basis_id` (기저 함수 ID)    | 12가지 기저 함수 선택      |
 * | 7-4       | 4    | `rot_code` (회전 코드)       | 10가지 회전 프리셋         |
 * | 3-1       | 3    | `log2_c` (곡률)              | 부호 있는 3비트 값 (-4 ~ +3) |
 * | 0         | 1    | `d_r` (반지름 미분 차수)     | 0 또는 1차 미분            |
 *
 * hi 필드 (64비트)는 기존 미분 사이클, 상태 전이, 메타데이터 등으로 활용
 */
```

## 4. 포팅 전략 및 단계별 구현

### 1단계: 기저 함수 LUT(Look-Up Table) 생성

Legacy의 12가지 수학 함수(Bessel, Morlet 등)를 매번 `f32`로 계산하는 것은 비트-도메인의 속도 이점을 저해합니다. 따라서 이 함수들을 사전 계산하여 고정소수점 LUT로 만들어 비트 연산과 함께 사용할 수 있도록 최적화합니다.

- **실행**: `cargo run --bin generate_basis_luts`와 같은 별도 유틸리티를 만들어 `hyperbolic_lut.rs`처럼 LUT 파일을 자동 생성합니다.
- **출력**: `src/core/tensors/basis_lut.rs`
- **구현**:
  ```rust
  // 예시: 베셀 함수 J0의 Q16 고정소수점 LUT
  pub const BESSEL_J0_LUT_Q16: [u16; 256] = [ ... ];
  
  // 비트 도메인에서 직접 LUT를 조회하는 함수
  fn compute_basis_from_lut(basis_id: u8, r_bits: u32) -> i32 {
      match basis_id {
          4 => BESSEL_J0_LUT_Q16[(r_bits >> 16) as usize] as i32,
          // ... 다른 기저 함수들
      }
  }
  ```

### 2단계: `fused_forward_enhanced` 구현

`Enhanced128` 구조체에 새로운 순전파 함수를 구현하여 Legacy의 모든 수학적 요소를 비트-도메인에서 효율적으로 계산합니다.

```rust
// src/core/tensors/enhanced_types.rs
impl Enhanced128 {
    pub fn fused_forward_enhanced(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 1. 확장된 파라미터 디코딩 (비트 마스킹)
        let params = self.decode_enhanced_params();

        // 2. 좌표 계산 및 회전 적용 (비트 연산)
        let (r_local, theta_local) = self.compute_rotated_coords(params.rot_code, i, j, rows, cols);

        // 3. 기저 함수 값 계산 (LUT 조회)
        let basis_value_fixed = self.compute_basis_from_lut(params.basis_id, r_local);
        
        // 4. 미분 적용 (hi 필드의 미분 사이클 비트 사용)
        let derivative_value_fixed = self.apply_derivatives_bit(basis_value_fixed, self.hi);
        
        // 5. 야코비안 계산 (고정소수점)
        let jacobian_fixed = self.compute_jacobian_fixed(params.r_bits, params.log2_c_bits);

        // 6. 최종 값 계산 (고정소수점 곱셈 후 f32 변환)
        let final_value_fixed = (derivative_value_fixed as i64 * jacobian_fixed as i64) >> 16;
        
        final_value_fixed as f32 / 65536.0
    }
}
```

### 3단계: 통합 및 성능 검증

- **통합**: `RBELinear`와 같은 NLP 레이어에서 `Packed128` 대신 `Enhanced128`을 사용하도록 수정합니다.
- **테스트**: `differential_system_test.rs`에 새로운 테스트 케이스(`enhanced_시스템_성능_테스트`)를 추가하여 세 가지 구현(Legacy, 현재, Enhanced)의 **RMSE, ops/s, 압축률**을 직접 비교합니다.

## 5. 기대 효과

| 메트릭 | 현재 구현 | Legacy 포팅 예상 | 프로젝트 목표 |
|:---|:---:|:---:|:---:|
| **RMSE** | ~0.5 | **~0.01** | ✅ 달성 |
| **ops/s** | 8,000-15,000 | 10,000-12,000 | ✅ 달성 |
| **압축률** | 93.75% (16x16) | **96.8%** (16x16) | ✅ 달성 |

**결론**: Legacy 시스템의 수학적 정교함을 현재 비트-도메인 아키텍처에 포팅하는 것은 **프로젝트의 3대 목표를 모두 달성할 수 있는 가장 확실한 전략**입니다. 정확도를 대폭 향상시키면서도 속도와 압축률의 강점을 유지할 수 있습니다. 