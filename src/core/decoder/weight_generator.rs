//! 5단계 가중치 생성 파이프라인

use crate::core::packed_params::{PoincarePackedBit128, PoincareQuadrant};
use super::cordic::{hyperbolic_cordic, POINCARE_BOUNDARY};
use libm;

/// 5단계 가중치 생성 파이프라인 (문서 3.3.1)
#[derive(Debug, Clone)]
pub struct WeightGenerator {
    // CORDIC 함수를 직접 사용하므로 필드 불필요
}

impl WeightGenerator {
    /// 새로운 가중치 생성기 생성
    pub fn new() -> Self {
        Self {
        }
    }

    /// 1단계: 이산 비트에서 초기 회전각 생성
    /// 문서 3.3.1.1: Discrete Bit Rotation Mapping
    pub fn discrete_rotation(&self, bits: u32) -> f32 {
        // 11비트를 [0, 2π) 범위로 정규화
        let normalized = (bits as f32) / (1u32 << 11) as f32;
        normalized * 2.0 * std::f32::consts::PI
    }

    /// 2단계: 쌍곡 CORDIC 회전
    /// 문서 3.3.1.2: Hyperbolic CORDIC Rotation
    pub fn hyperbolic_rotation(&self, r: f32, theta: f32, discrete_angle: f32) -> (f32, f32) {
        let x = r * libm::cosf(theta);
        let y = r * libm::sinf(theta);
        
        // rotation_sequence를 angle로 변환
        let target_angle = discrete_angle;
        
        hyperbolic_cordic(x, y, target_angle)
    }
    
    /// 단일 가중치 생성 (5단계 파이프라인)
    pub fn generate_weight(
        &self,
        packed: &PoincarePackedBit128,
        row: usize,
        col: usize,
        total_rows: usize,
        total_cols: usize,
    ) -> f32 {
        // 1단계: 비트 추출 (문서 3.3.2)
        let (quadrant, hyp_freq, geo_amp, basis_sel, cordic_seq) = self.extract_bits(packed);
        
        // 2단계: 좌표 정규화 (문서 3.3.3)
        let (x_norm, y_norm) = self.normalize_coordinates(row, col, total_rows, total_cols);
        
        // 3단계: CORDIC 회전 (문서 3.3.4)
        let (x_rotated, y_rotated) = self.apply_cordic_rotation(
            cordic_seq, x_norm, y_norm, packed.get_r_poincare(), packed.get_theta_poincare()
        );
        
        // 4단계: 기저함수 적용 (문서 3.3.5)
        let base_value = self.apply_basis_function(quadrant, x_rotated, y_rotated, hyp_freq);
        
        // 5단계: 연속 보정 (문서 3.3.6)
        let final_weight = self.apply_continuous_correction(base_value, geo_amp, basis_sel, x_rotated, y_rotated);
        
        // NaN 체크 및 안정성을 위한 클램핑
        if final_weight.is_finite() {
            final_weight.clamp(-1.0, 1.0)
        } else {
            // NaN이면 위치 기반 기본값 반환
            let fallback = ((row as f32 / total_rows as f32) - 0.5) * 0.1;
            fallback.clamp(-1.0, 1.0)
        }
    }
    
    /// 1단계: 구조화된 비트 추출
    fn extract_bits(&self, packed: &PoincarePackedBit128) -> (PoincareQuadrant, u16, u16, u8, u32) {
        let quadrant = packed.get_quadrant();
        let hyp_freq = packed.get_hyperbolic_frequency();
        let geo_amp = packed.get_geodesic_amplitude();
        let basis_sel = packed.get_basis_function_selector();
        let cordic_seq = packed.get_cordic_rotation_sequence();
        
        (quadrant, hyp_freq, geo_amp, basis_sel, cordic_seq)
    }
    
    /// 2단계: 좌표 정규화 (문서 3.3.3 공식)
    fn normalize_coordinates(&self, row: usize, col: usize, total_rows: usize, total_cols: usize) -> (f32, f32) {
        // 단일 점인 경우 중심으로 설정
        let x_norm = if total_cols > 1 {
            (2.0 * col as f32) / (total_cols - 1) as f32 - 1.0
        } else {
            0.0
        };
        
        let y_norm = if total_rows > 1 {
            (2.0 * row as f32) / (total_rows - 1) as f32 - 1.0
        } else {
            0.0
        };
        
        // 푸앵카레 볼 경계 조건 처리
        let r_max = (x_norm * x_norm + y_norm * y_norm).sqrt();
        if r_max >= POINCARE_BOUNDARY {
            let scale = POINCARE_BOUNDARY / r_max;
            (x_norm * scale, y_norm * scale)
        } else {
            (x_norm, y_norm)
        }
    }
    
    /// 3단계: CORDIC 쌍곡회전 적용
    fn apply_cordic_rotation(
        &self,
        cordic_seq: u32,
        x_norm: f32,
        y_norm: f32,
        r_poincare: f32,
        theta_poincare: f32,
    ) -> (f32, f32) {
        // 좌표에 따른 공간적 변조 추가 (다양성 확보)
        let spatial_modulation = libm::sinf(x_norm * 3.14159) * libm::cosf(y_norm * 2.71828);
        let coordinate_hash = libm::sinf(x_norm * 7.389 + y_norm * 5.772); // 더 다양한 해시
        
        // 초기 벡터 설정 (문서 3.3.4 + 개선)
        let base_angle = if x_norm.abs() < 1e-6 && y_norm.abs() < 1e-6 {
            0.0  // 중심점 처리
        } else {
            libm::atan2f(y_norm, x_norm)
        };
        
        let combined_angle = base_angle + theta_poincare + spatial_modulation * 0.2 + coordinate_hash * 0.15;
        
        // r_poincare에 위치별 변조 적용 (더 큰 다양성)
        let modulated_r = r_poincare * (1.0 + spatial_modulation * 0.3 + coordinate_hash * 0.25).clamp(0.1, 2.0);
        
        let initial_x = modulated_r * libm::cosf(combined_angle);
        let initial_y = modulated_r * libm::sinf(combined_angle);
        
        // CORDIC 회전 수행
        hyperbolic_cordic(initial_x, initial_y, combined_angle)
    }
    
    /// 4단계: 쌍곡 기저함수 적용 (문서 3.3.5 매핑 테이블)
    fn apply_basis_function(
        &self,
        quadrant: PoincareQuadrant,
        x_rotated: f32,
        y_rotated: f32,
        hyp_freq: u16,
    ) -> f32 {
        let r_final = libm::sqrtf(x_rotated * x_rotated + y_rotated * y_rotated);
        let theta_final = libm::atan2f(y_rotated, x_rotated);
        
        // 스케일링 팩터 계산 (더 섬세한 범위)
        let alpha = (hyp_freq as f32 / 4095.0) * 5.0 + 0.1; // [0.1, 5.1] 범위
        let scaled_r = alpha * r_final;
        
        // 각도에 따른 추가 변조 (더 강화)
        let angular_modulation = libm::sinf(theta_final * 2.0) * 0.2;
        let frequency_modulation = libm::sinf(hyp_freq as f32 * 0.001) * 0.15; // 주파수 기반 변조
        let modulated_input = scaled_r + angular_modulation + frequency_modulation;
        
        // 수치적 안정성을 위한 클램핑 (더 보수적)
        let safe_input = modulated_input.clamp(-3.0, 3.0);
        
        let base_result = match quadrant {
            PoincareQuadrant::First => {   // sinh 함수 (libm으로 정확하게)
                libm::sinhf(safe_input).clamp(-5.0, 5.0)
            },
            PoincareQuadrant::Second => {  // cosh 함수 (libm으로 정확하게)
                libm::coshf(safe_input).clamp(1.0, 10.0)
            },
            PoincareQuadrant::Third => {   // tanh 함수 (libm으로 정확하게)
                libm::tanhf(safe_input) // tanh는 자연스럽게 [-1,1] 범위
            },
            PoincareQuadrant::Fourth => {  // sech² 함수 (libm으로 정확하게)
                let cosh_val = libm::coshf(safe_input).max(1.0);
                let sech = 1.0 / cosh_val;
                (sech * sech).clamp(0.01, 1.0)
            }
        };
        
        // NaN 체크 및 최종 처리
        if !base_result.is_finite() {
            // NaN이면 안전한 기본값 반환
            return (r_final * 0.1).clamp(-1.0, 1.0);
        }
        
        // 미세한 노이즈 추가로 다양성 확보 (deterministic, 더 큰 변동)
        let position_hash = ((x_rotated * 1000.0) as i32 ^ (y_rotated * 1000.0) as i32) as f32;
        let noise = libm::sinf(position_hash * 0.00012345) * 0.05; // 5배 증가
        let final_result = base_result + noise;
        
        // 최종 NaN 체크
        if final_result.is_finite() {
            final_result
        } else {
            (r_final * 0.1).clamp(-1.0, 1.0)
        }
    }
    
    /// 5단계: 연속 파라미터 보정 (문서 3.3.6)
    fn apply_continuous_correction(
        &self,
        base_value: f32,
        geo_amp: u16,
        basis_sel: u8,
        x_rotated: f32,
        y_rotated: f32,
    ) -> f32 {
        // 진폭 함수 계산
        let amplitude = (geo_amp as f32 / 4095.0) * 2.0 - 1.0; // [-1, 1] 범위
        
        // 변조 함수 계산
        let theta_final = libm::atan2f(y_rotated, x_rotated);
        let modulation = match basis_sel {
            0 => 1.0, // 변조 없음
            s if s <= 31 => libm::cosf(s as f32 * theta_final),
            s => libm::sinf((s - 32) as f32 * theta_final),
        };
        
        let result = amplitude * base_value * modulation;
        
        // NaN 체크
        if result.is_finite() {
            result
        } else {
            // NaN이면 기본값 반환
            base_value * 0.1
        }
    }
    
    /// CORDIC 정확성 검증 (성능 테스트용)
    pub fn verify_cordic_accuracy(&self, test_cases: usize) -> f32 {
        let mut total_error = 0.0;
        
        for i in 0..test_cases {
            let angle = (i as f32 / test_cases as f32) * 2.0 * std::f32::consts::PI;
            let x = libm::cosf(angle) * 0.5;
            let y = libm::sinf(angle) * 0.5;
            
            let rotation_seq = (i as u32 * 12345) % u32::MAX;
            let (rotated_x, rotated_y) = hyperbolic_cordic(x, y, angle);
            
            // 기대값과 비교 (간단한 검증)
            let expected_magnitude = libm::sqrtf(x * x + y * y);
            let actual_magnitude = libm::sqrtf(rotated_x * rotated_x + rotated_y * rotated_y);
            
            let error = libm::fabsf(expected_magnitude - actual_magnitude);
            total_error += error;
        }
        
        total_error / test_cases as f32
    }
    
    /// 경계 안정성 테스트
    pub fn test_boundary_stability(&self) -> bool {
        let test_values = [
            (0.99, 0.0),
            (0.0, 0.99),
            (-0.99, 0.0),
            (0.0, -0.99),
            (0.7, 0.7),
            (-0.7, -0.7),
        ];
        
        for &(x, y) in &test_values {
            let (result_x, result_y) = hyperbolic_cordic(x, y, libm::atan2f(y, x));
            let magnitude = libm::sqrtf(result_x * result_x + result_y * result_y);
            
            if magnitude >= 1.0 || !result_x.is_finite() || !result_y.is_finite() {
                return false;
            }
        }
        
        true
    }
} 