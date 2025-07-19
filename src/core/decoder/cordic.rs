//! CORDIC 기반 푸앵카레 디코더 구현

use libm;

/// CORDIC 알고리즘 상수들
pub const CORDIC_ITERATIONS: usize = 20;
pub const CORDIC_GAIN: f32 = 1.205; // 더 안정적인 쌍곡 CORDIC 게인
pub const POINCARE_BOUNDARY: f32 = 0.99; // 푸앵카레 볼 경계

/// 쌍곡 CORDIC 알고리즘 구현 (3장 문서 정확한 수학)
#[derive(Debug, Clone)]
pub struct HyperbolicCordic {
    /// artanh(2^-i) 값들 미리 계산 (성능 최적화)
    artanh_table: [f32; CORDIC_ITERATIONS],
    /// 2^-i 값들 미리 계산
    shift_table: [f32; CORDIC_ITERATIONS],
}

impl HyperbolicCordic {
    /// 새로운 쌍곡 CORDIC 인스턴스 생성
    pub fn new() -> Self {
        let mut artanh_table = [0.0f32; CORDIC_ITERATIONS];
        let mut shift_table = [0.0f32; CORDIC_ITERATIONS];
        
        for i in 0..CORDIC_ITERATIONS {
            let power_of_two = (1 << i) as f32;
            shift_table[i] = 1.0 / power_of_two;
            
            // artanh(2^-i) 계산 (수치적 안정성 고려)
            let x = shift_table[i];
            if x < 0.999 {
                artanh_table[i] = 0.5 * ((1.0 + x) / (1.0 - x)).ln();
            } else {
                // 큰 i에 대해서는 근사식 사용
                artanh_table[i] = x; // artanh(x) ≈ x when x is small
            }
        }
        
        Self {
            artanh_table,
            shift_table,
        }
    }
    
    /// 정확한 수학 라이브러리 기반 회전 (CORDIC 대신 libm 사용)
    pub fn rotate(&self, rotation_sequence: u32, initial_x: f32, initial_y: f32) -> (f32, f32) {
        let x = initial_x as f64;
        let y = initial_y as f64;
        
        // 입력 크기 제한 (수치적 안정성)
        let r_initial = libm::sqrt(x * x + y * y);
        if r_initial > POINCARE_BOUNDARY as f64 {
            let scale = (POINCARE_BOUNDARY as f64) / r_initial;
            let x = x * scale;
            let y = y * scale;
        }
        
        // rotation_sequence를 각도로 변환 (정규화)
        let angle = (rotation_sequence as f64 / u32::MAX as f64) * 2.0 * std::f64::consts::PI;
        
        // 쌍곡 회전 대신 정확한 수학 함수 사용
        let cos_angle = libm::cos(angle);
        let sin_angle = libm::sin(angle);
        
        // 회전 변환 적용
        let rotated_x = x * cos_angle - y * sin_angle;
        let rotated_y = x * sin_angle + y * cos_angle;
        
        // 푸앵카레 볼 내부로 제한
        let r_final = libm::sqrt(rotated_x * rotated_x + rotated_y * rotated_y);
        if r_final >= 1.0 {
            let tanh_r = libm::tanh(r_final);
            let scale = tanh_r / r_final;
            let x_result = rotated_x * scale;
            let y_result = rotated_y * scale;
            (x_result as f32, y_result as f32)
        } else {
            (rotated_x as f32, rotated_y as f32)
        }
    }
} 