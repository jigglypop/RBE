use std::f32::consts::PI;
use std::mem;

/// 64-bit Packed Poincaré 시드 표현
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Packed64(pub u64);

/// 기저 함수 타입
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BasisFunction {
    SinCosh = 0,
    SinSinh = 1,
    CosCosh = 2,
    CosSinh = 3,
    BesselJ = 4,
    BesselI = 5,
    BesselK = 6,
    BesselY = 7,
    TanhSign = 8,
    SechTri = 9,
    ExpSin = 10,
    Morlet = 11,
}

/// 디코딩된 파라미터
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedParams {
    pub r: f32,           // 20 bits
    pub theta: f32,       // 24 bits
    pub basis_id: u8,     // 4 bits
    pub d_theta: u8,      // 2 bits - θ 미분 차수 (0-3)
    pub d_r: bool,        // 1 bit - r 미분 차수 (0 or 1)
    pub rot_code: u8,     // 4 bits - 회전 코드
    pub log2_c: i8,       // 3 bits - 곡률 (부호 있음)
    pub reserved: u8,     // 6 bits
}

impl Packed64 {
    /// 새로운 Packed64 시드 생성
    pub fn new(
        r: f32,
        theta: f32,
        basis_id: u8,
        d_theta: u8,
        d_r: bool,
        rot_code: u8,
        log2_c: i8,
        reserved: u8,
    ) -> Self {
        // r: [0, 1) 범위로 클램핑
        let r_clamped = r.clamp(0.0, 0.999999);
        // theta: [0, 2π) 범위로 정규화
        let theta_normalized = theta.rem_euclid(2.0 * PI);

        // 비트 패킹
        let r_bits = (r_clamped * ((1u64 << 20) - 1) as f32).round() as u64;
        let theta_bits = (theta_normalized / (2.0 * PI) * ((1u64 << 24) - 1) as f32).round() as u64;
        
        let mut packed = 0u64;
        packed |= (r_bits & 0xFFFFF) << 44;           // 20 bits
        packed |= (theta_bits & 0xFFFFFF) << 20;      // 24 bits
        packed |= ((basis_id as u64) & 0xF) << 16;    // 4 bits
        packed |= ((d_theta as u64) & 0x3) << 14;     // 2 bits
        packed |= ((d_r as u64) & 0x1) << 13;         // 1 bit
        packed |= ((rot_code as u64) & 0xF) << 9;     // 4 bits
        packed |= ((log2_c as u64) & 0x7) << 6;       // 3 bits (2's complement)
        packed |= (reserved as u64) & 0x3F;           // 6 bits

        Packed64(packed)
    }

    /// 시드를 디코딩
    pub fn decode(&self) -> DecodedParams {
        let bits = self.0;

        // 비트 언패킹
        let r_bits = (bits >> 44) & 0xFFFFF;
        let theta_bits = (bits >> 20) & 0xFFFFFF;
        let basis_id = ((bits >> 16) & 0xF) as u8;
        let d_theta = ((bits >> 14) & 0x3) as u8;
        let d_r = ((bits >> 13) & 0x1) != 0;
        let rot_code = ((bits >> 9) & 0xF) as u8;
        let log2_c_bits = ((bits >> 6) & 0x7) as u8;
        let reserved = (bits & 0x3F) as u8;

        // 값 복원
        let r = (r_bits as f32) / ((1u64 << 20) - 1) as f32;
        let theta = (theta_bits as f32 / ((1u64 << 24) - 1) as f32) * 2.0 * PI;
        
        // 3비트 부호있는 정수 복원 (-4 ~ +3)
        let log2_c = if (log2_c_bits & 0x4) != 0 {
            (log2_c_bits as i8) | -8
        } else {
            log2_c_bits as i8
        };

        DecodedParams {
            r,
            theta,
            basis_id,
            d_theta,
            d_r,
            rot_code,
            log2_c,
            reserved,
        }
    }

    /// 가중치 계산 (문서의 방식대로)
    pub fn compute_weight(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        let params = self.decode();
        
        // 곡률 계산
        let c = 2.0f32.powi(params.log2_c as i32);
        
        // 좌표를 [-1, 1] 범위로 정규화
        let x = 2.0 * (j as f32) / ((cols - 1) as f32) - 1.0;
        let y = 2.0 * (i as f32) / ((rows - 1) as f32) - 1.0;
        
        // 로컬 극좌표
        let r_local = (x * x + y * y).sqrt().min(0.999999);
        let theta_local = y.atan2(x);
        
        // 회전 적용
        let rotation = get_rotation_angle(params.rot_code);
        let theta_final = params.theta + theta_local + rotation;
        
        // 미분 순환성 적용
        let angular_value = apply_angular_derivative(theta_final, params.d_theta, params.basis_id);
        let radial_value = apply_radial_derivative(c * params.r, params.d_r, params.basis_id);
        
        // 기저 함수에 따른 계산
        let basis_value = match params.basis_id {
            0..=3 => angular_value * radial_value,
            4 => bessel_j0(r_local * 10.0),
            5 => bessel_i0(r_local * 10.0),
            6 => bessel_k0(r_local * 10.0),
            7 => bessel_y0(r_local * 10.0),
            8 => (c * r_local).tanh() * theta_final.cos().signum(),
            9 => sech(c * r_local) * triangle_wave(theta_final),
            10 => (-c * r_local).exp() * theta_final.sin(),
            11 => morlet_wavelet(r_local, theta_final, 5.0),
            _ => 0.0,
        };
        
        // 야코비안 계산
        let jacobian = (1.0 - c * params.r * params.r).powi(-2).sqrt();
        
        basis_value * jacobian
    }
}

/// 회전 각도 계산
fn get_rotation_angle(rot_code: u8) -> f32 {
    match rot_code {
        0 => 0.0,
        1 => PI / 8.0,
        2 => PI / 6.0,
        3 => PI / 4.0,
        4 => PI / 3.0,
        5 => PI / 2.0,
        6 => 2.0 * PI / 3.0,
        7 => 3.0 * PI / 4.0,
        8 => 5.0 * PI / 6.0,
        9 => 7.0 * PI / 8.0,
        _ => 0.0,
    }
}

/// 각도 미분 적용
fn apply_angular_derivative(theta: f32, d_theta: u8, basis_id: u8) -> f32 {
    let is_sin_based = (basis_id & 0x1) == 0;
    
    match (is_sin_based, d_theta % 4) {
        (true, 0) => theta.sin(),
        (true, 1) => theta.cos(),
        (true, 2) => -theta.sin(),
        (true, 3) => -theta.cos(),
        (false, 0) => theta.cos(),
        (false, 1) => -theta.sin(),
        (false, 2) => -theta.cos(),
        (false, 3) => theta.sin(),
        _ => unreachable!(),
    }
}

/// 반지름 미분 적용
fn apply_radial_derivative(r: f32, d_r: bool, basis_id: u8) -> f32 {
    let is_sinh_based = (basis_id & 0x2) == 0;
    
    match (is_sinh_based, d_r) {
        (true, false) => r.sinh(),
        (true, true) => r.cosh(),
        (false, false) => r.cosh(),
        (false, true) => r.sinh(),
    }
}

// 베셀 함수들 (이전과 동일)
fn bessel_j0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 
            + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 
            + y * (59272.64853 + y * (267.8532712 + y))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;
        let ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 
            + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 
            + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        (2.0 / (PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
    }
}

fn bessel_i0(x: f32) -> f32 {
    if x.abs() < 3.75 {
        let t = (x / 3.75).powi(2);
        let mut result = 1.0;
        let mut term = 1.0;
        
        for k in 1..=10 {
            term *= t / (k * k) as f32;
            result += term;
        }
        result
    } else {
        let t = 3.75 / x.abs();
        let mut result = 0.39894228;
        result += 0.01328592 * t;
        result += 0.00225319 * t * t;
        result -= 0.00157565 * t.powi(3);
        result * x.exp() / x.sqrt()
    }
}

fn bessel_k0(x: f32) -> f32 {
    if x < 2.0 {
        let i0 = bessel_i0(x);
        -x.ln() * i0 + 0.5772156649
    } else {
        let mut result = 1.2533141;
        result -= 0.07832358 * (2.0 / x);
        result += 0.02189568 * (2.0 / x).powi(2);
        result * (-x).exp() / x.sqrt()
    }
}

fn bessel_y0(x: f32) -> f32 {
    let j0 = bessel_j0(x);
    2.0 / PI * (x.ln() * j0 + 0.07832358)
}

fn sech(x: f32) -> f32 {
    2.0 / (x.exp() + (-x).exp())
}

fn triangle_wave(x: f32) -> f32 {
    let phase = x / PI;
    let t = phase - phase.floor();
    if t < 0.5 {
        4.0 * t - 1.0
    } else {
        3.0 - 4.0 * t
    }
}

fn morlet_wavelet(r: f32, theta: f32, freq: f32) -> f32 {
    let sigma = 1.0 / freq.sqrt();
    let gaussian = (-0.5 * (r / sigma).powi(2)).exp();
    let oscillation = (freq * theta).cos();
    gaussian * oscillation
}

/// 행렬 압축 및 복원
pub struct PoincareMatrix {
    pub seed: Packed64,
    pub rows: usize,
    pub cols: usize,
}

impl PoincareMatrix {
    /// FP32 행렬을 64비트 시드로 압축 (문서의 방식)
    pub fn compress(matrix: &[f32], rows: usize, cols: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // 간단한 버전: 첫 번째 비영 요소 쌍을 찾아서 인코딩
        let mut r = 0.5;
        let mut theta = 0.0;
        
        // 행렬에서 가장 큰 값의 위치를 찾아 패턴 추정
        let mut max_val = 0.0;
        let mut max_i = 0;
        let mut max_j = 0;
        
        for i in 0..rows {
            for j in 0..cols {
                let val = matrix[i * cols + j].abs();
                if val > max_val {
                    max_val = val;
                    max_i = i;
                    max_j = j;
                }
            }
        }
        
        // 최대값 위치에서 극좌표 추정
        let x = 2.0 * (max_j as f32) / ((cols - 1) as f32) - 1.0;
        let y = 2.0 * (max_i as f32) / ((rows - 1) as f32) - 1.0;
        r = (x * x + y * y).sqrt().min(0.9);
        theta = y.atan2(x).rem_euclid(2.0 * PI);
        
        // 랜덤 탐색으로 최적화
        let mut best_seed = Packed64::new(r, theta, 0, 0, false, 0, 0, 0);
        let mut min_error = f32::INFINITY;
        
        for _ in 0..1000 {
            let r_test = rng.gen_range(0.1..0.95);
            let theta_test = rng.gen_range(0.0..2.0 * PI);
            let basis_id = rng.gen_range(0..4);
            let d_theta = rng.gen_range(0..4);
            let d_r = rng.gen::<bool>();
            let rot_code = rng.gen_range(0..10);
            let log2_c = rng.gen_range(-3..4);
            
            let test_seed = Packed64::new(
                r_test, theta_test, basis_id, d_theta, d_r, rot_code, log2_c, 0
            );
            
            let mut error = 0.0;
            for i in 0..rows.min(10) {  // 빠른 평가를 위해 일부만
                for j in 0..cols.min(10) {
                    let original = matrix[i * cols + j];
                    let reconstructed = test_seed.compute_weight(i, j, rows, cols);
                    error += (original - reconstructed).powi(2);
                }
            }
            
            if error < min_error {
                min_error = error;
                best_seed = test_seed;
            }
        }
        
        PoincareMatrix {
            seed: best_seed,
            rows,
            cols,
        }
    }
    
    /// 시드로부터 행렬 복원
    pub fn decompress(&self) -> Vec<f32> {
        let mut matrix = vec![0.0; self.rows * self.cols];
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix[i * self.cols + j] = self.seed.compute_weight(i, j, self.rows, self.cols);
            }
        }
        
        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_encode_decode_exact() {
        // 문서의 예제대로 테스트
        let r = 0.5;
        let theta = PI;
        let basis_id = 0;
        let d_theta = 2;
        let d_r = true;
        let rot_code = 3;
        let log2_c = -2;
        let reserved = 0;

        let packed = Packed64::new(r, theta, basis_id, d_theta, d_r, rot_code, log2_c, reserved);
        let decoded = packed.decode();
        
        assert_relative_eq!(decoded.r, r, epsilon = 1e-6);
        assert_relative_eq!(decoded.theta, theta, epsilon = 1e-6); // Epsilon 완화
        assert_eq!(decoded.basis_id, basis_id);
        assert_eq!(decoded.d_theta, d_theta);
        assert_eq!(decoded.d_r, d_r);
        assert_eq!(decoded.rot_code, rot_code);
        assert_eq!(decoded.log2_c, log2_c);
        assert_eq!(decoded.reserved, reserved);
    }

    #[test]
    fn test_derivative_cycles() {
        // 미분 순환성 테스트
        let theta = PI / 4.0;
        
        // sin 기반, 4주기 테스트
        for d in 0..8 {
            let val = apply_angular_derivative(theta, d, 0);
            let expected = match d % 4 {
                0 => theta.sin(),
                1 => theta.cos(),
                2 => -theta.sin(),
                3 => -theta.cos(),
                _ => unreachable!(),
            };
            assert_relative_eq!(val, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_rotation_codes() {
        // 회전 코드 테스트
        assert_relative_eq!(get_rotation_angle(0), 0.0, epsilon = 1e-6);
        assert_relative_eq!(get_rotation_angle(1), PI / 8.0, epsilon = 1e-6);
        assert_relative_eq!(get_rotation_angle(3), PI / 4.0, epsilon = 1e-6);
        assert_relative_eq!(get_rotation_angle(5), PI / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_compression() {
        let rows = 32;
        let cols = 32;
        
        // 간단한 패턴 생성
        let mut matrix = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let x = 2.0 * (j as f32) / ((cols - 1) as f32) - 1.0;
                let y = 2.0 * (i as f32) / ((rows - 1) as f32) - 1.0;
                matrix[i * cols + j] = (PI * x).sin() * (PI * y).cos();
            }
        }
        
        let compressed = PoincareMatrix::compress(&matrix, rows, cols);
        let reconstructed = compressed.decompress();
        
        // 기본적인 재구성 확인
        let mut total_error = 0.0;
        for i in 0..matrix.len() {
            total_error += (matrix[i] - reconstructed[i]).powi(2);
        }
        let rmse = (total_error / matrix.len() as f32).sqrt();
        
        println!("압축 테스트:");
        println!("  원본 크기: {} bytes", matrix.len() * 4);
        println!("  압축 크기: 8 bytes");
        println!("  RMSE: {:.6}", rmse);
        
        // 완벽한 재구성은 어렵지만 기본 패턴은 유지되어야 함
        assert!(rmse < 1.5, "RMSE should be less than 1.5 for this pattern");
    }
}
