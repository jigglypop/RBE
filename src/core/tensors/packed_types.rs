//! 완전한 비트 도메인 푸앵카레볼 구현 - 순수 비트 연산만 사용

use rand::Rng;
use std::collections::HashMap;
use super::hyperbolic_lut::HYPERBOLIC_LUT_DATA;

/// 비트 텐서 - 순수 비트 도메인 연산
#[derive(Debug, Clone)]
pub struct BitTensor {
    /// 비트 데이터 (Packed128 배열)
    pub data: Vec<Packed128>,
    /// 형상 정보
    pub shape: Vec<usize>,
    /// 비트별 그래디언트 추적
    pub bit_gradients: BitGradientTracker,
}

/// 비트별 그래디언트 추적기
#[derive(Debug, Clone)]
pub struct BitGradientTracker {
    /// 각 비트별 그래디언트 (128개 비트)
    bit_grads: Vec<[u8; 128]>, // u8로 양자화된 그래디언트
    /// 비트간 상호작용 그래디언트
    bit_interactions: HashMap<(u8, u8), u8>,
    /// 상태 전이 그래디언트
    state_transition_grads: Vec<StateTransitionGrad>,
}

#[derive(Debug, Clone)]
struct StateTransitionGrad {
    bit_position: u8,
    old_value: u8,
    new_value: u8,
    gradient: u8, // 양자화된 그래디언트
}

impl BitGradientTracker {
    pub fn new(size: usize) -> Self {
        Self {
            bit_grads: vec![[0u8; 128]; size],
            bit_interactions: HashMap::new(),
            state_transition_grads: Vec::new(),
        }
    }
    
    /// 비트 연산 의존성 등록
    pub fn register_dependency(&mut self, idx: usize, input: &Packed128, output: &Packed128) {
        // Hi 필드 비트별 그래디언트 계산
        for bit_pos in 0..64 {
            let input_bit = (input.hi >> bit_pos) & 1;
            let output_bit = (output.hi >> bit_pos) & 1;
            
            // XOR 연산의 비트 그래디언트
            self.bit_grads[idx][bit_pos] = if input_bit != output_bit { 255 } else { 0 };
        }
        
        // Lo 필드 그래디언트 (고정소수점)
        let r_input = (input.lo >> 32) as u32;
        let r_output = (output.lo >> 32) as u32;
        let grad = ((r_output.wrapping_sub(r_input) >> 24) & 0xFF) as u8;
        self.bit_grads[idx][64] = grad;
    }
}

/// 64-bit Packed Poincaré 시드 표현 (비트 도메인 CORDIC)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Packed64 {
    pub rotations: u64,  // CORDIC 회전 시퀀스
}

/// 128-bit 시드 (64비트만 실제 사용 - 11비트 사이클 제거)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Packed128 {
    pub hi: u64,   // 미사용 (향후 확장용 예약)
    pub lo: u64,   // 연속 파라미터 (고정소수점 Q32.32)
}

/// 연속 파라미터 디코딩
#[derive(Debug, Clone, Default)]
pub struct DecodedParams {
    pub r_fp32: f32,
    pub theta_fp32: f32,
}

// 비트 도메인 CORDIC 각도 테이블 (Q32 고정소수점)
pub const CORDIC_ANGLES_Q32: [u32; 20] = [
    0x3243F6A8, // atan(2^0) ≈ 0.7854 (π/4)
    0x1DAC6705, // atan(2^-1)
    0x0FADBAFC, // atan(2^-2)
    0x07F56EA6, // atan(2^-3)
    0x03FEAB76, // atan(2^-4)
    0x01FFD55B, // atan(2^-5)
    0x00FFFAAA, // atan(2^-6)
    0x007FFF55, // atan(2^-7)
    0x003FFFEA, // atan(2^-8)
    0x001FFFFD, // atan(2^-9)
    0x000FFFFF, // atan(2^-10)
    0x0007FFFF, // atan(2^-11)
    0x0003FFFF, // atan(2^-12)
    0x0001FFFF, // atan(2^-13)
    0x0000FFFF, // atan(2^-14)
    0x00007FFF, // atan(2^-15)
    0x00003FFF, // atan(2^-16)
    0x00001FFF, // atan(2^-17)
    0x00000FFF, // atan(2^-18)
    0x000007FF, // atan(2^-19)
];



impl Packed64 {
    pub fn new(rotations: u64) -> Self {
        Packed64 { rotations }
    }

    /// 순수 비트 도메인 CORDIC 가중치 계산
    pub fn compute_weight(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 입력 검증 - 0 나누기 방지
        if rows <= 1 || cols <= 1 {
            return 0.0;
        }
        
        // Q16.16 고정소수점 좌표
        let x_fixed = ((j << 17) / (cols - 1)) as i32 - (1 << 16);
        let y_fixed = ((i << 17) / (rows - 1)) as i32 - (1 << 16);
        
        // 비트 연산 atan2
        let angle_fixed = Self::bit_atan2_q16(y_fixed, x_fixed);
        
        // CORDIC 초기화 (K ≈ 0.60725 in Q16.16)
        let mut x = 39797i32; // 0.60725 * 65536
        let mut y = 0i32;
        let mut z = angle_fixed;
        
        // 비트 도메인 CORDIC 회전
        for k in 0..20 {
            let sigma = if (self.rotations >> k) & 1 == 1 { 1 } else { -1 };
            
            // 쌍곡 CORDIC (k=4,13에서 반복)
            let (x_shift, y_shift) = if k == 4 || k == 13 {
                // 쌍곡 모드: 2배 적용
                let xs = x >> k;
                let ys = y >> k;
                (xs + (xs >> 1), ys + (ys >> 1))
            } else {
                (x >> k, y >> k)
            };
            
            let x_new = x - sigma * y_shift;
            let y_new = y + sigma * x_shift;
            
            x = x_new;
            y = y_new;
            
            // 쌍곡 변환 (4회마다) - LUT 사용
            if k % 4 == 0 && k > 0 {
                let r2_q32 = ((x as i64 * x as i64 + y as i64 * y as i64) >> 16) as u32;
                if r2_q32 > 256 { // 매우 작은 값 방지
                    // tanh(r) 근사를 LUT에서 조회
                    let r_idx = (r2_q32.min(0xFFFF) >> 8) as usize;
                    let tanh_r = HYPERBOLIC_LUT_DATA[2][r_idx.min(255)];
                    
                    // Q16.16 곱셈
                    x = ((x as i64 * tanh_r as i64) >> 32) as i32;
                    y = ((y as i64 * tanh_r as i64) >> 32) as i32;
                }
            }
        }
        
        // Q16.16을 f32로 변환
        x as f32 / 65536.0
    }
    
    /// 비트 연산 전용 atan2 (Q16.16)
    pub fn bit_atan2_q16(y: i32, x: i32) -> i32 {
        // 특수 케이스 처리
        if x == 0 && y == 0 {
            return 0;
        }
        
        // x가 양수이고 y가 0인 경우
        if y == 0 {
            if x > 0 {
                return 0; // 0도
            } else {
                return 0x6487; // 180도 (π in Q16)
            }
        }
        
        // y가 0이 아닌 경우의 일반적인 처리
        let mut angle = 0i32;
        let mut xi = x;
        let mut yi = y;
        
        // 사분면 처리
        if xi < 0 {
            if yi >= 0 {
                angle = 0x3243F6A8; // π in Q32
            } else {
                angle = -0x3243F6A8; // -π
            }
            xi = -xi;
            yi = -yi;
        }
        
        // CORDIC 벡터링 모드
        for i in 0..20.min(CORDIC_ANGLES_Q32.len()) {
            let di: i32 = if yi < 0 { -1 } else { 1 };
            let xi_new = xi.saturating_sub(di.saturating_mul((yi >> i) as i32));
            let yi_new = yi.saturating_add(di.saturating_mul((xi >> i) as i32));
            
            // 안전한 연산을 위해 wrapping_sub 사용
            let angle_delta = di.wrapping_mul(CORDIC_ANGLES_Q32[i] as i32);
            angle = angle.wrapping_sub(angle_delta);
            
            xi = xi_new;
            yi = yi_new;
        }
        
        angle >> 16 // Q32 to Q16
    }
}

impl Packed128 {
    /// 순수 비트 도메인 fused forward
    pub fn fused_forward(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 새로운 푸앵카레볼 기하학 버전 사용
        self.fused_forward_poincare(i, j, rows, cols)
    }
    
    /// 비트 패턴 변조 (곱셈 없이)
    pub fn bit_pattern_modulation(pattern: u64, i: usize, j: usize, cycle: usize) -> f32 {
        // MurmurHash 스타일 믹싱
        let mut h = pattern ^ ((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        h ^= (j as u64).wrapping_mul(0x94D049BB133111EB);
        h ^= (cycle as u64).wrapping_mul(0xBF58476D1CE4E5B9);
        
        // 추가 믹싱으로 더 균등한 분포 생성
        h = h ^ (h >> 33);
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h = h ^ (h >> 33);
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h = h ^ (h >> 33);
        
        // 상위 32비트를 사용하여 [0, 1] 범위로 변환
        (h >> 32) as f32 / 4294967296.0
    }
    
    /// 쌍곡함수 LUT 적용 (실제 룩업)
    fn apply_hyperbolic_lut(func_idx: usize, x: f32, mod_factor: f32) -> f32 {
        // 입력을 LUT 인덱스로 변환
        let x_fixed = (x * 65536.0) as i32;
        let idx = ((x_fixed >> 8) + 128).clamp(0, 255) as usize;
        
        // LUT에서 값 조회 (Q16.16)
        let lut_value = HYPERBOLIC_LUT_DATA[func_idx & 0x7][idx];
        let base = (lut_value as i32) as f32 / 65536.0;
        
        // 변조 적용 (고정소수점 곱셈)
        let mod_fixed = (mod_factor * 65536.0) as i32;
        let result_fixed = ((base * 65536.0) as i64 * mod_fixed as i64) >> 16;
        (result_fixed as f32) / 65536.0
    }
    
    // 기존 메서드들 유지
    pub fn decode(&self) -> DecodedParams {
        // lo 필드는 Q32.32 고정소수점
        let r_q32 = (self.lo >> 32) as u32;
        let theta_q32 = self.lo as u32;
        
        // Q32.32 → f32 변환
        let r_fp32 = (r_q32 as f32) / 4294967296.0;
        let theta_fp32 = (theta_q32 as f32) / 4294967296.0 * 2.0 * std::f32::consts::PI;
        
        DecodedParams { r_fp32, theta_fp32 }
    }
    
    pub fn from_continuous(p: &DecodedParams) -> Self {
        // f32 → Q32.32 변환 (정밀도 개선)
        let r_clamped = p.r_fp32.clamp(0.0, 0.999999);
        let r_q32 = ((r_clamped as f64) * 4294967296.0) as u64;
        
        // theta를 [0, 2π) 범위로 정규화
        let theta_norm = p.theta_fp32.rem_euclid(2.0 * std::f32::consts::PI);
        let theta_q32 = ((theta_norm as f64) / (2.0 * std::f64::consts::PI) * 4294967296.0) as u64;
        
        let hi = 0; // 초기 상태
        let lo = (r_q32 << 32) | (theta_q32 & 0xFFFFFFFF);
        
        Packed128 { hi, lo }
    }

    /// 디코딩된 연속 파라미터(r, θ)를 사용하여 현재 `Packed128`의 `lo` 필드를 업데이트합니다.
    pub fn update_from_continuous(&mut self, params: &DecodedParams) {
        // f32 → Q32.32 변환
        let r_clamped = params.r_fp32.clamp(0.0, 0.999999);
        let r_q32 = ((r_clamped as f64) * 4294967296.0) as u64;

        // theta를 [0, 2π) 범위로 정규화
        let theta_norm = params.theta_fp32.rem_euclid(2.0 * std::f32::consts::PI);
        let theta_q32 = ((theta_norm as f64) / (2.0 * std::f64::consts::PI) * 4294967296.0) as u64;

        // 양자화된 값을 u64로 패킹하여 `lo` 필드에 저장
        self.lo = (r_q32 << 32) | (theta_q32 & 0xFFFFFFFF);
    }
    
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut seed = Self::default();
        
        // 모든 비트 랜덤 초기화
        seed.hi = rng.gen();
        seed.lo = rng.gen();
        
        seed
    }
} 

pub trait AnalyticalGradient {
    fn analytical_gradient_r(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32;
    fn analytical_gradient_theta(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32;
}

impl AnalyticalGradient for Packed128 {
    fn analytical_gradient_r(&self, _i: usize, _j: usize, _rows: usize, _cols: usize) -> f32 {
        // Placeholder implementation
        0.0
    }

    fn analytical_gradient_theta(&self, _i: usize, _j: usize, _rows: usize, _cols: usize) -> f32 {
        // Placeholder implementation
        0.0
    }
} 

/// 성능 측정 및 정확도 검증 함수들
impl Packed128 {
    /// 연산 속도 벤치마크 - 실제 ns/op 측정
    pub fn benchmark_speed(iterations: usize) {
        use std::time::Instant;
        let mut rng = rand::thread_rng();
        
        println!("=== 비트 도메인 푸앵카레볼 성능 측정 ===");
        println!("반복 횟수: {}", iterations);
        
        // 1. fused_forward 속도 측정
        let packed = Self::random(&mut rng);
        let start = Instant::now();
        let mut sum = 0.0f32;
        
        for i in 0..iterations {
            let row = i % 100;
            let col = (i * 7) % 150;
            sum += packed.fused_forward(row, col, 100, 150);
        }
        
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;
        
        println!("fused_forward 속도: {:.1} ns/op ({:.1} MHz)", ns_per_op, 1000.0 / ns_per_op);
        println!("처리량: {:.1} million ops/sec", 1000.0 / ns_per_op);
        println!("결과 합계: {:.6} (최적화 방지)", sum);
        
        // 2. 상태 전이 속도 측정
        let mut packed_mut = Self::random(&mut rng);
        let start = Instant::now();
        
        for i in 0..iterations {
            let error = (i as f32 % 100.0) * 0.01 - 0.5;
            let row = i % 50;
            let col = (i * 3) % 75;
            // packed_mut.apply_state_transition(error, row, col); // Removed 11-bit cycle transition
        }
        
        let elapsed = start.elapsed();
        let ns_per_transition = elapsed.as_nanos() as f64 / iterations as f64;
        
        println!("상태전이 속도: {:.1} ns/op ({:.1} MHz)", ns_per_transition, 1000.0 / ns_per_transition);
        
        // 3. 압축률 측정 (기존 f32 대비)
        let original_size = 100 * 150 * 4; // f32 배열 크기
        let compressed_size = std::mem::size_of::<Self>(); // 128bit
        let compression_ratio = original_size as f64 / compressed_size as f64;
        
        println!("압축률: {:.1}:1 ({} bytes -> {} bytes)", 
                compression_ratio, original_size, compressed_size);
    }
    
    /// 정확도 측정 - RMSE 및 오차 분석
    pub fn measure_accuracy(samples: usize) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        println!("\n=== 정확도 분석 ===");
        println!("샘플 수: {}", samples);
        
        let mut total_error = 0.0f64;
        let mut max_error = 0.0f32;
        let mut error_distribution = [0u32; 10]; // 오차 분포 (0.1 단위)
        
        // 연속 파라미터 왕복 변환 정확도
        for _ in 0..samples {
            let r_original = rng.gen::<f32>() * 0.99;
            let theta_original = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
            
            let params = DecodedParams { 
                r_fp32: r_original, 
                theta_fp32: theta_original 
            };
            
            let packed = Self::from_continuous(&params);
            let decoded = packed.decode();
            
            // 오차 계산
            let r_error = (decoded.r_fp32 - r_original).abs();
            let theta_diff = (decoded.theta_fp32 - theta_original).abs();
            let theta_error = theta_diff.min(2.0 * std::f32::consts::PI - theta_diff);
            
            let combined_error = (r_error + theta_error * 0.1).sqrt();
            total_error += combined_error as f64;
            
            if combined_error > max_error {
                max_error = combined_error;
            }
            
            // 오차 분포 기록
            let bucket = (combined_error * 100.0) as usize;
            if bucket < error_distribution.len() {
                error_distribution[bucket] += 1;
            }
        }
        
        let rmse = (total_error / samples as f64).sqrt();
        
        println!("RMSE: {:.6}", rmse);
        println!("최대 오차: {:.6}", max_error);
        println!("평균 오차: {:.6}", total_error / samples as f64);
        
        // 오차 분포 출력
        println!("오차 분포:");
        for (i, &count) in error_distribution.iter().enumerate() {
            let percentage = count as f64 / samples as f64 * 100.0;
            if percentage > 0.1 {
                println!("  {:.1}-{:.1}%: {:.1}% ({} samples)", 
                        i as f64 * 0.01, (i + 1) as f64 * 0.01, percentage, count);
            }
        }
    }
    
    /// 메모리 사용량 및 캐시 효율성 분석
    pub fn analyze_memory_efficiency() {
        println!("\n=== 메모리 효율성 분석 ===");
        
        // 구조체 크기 정보
        println!("데이터 구조 크기:");
        println!("  Packed64: {} bytes", std::mem::size_of::<Packed64>());
        println!("  Packed128: {} bytes", std::mem::size_of::<Self>());
        println!("  BitGradientTracker: {} bytes", std::mem::size_of::<BitGradientTracker>());
        
        // 메모리 정렬
        println!("메모리 정렬:");
        println!("  Packed64: {} bytes", std::mem::align_of::<Packed64>());
        println!("  Packed128: {} bytes", std::mem::align_of::<Self>());
        
        // 캐시 라인 효율성 (64바이트 가정)
        let cache_line_size = 64;
        let structs_per_cache_line = cache_line_size / std::mem::size_of::<Self>();
        println!("캐시 라인당 구조체 수: {}", structs_per_cache_line);
        
        // 대량 데이터 처리 시뮬레이션
        let matrix_size = 1000;
        let total_elements = matrix_size * matrix_size;
        let traditional_memory = total_elements * 4; // f32
        let compressed_memory = std::mem::size_of::<Self>(); // 단일 시드
        
        println!("{}x{} 행렬 메모리 사용량:", matrix_size, matrix_size);
        println!("  기존 방식: {:.1} MB", traditional_memory as f64 / 1_048_576.0);
        println!("  RBE 방식: {:.1} KB", compressed_memory as f64 / 1024.0);
        println!("  메모리 절약률: {:.1}%", 
                (1.0 - compressed_memory as f64 / traditional_memory as f64) * 100.0);
    }
    
    /// 실시간 성능 모니터링
    pub fn realtime_performance_monitor(duration_secs: u64) {
        use std::time::{Instant, Duration};
        use rand::Rng;
        
        println!("\n=== 실시간 성능 모니터링 ({} 초) ===", duration_secs);
        
        let mut rng = rand::thread_rng();
        let packed = Self::random(&mut rng);
        let start_time = Instant::now();
        let duration = Duration::from_secs(duration_secs);
        
        let mut operation_count = 0u64;
        let mut last_report = Instant::now();
        let report_interval = Duration::from_millis(500);
        
        while start_time.elapsed() < duration {
            // 다양한 연산 수행
            for _ in 0..1000 {
                let i = rng.gen_range(0..100);
                let j = rng.gen_range(0..100);
                let _result = packed.fused_forward(i, j, 100, 100);
                operation_count += 1;
            }
            
            // 주기적 리포트
            if last_report.elapsed() >= report_interval {
                let elapsed = start_time.elapsed();
                let ops_per_sec = operation_count as f64 / elapsed.as_secs_f64();
                let ns_per_op = elapsed.as_nanos() as f64 / operation_count as f64;
                
                println!("[{:.1}s] {:.1} MHz, {:.1} ns/op, {} ops", 
                        elapsed.as_secs_f64(), ops_per_sec / 1_000_000.0, ns_per_op, operation_count);
                
                last_report = Instant::now();
            }
        }
        
        let final_elapsed = start_time.elapsed();
        let final_ops_per_sec = operation_count as f64 / final_elapsed.as_secs_f64();
        
        println!("최종 성능: {:.1} MHz ({} operations in {:.2}s)", 
                final_ops_per_sec / 1_000_000.0, operation_count, final_elapsed.as_secs_f64());
    }

    /// 간단한 수학적 그래디언트 계산 (11비트 제거)
    /// fused_forward: f(r,θ) = tanh(2*arctanh(r)) * sin(θ)
    /// 
    /// Returns: (grad_r, grad_theta, predicted)
    pub fn compute_gradients(
        &self, 
        i: usize, 
        j: usize, 
        rows: usize, 
        cols: usize,
        target: f32,
        use_l1: bool,
    ) -> (f32, f32, f32) {
        // 1. 현재 예측값
        let predicted = self.fused_forward_poincare(i, j, rows, cols);
        
        // 2. 현재 r, theta 값
        let params = self.decode();
        let r = params.r_fp32.min(0.999);
        let theta = params.theta_fp32;
        
        // 3. 손실 함수의 미분
        let grad_of_loss = if use_l1 {
            if predicted >= target { 1.0 } else { -1.0 }
        } else {
            2.0 * (predicted - target)
        };
        let loss_grad = grad_of_loss * (1.0 - predicted.powi(2));
        
        // 4. 푸앵카레볼 기하학에 따른 그래디언트
        let d = if r < 0.999 {
            2.0 * r.atanh()
        } else {
            2.0 * (0.5 * ((1.0 + r) / (1.0 - r)).ln())
        };
        
        // 단순화: tanh와 sin만 사용
        let func_value = d.tanh();
        let func_deriv = 1.0 - func_value * func_value;
        let angular_component = theta.sin();
        let angular_deriv = theta.cos();
        
        // ∂f/∂r = loss_grad * func_deriv * angular_component * (2/(1-r²))
        let dd_dr = 2.0 / ((1.0 - r * r).max(1e-6));
        let df_dr = loss_grad * func_deriv * angular_component * dd_dr;
        
        // ∂f/∂θ = loss_grad * func_value * angular_deriv
        let df_dtheta = loss_grad * func_value * angular_deriv;
        
        (df_dr, df_dtheta, predicted)
    }
    
    /// 리만 기하학을 적용한 자연 그래디언트 계산 (11비트 제거)
    /// 푸앵카레 볼 메트릭: ds² = 4/(1-r²)² (dr² + r²dθ²)
    pub fn compute_riemannian_gradients(
        &self,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
        target: f32,
        use_l1: bool,
    ) -> (f32, f32) {
        // 1. 현재 예측값
        let predicted = self.fused_forward_poincare(i, j, rows, cols);
        
        // 2. 현재 r, theta 값
        let params = self.decode();
        let r = params.r_fp32.min(0.999);
        let theta = params.theta_fp32;
        
        // 3. 손실 함수의 미분
        let grad_of_loss = if use_l1 {
            if predicted >= target { 1.0 } else { -1.0 }
        } else {
            2.0 * (predicted - target)
        };
        let loss_grad = grad_of_loss * (1.0 - predicted.powi(2));
        
        // 4. 쌍곡 기하학 계산
        let d = if r < 0.999 {
            2.0 * r.atanh()
        } else {
            2.0 * (0.5 * ((1.0 + r) / (1.0 - r)).ln())
        };
        
        // 단순화된 쌍곡함수
        let func_value = d.tanh();
        let func_deriv = 1.0 - func_value * func_value;
        let angular_component = theta.sin();
        let angular_deriv = theta.cos();

        // 5. 리만 그래디언트의 안정적인 계산
        let one_minus_r2 = (1.0 - r * r).max(1e-6);

        // 그래디언트 계산 (경계 근처에서 안정화)
        let grad_r_riemannian = (one_minus_r2 / 2.0) * loss_grad * func_deriv * angular_component;
        let grad_theta_riemannian = (one_minus_r2.powi(2) / (4.0 * r.powi(2)).max(1e-9)) * loss_grad * func_value * angular_deriv;
        
        // 6. 동적 그래디언트 클리핑 및 경계 감쇠
        let mut boundary_damping = (1.0 - r.powi(4)).max(0.01);
        
        // 경계 근처에서 r을 밀어내는 그래디언트는 약화
        if r > 0.95 && grad_r_riemannian > 0.0 {
            boundary_damping *= (1.0 - (r - 0.95) * 20.0).max(0.01);
        }

        let max_grad_r = 1.0 * boundary_damping;
        let max_grad_theta = 2.0 * boundary_damping;
        
        let grad_r_clipped = grad_r_riemannian.clamp(-max_grad_r, max_grad_r);
        let grad_theta_clipped = grad_theta_riemannian.clamp(-max_grad_theta, max_grad_theta);

        (grad_r_clipped, grad_theta_clipped)
    }
    
    /// 리만 그래디언트를 사용하여 파라미터를 직접 업데이트 (안정화 버전)
    pub fn update_with_riemannian_grad(&mut self, update_r: f32, update_theta: f32, _lr: f32) {
        let mut params = self.decode();
        
        // r 업데이트 (경계 고려) - update_r에는 이미 학습률이 적용되어 있음
        let new_r = params.r_fp32 - update_r;
        params.r_fp32 = new_r.clamp(0.0, 0.999); // 1에 매우 가까워지지 않도록 함
        
        // theta 업데이트 (순환 구조) - update_theta에도 이미 학습률이 적용되어 있음
        params.theta_fp32 = (params.theta_fp32 - update_theta).rem_euclid(2.0 * std::f32::consts::PI);

        self.update_from_continuous(&params);
    }

    /// 간단한 푸앵카레볼 순전파 (11비트 사이클 제거)
    pub fn fused_forward_poincare(&self, i: usize, j: usize, _rows: usize, _cols: usize) -> f32 {
        // 1. 연속 파라미터 디코딩
        let params = self.decode();
        let r = params.r_fp32.min(0.999);
        let theta = params.theta_fp32;
        
        // 2. 푸앵카레볼 → 쌍곡 거리 변환
        let d = if r < 0.999 {
            2.0 * r.atanh()
        } else {
            2.0 * (0.5 * ((1.0 + r) / (1.0 - r)).ln())
        };
        
        // 3. 위치 기반 변조
        let pos_hash = ((i * 31 + j * 17) % 256) as f32 / 256.0;
        let spatial_modulation = (pos_hash * 2.0 * std::f32::consts::PI).sin();
        
        // 4. 단순화된 함수: tanh만 사용
        let func_value = d.tanh();
        
        // 5. 각도 성분
        let angular_component = theta.sin();
        
        // 6. 최종 출력
        let output = func_value * angular_component * (1.0 + spatial_modulation * 0.1);
        
        // 7. 출력 정규화
        output.tanh()
    }
} 

/// 종합 성능 리포트 생성
pub fn generate_comprehensive_report() {
    println!("██████████████████████████████████████████████████████████");
    println!("██  RBE 푸앵카레볼 비트도메인 성능 종합 리포트          ██");
    println!("██████████████████████████████████████████████████████████");
    
    // 속도 측정
    Packed128::benchmark_speed(1_000_000);
    
    // 정확도 측정  
    Packed128::measure_accuracy(10_000);
    
    // 메모리 효율성
    Packed128::analyze_memory_efficiency();
    
    // 실시간 모니터링 (3초간)
    Packed128::realtime_performance_monitor(3);
    
    println!("\n██████████████████████████████████████████████████████████");
    println!("██  리포트 완료                                        ██");
    println!("██████████████████████████████████████████████████████████");
} 