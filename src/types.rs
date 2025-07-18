use rand::Rng;
use std::f32;
use crate::math::{ste_quant_q0x, ste_quant_phase};
use nalgebra::{DMatrix, DVector};

/// 64-bit Packed Poincaré 시드 표현 (CORDIC 통합)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Packed64 {
    pub rotations: u64,  // CORDIC 회전 시퀀스
}

impl Packed64 {
    pub fn new(rotations: u64) -> Self {
        Packed64 { rotations }
    }

    pub fn compute_weight(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 1. hi 비트필드에서 r, theta 디코딩
        let r_quant = (self.rotations >> 44) & 0xFFFFF; // 20 bits
        let theta_quant = (self.rotations >> 20) & 0xFFFFFF; // 24 bits
        
        let r_val = r_quant as f32 / ((1u64 << 20) - 1) as f32; // [0, 1] 범위로 정규화
        let theta_val = (theta_quant as f32 / ((1u64 << 24) - 1) as f32) * 2.0 * std::f32::consts::PI; // [0, 2PI] 범위로 정규화

        let rotations = self.rotations;

        // 2. 좌표 기반 초기 각도 계산
        let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        let base_angle = y_norm.atan2(x_norm);
        
        // 3. r, theta를 적용하여 초기 벡터 (x, y) 설정
        let mut x = r_val * (base_angle + theta_val).cos();
        let mut y = r_val * (base_angle + theta_val).sin();

        for k in 0..20 { // CORDIC 반복 횟수를 r, theta를 제외한 나머지 비트(20)만큼으로 조정
            let sigma = if (rotations >> k) & 1 == 1 { 1.0 } else { -1.0 };
            
            let power_of_2 = (2.0f32).powi(-(k as i32));

            let x_new = x - sigma * y * power_of_2;
            let y_new = y + sigma * x * power_of_2;
            
            x = x_new;
            y = y_new;

            // 쌍곡 변환 추가
            if k % 4 == 0 {
                let r = (x*x + y*y).sqrt();
                if r > 1e-9 { // 0에 가까운 값 방지
                    let tanh_r = r.tanh();
                    x *= tanh_r;
                    y *= tanh_r;
                }
            }
        }
        
        // CORDIC 게인 보정.
        let gain = 1.64676; 
        x / gain
    }
}

/// 128-bit 시드 (Seed0: 비트필드, Seed1: 연속 FP32×2)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Packed128 {
    pub hi: u64,   // Seed0 : 기존 Packed64 비트필드
    pub lo: u64,   // Seed1 : [63:32] r_fp32  |  [31:0] θ_fp32
}

/// 연속 파라미터까지 포함해 디코딩
#[derive(Debug, Clone, Default)]
pub struct DecodedParams { // DecodedParams128 -> DecodedParams. 기존것과 통합
    // pub base: DecodedParams, // 기존 필드 모두
    pub r_fp32: f32,
    pub theta_fp32: f32,
}

impl Packed128 {
    /// Seed0+1 디코딩
    pub fn decode(&self) -> DecodedParams {
        // let base = Packed64(self.hi).decode(); // Packed64에는 decode가 없으므로 일단 주석처리.
        let r_fp32     = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        DecodedParams { r_fp32, theta_fp32, ..Default::default() }
    }

    /// 연속 파라미터 → 128 bit 시드
    pub fn from_continuous(p: &DecodedParams) -> Self {
        // new.md의 비트 레이아웃에 따라 hi 필드를 구성합니다.
        // r (Q0.20) -> [63:44], theta (Q0.24) -> [43:20]
        let r_quant = ste_quant_q0x(p.r_fp32, 20);
        let theta_quant = ste_quant_phase(p.theta_fp32, 24);

        let hi = (r_quant << 44) | (theta_quant << 20); // 다른 필드는 0으로 가정
        let lo = ((p.r_fp32.to_bits() as u64) << 32) | p.theta_fp32.to_bits() as u64;
        
        Packed128 { hi, lo }
    }

    /// 무작위 초기화
    pub fn random(rng: &mut impl Rng) -> Self {
        let r = 0.8 + rng.gen::<f32>() * 0.2; // [0.8, 1.0] 범위로 증가
        let theta: f32 = rng.gen_range(-0.5..0.5); // [-0.5, 0.5] 범위
        
        let lo = ((r.to_bits() as u64) << 32) | theta.to_bits() as u64;
        let hi = rng.gen::<u64>() & 0xFFFFF; // 하위 20비트만 랜덤
        
        Packed128 { hi, lo }
    }

    /// 정밀한 순전파: hi(상태 전이) + lo(연속) 융합
    /// 상태 전이 미분의 핵심 구현
    #[inline(always)]
    pub fn fused_forward(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 1. 연속 파라미터 추출 (lo)
        let r_fp32 = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        
        // 2. 상태 비트 추출 (hi)
        let state_bits = self.hi & 0xFFFFF; // 하위 20비트
        
        // 3. 좌표 정규화
        let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        
        // 4. 연속 기저 패턴 계산
        let dist = (x_norm * x_norm + y_norm * y_norm).sqrt();
        let base_angle = y_norm.atan2(x_norm);
        let base_pattern = (r_fp32 - dist * r_fp32 + theta_fp32).clamp(0.0, 1.0);
        
        // 5. 상태 기반 함수 선택 및 변조
        let coord_hash = ((i * 31 + j) & 0x3) as u64; // 위치별 상태 선택
        let state_selector = (state_bits >> (coord_hash * 2)) & 0x3;
        
        let modulated_value = match state_selector {
            0 => base_pattern * (base_angle + theta_fp32).sin().abs(),      // sin 상태
            1 => base_pattern * (base_angle + theta_fp32).cos().abs(),      // cos 상태  
            2 => base_pattern * (dist * r_fp32 + theta_fp32).tanh(),        // tanh 상태
            3 => base_pattern * (1.0 - (dist * r_fp32).exp() * 0.5),       // exp 상태
            _ => base_pattern,
        };
        
        // 6. 고주파 세부사항 추가 (나머지 상태 비트 활용)
        let detail_bits = (state_bits >> 8) & 0xFFF; // 상위 12비트
        let detail_factor = 1.0 + 0.1 * (detail_bits as f32 / 4095.0 - 0.5);
        
        (modulated_value * detail_factor).clamp(0.0, 1.0)
    }

    /// 추론 전용: hi(Seed0) → weight
    #[inline(always)]
    pub fn compute_weight(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        Packed64{ rotations: self.hi }.compute_weight(i, j, rows, cols)
    }
    
    /// 학습 전용: lo(Seed1)의 연속 FP32 직접 사용
    #[inline(always)]
    pub fn compute_weight_continuous(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // lo에서 r_fp32, theta_fp32 직접 추출
        let r_fp32 = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        
        // 좌표를 [-1, 1] 범위로 정규화
        let x = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        
        // 극좌표로 변환
        let dist = (x*x + y*y).sqrt();
        
        // radial gradient와 유사한 함수
        // r_fp32는 스케일, theta_fp32는 오프셋으로 사용
        let value = (r_fp32 - dist * r_fp32 + theta_fp32).max(0.0).min(1.0);
        
        value
    }

    /// 고급 상태 전이 미분: 다중 비트 상태와 함수별 전이 규칙
    pub fn advanced_state_transition(&mut self, gradient_signal: f32, i: usize, j: usize) {
        // 다중 해시를 사용한 위치별 상태 관리
        let primary_hash = ((i * 31 + j) & 0x7) as u64; // 3비트 선택자
        let secondary_hash = ((i * 17 + j * 13) & 0x3) as u64; // 2비트 보조 선택자
        
        // 주요 상태 (3비트): 8가지 함수 상태
        let primary_bit_pos = primary_hash * 3;
        let current_primary_state = (self.hi >> primary_bit_pos) & 0x7;
        
        // 보조 상태 (2비트): 4가지 변조 상태
        let secondary_bit_pos = (primary_hash + 8) * 2;
        let current_secondary_state = (self.hi >> secondary_bit_pos) & 0x3;
        
        // 그래디언트 강도에 따른 전이 결정
        let abs_gradient = gradient_signal.abs();
        let gradient_sign = gradient_signal.signum();
        
        let new_primary_state = if abs_gradient > 0.2 {
            // 강한 그래디언트: 명확한 미분 전이
            match (current_primary_state, gradient_sign > 0.0) {
                (0, true) => 1,   // sin -> cos (미분)
                (1, true) => 0,   // cos -> -sin (미분, 부호는 별도 처리)
                (2, true) => 3,   // tanh -> sech^2 (미분)
                (3, true) => 2,   // sech^2 -> tanh (역미분)
                (4, true) => 5,   // exp -> exp (자기 자신)
                (5, true) => 6,   // log -> 1/x (미분)
                (6, true) => 7,   // 1/x -> -1/x^2 (미분)
                (7, true) => 4,   // 다항식 -> exp (전이)
                
                // 음의 그래디언트: 역방향 전이
                (0, false) => 7,  // sin -> 다항식
                (1, false) => 6,  // cos -> 1/x
                (2, false) => 5,  // tanh -> log
                (3, false) => 4,  // sech^2 -> exp
                (4, false) => 3,  // exp -> sech^2
                (5, false) => 2,  // log -> tanh
                (6, false) => 1,  // 1/x -> cos
                (7, false) => 0,  // 다항식 -> sin
                _ => current_primary_state,
            }
        } else if abs_gradient > 0.05 {
            // 중간 그래디언트: 보조 상태 기반 미세 조정
            match current_secondary_state {
                0 => (current_primary_state + 1) & 0x7, // 순환 전진
                1 => if current_primary_state > 0 { current_primary_state - 1 } else { 7 }, // 순환 후진
                2 => current_primary_state ^ 0x1, // 비트 토글
                3 => current_primary_state ^ 0x2, // 다른 비트 토글
                _ => current_primary_state,
            }
        } else {
            current_primary_state // 약한 그래디언트는 상태 유지
        };
        
        // 보조 상태 업데이트 (적응적 학습)
        let new_secondary_state = if abs_gradient > 0.1 {
            // 그래디언트 방향에 따른 보조 상태 조정
            if gradient_sign > 0.0 {
                (current_secondary_state + 1) & 0x3
            } else {
                if current_secondary_state > 0 { current_secondary_state - 1 } else { 3 }
            }
        } else {
            current_secondary_state
        };
        
        // 비트 필드 업데이트
        self.hi = (self.hi & !(0x7 << primary_bit_pos)) | (new_primary_state << primary_bit_pos);
        self.hi = (self.hi & !(0x3 << secondary_bit_pos)) | (new_secondary_state << secondary_bit_pos);
    }

    /// 상태 전이 미분: 그래디언트 신호에 따른 비트 상태 업데이트
    pub fn apply_state_transition(&mut self, gradient_signal: f32, i: usize, j: usize) {
        let coord_hash = ((i * 31 + j) & 0x3) as u64;
        let bit_pos = coord_hash * 2;
        let current_state = (self.hi >> bit_pos) & 0x3;
        
        // 그래디언트 신호에 따른 상태 전이 결정
        let new_state = if gradient_signal > 0.1 {
            // 양의 그래디언트: sin -> cos, tanh -> exp 등으로 전이
            match current_state {
                0 => 1, // sin -> cos (미분)
                1 => 0, // cos -> -sin (미분)
                2 => 3, // tanh -> sech^2 (미분 근사)
                3 => 2, // exp -> exp (자기 자신이 미분)
                _ => current_state,
            }
        } else if gradient_signal < -0.1 {
            // 음의 그래디언트: 역방향 전이
            match current_state {
                0 => 3, // sin -> exp
                1 => 2, // cos -> tanh
                2 => 1, // tanh -> cos
                3 => 0, // exp -> sin
                _ => current_state,
            }
        } else {
            current_state // 작은 그래디언트는 상태 유지
        };
        
        // 비트 업데이트
        self.hi = (self.hi & !(0x3 << bit_pos)) | (new_state << bit_pos);
    }

    /// 상태별 함수 계산 (고급 버전) - 안정성 개선
    pub fn compute_state_function(&self, state: u64, input: f32, phase: f32) -> f32 {
        let safe_input = input.clamp(-10.0, 10.0); // 입력 범위 제한
        let safe_phase = phase.clamp(-10.0, 10.0); // 위상 범위 제한
        
        let result = match state {
            0 => (safe_input + safe_phase).sin(),                    // sin 상태
            1 => (safe_input + safe_phase).cos(),                    // cos 상태
            2 => (safe_input * safe_phase).tanh(),                   // tanh 상태
            3 => {
                let cosh_val = (safe_input * safe_phase).cosh();
                1.0 / (cosh_val * cosh_val).max(1e-6)               // sech^2 상태 (0 나누기 방지)
            },
            4 => (safe_input * safe_phase * 0.1).exp().min(10.0),   // exp 상태 (폭발 방지)
            5 => {
                let arg = (safe_input * safe_phase).abs() + 1e-6;   // 음수와 0 방지
                arg.ln().clamp(-10.0, 10.0)                         // log 상태
            },
            6 => {
                let denom = safe_input * safe_phase + 1e-3;         // 0 나누기 방지
                (1.0 / denom).clamp(-100.0, 100.0)                 // 1/x 상태
            },
            7 => {
                let linear = safe_input * safe_phase;
                let quadratic = 0.1 * safe_input * safe_input;     // 계수 축소
                (linear + quadratic).clamp(-10.0, 10.0)            // 다항식 상태
            },
            _ => safe_input * safe_phase, // 기본값
        };
        
        // 최종 안전성 검사
        if result.is_finite() {
            result.clamp(-1.0, 1.0) // 출력 범위 [-1, 1] 제한
        } else {
            0.0 // NaN이나 무한대면 0으로 대체
        }
    }

    /// 개선된 융합 순전파: 고급 상태 전이 적용
    #[inline(always)]
    pub fn fused_forward_advanced(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 1. 연속 파라미터 추출 (lo)
        let r_fp32 = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        
        // 2. 상태 비트 추출 (hi) - 고급 버전
        let state_bits = self.hi & 0xFFFFF; // 하위 20비트
        
        // 3. 좌표 정규화
        let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        
        // 4. 연속 기저 패턴 계산
        let dist = (x_norm * x_norm + y_norm * y_norm).sqrt();
        let base_angle = y_norm.atan2(x_norm);
        let base_pattern = (r_fp32 - dist * r_fp32 + theta_fp32).clamp(0.0, 1.0);
        
        // 5. 다중 상태 기반 함수 선택 및 변조
        let primary_hash = ((i * 31 + j) & 0x7) as u64;
        let secondary_hash = ((i * 17 + j * 13) & 0x3) as u64;
        
        let primary_state = (state_bits >> (primary_hash * 3)) & 0x7;
        let secondary_state = (state_bits >> ((primary_hash + 8) * 2)) & 0x3;
        
        // 주요 함수 계산
        let primary_value = self.compute_state_function(
            primary_state, 
            base_angle + theta_fp32 * 0.5, 
            r_fp32
        );
        
        // 보조 변조 적용
        let modulation_factor = match secondary_state {
            0 => 1.0,                                    // 변조 없음
            1 => 0.8 + 0.4 * (dist * 3.14159).sin(),    // 사인 변조
            2 => 1.0 - 0.3 * dist,                      // 거리 기반 감쇠
            3 => (1.0 + (base_angle * 2.0).cos()) * 0.5, // 각도 기반 변조
            _ => 1.0,
        };
        
        let modulated_value = base_pattern * primary_value.abs() * modulation_factor;
        
        // 6. 고주파 세부사항 추가 (나머지 상태 비트 활용)
        let detail_bits = (state_bits >> 16) & 0xF; // 상위 4비트
        let detail_factor = 1.0 + 0.05 * (detail_bits as f32 / 15.0 - 0.5);
        
        (modulated_value * detail_factor).clamp(0.0, 1.0)
    }
}

/// 잔차 인코딩에 사용할 변환 타입
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformType {
    Dct,
    Dwt,
    Adaptive,
}

/// DCT/웨이블릿 잔차 계수
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualCoefficient {
    pub index: (u16, u16), // 블록 내 좌표 (최대 65535x65535)
    pub value: f32,
}

/// RBE 기본 패턴 + 잔차 계수를 포함하는 하이브리드 압축 블록
#[derive(Debug, Clone, PartialEq)]
pub struct HybridEncodedBlock {
    /// RBE 기본 패턴을 생성하는 8개의 연속 파라미터
    pub rbe_params: RbeParameters,
    /// 잔차 보정을 위한 상위 K개의 DCT 또는 웨이블릿 계수
    pub residuals: Vec<ResidualCoefficient>,
    /// 블록의 원래 크기
    pub rows: usize,
    pub cols: usize,
    /// 적용된 변환 타입
    pub transform_type: TransformType,
}

/// 단일 인코딩 블록의 그래디언트를 저장하는 구조체
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedBlockGradients {
    pub rbe_params_grad: RbeParameters,
    pub residuals_grad: Vec<ResidualCoefficient>,
}

/// RBE 기본 패턴 파라미터 타입 별칭
pub type RbeParameters = [f32; 8];

/// 기저 함수 타입 (기존 유지)
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

/// 행렬 압축 및 복원 (PoincareMatrix가 Packed128을 사용하도록 변경)
pub struct PoincareMatrix {
    pub seed: Packed128,
    pub rows: usize,
    pub cols: usize,
} 