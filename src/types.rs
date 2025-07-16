use rand::Rng;
use std::f32;
use crate::math::{ste_quant_q0x, ste_quant_phase};

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
}

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