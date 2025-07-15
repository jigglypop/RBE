use crate::types::{Packed64, DecodedParams};
use std::f32::consts::PI;

impl Packed64 {
    pub fn new(
        r: f32, theta: f32, basis_id: u8, freq_x: u8, freq_y: u8,
        amplitude: f32, offset: f32, pattern_mix: u8, decay_rate: f32,
        d_theta: u8, d_r: bool, log2_c: i8
    ) -> Self {
        // r: [0, 1) 범위로 클램핑
        let r_clamped = r.clamp(0.0, 0.99999);
        // theta: [0, 2π) 범위로 정규화
        let theta_normalized = theta.rem_euclid(2.0 * PI);
        // amplitude, offset, decay_rate 정규화
        let amplitude_normalized = ((amplitude.clamp(0.25, 4.0) - 0.25) / 3.75 * 63.0) as u64;
        let offset_normalized = ((offset.clamp(-2.0, 2.0) + 2.0) / 4.0 * 63.0) as u64;
        let decay_normalized = (decay_rate.clamp(0.0, 4.0) / 4.0 * 31.0) as u64;

        // 비트 패킹
        let r_bits = (r_clamped * ((1u64 << 12) - 1) as f32).round().min(4094.0) as u64;
        let theta_bits = (theta_normalized / (2.0 * PI) * ((1u64 << 12) - 1) as f32).round() as u64;
        
        let mut packed = 0u64;
        packed |= (r_bits & 0xFFF) << 52;
        packed |= (theta_bits & 0xFFF) << 40;
        packed |= ((basis_id as u64) & 0xF) << 36;
        packed |= ((freq_x as u64) & 0x1F) << 31;
        packed |= ((freq_y as u64) & 0x1F) << 26;
        packed |= (amplitude_normalized & 0x3F) << 20;
        packed |= (offset_normalized & 0x3F) << 14;
        packed |= ((pattern_mix as u64) & 0xF) << 10;
        packed |= (decay_normalized & 0x1F) << 5;
        packed |= ((d_theta as u64) & 0x3) << 3;
        packed |= ((d_r as u64) & 0x1) << 2;
        
        let log2_c_bits = match log2_c { -2 => 0, -1 => 1, 0 => 2, 1 => 3, _ => 2 };
        packed |= log2_c_bits & 0x3;

        Packed64(packed)
    }

    /// DecodedParams로부터 Packed64를 생성합니다.
    pub fn from_params(params: &DecodedParams) -> Self {
        Packed64::new(
            params.r, params.theta, params.basis_id, params.freq_x, params.freq_y,
            params.amplitude, params.offset, params.pattern_mix, params.decay_rate,
            params.d_theta, params.d_r, params.log2_c,
        )
    }
} 