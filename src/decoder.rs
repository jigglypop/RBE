use std::f32::consts::PI;
use crate::types::{Packed64, DecodedParams};

impl Packed64 {
    /// 시드를 디코딩 - 개선된 비트 할당
    pub fn decode(&self) -> DecodedParams {
        let bits = self.0;

        // 비트 언패킹 - 새로운 레이아웃
        let r_bits = (bits >> 52) & 0xFFF;           // 12 bits
        let theta_bits = (bits >> 40) & 0xFFF;       // 12 bits
        let basis_id = ((bits >> 36) & 0xF) as u8;   // 4 bits
        let freq_x = ((bits >> 31) & 0x1F) as u8;    // 5 bits
        let freq_y = ((bits >> 26) & 0x1F) as u8;    // 5 bits
        let amplitude_bits = (bits >> 20) & 0x3F;    // 6 bits
        let offset_bits = (bits >> 14) & 0x3F;       // 6 bits
        let pattern_mix = ((bits >> 10) & 0xF) as u8; // 4 bits
        let decay_bits = (bits >> 5) & 0x1F;         // 5 bits
        let d_theta = ((bits >> 3) & 0x3) as u8;     // 2 bits
        let d_r = ((bits >> 2) & 0x1) != 0;          // 1 bit
        let log2_c_bits = (bits & 0x3) as u8;        // 2 bits

        // 값 복원
        let r = (r_bits as f32) / ((1u64 << 12) - 1) as f32;
        let theta = (theta_bits as f32 / ((1u64 << 12) - 1) as f32) * 2.0 * PI;
        
        // amplitude: 6비트 -> [0.25, 4.0]
        let amplitude = 0.25 + (amplitude_bits as f32 / 63.0) * 3.75;
        
        // offset: 6비트 -> [-2.0, 2.0]
        let offset = (offset_bits as f32 / 63.0) * 4.0 - 2.0;
        
        // decay_rate: 5비트 -> [0.0, 4.0]
        let decay_rate = (decay_bits as f32 / 31.0) * 4.0;
        
        // 2비트 부호있는 정수 복원 (-2 ~ +1)
        let log2_c = match log2_c_bits {
            0 => -2,
            1 => -1,
            2 => 0,
            3 => 1,
            _ => 0,
        };

        DecodedParams {
            r,
            theta,
            basis_id,
            freq_x: freq_x.max(1),  // 최소 1
            freq_y: freq_y.max(1),  // 최소 1
            amplitude,
            offset,
            pattern_mix,
            decay_rate,
            d_theta,
            d_r,
            log2_c,
        }
    }
} 