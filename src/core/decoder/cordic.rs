//! CORDIC 기반 푸앵카레 디코더 구현

use libm;

/// 6.1 CORDIC 알고리즘 푸앵카레 볼 버전
/// 
/// CORDIC (COordinate Rotation DIgital Computer) 알고리즘을 
/// 푸앵카레 볼 좌표계에 맞게 확장한 구현

pub const POINCARE_BOUNDARY: f32 = 0.9999999; // f32 정밀도에 맞는 푸앵카레 볼 경계

/// CORDIC용 미리 계산된 아탄젠트 테이블
pub const CORDIC_ATAN_TABLE: [f32; 32] = [
    0.7853981634, 0.4636476090, 0.2449786631, 0.1243549945,
    0.0624188100, 0.0312398334, 0.0156237286, 0.0078123493,
    0.0039062302, 0.0019531226, 0.0009765622, 0.0004882812,
    0.0002441406, 0.0001220703, 0.0000610352, 0.0000305176,
    0.0000152588, 0.0000076294, 0.0000038147, 0.0000019073,
    0.0000009537, 0.0000004768, 0.0000002384, 0.0000001192,
    0.0000000596, 0.0000000298, 0.0000000149, 0.0000000075,
    0.0000000037, 0.0000000019, 0.0000000009, 0.0000000005,
];

/// 하이퍼볼릭 CORDIC 계산
pub fn hyperbolic_cordic(x: f32, y: f32, target_angle: f32) -> (f32, f32) {
    if x < 0.999 {
        let mut current_x = x;
        let mut current_y = y;
        let mut z = target_angle;
        
        for i in 0..16 {
            let d = if z >= 0.0 { 1.0 } else { -1.0 };
            let shift = 1.0 / (1 << i) as f32;
            
            let new_x = current_x + d * current_y * shift;
            let new_y = current_y + d * current_x * shift;
            
            current_x = new_x;
            current_y = new_y;
            z -= d * CORDIC_ATAN_TABLE[i];
        }
        
        (current_x, current_y)
    } else {
        (x, y)
    }
} 