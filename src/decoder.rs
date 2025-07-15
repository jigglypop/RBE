use crate::types::Packed64;

impl Packed64 {
    /// 인코딩된 `u64` 값을 그대로 반환합니다.
    /// CORDIC 모델에서는 이 `rotations`값이 모든 정보를 담고 있습니다.
    pub fn decode(&self) -> u64 {
        self.rotations
    }
} 