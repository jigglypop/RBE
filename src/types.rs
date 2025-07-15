/// 64-bit Packed Poincaré 시드 표현
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
    pub r: f32,           // 12 bits (reduced from 20)
    pub theta: f32,       // 12 bits (reduced from 24)
    pub basis_id: u8,     // 4 bits
    pub freq_x: u8,       // 5 bits - X축 주파수 (1-32)
    pub freq_y: u8,       // 5 bits - Y축 주파수 (1-32)
    pub amplitude: f32,   // 6 bits - 진폭 스케일 (0.25-4.0)
    pub offset: f32,      // 6 bits - DC 오프셋 (-2.0 to +2.0)
    pub pattern_mix: u8,  // 4 bits - 패턴 믹싱 ID
    pub decay_rate: f32,  // 5 bits - 감쇠율 (0.0-4.0)
    pub d_theta: u8,      // 2 bits - θ 미분 차수 (0-3)
    pub d_r: bool,        // 1 bit - r 미분 차수 (0 or 1)
    pub log2_c: i8,       // 2 bits - 곡률 (부호 있음, -2 to +1)
}

/// 행렬 압축 및 복원
pub struct PoincareMatrix {
    pub seed: Packed64,
    pub rows: usize,
    pub cols: usize,
} 