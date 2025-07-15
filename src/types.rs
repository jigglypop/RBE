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
        let rotations = self.rotations;

        // 좌표(i, j)를 [-1.0, 1.0] 범위의 정규화된 좌표로 변환하여 초기 벡터로 사용합니다.
        // 이렇게 함으로써 각 좌표마다 다른 초기값으로 CORDIC 연산을 시작하게 됩니다.
        let mut x = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let mut y = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;

        for k in 0..64 {
            let sigma = if (rotations >> k) & 1 == 1 { 1.0 } else { -1.0 };
            
            let power_of_2 = (2.0f32).powi(-(k as i32));
            // let angle_k = power_of_2.atan(); // 사용되지 않으므로 제거

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
        
        // CORDIC 게인 보정. 초기 벡터가 (1,0)이 아니므로 이득 보정이 다를 수 있으나,
        // 유전 알고리즘이 최적의 시드를 찾을 것이므로 일단 기존 값을 사용합니다.
        let gain = 1.64676; 
        x / gain
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

/// 행렬 압축 및 복원 (기존 유지)
pub struct PoincareMatrix {
    pub seed: Packed64,
    pub rows: usize,
    pub cols: usize,
} 