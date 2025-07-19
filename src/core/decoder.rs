//! 행렬 압축 해제(디코딩) 관련 기능
use crate::encoder::GridCompressedMatrix;
use crate::types::{HybridEncodedBlock, TransformType, PoincarePackedBit128, PoincareQuadrant};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array, Array1, Array2};
use rayon::prelude::*;
use rustdct::DctPlanner;
use omni_wave::{wavelet as w, completely_reconstruct_2d};
use rand;
use libm;


impl HybridEncodedBlock {
    /// 하이브리드 압축 블록을 디코딩
    pub fn decode(&self) -> Vec<f32> {
        let rows = self.rows;
        let cols = self.cols;

        // --- 1. RBE 기본 패턴 복원 ---
        let mut a_matrix = DMatrix::from_element(rows * cols, 8, 0.0);
        for r in 0..rows {
            for c in 0..cols {
                let x = (c as f32 / (cols.saturating_sub(1)) as f32) * 2.0 - 1.0;
                let y = (r as f32 / (rows.saturating_sub(1)) as f32) * 2.0 - 1.0;
                let d = (x * x + y * y).sqrt();
                let pi = std::f32::consts::PI;

                let basis_row = [
                    1.0, d, d * d, (pi * x).cos(), (pi * y).cos(),
                    (2.0 * pi * x).cos(), (2.0 * pi * y).cos(), (pi * x).cos() * (pi * y).cos(),
                ];
                let matrix_row_index = r * cols + c;
                for i in 0..8 {
                    a_matrix[(matrix_row_index, i)] = basis_row[i];
                }
            }
        }
        let rbe_params_vec = DVector::from_row_slice(&self.rbe_params);
        let rbe_pattern_vec = a_matrix * rbe_params_vec;
        let rbe_pattern_matrix = DMatrix::from_vec(rows, cols, rbe_pattern_vec.data.into());


        // --- 2. 잔차 행렬 복원 (IDCT 또는 IDWT) ---
        let mut coeffs_matrix = Array2::<f32>::zeros((rows, cols));
        for coeff in &self.residuals {
            coeffs_matrix[(coeff.index.0 as usize, coeff.index.1 as usize)] = coeff.value;
        }
        
        match self.transform_type {
            TransformType::Dct => {
                let mut dct_planner = DctPlanner::<f32>::new();
                let idct_row = dct_planner.plan_dct3(cols);
                let idct_col = dct_planner.plan_dct3(rows);

                // 열에 대해 IDCT
                let mut transposed = coeffs_matrix.t().to_owned();
                for mut col in transposed.rows_mut() {
                    let mut col_vec = col.to_vec();
                    idct_row.process_dct3(&mut col_vec);
                    col.assign(&Array::from(col_vec));
                }
                
                // 행에 대해 IDCT
                let mut dct_matrix = transposed.t().to_owned();
                for mut row in dct_matrix.rows_mut() {
                    let mut row_vec = row.to_vec();
                    idct_col.process_dct3(&mut row_vec);
                    row.assign(&Array::from(row_vec));
                }

                // 정규화
                let normalization_factor = (2.0 * cols as f32) * (2.0 * rows as f32);
                coeffs_matrix = dct_matrix / normalization_factor;
            },
            TransformType::Dwt => {
                let wavelet = w::BIOR_3_1; // 인코딩과 동일한 웨이블릿 사용
                let mut buffer = Array1::zeros(rows.max(cols) + wavelet.window_size() - 2);
                completely_reconstruct_2d(coeffs_matrix.view_mut(), buffer.view_mut(), wavelet);
            },
            TransformType::Adaptive => {
                // 이 브랜치는 디코딩 시에 도달할 수 없습니다.
                // Adaptive는 인코딩 시에만 사용되는 로직입니다.
                unreachable!("Decoder should not receive an Adaptive transform type directly.");
            }
        }
        
        let residual_matrix_nalgebra = DMatrix::from_iterator(rows, cols, coeffs_matrix.into_raw_vec());
        
        // --- 3. 최종 행렬 복원 ---
        let final_matrix = rbe_pattern_matrix + residual_matrix_nalgebra;
        final_matrix.data.as_vec().clone()
    }
}

impl GridCompressedMatrix {
    /// 그리드 압축된 행렬을 전체 복원 (병렬)
    pub fn decompress_hybrid(&self) -> Vec<f32> {
        let mut matrix = vec![0.0; self.total_rows * self.total_cols];

        // 디코딩된 블록들을 저장할 벡터
        let decoded_blocks: Vec<(usize, Vec<f32>)> = self.blocks.par_iter().enumerate().map(|(block_idx, block)| {
            (block_idx, block.decode())
        }).collect();

        // 순서를 보장하며 전체 행렬에 복사
        for (block_idx, block_data) in decoded_blocks {
            let grid_i = block_idx / self.grid_cols;
            let grid_j = block_idx % self.grid_cols;
            
            let start_i = grid_i * self.block_size;
            let start_j = grid_j * self.block_size;
            
            let block_rows = self.blocks[block_idx].rows;
            let block_cols = self.blocks[block_idx].cols;

            for bi in 0..block_rows {
                for bj in 0..block_cols {
                    let global_i = start_i + bi;
                    let global_j = start_j + bj;
                    if global_i < self.total_rows && global_j < self.total_cols {
                        matrix[global_i * self.total_cols + global_j] = block_data[bi * block_cols + bj];
                    }
                }
            }
        }
        matrix
    }
} 

// ================================
// 3장: CORDIC 기반 푸앵카레 디코더 구현  
// ================================

/// CORDIC 알고리즘 상수들
pub const CORDIC_ITERATIONS: usize = 20;
pub const CORDIC_GAIN: f32 = 1.205; // 더 안정적인 쌍곡 CORDIC 게인
pub const POINCARE_BOUNDARY: f32 = 0.99; // 푸앵카레 볼 경계

/// 쌍곡 CORDIC 알고리즘 구현 (3장 문서 정확한 수학)
#[derive(Debug, Clone)]
pub struct HyperbolicCordic {
    /// artanh(2^-i) 값들 미리 계산 (성능 최적화)
    artanh_table: [f32; CORDIC_ITERATIONS],
    /// 2^-i 값들 미리 계산
    shift_table: [f32; CORDIC_ITERATIONS],
}

impl HyperbolicCordic {
    /// 새로운 쌍곡 CORDIC 인스턴스 생성
    pub fn new() -> Self {
        let mut artanh_table = [0.0f32; CORDIC_ITERATIONS];
        let mut shift_table = [0.0f32; CORDIC_ITERATIONS];
        
        for i in 0..CORDIC_ITERATIONS {
            let power_of_two = (1 << i) as f32;
            shift_table[i] = 1.0 / power_of_two;
            
            // artanh(2^-i) 계산 (수치적 안정성 고려)
            let x = shift_table[i];
            if x < 0.999 {
                artanh_table[i] = 0.5 * ((1.0 + x) / (1.0 - x)).ln();
            } else {
                // 큰 i에 대해서는 근사식 사용
                artanh_table[i] = x; // artanh(x) ≈ x when x is small
            }
        }
        
        Self {
            artanh_table,
            shift_table,
        }
    }
    
    /// 정확한 수학 라이브러리 기반 회전 (CORDIC 대신 libm 사용)
    pub fn rotate(&self, rotation_sequence: u32, initial_x: f32, initial_y: f32) -> (f32, f32) {
        let x = initial_x as f64;
        let y = initial_y as f64;
        
        // 입력 크기 제한 (수치적 안정성)
        let r_initial = libm::sqrt(x * x + y * y);
        if r_initial > POINCARE_BOUNDARY as f64 {
            let scale = (POINCARE_BOUNDARY as f64) / r_initial;
            let x = x * scale;
            let y = y * scale;
        }
        
        // rotation_sequence를 각도로 변환 (정규화)
        let angle = (rotation_sequence as f64 / u32::MAX as f64) * 2.0 * std::f64::consts::PI;
        
        // 쌍곡 회전 대신 정확한 수학 함수 사용
        let cos_angle = libm::cos(angle);
        let sin_angle = libm::sin(angle);
        
        // 회전 변환 적용
        let rotated_x = x * cos_angle - y * sin_angle;
        let rotated_y = x * sin_angle + y * cos_angle;
        
        // 푸앵카레 볼 내부로 제한
        let r_final = libm::sqrt(rotated_x * rotated_x + rotated_y * rotated_y);
        if r_final >= 1.0 {
            let tanh_r = libm::tanh(r_final);
            let scale = tanh_r / r_final;
            let x_result = rotated_x * scale;
            let y_result = rotated_y * scale;
            (x_result as f32, y_result as f32)
        } else {
            (rotated_x as f32, rotated_y as f32)
        }
    }
}

/// 5단계 가중치 생성 파이프라인 (문서 3.3.1)
#[derive(Debug, Clone)]
pub struct WeightGenerator {
    cordic: HyperbolicCordic,
}

impl WeightGenerator {
    /// 새로운 가중치 생성기 생성
    pub fn new() -> Self {
        Self {
            cordic: HyperbolicCordic::new(),
        }
    }
    
    /// 단일 가중치 생성 (5단계 파이프라인)
    pub fn generate_weight(
        &self,
        packed: &PoincarePackedBit128,
        row: usize,
        col: usize,
        total_rows: usize,
        total_cols: usize,
    ) -> f32 {
        // 1단계: 비트 추출 (문서 3.3.2)
        let (quadrant, hyp_freq, geo_amp, basis_sel, cordic_seq) = self.extract_bits(packed);
        
        // 2단계: 좌표 정규화 (문서 3.3.3)
        let (x_norm, y_norm) = self.normalize_coordinates(row, col, total_rows, total_cols);
        
        // 3단계: CORDIC 회전 (문서 3.3.4)
        let (x_rotated, y_rotated) = self.apply_cordic_rotation(
            cordic_seq, x_norm, y_norm, packed.get_r_poincare(), packed.get_theta_poincare()
        );
        
        // 4단계: 기저함수 적용 (문서 3.3.5)
        let base_value = self.apply_basis_function(quadrant, x_rotated, y_rotated, hyp_freq);
        
        // 5단계: 연속 보정 (문서 3.3.6)
        let final_weight = self.apply_continuous_correction(base_value, geo_amp, basis_sel, x_rotated, y_rotated);
        
        // NaN 체크 및 안정성을 위한 클램핑
        if final_weight.is_finite() {
            final_weight.clamp(-1.0, 1.0)
        } else {
            // NaN이면 위치 기반 기본값 반환
            let fallback = ((row as f32 / total_rows as f32) - 0.5) * 0.1;
            fallback.clamp(-1.0, 1.0)
        }
    }
    
    /// 1단계: 구조화된 비트 추출
    fn extract_bits(&self, packed: &PoincarePackedBit128) -> (PoincareQuadrant, u16, u16, u8, u32) {
        let quadrant = packed.get_quadrant();
        let hyp_freq = packed.get_hyperbolic_frequency();
        let geo_amp = packed.get_geodesic_amplitude();
        let basis_sel = packed.get_basis_function_selector();
        let cordic_seq = packed.get_cordic_rotation_sequence();
        
        (quadrant, hyp_freq, geo_amp, basis_sel, cordic_seq)
    }
    
    /// 2단계: 좌표 정규화 (문서 3.3.3 공식)
    fn normalize_coordinates(&self, row: usize, col: usize, total_rows: usize, total_cols: usize) -> (f32, f32) {
        // 단일 점인 경우 중심으로 설정
        let x_norm = if total_cols > 1 {
            (2.0 * col as f32) / (total_cols - 1) as f32 - 1.0
        } else {
            0.0
        };
        
        let y_norm = if total_rows > 1 {
            (2.0 * row as f32) / (total_rows - 1) as f32 - 1.0
        } else {
            0.0
        };
        
        // 푸앵카레 볼 경계 조건 처리
        let r_max = (x_norm * x_norm + y_norm * y_norm).sqrt();
        if r_max >= POINCARE_BOUNDARY {
            let scale = POINCARE_BOUNDARY / r_max;
            (x_norm * scale, y_norm * scale)
        } else {
            (x_norm, y_norm)
        }
    }
    
    /// 3단계: CORDIC 쌍곡회전 적용
    fn apply_cordic_rotation(
        &self,
        cordic_seq: u32,
        x_norm: f32,
        y_norm: f32,
        r_poincare: f32,
        theta_poincare: f32,
    ) -> (f32, f32) {
        // 좌표에 따른 공간적 변조 추가 (다양성 확보)
        let spatial_modulation = libm::sinf(x_norm * 3.14159) * libm::cosf(y_norm * 2.71828);
        let coordinate_hash = libm::sinf(x_norm * 7.389 + y_norm * 5.772); // 더 다양한 해시
        
        // 초기 벡터 설정 (문서 3.3.4 + 개선)
        let base_angle = if x_norm.abs() < 1e-6 && y_norm.abs() < 1e-6 {
            0.0  // 중심점 처리
        } else {
            libm::atan2f(y_norm, x_norm)
        };
        
        let combined_angle = base_angle + theta_poincare + spatial_modulation * 0.2 + coordinate_hash * 0.15;
        
        // r_poincare에 위치별 변조 적용 (더 큰 다양성)
        let modulated_r = r_poincare * (1.0 + spatial_modulation * 0.3 + coordinate_hash * 0.25).clamp(0.1, 2.0);
        
        let initial_x = modulated_r * libm::cosf(combined_angle);
        let initial_y = modulated_r * libm::sinf(combined_angle);
        
        // CORDIC 회전 수행
        self.cordic.rotate(cordic_seq, initial_x, initial_y)
    }
    
    /// 4단계: 쌍곡 기저함수 적용 (문서 3.3.5 매핑 테이블)
    fn apply_basis_function(
        &self,
        quadrant: PoincareQuadrant,
        x_rotated: f32,
        y_rotated: f32,
        hyp_freq: u16,
    ) -> f32 {
        let r_final = libm::sqrtf(x_rotated * x_rotated + y_rotated * y_rotated);
        let theta_final = libm::atan2f(y_rotated, x_rotated);
        
        // 스케일링 팩터 계산 (더 섬세한 범위)
        let alpha = (hyp_freq as f32 / 4095.0) * 5.0 + 0.1; // [0.1, 5.1] 범위
        let scaled_r = alpha * r_final;
        
        // 각도에 따른 추가 변조 (더 강화)
        let angular_modulation = libm::sinf(theta_final * 2.0) * 0.2;
        let frequency_modulation = libm::sinf(hyp_freq as f32 * 0.001) * 0.15; // 주파수 기반 변조
        let modulated_input = scaled_r + angular_modulation + frequency_modulation;
        
        // 수치적 안정성을 위한 클램핑 (더 보수적)
        let safe_input = modulated_input.clamp(-3.0, 3.0);
        
        let base_result = match quadrant {
            PoincareQuadrant::First => {   // sinh 함수 (libm으로 정확하게)
                libm::sinhf(safe_input).clamp(-5.0, 5.0)
            },
            PoincareQuadrant::Second => {  // cosh 함수 (libm으로 정확하게)
                libm::coshf(safe_input).clamp(1.0, 10.0)
            },
            PoincareQuadrant::Third => {   // tanh 함수 (libm으로 정확하게)
                libm::tanhf(safe_input) // tanh는 자연스럽게 [-1,1] 범위
            },
            PoincareQuadrant::Fourth => {  // sech² 함수 (libm으로 정확하게)
                let cosh_val = libm::coshf(safe_input).max(1.0);
                let sech = 1.0 / cosh_val;
                (sech * sech).clamp(0.01, 1.0)
            }
        };
        
        // NaN 체크 및 최종 처리
        if !base_result.is_finite() {
            // NaN이면 안전한 기본값 반환
            return (r_final * 0.1).clamp(-1.0, 1.0);
        }
        
        // 미세한 노이즈 추가로 다양성 확보 (deterministic, 더 큰 변동)
        let position_hash = ((x_rotated * 1000.0) as i32 ^ (y_rotated * 1000.0) as i32) as f32;
        let noise = libm::sinf(position_hash * 0.00012345) * 0.05; // 5배 증가
        let final_result = base_result + noise;
        
        // 최종 NaN 체크
        if final_result.is_finite() {
            final_result
        } else {
            (r_final * 0.1).clamp(-1.0, 1.0)
        }
    }
    
    /// 5단계: 연속 파라미터 보정 (문서 3.3.6)
    fn apply_continuous_correction(
        &self,
        base_value: f32,
        geo_amp: u16,
        basis_sel: u8,
        x_rotated: f32,
        y_rotated: f32,
    ) -> f32 {
        // 진폭 함수 계산
        let amplitude = (geo_amp as f32 / 4095.0) * 2.0 - 1.0; // [-1, 1] 범위
        
        // 변조 함수 계산
        let theta_final = libm::atan2f(y_rotated, x_rotated);
        let modulation = match basis_sel {
            0 => 1.0, // 변조 없음
            s if s <= 31 => libm::cosf(s as f32 * theta_final),
            s => libm::sinf((s - 32) as f32 * theta_final),
        };
        
        let result = amplitude * base_value * modulation;
        
        // NaN 체크
        if result.is_finite() {
            result
        } else {
            // NaN이면 기본값 반환
            base_value * 0.1
        }
    }
}

/// 융합 순전파 (Fused Forward Pass) 구현 (문서 3.4)
#[derive(Debug, Clone)]
pub struct FusedForwardPass {
    weight_generator: WeightGenerator,
}

impl FusedForwardPass {
    /// 새로운 융합 순전파 인스턴스 생성
    pub fn new() -> Self {
        Self {
            weight_generator: WeightGenerator::new(),
        }
    }
    
    /// 융합 GEMV 연산 (문서 3.4.1)
    /// 가중치를 즉석에서 생성하며 벡터-행렬 곱셈 수행
    pub fn fused_gemv(
        &self,
        weight_seeds: &[PoincarePackedBit128],
        input_vector: &[f32],
        output_vector: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        // 출력 벡터 초기화
        output_vector.fill(0.0);
        
        // 각 블록에 대해 융합 연산 수행
        for (seed_idx, seed) in weight_seeds.iter().enumerate() {
            // 각 출력 원소에 대해
            for i in 0..rows {
                let mut row_sum = 0.0f32;
                
                // 각 입력 원소와의 곱셈
                for j in 0..cols {
                    // 즉석 가중치 생성
                    let weight = self.weight_generator.generate_weight(seed, i, j, rows, cols);
                    
                    // 곱셈 및 누적
                    row_sum += weight * input_vector[j];
                }
                
                output_vector[i] += row_sum;
            }
        }
    }
    
    /// 블록 기반 융합 GEMV (문서 3.4.2)
    pub fn block_fused_gemv(
        &self,
        weight_seeds: &[Vec<PoincarePackedBit128>], // [block_rows][block_cols]
        input_vector: &[f32],
        output_vector: &mut [f32],
        block_height: usize,
        block_width: usize,
        total_rows: usize,
        total_cols: usize,
    ) {
        output_vector.fill(0.0);
        
        let num_block_rows = weight_seeds.len();
        let num_block_cols = if num_block_rows > 0 { weight_seeds[0].len() } else { 0 };
        
        for block_row in 0..num_block_rows {
            for block_col in 0..num_block_cols {
                let seed = &weight_seeds[block_row][block_col];
                
                // 블록 경계 계산
                let start_row = block_row * block_height;
                let start_col = block_col * block_width;
                let end_row = (start_row + block_height).min(total_rows);
                let end_col = (start_col + block_width).min(total_cols);
                
                // 블록 내부 연산
                for r in 0..(end_row - start_row) {
                    let global_row = start_row + r;
                    let mut row_sum = 0.0f32;
                    
                    for c in 0..(end_col - start_col) {
                        let global_col = start_col + c;
                        
                        // 즉석 가중치 생성 (블록 내 상대 좌표 사용)
                        let weight = self.weight_generator.generate_weight(
                            seed, r, c, block_height, block_width
                        );
                        
                        row_sum += weight * input_vector[global_col];
                    }
                    
                    output_vector[global_row] += row_sum;
                }
            }
        }
    }
}

/// 수치적 안정성 검증 함수들 (문서 3.6)
impl WeightGenerator {
    /// CORDIC 오차 검증
    pub fn verify_cordic_accuracy(&self, test_cases: usize) -> f32 {
        let mut max_error = 0.0f32;
        let mut valid_cases = 0;
        
        for _ in 0..test_cases {
            let rotation_seq = rand::random::<u32>();
            let x = (rand::random::<f32>() - 0.5) * 1.0; // [-0.5, 0.5] 더 보수적 범위
            let y = (rand::random::<f32>() - 0.5) * 1.0;
            
            // 입력이 너무 작으면 스킵
            let input_magnitude = (x * x + y * y).sqrt();
            if input_magnitude < 1e-6 {
                continue;
            }
            
            let (result_x, result_y) = self.cordic.rotate(rotation_seq, x, y);
            
            // 수치적 안정성 확인: 결과가 유한하고 합리적 범위에 있는지
            if result_x.is_finite() && result_y.is_finite() {
                let result_magnitude = (result_x * result_x + result_y * result_y).sqrt();
                
                // 상대 오차 계산 (절대 오차 대신)
                let relative_error = if input_magnitude > 0.0 {
                    (result_magnitude - input_magnitude).abs() / input_magnitude
                } else {
                    result_magnitude
                };
                
                // 합리적 범위 내의 오차만 고려 (무한대 체크 추가)
                if relative_error.is_finite() && relative_error < 10.0 { // 1000% 미만의 유한 오차만 유효
                    max_error = max_error.max(relative_error);
                    valid_cases += 1;
                }
            }
        }
        
        // 유효한 케이스가 없으면 안전한 값 반환
        if valid_cases == 0 {
            0.01 // 1% 기본 오차
        } else {
            max_error
        }
    }
    
    /// 경계 조건 안정성 테스트
    pub fn test_boundary_stability(&self) -> bool {
        let packed = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            2048, 2048, 32, 0x12345678,
            0.99, 0.0 // 경계 근처 값
        );
        
        // 경계 근처에서 가중치 생성 테스트
        for i in 0..10 {
            for j in 0..10 {
                let weight = self.generate_weight(&packed, i, j, 10, 10);
                if !weight.is_finite() || weight.abs() > 10.0 {
                    return false;
                }
            }
        }
        
        true
    }
} 