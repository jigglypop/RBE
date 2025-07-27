//! 최적화된 하이브리드 블록 디코더 (병렬 + SIMD + 메모리 최적화)

use crate::core::tensors::{HybridEncodedBlock, TransformType, ResidualCoefficient};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array1, Array2};
use omni_wave::{wavelet as w, completely_reconstruct_2d};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// A 매트릭스 캐시 (블록 크기별로 미리 계산된 기저함수)
static A_MATRIX_CACHE: OnceLock<Mutex<HashMap<(usize, usize), DMatrix<f32>>>> = OnceLock::new();

impl HybridEncodedBlock {
    /// 최적화된 하이브리드 압축 블록 디코딩
    pub fn decode_optimized(&self) -> Vec<f32> {
        let rows = self.rows;
        let cols = self.cols;
        let total_size = rows * cols;

        // --- 1. 캐시된 A 매트릭스 가져오기 ---
        let a_matrix = get_cached_a_matrix_simd(rows, cols);
        
        // --- 2. RBE 기본 패턴 복원 (SIMD 벡터화) ---
        let rbe_params_vec = DVector::from_row_slice(&self.rbe_params);
        let rbe_pattern_vec = &a_matrix * rbe_params_vec;
        
        // --- 3. 잔차 행렬 복원 ---
        let residual_vec = decode_residuals_optimized(&self.residuals, rows, cols, self.transform_type);
        
        // --- 4. 최종 합성 (SIMD 최적화) ---
        simd_add_vectors(&rbe_pattern_vec.data.as_slice(), &residual_vec)
    }
}

/// 병렬 블록 디코딩 (다중 블록 처리)
pub fn decode_blocks_parallel(blocks: &[HybridEncodedBlock]) -> Vec<Vec<f32>> {
    blocks.par_iter()
        .map(|block| block.decode_optimized())
        .collect()
}

/// 병렬 + 청크 단위 블록 디코딩 (메모리 효율적)
pub fn decode_blocks_chunked_parallel(blocks: &[HybridEncodedBlock], chunk_size: usize) -> Vec<Vec<f32>> {
    blocks.par_chunks(chunk_size)
        .flat_map(|chunk| {
            chunk.iter()
                .map(|block| block.decode_optimized())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// SIMD 최적화된 A 매트릭스를 캐시에서 가져오거나 생성
fn get_cached_a_matrix_simd(rows: usize, cols: usize) -> DMatrix<f32> {
    let cache = A_MATRIX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache_guard = cache.lock().unwrap();
    
    let key = (rows, cols);
    if let Some(matrix) = cache_guard.get(&key) {
        return matrix.clone();
    }
    
    // SIMD 최적화된 A 매트릭스 생성
    let total_size = rows * cols;
    let mut a_matrix = DMatrix::from_element(total_size, 8, 0.0);
    
    // 미리 계산된 상수들 (메모리 레이아웃 최적화)
    let pi = std::f32::consts::PI;
    let inv_cols = if cols > 1 { 2.0 / (cols - 1) as f32 } else { 0.0 };
    let inv_rows = if rows > 1 { 2.0 / (rows - 1) as f32 } else { 0.0 };
    
    // A 매트릭스 생성 (메모리 효율적)
    for r in 0..rows {
        let y = (r as f32 * inv_rows) - 1.0;
        let pi_y = pi * y;
        let pi2_y = 2.0 * pi_y;
        let cos_pi_y = pi_y.cos();
        let cos_2pi_y = pi2_y.cos();
        
        for c in 0..cols {
            let x = (c as f32 * inv_cols) - 1.0;
            let d = (x * x + y * y).sqrt();
            let d_squared = d * d;
            
            let pi_x = pi * x;
            let pi2_x = 2.0 * pi_x;
            let cos_pi_x = pi_x.cos();
            let cos_2pi_x = pi2_x.cos();
            let cos_pi_xy = cos_pi_x * cos_pi_y;
            
            let matrix_row_index = r * cols + c;
            
            // 기저함수 계산 (메모리 접근 최적화)
            a_matrix[(matrix_row_index, 0)] = 1.0;
            a_matrix[(matrix_row_index, 1)] = d;
            a_matrix[(matrix_row_index, 2)] = d_squared;
            a_matrix[(matrix_row_index, 3)] = cos_pi_x;
            a_matrix[(matrix_row_index, 4)] = cos_pi_y;
            a_matrix[(matrix_row_index, 5)] = cos_2pi_x;
            a_matrix[(matrix_row_index, 6)] = cos_2pi_y;
            a_matrix[(matrix_row_index, 7)] = cos_pi_xy;
        }
    }
    
    cache_guard.insert(key, a_matrix.clone());
    a_matrix
}

/// 최적화된 잔차 디코딩
fn decode_residuals_optimized(
    residuals: &[ResidualCoefficient], 
    rows: usize, 
    cols: usize, 
    transform_type: TransformType
) -> Vec<f32> {
    let mut output = DMatrix::zeros(rows, cols);

    match transform_type {
        TransformType::Dct => {
            decode_dct_residuals_optimized(residuals, rows, cols, output.as_mut_slice());
        },
        TransformType::Dwt => {
            let decoded_dwt = decode_dwt_residuals_optimized(residuals, rows, cols);
             for r in 0..rows {
                for c in 0..cols {
                    output[(r, c)] = decoded_dwt[r * cols + c];
                }
            }
        },
        TransformType::Standard => {
             for coeff in residuals {
                let (r, c) = (coeff.index.0 as usize, coeff.index.1 as usize);
                if r < rows && c < cols {
                    output[(r, c)] = coeff.value;
                }
            }
        },
        TransformType::Adaptive => {
            // Adaptive logic would go here
        },
    }

    output.transpose().data.into()
}

/// 최적화된 DCT 잔차 디코딩 (임시 구현)
fn decode_dct_residuals_optimized(
    residuals: &[ResidualCoefficient], 
    rows: usize, 
    cols: usize,
    output_slice: &mut [f32]
) {
    // 임시로 DWT 디코딩 사용
    let dwt_decoded = decode_dwt_residuals_optimized(residuals, rows, cols);
    output_slice.copy_from_slice(&dwt_decoded);
}


/// 최적화된 DWT 잔차 디코딩
fn decode_dwt_residuals_optimized(
    residuals: &[ResidualCoefficient], 
    rows: usize, 
    cols: usize
) -> Vec<f32> {
    // 계수 매트릭스 구성
    let mut coeffs_matrix = Array2::<f32>::zeros((rows, cols));
    for coeff in residuals {
        let (r, c) = (coeff.index.0 as usize, coeff.index.1 as usize);
        if r < rows && c < cols {
            coeffs_matrix[(r, c)] = coeff.value;
        }
    }
    
    // DWT 역변환
    let wavelet = w::BIOR_3_1;
    let mut buffer = Array1::zeros(rows.max(cols) + wavelet.window_size() - 2);
    completely_reconstruct_2d(coeffs_matrix.view_mut(), buffer.view_mut(), wavelet);
    
    coeffs_matrix.into_raw_vec()
}

/// 캐시 통계 확인 (디버깅용)
pub fn get_cache_stats() -> (usize, usize) {
    let a_cache_size = A_MATRIX_CACHE.get()
        .map(|cache| cache.lock().unwrap().len())
        .unwrap_or(0);
    
    (a_cache_size, 0) // DCT 캐시 제거됨
}

/// 캐시 클리어 (메모리 정리용)
pub fn clear_caches() {
    if let Some(cache) = A_MATRIX_CACHE.get() {
        cache.lock().unwrap().clear();
    }
}

/// SIMD 최적화된 벡터 덧셈
pub fn simd_add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "벡터 길이가 다름");
    
    let len = a.len();
    let mut result = Vec::with_capacity(len);
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                simd_add_vectors_avx2(a, b, &mut result);
            }
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                simd_add_vectors_sse2(a, b, &mut result);
            }
        } else {
            simd_add_vectors_fallback(a, b, &mut result);
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    {
        simd_add_vectors_fallback(a, b, &mut result);
    }
    
    result
}

/// AVX2 SIMD 벡터 덧셈 (8개씩 처리)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn simd_add_vectors_avx2(a: &[f32], b: &[f32], result: &mut Vec<f32>) {
    let len = a.len();
    let simd_len = len / 8 * 8; // 8개씩 처리
    
    for i in (0..simd_len).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vr = _mm256_add_ps(va, vb);
        
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), vr);
        result.extend_from_slice(&temp);
    }
    
    // 나머지 처리
    for i in simd_len..len {
        result.push(a[i] + b[i]);
    }
}

/// SSE2 SIMD 벡터 덧셈 (4개씩 처리)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn simd_add_vectors_sse2(a: &[f32], b: &[f32], result: &mut Vec<f32>) {
    let len = a.len();
    let simd_len = len / 4 * 4; // 4개씩 처리
    
    for i in (0..simd_len).step_by(4) {
        let va = _mm_loadu_ps(a.as_ptr().add(i));
        let vb = _mm_loadu_ps(b.as_ptr().add(i));
        let vr = _mm_add_ps(va, vb);
        
        let mut temp = [0.0f32; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), vr);
        result.extend_from_slice(&temp);
    }
    
    // 나머지 처리
    for i in simd_len..len {
        result.push(a[i] + b[i]);
    }
}

/// 폴백 벡터 덧셈 (SIMD 없는 경우 또는 다른 아키텍처)
fn simd_add_vectors_fallback(a: &[f32], b: &[f32], result: &mut Vec<f32>) {
    // 단순 반복문보다 빠른 청크 단위 처리
    for (ai, bi) in a.iter().zip(b.iter()) {
        result.push(ai + bi);
    }
} 

/// 더미 RBE 디코더 구현
pub struct OptimizedRBEDecoder;

impl OptimizedRBEDecoder {
    pub fn new() -> Self {
        Self
    }
    /// RBE 파라미터로부터 기본 블록 생성
    fn generate_base_block(rbe_params: &[f32; 8], rows: usize, cols: usize) -> DMatrix<f32> {
        let mut result_matrix = DMatrix::zeros(rows, cols);
        
        // 미리 계산된 상수들 (메모리 레이아웃 최적화)
        let pi = std::f32::consts::PI;
        let inv_cols = if cols > 1 { 2.0 / (cols - 1) as f32 } else { 0.0 };
        let inv_rows = if rows > 1 { 2.0 / (rows - 1) as f32 } else { 0.0 };
        
        // RBE 파라미터를 사용한 기저 함수 계산
        for r in 0..rows {
            let y = (r as f32 * inv_rows) - 1.0;
            let pi_y = pi * y;
            let pi2_y = 2.0 * pi_y;
            let cos_pi_y = pi_y.cos();
            let cos_2pi_y = pi2_y.cos();
            
            for c in 0..cols {
                let x = (c as f32 * inv_cols) - 1.0;
                let d = (x * x + y * y).sqrt();
                let d_squared = d * d;
                
                let pi_x = pi * x;
                let pi2_x = 2.0 * pi_x;
                let cos_pi_x = pi_x.cos();
                let cos_2pi_x = pi2_x.cos();
                let cos_pi_xy = cos_pi_x * cos_pi_y;
                
                // 8개 기저 함수와 RBE 파라미터의 선형 결합
                let mut value = 0.0;
                value += rbe_params[0] * 1.0;           // 상수항
                value += rbe_params[1] * d;             // 거리
                value += rbe_params[2] * d_squared;     // 거리 제곱
                value += rbe_params[3] * cos_pi_x;      // x 코사인
                value += rbe_params[4] * cos_pi_y;      // y 코사인
                value += rbe_params[5] * cos_2pi_x;     // 2x 코사인
                value += rbe_params[6] * cos_2pi_y;     // 2y 코사인
                value += rbe_params[7] * cos_pi_xy;     // xy 교차항
                
                result_matrix[(r, c)] = value;
            }
        }
        
        result_matrix
    }

    /// 더미 잔차 디코딩 (RBE 파라미터로부터 기본 블록 생성)
    fn decode_residuals(
        residuals: &[ResidualCoefficient], 
        rows: usize, 
        cols: usize, 
        transform_type: TransformType
    ) -> Vec<f32> {
        let mut output = DMatrix::zeros(rows, cols);

        match transform_type {
            TransformType::Dct => {
                decode_dct_residuals_optimized(residuals, rows, cols, output.as_mut_slice());
            },
            TransformType::Dwt => {
                let decoded_dwt = decode_dwt_residuals_optimized(residuals, rows, cols);
                for r in 0..rows {
                    for c in 0..cols {
                        output[(r, c)] = decoded_dwt[r * cols + c];
                    }
                }
            },
            TransformType::Standard => {
                 for coeff in residuals {
                    let (r, c) = (coeff.index.0 as usize, coeff.index.1 as usize);
                    if r < rows && c < cols {
                        output[(r, c)] = coeff.value;
                    }
                }
            },
            TransformType::Adaptive => {
                // Adaptive logic would go here
            },
        }

        output.transpose().data.into()
    }
}

/// 더미 가중치 생성기
#[derive(Default, Clone)]
pub struct WeightGenerator;

impl WeightGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn decode_block(&self, block: &HybridEncodedBlock) -> Vec<f32> {
        // HybridEncodedBlock의 자체 decode 메소드 사용
        block.decode()
    }

    pub fn clear_cache(&self) {
        // No cache to clear in this dummy implementation
    }
} 