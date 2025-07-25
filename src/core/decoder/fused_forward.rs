//! 융합 순전파 (Fused Forward Pass) 구현

use crate::core::packed_params::{PoincarePackedBit128, PoincareQuadrant};
use super::weight_generator::WeightGenerator;

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
        for (_seed_idx, seed) in weight_seeds.iter().enumerate() {
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
    /// 경계 조건 안정성 테스트
    pub fn test_boundary_stability_extended(&self) -> bool {
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