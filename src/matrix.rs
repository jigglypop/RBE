use crate::math::adam_update;
use crate::types::PoincareMatrix;
use std::f32::consts::PI;

impl PoincareMatrix {
    /// Adam + 128bit 연속 파라미터 학습
    pub fn train_with_adam128(
        &self,
        target: &[f32],
        rows: usize,
        cols: usize,
        epochs: usize,
        lr: f32,
    ) -> Self {
        // ① lo에서 연속 파라미터 직접 추출
        let mut r_fp32 = f32::from_bits((self.seed.lo >> 32) as u32);
        let mut theta_fp32 = f32::from_bits(self.seed.lo as u32);

        // ② Adam 모멘텀
        let mut m_r = 0.0; let mut v_r = 0.0;
        let mut m_th= 0.0; let mut v_th= 0.0;

        for ep in 1..=epochs {
            // --- forward: 연속 값으로 직접 weight 생성 ---
            let mut current_seed = self.seed;
            current_seed.lo = ((r_fp32.to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
            
            let mut pred = Vec::with_capacity(target.len());
            for i in 0..rows { 
                for j in 0..cols {
                    pred.push(current_seed.compute_weight_continuous(i, j, rows, cols));
                }
            }

            // --- gradient 계산 (수치 미분) ---
            let mut g_r = 0.0; 
            let mut g_th = 0.0;
            let eps = 1e-3;  // 1e-4 -> 1e-3으로 증가
            
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    let diff = pred[idx] - target[idx];
                    
                    // r에 대한 그래디언트
                    let mut seed_r_plus = current_seed;
                    seed_r_plus.lo = (((r_fp32 + eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                    let w_r_plus = seed_r_plus.compute_weight_continuous(i, j, rows, cols);
                    
                    let mut seed_r_minus = current_seed;
                    seed_r_minus.lo = (((r_fp32 - eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                    let w_r_minus = seed_r_minus.compute_weight_continuous(i, j, rows, cols);
                    
                    let dr = (w_r_plus - w_r_minus) / (2.0 * eps);
                    g_r += diff * dr;
                    
                    // theta에 대한 그래디언트
                    let mut seed_th_plus = current_seed;
                    seed_th_plus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 + eps).to_bits() as u64;
                    let w_th_plus = seed_th_plus.compute_weight_continuous(i, j, rows, cols);
                    
                    let mut seed_th_minus = current_seed;
                    seed_th_minus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 - eps).to_bits() as u64;
                    let w_th_minus = seed_th_minus.compute_weight_continuous(i, j, rows, cols);
                    
                    let dth = (w_th_plus - w_th_minus) / (2.0 * eps);
                    g_th += diff * dth;
                }
            }

            // --- Adam 업데이트 ---
            adam_update(&mut r_fp32, &mut m_r, &mut v_r, g_r, lr, ep as i32);
            adam_update(&mut theta_fp32, &mut m_th, &mut v_th, g_th, lr, ep as i32);
            r_fp32 = r_fp32.clamp(0.1, 1.0);  // 최소값을 0.1로 변경
            theta_fp32 = theta_fp32.rem_euclid(2.0*PI);

            // 로그
            if ep%100==0 || ep==epochs {  // 50 -> 100으로 변경
                current_seed.lo = ((r_fp32.to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                let rmse = {
                    let mut err = 0.0;
                    for i in 0..rows {
                        for j in 0..cols {
                            let idx = i * cols + j;
                            let w = current_seed.compute_weight_continuous(i, j, rows, cols);
                            err += (target[idx] - w).powi(2);
                        }
                    }
                    (err / target.len() as f32).sqrt()
                };
                println!("epoch {:3}/{}, RMSE={:.5}, r={:.4}, theta={:.4}, grad_r={:.6}, grad_theta={:.6}", 
                         ep, epochs, rmse, r_fp32, theta_fp32, g_r, g_th);
            }
        }

        // ③ 최종 시드 생성
        let mut final_seed = self.seed;
        final_seed.lo = ((r_fp32.to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
        
        // hi 필드도 업데이트 (양자화된 값 저장)
        let r_quant = (r_fp32.clamp(0.0, 1.0) * ((1u64 << 20) - 1) as f32) as u64;
        let theta_quant = ((theta_fp32.rem_euclid(2.0 * PI) / (2.0 * PI)) * ((1u64 << 24) - 1) as f32) as u64;
        final_seed.hi = (r_quant << 44) | (theta_quant << 20) | (self.seed.hi & 0xFFFFF);
        
        PoincareMatrix { seed: final_seed, rows: self.rows, cols: self.cols }
    }
}

// ============================================================================
// 6장: 대규모 행렬 연산: 푸앵카레 볼 기반 선형대수 최적화
// ============================================================================

use crate::types::Packed128;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::sync::mpsc;

/// 6.2 계층적 블록 분할 시스템
/// 
/// 4단계 분할 구조로 대규모 행렬을 효율적으로 처리합니다.
/// L1: 4096×4096 → L2: 1024×1024 → L3: 256×256 → L4: 64×64
#[derive(Debug, Clone)]
pub struct HierarchicalBlockMatrix {
    /// 전체 행렬 크기
    pub total_rows: usize,
    pub total_cols: usize,
    /// 4단계 블록 구조
    pub l1_blocks: Vec<Vec<L1Block>>,
    /// 품질 등급별 설정
    pub quality_level: QualityLevel,
    /// 오차 제어 시스템
    pub error_controller: ErrorController,
}

/// 6.2.1 품질 등급 정의
#[derive(Debug, Clone, Copy)]
pub enum QualityLevel {
    Ultra,   // PSNR > 50 dB, 32×32 블록
    High,    // PSNR > 40 dB, 64×64 블록
    Medium,  // PSNR > 30 dB, 128×128 블록
    Low,     // PSNR > 20 dB, 256×256 블록
}

impl QualityLevel {
    /// 품질 등급에 따른 최적 블록 크기 반환
    pub fn optimal_block_size(&self) -> usize {
        match self {
            QualityLevel::Ultra => 32,
            QualityLevel::High => 64,
            QualityLevel::Medium => 128,
            QualityLevel::Low => 256,
        }
    }
    
    /// 목표 PSNR 값
    pub fn target_psnr(&self) -> f32 {
        match self {
            QualityLevel::Ultra => 50.0,
            QualityLevel::High => 40.0,
            QualityLevel::Medium => 30.0,
            QualityLevel::Low => 20.0,
        }
    }
    
    /// 압축률
    pub fn compression_ratio(&self) -> f32 {
        match self {
            QualityLevel::Ultra => 200.0,
            QualityLevel::High => 500.0,
            QualityLevel::Medium => 1000.0,
            QualityLevel::Low => 2000.0,
        }
    }
}

/// 6.2.1 L1 블록 (최상위 레벨)
#[derive(Debug, Clone)]
pub struct L1Block {
    /// 블록 위치
    pub row_start: usize,
    pub col_start: usize,
    pub rows: usize,
    pub cols: usize,
    /// L2 하위 블록들
    pub l2_blocks: Vec<Vec<L2Block>>,
    /// 전체 블록을 표현하는 단일 파라미터
    pub global_params: Packed128,
}

/// L2 블록 (1024×1024)
#[derive(Debug, Clone)]
pub struct L2Block {
    pub row_start: usize,
    pub col_start: usize,
    pub rows: usize,
    pub cols: usize,
    pub l3_blocks: Vec<Vec<L3Block>>,
    pub macro_params: Packed128,
}

/// L3 블록 (256×256)
#[derive(Debug, Clone)]
pub struct L3Block {
    pub row_start: usize,
    pub col_start: usize,
    pub rows: usize,
    pub cols: usize,
    pub l4_blocks: Vec<Vec<L4Block>>,
    pub mid_params: Packed128,
}

/// L4 블록 (64×64, 최소 단위)
#[derive(Debug, Clone)]
pub struct L4Block {
    pub row_start: usize,
    pub col_start: usize,
    pub rows: usize,
    pub cols: usize,
    pub detail_params: Packed128,
}

/// 6.2.4 오차 제어 시스템
#[derive(Debug, Clone)]
pub struct ErrorController {
    /// 전체 오차 임계값
    pub global_error_threshold: f32,
    /// 블록별 오차 맵
    pub block_errors: HashMap<(usize, usize), f32>,
    /// 오차 가중치
    pub error_weights: Vec<f32>,
}

impl ErrorController {
    /// 새로운 오차 제어기 생성
    pub fn new(error_threshold: f32) -> Self {
        Self {
            global_error_threshold: error_threshold,
            block_errors: HashMap::new(),
            error_weights: Vec::new(),
        }
    }
    
    /// 6.2.4 전체 오차 계산
    /// E_total = √(Σ w_i² E_i²)
    pub fn compute_total_error(&self) -> f32 {
        let mut weighted_error_sum = 0.0;
        let mut total_weight = 0.0;
        
        for (block_id, &error) in &self.block_errors {
            let weight = self.get_block_weight(block_id);
            weighted_error_sum += weight * weight * error * error;
            total_weight += weight * weight;
        }
        
        if total_weight > 0.0 {
            (weighted_error_sum / total_weight).sqrt()
        } else {
            0.0
        }
    }
    
    /// 블록 가중치 계산 (블록 크기에 비례)
    fn get_block_weight(&self, block_id: &(usize, usize)) -> f32 {
        // 간단화: 모든 블록의 가중치를 1.0으로 설정
        1.0
    }
    
    /// 블록 오차 업데이트
    pub fn update_block_error(&mut self, block_id: (usize, usize), error: f32) {
        self.block_errors.insert(block_id, error);
    }
    
    /// 블록 분할 필요성 판단
    pub fn should_subdivide(&self, block_id: (usize, usize), current_level: usize) -> bool {
        if current_level >= 4 {
            return false; // 최대 깊이 도달
        }
        
        if let Some(&error) = self.block_errors.get(&block_id) {
            error > self.global_error_threshold
        } else {
            true // 오차 정보가 없으면 분할
        }
    }
}

impl HierarchicalBlockMatrix {
    /// 새로운 계층적 블록 행렬 생성
    pub fn new(rows: usize, cols: usize, quality: QualityLevel) -> Self {
        let error_threshold = match quality {
            QualityLevel::Ultra => 1e-4,
            QualityLevel::High => 1e-3,
            QualityLevel::Medium => 1e-2,
            QualityLevel::Low => 1e-1,
        };
        
        Self {
            total_rows: rows,
            total_cols: cols,
            l1_blocks: Vec::new(),
            quality_level: quality,
            error_controller: ErrorController::new(error_threshold),
        }
    }
    
    /// 6.2.2 적응적 블록 분할 수행
    pub fn adaptive_partition(&mut self, source_matrix: &[f32]) {
        let l1_block_size = 4096;
        
        // L1 블록들 생성
        for i in (0..self.total_rows).step_by(l1_block_size) {
            let mut l1_row = Vec::new();
            
            for j in (0..self.total_cols).step_by(l1_block_size) {
                let rows = (l1_block_size).min(self.total_rows - i);
                let cols = (l1_block_size).min(self.total_cols - j);
                
                let l1_block = self.create_l1_block(source_matrix, i, j, rows, cols);
                l1_row.push(l1_block);
            }
            
            self.l1_blocks.push(l1_row);
        }
    }
    
    /// L1 블록 생성
    fn create_l1_block(&mut self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                       rows: usize, cols: usize) -> L1Block {
        // 전역 파라미터 추정
        let global_params = self.estimate_global_parameters(source_matrix, row_start, col_start, rows, cols);
        
        // 압축 오차 계산
        let compression_error = self.compute_compression_error(source_matrix, &global_params, 
                                                             row_start, col_start, rows, cols);
        
        // 오차 기록
        self.error_controller.update_block_error((row_start, col_start), compression_error);
        
        // L2 블록들 생성 (재귀적 분할)
        let l2_blocks = if self.error_controller.should_subdivide((row_start, col_start), 1) {
            self.create_l2_blocks(source_matrix, row_start, col_start, rows, cols)
        } else {
            Vec::new() // 분할 불필요
        };
        
        L1Block {
            row_start,
            col_start,
            rows,
            cols,
            l2_blocks,
            global_params,
        }
    }
    
    /// L2 블록들 생성
    fn create_l2_blocks(&mut self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                        rows: usize, cols: usize) -> Vec<Vec<L2Block>> {
        let l2_block_size = 1024;
        let mut l2_blocks = Vec::new();
        
        for i in (0..rows).step_by(l2_block_size) {
            let mut l2_row = Vec::new();
            
            for j in (0..cols).step_by(l2_block_size) {
                let sub_rows = l2_block_size.min(rows - i);
                let sub_cols = l2_block_size.min(cols - j);
                
                let l2_block = self.create_l2_block(source_matrix, 
                                                  row_start + i, col_start + j, 
                                                  sub_rows, sub_cols);
                l2_row.push(l2_block);
            }
            
            l2_blocks.push(l2_row);
        }
        
        l2_blocks
    }
    
    /// L2 블록 생성
    fn create_l2_block(&mut self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                       rows: usize, cols: usize) -> L2Block {
        let macro_params = self.estimate_local_parameters(source_matrix, row_start, col_start, rows, cols);
        let compression_error = self.compute_compression_error(source_matrix, &macro_params, 
                                                             row_start, col_start, rows, cols);
        
        self.error_controller.update_block_error((row_start, col_start), compression_error);
        
        let l3_blocks = if self.error_controller.should_subdivide((row_start, col_start), 2) {
            self.create_l3_blocks(source_matrix, row_start, col_start, rows, cols)
        } else {
            Vec::new()
        };
        
        L2Block {
            row_start,
            col_start,
            rows,
            cols,
            l3_blocks,
            macro_params,
        }
    }
    
    /// L3 블록들 생성
    fn create_l3_blocks(&mut self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                        rows: usize, cols: usize) -> Vec<Vec<L3Block>> {
        let l3_block_size = 256;
        let mut l3_blocks = Vec::new();
        
        for i in (0..rows).step_by(l3_block_size) {
            let mut l3_row = Vec::new();
            
            for j in (0..cols).step_by(l3_block_size) {
                let sub_rows = l3_block_size.min(rows - i);
                let sub_cols = l3_block_size.min(cols - j);
                
                let l3_block = self.create_l3_block(source_matrix, 
                                                  row_start + i, col_start + j, 
                                                  sub_rows, sub_cols);
                l3_row.push(l3_block);
            }
            
            l3_blocks.push(l3_row);
        }
        
        l3_blocks
    }
    
    /// L3 블록 생성
    fn create_l3_block(&mut self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                       rows: usize, cols: usize) -> L3Block {
        let mid_params = self.estimate_local_parameters(source_matrix, row_start, col_start, rows, cols);
        let compression_error = self.compute_compression_error(source_matrix, &mid_params, 
                                                             row_start, col_start, rows, cols);
        
        self.error_controller.update_block_error((row_start, col_start), compression_error);
        
        let l4_blocks = if self.error_controller.should_subdivide((row_start, col_start), 3) {
            self.create_l4_blocks(source_matrix, row_start, col_start, rows, cols)
        } else {
            Vec::new()
        };
        
        L3Block {
            row_start,
            col_start,
            rows,
            cols,
            l4_blocks,
            mid_params,
        }
    }
    
    /// L4 블록들 생성 (최소 단위)
    fn create_l4_blocks(&mut self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                        rows: usize, cols: usize) -> Vec<Vec<L4Block>> {
        let l4_block_size = self.quality_level.optimal_block_size();
        let mut l4_blocks = Vec::new();
        
        for i in (0..rows).step_by(l4_block_size) {
            let mut l4_row = Vec::new();
            
            for j in (0..cols).step_by(l4_block_size) {
                let sub_rows = l4_block_size.min(rows - i);
                let sub_cols = l4_block_size.min(cols - j);
                
                let detail_params = self.estimate_local_parameters(source_matrix, 
                                                                 row_start + i, col_start + j, 
                                                                 sub_rows, sub_cols);
                
                let l4_block = L4Block {
                    row_start: row_start + i,
                    col_start: col_start + j,
                    rows: sub_rows,
                    cols: sub_cols,
                    detail_params,
                };
                
                l4_row.push(l4_block);
            }
            
            l4_blocks.push(l4_row);
        }
        
        l4_blocks
    }
    
    /// 전역 파라미터 추정 (SVD 기반)
    fn estimate_global_parameters(&self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                                rows: usize, cols: usize) -> Packed128 {
        // 간단한 평균값 기반 추정 (실제로는 SVD 사용)
        let mut sum = 0.0;
        let mut count = 0;
        
        for i in 0..rows {
            for j in 0..cols {
                if row_start + i < self.total_rows && col_start + j < self.total_cols {
                    let idx = (row_start + i) * self.total_cols + (col_start + j);
                    if idx < source_matrix.len() {
                        sum += source_matrix[idx];
                        count += 1;
                    }
                }
            }
        }
        
        let average = if count > 0 { sum / count as f32 } else { 0.0 };
        
        // 평균값을 기반으로 파라미터 생성
        let r_fp32 = (average.abs().clamp(0.1, 1.0));
        let theta_fp32 = if average >= 0.0 { 0.0 } else { std::f32::consts::PI };
        
        Packed128 {
            hi: 0x12345678,  // 기본 상태 비트
            lo: ((r_fp32.to_bits() as u64) << 32) | theta_fp32.to_bits() as u64,
        }
    }
    
    /// 지역 파라미터 추정
    fn estimate_local_parameters(&self, source_matrix: &[f32], row_start: usize, col_start: usize, 
                               rows: usize, cols: usize) -> Packed128 {
        // 지역적 특성을 고려한 파라미터 추정
        let mut sum = 0.0;
        let mut variance = 0.0;
        let mut count = 0;
        
        // 1차 통계량 계산
        for i in 0..rows {
            for j in 0..cols {
                if row_start + i < self.total_rows && col_start + j < self.total_cols {
                    let idx = (row_start + i) * self.total_cols + (col_start + j);
                    if idx < source_matrix.len() {
                        sum += source_matrix[idx];
                        count += 1;
                    }
                }
            }
        }
        
        let mean = if count > 0 { sum / count as f32 } else { 0.0 };
        
        // 2차 통계량 계산
        for i in 0..rows {
            for j in 0..cols {
                if row_start + i < self.total_rows && col_start + j < self.total_cols {
                    let idx = (row_start + i) * self.total_cols + (col_start + j);
                    if idx < source_matrix.len() {
                        let diff = source_matrix[idx] - mean;
                        variance += diff * diff;
                    }
                }
            }
        }
        
        let std_dev = if count > 1 { (variance / (count - 1) as f32).sqrt() } else { 0.1 };
        
        // 통계량을 기반으로 파라미터 생성
        let r_fp32 = (mean.abs() + std_dev).clamp(0.1, 0.99);
        let theta_fp32 = (mean.atan2(std_dev)).rem_euclid(2.0 * std::f32::consts::PI);
        
        // 상태 비트는 블록 위치에 따라 설정
        let state_hash = ((row_start * 31 + col_start) % 256) as u64;
        
        Packed128 {
            hi: state_hash << 8,
            lo: ((r_fp32.to_bits() as u64) << 32) | theta_fp32.to_bits() as u64,
        }
    }
    
    /// 압축 오차 계산
    fn compute_compression_error(&self, source_matrix: &[f32], params: &Packed128, 
                               row_start: usize, col_start: usize, 
                               rows: usize, cols: usize) -> f32 {
        let mut error_sum = 0.0;
        let mut count = 0;
        
        for i in 0..rows {
            for j in 0..cols {
                if row_start + i < self.total_rows && col_start + j < self.total_cols {
                    let idx = (row_start + i) * self.total_cols + (col_start + j);
                    if idx < source_matrix.len() {
                        let original = source_matrix[idx];
                        let reconstructed = params.fused_forward(i, j, rows, cols);
                        let error = (original - reconstructed).abs();
                        error_sum += error * error;
                        count += 1;
                    }
                }
            }
        }
        
        if count > 0 {
            (error_sum / count as f32).sqrt() // RMSE
        } else {
            0.0
        }
    }
    
    /// 6.4 블록별 병렬 GEMV 연산
    pub fn parallel_gemv(&self, input: &[f32], output: &mut [f32], num_threads: usize) {
        let (sender, receiver) = mpsc::channel();
        let input_arc = Arc::new(input.to_vec());
        let output_arc = Arc::new(Mutex::new(vec![0.0; output.len()]));
        
        let mut handles = Vec::new();
        
        // 스레드 풀로 L1 블록들을 병렬 처리
        for l1_row in &self.l1_blocks {
            for l1_block in l1_row {
                let sender_clone = sender.clone();
                let input_clone = Arc::clone(&input_arc);
                let output_clone = Arc::clone(&output_arc);
                let block_clone = l1_block.clone();
                
                let handle = thread::spawn(move || {
                    let result = Self::process_l1_block(&block_clone, &input_clone);
                    sender_clone.send((block_clone.row_start, result)).unwrap();
                });
                
                handles.push(handle);
            }
        }
        
        // 결과 수집
        drop(sender);
        for _ in &self.l1_blocks {
            for _ in &self.l1_blocks[0] {
                if let Ok((row_start, block_result)) = receiver.recv() {
                    let mut output_lock = output_arc.lock().unwrap();
                    for (i, value) in block_result.iter().enumerate() {
                        if row_start + i < output_lock.len() {
                            output_lock[row_start + i] += value;
                        }
                    }
                }
            }
        }
        
        // 결과 복사
        let final_output = output_arc.lock().unwrap();
        output.copy_from_slice(&final_output);
        
        // 스레드 정리
        for handle in handles {
            handle.join().unwrap();
        }
    }
    
    /// L1 블록 처리
    fn process_l1_block(block: &L1Block, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; block.rows];
        
        // L2 블록들이 있으면 계층적 처리, 없으면 직접 처리
        if !block.l2_blocks.is_empty() {
            for l2_row in &block.l2_blocks {
                for l2_block in l2_row {
                    let l2_result = Self::process_l2_block(l2_block, input);
                    for (i, value) in l2_result.iter().enumerate() {
                        let global_i = l2_block.row_start - block.row_start + i;
                        if global_i < result.len() {
                            result[global_i] += value;
                        }
                    }
                }
            }
        } else {
            // 직접 전역 파라미터로 처리
            for i in 0..block.rows {
                for j in 0..block.cols {
                    if block.col_start + j < input.len() {
                        let weight = block.global_params.fused_forward(i, j, block.rows, block.cols);
                        result[i] += weight * input[block.col_start + j];
                    }
                }
            }
        }
        
        result
    }
    
    /// L2 블록 처리
    fn process_l2_block(block: &L2Block, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; block.rows];
        
        if !block.l3_blocks.is_empty() {
            for l3_row in &block.l3_blocks {
                for l3_block in l3_row {
                    let l3_result = Self::process_l3_block(l3_block, input);
                    for (i, value) in l3_result.iter().enumerate() {
                        let global_i = l3_block.row_start - block.row_start + i;
                        if global_i < result.len() {
                            result[global_i] += value;
                        }
                    }
                }
            }
        } else {
            for i in 0..block.rows {
                for j in 0..block.cols {
                    if block.col_start + j < input.len() {
                        let weight = block.macro_params.fused_forward(i, j, block.rows, block.cols);
                        result[i] += weight * input[block.col_start + j];
                    }
                }
            }
        }
        
        result
    }
    
    /// L3 블록 처리
    fn process_l3_block(block: &L3Block, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; block.rows];
        
        if !block.l4_blocks.is_empty() {
            for l4_row in &block.l4_blocks {
                for l4_block in l4_row {
                    for i in 0..l4_block.rows {
                        for j in 0..l4_block.cols {
                            if l4_block.col_start + j < input.len() {
                                let weight = l4_block.detail_params.fused_forward(i, j, l4_block.rows, l4_block.cols);
                                let global_i = l4_block.row_start - block.row_start + i;
                                if global_i < result.len() {
                                    result[global_i] += weight * input[l4_block.col_start + j];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for i in 0..block.rows {
                for j in 0..block.cols {
                    if block.col_start + j < input.len() {
                        let weight = block.mid_params.fused_forward(i, j, block.rows, block.cols);
                        result[i] += weight * input[block.col_start + j];
                    }
                }
            }
        }
        
        result
    }
    
    /// 메모리 사용량 계산
    pub fn memory_usage(&self) -> (usize, f32) {
        let mut total_blocks = 0;
        let mut total_bytes = 0;
        
        for l1_row in &self.l1_blocks {
            for l1_block in l1_row {
                total_blocks += 1;
                total_bytes += 16; // Packed128 크기
                
                for l2_row in &l1_block.l2_blocks {
                    for l2_block in l2_row {
                        total_blocks += 1;
                        total_bytes += 16;
                        
                        for l3_row in &l2_block.l3_blocks {
                            for l3_block in l3_row {
                                total_blocks += 1;
                                total_bytes += 16;
                                
                                for l4_row in &l3_block.l4_blocks {
                                    for l4_block in l4_row {
                                        total_blocks += 1;
                                        total_bytes += 16;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 원본 행렬 대비 압축률 계산
        let original_bytes = self.total_rows * self.total_cols * 4; // f32 크기
        let compression_ratio = original_bytes as f32 / total_bytes as f32;
        
        (total_bytes, compression_ratio)
    }
    
    /// 품질 통계 계산
    pub fn quality_statistics(&self) -> QualityStats {
        let total_error = self.error_controller.compute_total_error();
        let psnr = if total_error > 0.0 {
            20.0 * (-total_error).log10()
        } else {
            f32::INFINITY
        };
        
        let (memory_bytes, compression_ratio) = self.memory_usage();
        
        QualityStats {
            total_error,
            psnr,
            compression_ratio,
            memory_usage_bytes: memory_bytes,
            total_blocks: self.count_total_blocks(),
        }
    }
    
    /// 전체 블록 개수 계산
    fn count_total_blocks(&self) -> usize {
        let mut count = 0;
        
        for l1_row in &self.l1_blocks {
            for l1_block in l1_row {
                count += 1;
                
                for l2_row in &l1_block.l2_blocks {
                    for l2_block in l2_row {
                        count += 1;
                        
                        for l3_row in &l2_block.l3_blocks {
                            for l3_block in l3_row {
                                count += 1;
                                count += l3_block.l4_blocks.iter().map(|row| row.len()).sum::<usize>();
                            }
                        }
                    }
                }
            }
        }
        
        count
    }
}

/// 품질 통계 구조체
#[derive(Debug, Clone)]
pub struct QualityStats {
    pub total_error: f32,
    pub psnr: f32,
    pub compression_ratio: f32,
    pub memory_usage_bytes: usize,
    pub total_blocks: usize,
}

impl QualityStats {
    /// 품질 보고서 출력
    pub fn print_report(&self) {
        println!("=== 품질 통계 보고서 ===");
        println!("총 오차: {:.6}", self.total_error);
        println!("PSNR: {:.2} dB", self.psnr);
        println!("압축률: {:.1}:1", self.compression_ratio);
        println!("메모리 사용량: {:.2} KB", self.memory_usage_bytes as f32 / 1024.0);
        println!("총 블록 수: {}", self.total_blocks);
        
        // 압축 효율성 등급
        let efficiency_grade = if self.compression_ratio > 1000.0 {
            "A+"
        } else if self.compression_ratio > 500.0 {
            "A"
        } else if self.compression_ratio > 200.0 {
            "B"
        } else {
            "C"
        };
        
        println!("압축 효율성 등급: {}", efficiency_grade);
    }
}

 