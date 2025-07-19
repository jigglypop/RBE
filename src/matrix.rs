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
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;

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
#[derive(Debug, Clone, Copy, PartialEq)]
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

impl L1Block {
    pub fn new() -> Self {
        // 4×4 L2 블록들로 구성 (4096 / 1024 = 4)
        let mut l2_blocks = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut row = Vec::with_capacity(4);
            for _ in 0..4 {
                row.push(L2Block::new());
            }
            l2_blocks.push(row);
        }
        
        Self {
            row_start: 0,
            col_start: 0,
            rows: 4096,
            cols: 4096,
            l2_blocks,
            global_params: Packed128 { hi: 0, lo: 0 },
        }
    }
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

impl L2Block {
    pub fn new() -> Self {
        // 4×4 L3 블록들로 구성 (1024 / 256 = 4)
        let mut l3_blocks = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut row = Vec::with_capacity(4);
            for _ in 0..4 {
                row.push(L3Block::new());
            }
            l3_blocks.push(row);
        }
        
        Self {
            row_start: 0,
            col_start: 0,
            rows: 1024,
            cols: 1024,
            l3_blocks,
            macro_params: Packed128 { hi: 0, lo: 0 },
        }
    }
}

/// L3 블록 (256×256)
#[derive(Debug, Clone)]
pub struct L3Block {
    pub row_start: usize,
    pub col_start: usize,
    pub rows: usize,
    pub cols: usize,
    pub l4_blocks: Vec<Vec<Packed128>>, // L4Block → Packed128으로 변경
    pub mid_params: Packed128,
}

impl L3Block {
    pub fn new() -> Self {
        // 4×4 L4 블록들로 구성 (256 / 64 = 4)
        let mut l4_blocks = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut row = Vec::with_capacity(4);
            for _ in 0..4 {
                row.push(Packed128 { hi: 0, lo: 0 });
            }
            l4_blocks.push(row);
        }
        
        Self {
            row_start: 0,
            col_start: 0,
            rows: 256,
            cols: 256,
            l4_blocks,
            mid_params: Packed128 { hi: 0, lo: 0 },
        }
    }
}

/// L4 블록은 이제 Packed128 타입으로 직접 사용
pub type L4Block = Packed128;

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
        let l1_blocks_rows = (rows + 4095) / 4096;
        let l1_blocks_cols = (cols + 4095) / 4096;
        
        let mut l1_blocks = Vec::with_capacity(l1_blocks_rows);
        
        for _ in 0..l1_blocks_rows {
            let mut row = Vec::with_capacity(l1_blocks_cols);
            for _ in 0..l1_blocks_cols {
                row.push(L1Block::new());
            }
            l1_blocks.push(row);
        }
        
        Self {
            total_rows: rows,
            total_cols: cols,
            l1_blocks,
            quality_level: quality,
            error_controller: ErrorController::new(0.0), // 0.0으로 변경
        }
    }
    
    /// 압축된 크기 계산 (바이트)
    pub fn compressed_size(&self) -> usize {
        let mut total_size = 0;
        
        for l1_row in &self.l1_blocks {
            for l1_block in l1_row {
                // L1 블록 헤더
                total_size += std::mem::size_of::<L1Block>();
                
                // L2 블록들
                for l2_row in &l1_block.l2_blocks {
                    for l2_block in l2_row {
                        total_size += std::mem::size_of::<L2Block>();
                        
                        // L3 블록들
                        for l3_row in &l2_block.l3_blocks {
                            for l3_block in l3_row {
                                total_size += std::mem::size_of::<L3Block>();
                                
                                // L4 블록들 (Packed128)
                                total_size += l3_block.l4_blocks.len() * l3_block.l4_blocks[0].len() * 16; // Packed128 크기
                            }
                        }
                    }
                }
            }
        }
        
        total_size
    }
    
    /// Dense 행렬에서 RBE 인코딩 (진행률 바 지원)
    pub fn encode_from_dense(&mut self, matrix: &[Vec<f32>], epoch_progress: Option<&indicatif::ProgressBar>, main_progress: Option<&indicatif::ProgressBar>) -> Result<(), String> {
        if matrix.len() != self.total_rows {
            return Err(format!("행 수 불일치: {} vs {}", matrix.len(), self.total_rows));
        }
        
        if !matrix.is_empty() && matrix[0].len() != self.total_cols {
            return Err(format!("열 수 불일치: {} vs {}", matrix[0].len(), self.total_cols));
        }
        
        // L1 블록 크기를 더 작게 조정 (원래 4096 → 512)
        let l1_block_size = 512;
        
        // 블록별로 인코딩
        for (l1_i, l1_row) in self.l1_blocks.iter_mut().enumerate() {
            for (l1_j, l1_block) in l1_row.iter_mut().enumerate() {
                let l1_start_row = l1_i * l1_block_size;
                let l1_start_col = l1_j * l1_block_size;
                
                // L1 블록 영역의 데이터 추출 및 인코딩
                Self::encode_l1_block(l1_block, matrix, l1_start_row, l1_start_col, epoch_progress, main_progress)?;
            }
        }
        
        Ok(())
    }
    
    /// RBE에서 Dense 행렬로 디코딩
    pub fn decode_to_dense(&self) -> Result<Vec<Vec<f32>>, String> {
        let mut result = vec![vec![0.0; self.total_cols]; self.total_rows];
        
        // L1 블록 크기를 더 작게 조정 (원래 4096 → 512)
        let l1_block_size = 512;
        
        for (l1_i, l1_row) in self.l1_blocks.iter().enumerate() {
            for (l1_j, l1_block) in l1_row.iter().enumerate() {
                let l1_start_row = l1_i * l1_block_size;
                let l1_start_col = l1_j * l1_block_size;
                
                // L1 블록 디코딩
                Self::decode_l1_block(l1_block, &mut result, l1_start_row, l1_start_col)?;
            }
        }
        
        Ok(result)
    }
    
    /// L1 블록 인코딩
    fn encode_l1_block(
        l1_block: &mut L1Block,
        matrix: &[Vec<f32>],
        start_row: usize,
        start_col: usize,
        epoch_progress: Option<&indicatif::ProgressBar>,
        main_progress: Option<&indicatif::ProgressBar>
    ) -> Result<(), String> {
        // L2 블록 크기를 더 작게 조정 (원래 1024 → 128)
        let l2_block_size = 128;
        
        for (l2_i, l2_row) in l1_block.l2_blocks.iter_mut().enumerate() {
            for (l2_j, l2_block) in l2_row.iter_mut().enumerate() {
                let l2_start_row = start_row + l2_i * l2_block_size;
                let l2_start_col = start_col + l2_j * l2_block_size;
                
                // L2 블록 인코딩
                Self::encode_l2_block(l2_block, matrix, l2_start_row, l2_start_col, epoch_progress, main_progress)?;
            }
        }
        Ok(())
    }
    
    /// L2 블록 인코딩
    fn encode_l2_block(
        l2_block: &mut L2Block,
        matrix: &[Vec<f32>],
        start_row: usize,
        start_col: usize,
        epoch_progress: Option<&indicatif::ProgressBar>,
        main_progress: Option<&indicatif::ProgressBar>
    ) -> Result<(), String> {
        // L3 블록 크기를 더 작게 조정 (원래 256 → 64)
        let l3_block_size = 64;
        
        for (l3_i, l3_row) in l2_block.l3_blocks.iter_mut().enumerate() {
            for (l3_j, l3_block) in l3_row.iter_mut().enumerate() {
                let l3_start_row = start_row + l3_i * l3_block_size;
                let l3_start_col = start_col + l3_j * l3_block_size;
                
                // L3 블록 인코딩
                Self::encode_l3_block(l3_block, matrix, l3_start_row, l3_start_col, epoch_progress, main_progress)?;
            }
        }
        Ok(())
    }
    
    /// L3 블록 인코딩 (실시간 진행률 바 지원)
    fn encode_l3_block(
        l3_block: &mut L3Block,
        matrix: &[Vec<f32>],
        start_row: usize,
        start_col: usize,
        epoch_progress: Option<&indicatif::ProgressBar>,
        main_progress: Option<&indicatif::ProgressBar>
    ) -> Result<(), String> {
        // 실제 블록 크기는 32x32 (테스트에서 확인된 크기)
        let actual_block_size = 32;
        
        for (l4_i, l4_row) in l3_block.l4_blocks.iter_mut().enumerate() {
            for (l4_j, l4_block) in l4_row.iter_mut().enumerate() {
                let l4_start_row = start_row + l4_i * actual_block_size;
                let l4_start_col = start_col + l4_j * actual_block_size;
                
                // 32×32 블록 데이터 추출
                let mut block_data = Vec::new();
                for i in 0..actual_block_size {
                    for j in 0..actual_block_size {
                        let row = l4_start_row + i;
                        let col = l4_start_col + j;
                        
                        if row < matrix.len() && col < matrix[0].len() {
                            block_data.push(matrix[row][col]);
                        } else {
                            block_data.push(0.0); // 패딩
                        }
                    }
                }
                
                // 실제 RBE 학습으로 최적 파라미터 찾기 (효율적 100 에포크)
                let mut best_seed = Packed128::random(&mut rand::thread_rng());
                
                // 초기 파라미터 설정
                let initial_r = 0.5f32;
                let initial_theta = 0.0f32;
                best_seed.lo = ((initial_r.to_bits() as u64) << 32) | initial_theta.to_bits() as u64;
                
                // 🚀 5000 에포크 RBE 학습으로 RMSE 최소화 (안정적 고품질 변환)
                let mut learning_rate = 0.005; // RMSE 안정성 우선 학습률 (사용자 요청)
                let epochs = 5000; // 사용자 요청: 5000 에포크로 균형잡힌 품질
                let mut best_rmse = f32::INFINITY;
                let mut no_improvement_count = 0;
                
                // 🎯 진행률 바 초기화 (있는 경우)
                if let Some(progress) = epoch_progress {
                    progress.reset();
                    progress.set_length(epochs as u64);
                    progress.set_message("RBE 학습 시작...");
                }
                
                for epoch in 1..=epochs {
                    // 현재 예측 생성
                    let mut predicted = vec![0.0; actual_block_size * actual_block_size];
                    for i in 0..actual_block_size {
                        for j in 0..actual_block_size {
                            let idx = i * actual_block_size + j;
                            predicted[idx] = best_seed.fused_forward(i, j, actual_block_size, actual_block_size);
                        }
                    }
                    
                    // fused_backward로 파라미터 최적화
                    let (mse, rmse) = crate::math::fused_backward(
                        &block_data,
                        &predicted,
                        &mut best_seed,
                        actual_block_size,
                        actual_block_size,
                        learning_rate
                    );
                    
                    // 적응적 학습률 조정
                    if rmse < best_rmse {
                        best_rmse = rmse;
                        no_improvement_count = 0;
                        learning_rate *= 1.02; // 성능 향상시 학습률 약간 증가
                    } else {
                        no_improvement_count += 1;
                        if no_improvement_count > 10 { // 더 빠른 적응
                            learning_rate *= 0.9; // 개선 없으면 학습률 감소
                            no_improvement_count = 0;
                        }
                    }
                    
                    // 🔥 실시간 진행률 바 업데이트 (매 에포크마다)
                    if let Some(progress) = epoch_progress {
                        progress.set_position(epoch as u64);
                        
                        // 품질 등급 계산 (RMSE 기반)
                        let quality_grade = if rmse < 0.01 { "🟢 우수" } 
                                           else if rmse < 0.05 { "🟡 양호" } 
                                           else if rmse < 0.1 { "🟠 보통" } 
                                           else { "🔴 개선중" };
                        
                        progress.set_message(format!(
                            "{:.4} | LR: {:.4} | Best: {:.4} | 개선: {}회", 
                            rmse, learning_rate, best_rmse, epochs as i32 - no_improvement_count
                        ));
                        
                        // 10 에포크마다 상세 정보 업데이트
                        if epoch % 10 == 0 {
                            progress.set_message(format!(
                                "{} | RMSE: {:.6} | LR: {:.4} | 최고: {:.6}", 
                                quality_grade, rmse, learning_rate, best_rmse
                            ));
                        }
                    }
                    
                    // 🎯 조기 종료 조건들 (품질 우선 + 효율적 수렴)
                    if rmse < 0.005 {  // 매우 좋은 품질 달성
                        if let Some(progress) = epoch_progress {
                            progress.finish_with_message("🎉 목표 품질 달성! 조기 종료".to_string());
                        }
                        break;
                    }
                    if no_improvement_count > 50 {  // 50 에포크 동안 개선 없음 (더 인내심)
                        if let Some(progress) = epoch_progress {
                            progress.finish_with_message(format!("⏹️ 수렴 완료 | 최종 RMSE: {:.6}", best_rmse));
                        }
                        break;
                    }
                }
                
                // 🏁 정상 완료 시 진행률 바 마무리
                if let Some(progress) = epoch_progress {
                    if !progress.is_finished() {
                        progress.finish_with_message(format!("✅ 학습 완료 | 최종 RMSE: {:.6}", best_rmse));
                    }
                }
                
                // 학습 완료 시 최종 상태
                // if !progress.is_finished() {
                //     progress.finish_with_message(format!("✅ 학습 완료: Best RMSE: {:.6}", best_rmse));
                // }
                
                // 🔥 상위 진행률 바 업데이트 (각 L4 블록 완료시마다)
                if let Some(main_prog) = main_progress {
                    main_prog.inc(1); // L4 블록 하나씩 완료 증가
                }
                
                *l4_block = best_seed;
            }
        }
        Ok(())
    }
    
    /// L1 블록 디코딩
    fn decode_l1_block(
        l1_block: &L1Block,
        result: &mut [Vec<f32>],
        start_row: usize,
        start_col: usize
    ) -> Result<(), String> {
        // L2 블록 크기를 더 작게 조정 (원래 1024 → 128)
        let l2_block_size = 128;
        
        for (l2_i, l2_row) in l1_block.l2_blocks.iter().enumerate() {
            for (l2_j, l2_block) in l2_row.iter().enumerate() {
                let l2_start_row = start_row + l2_i * l2_block_size;
                let l2_start_col = start_col + l2_j * l2_block_size;
                
                Self::decode_l2_block(l2_block, result, l2_start_row, l2_start_col)?;
            }
        }
        Ok(())
    }
    
    /// L2 블록 디코딩
    fn decode_l2_block(
        l2_block: &L2Block,
        result: &mut [Vec<f32>],
        start_row: usize,
        start_col: usize
    ) -> Result<(), String> {
        // L3 블록 크기를 더 작게 조정 (원래 256 → 64)
        let l3_block_size = 64;
        
        for (l3_i, l3_row) in l2_block.l3_blocks.iter().enumerate() {
            for (l3_j, l3_block) in l3_row.iter().enumerate() {
                let l3_start_row = start_row + l3_i * l3_block_size;
                let l3_start_col = start_col + l3_j * l3_block_size;
                
                Self::decode_l3_block(l3_block, result, l3_start_row, l3_start_col)?;
            }
        }
        Ok(())
    }
    
    /// L3 블록 디코딩
    fn decode_l3_block(
        l3_block: &L3Block,
        result: &mut [Vec<f32>],
        start_row: usize,
        start_col: usize
    ) -> Result<(), String> {
        // 실제 블록 크기는 32x32 (테스트에서 확인된 크기)
        let actual_block_size = 32;
        
        for (l4_i, l4_row) in l3_block.l4_blocks.iter().enumerate() {
            for (l4_j, l4_block) in l4_row.iter().enumerate() {
                let l4_start_row = start_row + l4_i * actual_block_size;
                let l4_start_col = start_col + l4_j * actual_block_size;
                
                // Packed128에서 32×32 블록 복원
                for i in 0..actual_block_size {
                    for j in 0..actual_block_size {
                        let row = l4_start_row + i;
                        let col = l4_start_col + j;
                        
                        if row < result.len() && col < result[0].len() {
                            // fused_forward로 값 생성
                            result[row][col] = l4_block.fused_forward(i, j, actual_block_size, actual_block_size);
                        }
                    }
                }
            }
        }
        Ok(())
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
                        rows: usize, cols: usize) -> Vec<Vec<Packed128>> {
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
                
                let l4_block = Packed128 {
                    hi: 0x12345678,  // 기본 상태 비트
                    lo: ((detail_params.lo >> 32) as u64) | (detail_params.hi << 32),
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
            for (l4_i, l4_row) in block.l4_blocks.iter().enumerate() {
                for (l4_j, l4_block) in l4_row.iter().enumerate() {
                    // L4 블록의 위치와 크기 계산 (64×64)
                    let l4_row_start = l4_i * 64;
                    let l4_col_start = l4_j * 64;
                    let l4_rows = 64.min(block.rows - l4_row_start);
                    let l4_cols = 64.min(input.len() - (block.col_start + l4_col_start));
                    
                    for i in 0..l4_rows {
                        for j in 0..l4_cols {
                            let input_idx = block.col_start + l4_col_start + j;
                            if input_idx < input.len() {
                                let weight = l4_block.fused_forward(i, j, 64, 64);
                                let result_idx = l4_row_start + i;
                                if result_idx < result.len() {
                                    result[result_idx] += weight * input[input_idx];
                                }
                            }
                        }
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
        
        // PSNR 계산: 20 * log10(MAX_VALUE / RMS_ERROR)
        // 여기서 MAX_VALUE = 1.0 (정규화된 값 기준)
        let psnr = if total_error > 1e-10 {
            20.0 * (1.0 / total_error).log10()
        } else {
            f32::INFINITY // 완벽한 복원 시
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

 