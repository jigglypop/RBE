use crate::types::*;
use crate::matrix::*;
use crate::encoder::*;
use crate::llm::llm_analyzer::*;
use std::collections::HashMap;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};

/// RBE 변환기
pub struct RBEConverter {
    /// 변환 설정
    pub config: ConversionConfig,
    
    /// 변환된 레이어들
    pub converted_layers: HashMap<usize, ConvertedLayer>,
    
    /// 변환 성능 통계
    pub conversion_stats: ConversionStatistics,
}

/// 변환 설정
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// 배치 변환 크기
    pub batch_size: usize,
    
    /// 병렬 스레드 수
    pub num_threads: usize,
    
    /// 품질 우선 모드
    pub quality_priority: bool,
    
    /// 메모리 제한 (bytes)
    pub memory_limit: usize,
    
    /// 변환 정밀도
    pub precision_mode: PrecisionMode,
}

#[derive(Debug, Clone)]
pub enum PrecisionMode {
    Fast,      // 빠른 변환, 낮은 정밀도
    Balanced,  // 균형잡힌 변환
    Precise,   // 정밀한 변환, 높은 품질
}

/// 변환된 레이어
#[derive(Debug, Clone)]
pub struct ConvertedLayer {
    /// 원본 레이어 ID
    pub original_layer_id: usize,
    
    /// 레이어 타입
    pub layer_type: LayerType,
    
    /// RBE 인코딩된 가중치
    pub rbe_weights: Vec<HierarchicalBlockMatrix>,
    
    /// 바이어스 (압축하지 않음)
    pub biases: Option<Vec<f32>>,
    
    /// 변환 메타데이터
    pub metadata: ConversionMetadata,
    
    /// 품질 검증 결과
    pub quality_metrics: QualityMetrics,
}

/// 변환 메타데이터
#[derive(Debug, Clone)]
pub struct ConversionMetadata {
    /// 원본 크기 (bytes)
    pub original_size: usize,
    
    /// 압축 후 크기 (bytes)
    pub compressed_size: usize,
    
    /// 실제 압축률
    pub actual_compression_ratio: f32,
    
    /// 변환 시간 (ms)
    pub conversion_time_ms: u128,
    
    /// 블록 구성 정보
    pub block_configuration: BlockConfig,
}

/// 블록 구성 정보
#[derive(Debug, Clone)]
pub struct BlockConfig {
    /// 블록 크기
    pub block_size: usize,
    
    /// 블록 개수
    pub num_blocks: usize,
    
    /// 품질 레벨
    pub quality_level: QualityLevel,
    
    /// 계층 구조
    pub hierarchy_levels: usize,
}

/// 품질 검증 메트릭
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// MSE (평균 제곱 오차)
    pub mse: f32,
    
    /// PSNR (피크 신호 대 잡음비)
    pub psnr: f32,
    
    /// 코사인 유사도
    pub cosine_similarity: f32,
    
    /// 프로베니우스 노름 비율
    pub frobenius_ratio: f32,
    
    /// 품질 점수 (0-100)
    pub quality_score: f32,
}

/// 변환 통계
#[derive(Debug, Clone)]
pub struct ConversionStatistics {
    /// 총 변환 시간
    pub total_time_ms: u128,
    
    /// 변환된 레이어 수
    pub converted_layers: usize,
    
    /// 총 원본 크기
    pub total_original_size: usize,
    
    /// 총 압축 크기
    pub total_compressed_size: usize,
    
    /// 평균 압축률
    pub average_compression_ratio: f32,
    
    /// 평균 품질 점수
    pub average_quality_score: f32,
}

impl RBEConverter {
    /// 새로운 RBE 변환기 생성
    pub fn new(config: ConversionConfig) -> Self {
        // Rayon 스레드 풀 설정
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build_global()
            .unwrap_or_else(|_| {
                println!("Warning: Rayon 스레드 풀 설정 실패, 기본값 사용");
            });
        
        Self {
            config,
            converted_layers: HashMap::new(),
            conversion_stats: ConversionStatistics::default(),
        }
    }
    
    /// GPT-2 FFN 레이어를 RBE로 변환
    pub fn convert_ffn_layer(
        &mut self, 
        layer_info: &LayerParameterInfo,
        w1_weights: &[f32],  // 768 × 3072
        w2_weights: &[f32],  // 3072 × 768
        bias1: Option<&[f32]>,
        bias2: Option<&[f32]>
    ) -> Result<(), String> {
        
        let start_time = std::time::Instant::now();
        
        println!("=== FFN 레이어 {} RBE 변환 시작 ===", layer_info.layer_id);
        
        // 1. 적응적 블록 크기 결정
        let block_config = self.determine_optimal_block_config(layer_info)?;
        
        // 🎯 통합 진행률 바 생성
        // W1: 768×3072, W2: 3072×768에 대한 총 블록 수 계산
        let w1_blocks = ((768 + block_config.block_size - 1) / block_config.block_size) * 
                        ((3072 + block_config.block_size - 1) / block_config.block_size);
        let w2_blocks = ((3072 + block_config.block_size - 1) / block_config.block_size) * 
                        ((768 + block_config.block_size - 1) / block_config.block_size);
        let total_blocks = w1_blocks + w2_blocks;
        
        let progress = ProgressBar::new(total_blocks as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("🔥 [{elapsed_precise}] [{bar:50.cyan/blue}] {pos:>4}/{len:4} ({percent:>3}%) {msg}")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏ ")
        );
        progress.set_message("FFN 레이어 변환 준비 중...");
        
        // 2. W1 행렬 변환 (768 → 3072)
        progress.set_message(format!("W1 행렬 변환 중... (768×3072, {}블록)", w1_blocks));
        let w1_rbe = self.convert_weight_matrix_with_progress(
            w1_weights, 
            768, 
            3072, 
            &block_config,
            "W1",
            &progress
        )?;
        
        // 3. W2 행렬 변환 (3072 → 768)  
        progress.set_message(format!("W2 행렬 변환 중... (3072×768, {}블록)", w2_blocks));
        let w2_rbe = self.convert_weight_matrix_with_progress(
            w2_weights, 
            3072, 
            768, 
            &block_config,
            "W2",
            &progress
        )?;
        
        // 4. 품질 검증
        let quality_w1 = self.verify_conversion_quality(w1_weights, &w1_rbe, 768, 3072)?;
        let quality_w2 = self.verify_conversion_quality(w2_weights, &w2_rbe, 3072, 768)?;
        
        // 5. 통합 품질 메트릭 계산
        let combined_quality = QualityMetrics {
            mse: (quality_w1.mse + quality_w2.mse) / 2.0,
            psnr: (quality_w1.psnr + quality_w2.psnr) / 2.0,
            cosine_similarity: (quality_w1.cosine_similarity + quality_w2.cosine_similarity) / 2.0,
            frobenius_ratio: (quality_w1.frobenius_ratio + quality_w2.frobenius_ratio) / 2.0,
            quality_score: (quality_w1.quality_score + quality_w2.quality_score) / 2.0,
        };
        
        // 🎉 진행률 바 완료 처리
        let original_size = (w1_weights.len() + w2_weights.len()) * 4;
        let compressed_size = w1_rbe.compressed_size() + w2_rbe.compressed_size();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        progress.finish_with_message(format!(
            "✅ FFN Layer {} 완료! | 품질: {:.1}/100 | 압축: {:.1}:1", 
            layer_info.layer_id, combined_quality.quality_score, compression_ratio
        ));
        
        // 6. 바이어스 처리
        let combined_biases = if bias1.is_some() || bias2.is_some() {
            let mut biases = Vec::new();
            if let Some(b1) = bias1 {
                biases.extend_from_slice(b1);
            }
            if let Some(b2) = bias2 {
                biases.extend_from_slice(b2);
            }
            Some(biases)
        } else {
            None
        };
        
        // 7. 메타데이터 생성
        let original_size = (w1_weights.len() + w2_weights.len()) * 4; // f32 = 4 bytes
        let compressed_size = w1_rbe.compressed_size() + w2_rbe.compressed_size();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        let metadata = ConversionMetadata {
            original_size,
            compressed_size,
            actual_compression_ratio: compression_ratio,
            conversion_time_ms: start_time.elapsed().as_millis(),
            block_configuration: block_config,
        };
        
        // 8. 변환된 레이어 저장
        let quality_score = combined_quality.quality_score; // 미리 저장
        let converted_layer = ConvertedLayer {
            original_layer_id: layer_info.layer_id,
            layer_type: layer_info.layer_type.clone(),
            rbe_weights: vec![w1_rbe, w2_rbe],
            biases: combined_biases,
            metadata,
            quality_metrics: combined_quality,
        };
        
        self.converted_layers.insert(layer_info.layer_id, converted_layer);
        
        // 9. 통계 업데이트
        self.update_conversion_stats(original_size, compressed_size, start_time.elapsed().as_millis());
        
        println!("✓ FFN 레이어 {} 변환 완료:", layer_info.layer_id);
        println!("  압축률: {:.1}:1", compression_ratio);
        println!("  품질 점수: {:.1}/100", quality_score);
        println!("  변환 시간: {}ms", start_time.elapsed().as_millis());
        
        Ok(())
    }
    
    /// Attention 레이어를 RBE로 변환
    pub fn convert_attention_layer(
        &mut self,
        layer_info: &LayerParameterInfo,
        q_weights: &[f32],  // 768 × 768
        k_weights: &[f32],  // 768 × 768
        v_weights: &[f32],  // 768 × 768
        o_weights: &[f32],  // 768 × 768
        biases: Option<&[f32]>
    ) -> Result<(), String> {
        
        let start_time = std::time::Instant::now();
        
        println!("=== Attention 레이어 {} RBE 변환 시작 ===", layer_info.layer_id);
        
        // Attention은 더 보수적인 블록 설정 사용
        let mut block_config = self.determine_optimal_block_config(layer_info)?;
        block_config.quality_level = QualityLevel::Ultra; // 높은 품질 유지
        block_config.block_size = (block_config.block_size * 2).min(128); // 더 큰 블록
        
        // 병렬 변환
        let weight_matrices = vec![
            ("Q", q_weights),
            ("K", k_weights), 
            ("V", v_weights),
            ("O", o_weights),
        ];
        
        let converted_matrices: Result<Vec<_>, String> = weight_matrices
            .into_par_iter()
            .map(|(name, weights)| {
                println!("{} 행렬 변환 중... (768×768)", name);
                self.convert_weight_matrix(weights, 768, 768, &block_config, name)
            })
            .collect();
        
        let rbe_matrices = converted_matrices?;
        
        // 품질 검증 (병렬)
        let quality_checks: Vec<QualityMetrics> = [q_weights, k_weights, v_weights, o_weights]
            .par_iter()
            .zip(rbe_matrices.par_iter())
            .map(|(original, rbe)| {
                self.verify_conversion_quality(original, rbe, 768, 768).unwrap_or_default()
            })
            .collect();
        
        // 평균 품질 계산
        let avg_quality = QualityMetrics {
            mse: quality_checks.iter().map(|q| q.mse).sum::<f32>() / 4.0,
            psnr: quality_checks.iter().map(|q| q.psnr).sum::<f32>() / 4.0,
            cosine_similarity: quality_checks.iter().map(|q| q.cosine_similarity).sum::<f32>() / 4.0,
            frobenius_ratio: quality_checks.iter().map(|q| q.frobenius_ratio).sum::<f32>() / 4.0,
            quality_score: quality_checks.iter().map(|q| q.quality_score).sum::<f32>() / 4.0,
        };
        
        // 메타데이터 생성
        let original_size = (q_weights.len() + k_weights.len() + v_weights.len() + o_weights.len()) * 4;
        let compressed_size: usize = rbe_matrices.iter().map(|m| m.compressed_size()).sum();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        let metadata = ConversionMetadata {
            original_size,
            compressed_size,
            actual_compression_ratio: compression_ratio,
            conversion_time_ms: start_time.elapsed().as_millis(),
            block_configuration: block_config,
        };
        
        // 변환된 레이어 저장
        let avg_quality_score = avg_quality.quality_score; // 미리 저장
        let converted_layer = ConvertedLayer {
            original_layer_id: layer_info.layer_id,
            layer_type: layer_info.layer_type.clone(),
            rbe_weights: rbe_matrices,
            biases: biases.map(|b| b.to_vec()),
            metadata,
            quality_metrics: avg_quality,
        };
        
        self.converted_layers.insert(layer_info.layer_id, converted_layer);
        self.update_conversion_stats(original_size, compressed_size, start_time.elapsed().as_millis());
        
        println!("✓ Attention 레이어 {} 변환 완료:", layer_info.layer_id);
        println!("  압축률: {:.1}:1", compression_ratio);
        println!("  품질 점수: {:.1}/100", avg_quality_score);
        
        Ok(())
    }
    
    /// 임베딩 레이어를 RBE로 변환
    pub fn convert_embedding_layer(
        &mut self,
        layer_info: &LayerParameterInfo,
        embedding_weights: &[f32], // vocab_size × hidden_size
        vocab_size: usize,
        hidden_size: usize
    ) -> Result<(), String> {
        
        let start_time = std::time::Instant::now();
        
        println!("=== 임베딩 레이어 {} RBE 변환 시작 ===", layer_info.layer_id);
        println!("  크기: {}×{}", vocab_size, hidden_size);
        
        // 임베딩 특화 블록 설정 (희소성 활용)
        let mut block_config = self.determine_optimal_block_config(layer_info)?;
        block_config.quality_level = QualityLevel::High;
        
        // 임베딩은 보통 희소하므로 더 큰 압축률 적용
        let rbe_matrix = self.convert_weight_matrix(
            embedding_weights, 
            vocab_size, 
            hidden_size, 
            &block_config,
            "embedding"
        )?;
        
        // 품질 검증
        let quality = self.verify_conversion_quality(
            embedding_weights, 
            &rbe_matrix, 
            vocab_size, 
            hidden_size
        )?;
        
        // 메타데이터
        let original_size = embedding_weights.len() * 4;
        let compressed_size = rbe_matrix.compressed_size();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        let metadata = ConversionMetadata {
            original_size,
            compressed_size,
            actual_compression_ratio: compression_ratio,
            conversion_time_ms: start_time.elapsed().as_millis(),
            block_configuration: block_config,
        };
        
        let converted_layer = ConvertedLayer {
            original_layer_id: layer_info.layer_id,
            layer_type: layer_info.layer_type.clone(),
            rbe_weights: vec![rbe_matrix],
            biases: None,
            metadata,
            quality_metrics: quality.clone(), // clone 사용
        };
        
        self.converted_layers.insert(layer_info.layer_id, converted_layer);
        self.update_conversion_stats(original_size, compressed_size, start_time.elapsed().as_millis());
        
        println!("✓ 임베딩 레이어 {} 변환 완료:", layer_info.layer_id);
        println!("  압축률: {:.1}:1", compression_ratio);
        println!("  품질 점수: {:.1}/100", quality.quality_score);
        
        Ok(())
    }
    
    /// 최적 블록 구성 결정
    pub fn determine_optimal_block_config(&self, layer_info: &LayerParameterInfo) -> Result<BlockConfig, String> {
        // 레이어 타입과 크기에 따른 적응적 블록 크기
        let (block_size, quality_level) = match layer_info.layer_type {
            LayerType::FFN => {
                // FFN은 가장 적극적으로 압축
                if layer_info.target_compression_ratio > 1000.0 {
                    (16, QualityLevel::Medium)
                } else if layer_info.target_compression_ratio > 500.0 {
                    (32, QualityLevel::High)
                } else {
                    (64, QualityLevel::Ultra)
                }
            },
            LayerType::Attention => {
                // Attention은 보수적으로 압축
                if layer_info.target_compression_ratio > 400.0 {
                    (64, QualityLevel::High)
                } else {
                    (128, QualityLevel::Ultra)
                }
            },
            LayerType::TokenEmbedding | LayerType::Output => {
                // 임베딩은 중간 정도
                (32, QualityLevel::High)
            },
            _ => (64, QualityLevel::High),
        };
        
        // 메모리 제한 고려
        let adjusted_block_size = if self.config.memory_limit > 0 {
            let memory_per_block = block_size * block_size * 16; // Packed128 크기
            let max_blocks = self.config.memory_limit / memory_per_block;
            if max_blocks < 100 { // 최소 블록 수 보장
                (block_size / 2).max(8)
            } else {
                block_size
            }
        } else {
            block_size
        };
        
        Ok(BlockConfig {
            block_size: adjusted_block_size,
            num_blocks: 0, // 나중에 계산
            quality_level,
            hierarchy_levels: 4, // 4단계 계층 구조 사용
        })
    }
    
    /// 가중치 행렬을 RBE로 변환
    fn convert_weight_matrix(
        &self,
        weights: &[f32],
        rows: usize,
        cols: usize,
        block_config: &BlockConfig,
        matrix_name: &str
    ) -> Result<HierarchicalBlockMatrix, String> {
        
        println!("  {} 행렬 변환: {}×{}, 블록크기: {}", 
                 matrix_name, rows, cols, block_config.block_size);
        
        // HierarchicalBlockMatrix 생성
        let mut block_matrix = HierarchicalBlockMatrix::new(
            rows,
            cols,
            block_config.quality_level.clone()
        );
        
        // 가중치 데이터를 행렬 형태로 재구성
        let weight_matrix: Vec<Vec<f32>> = (0..rows)
            .map(|i| {
                weights[i * cols..(i + 1) * cols].to_vec()
            })
            .collect();
        
        // RBE 인코딩 수행
        block_matrix.encode_from_dense(&weight_matrix)
            .map_err(|e| format!("RBE 인코딩 실패 ({}): {}", matrix_name, e))?;
        
        println!("  ✓ {} 변환 완료", matrix_name);
        
        Ok(block_matrix)
    }
    
    /// 진행률 바 지원 가중치 행렬 RBE 변환
    fn convert_weight_matrix_with_progress(
        &self,
        weights: &[f32],
        rows: usize,
        cols: usize,
        block_config: &BlockConfig,
        matrix_name: &str,
        progress: &ProgressBar
    ) -> Result<HierarchicalBlockMatrix, String> {
        
        // 예상 블록 수 계산
        let expected_blocks = ((rows + block_config.block_size - 1) / block_config.block_size) * 
                             ((cols + block_config.block_size - 1) / block_config.block_size);
        
        // HierarchicalBlockMatrix 생성
        let mut block_matrix = HierarchicalBlockMatrix::new(
            rows,
            cols,
            block_config.quality_level.clone()
        );
        
        // 가중치 데이터를 행렬 형태로 재구성
        let weight_matrix: Vec<Vec<f32>> = (0..rows)
            .map(|i| {
                weights[i * cols..(i + 1) * cols].to_vec()
            })
            .collect();
        
        // RBE 인코딩 수행 (시뮬레이션된 진행률)
        progress.set_message(format!("{}: 블록 인코딩 시작...", matrix_name));
        
        // 인코딩 시작
        let start_time = std::time::Instant::now();
        block_matrix.encode_from_dense(&weight_matrix)
            .map_err(|e| format!("RBE 인코딩 실패 ({}): {}", matrix_name, e))?;
        let elapsed = start_time.elapsed();
        
        // 예상 블록 수만큼 진행률 증가
        progress.inc(expected_blocks as u64);
        
        // 품질 통계 가져오기
        let stats = block_matrix.quality_statistics();
        let rmse = stats.total_error.sqrt(); // RMSE 계산
        
        progress.set_message(format!(
            "{}: ✅ 완료 | RMSE: {:.6} | 압축: {:.1}:1 | 시간: {:?}", 
            matrix_name, rmse, stats.compression_ratio, elapsed
        ));
        
        Ok(block_matrix)
    }
    
    /// 변환 품질 검증
    fn verify_conversion_quality(
        &self,
        original: &[f32],
        rbe_matrix: &HierarchicalBlockMatrix,
        rows: usize,
        cols: usize
    ) -> Result<QualityMetrics, String> {
        
        // RBE에서 복원
        let reconstructed = rbe_matrix.decode_to_dense()?;
        
        // 1차원 배열로 변환
        let reconstructed_flat: Vec<f32> = reconstructed
            .iter()
            .flat_map(|row| row.iter())
            .cloned()
            .collect();
        
        // MSE 계산
        let mse = original.iter()
            .zip(reconstructed_flat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;
        
        // PSNR 계산
        let max_val = original.iter().cloned().fold(0.0f32, f32::max);
        let psnr = if mse > 0.0 {
            20.0 * (max_val / mse.sqrt()).log10()
        } else {
            100.0 // 완벽한 복원
        };
        
        // 코사인 유사도 계산
        let dot_product: f32 = original.iter()
            .zip(reconstructed_flat.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let norm_original: f32 = original.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let norm_reconstructed: f32 = reconstructed_flat.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        
        let cosine_similarity = if norm_original > 0.0 && norm_reconstructed > 0.0 {
            dot_product / (norm_original * norm_reconstructed)
        } else {
            0.0
        };
        
        // 프로베니우스 노름 비율
        let frobenius_ratio = norm_reconstructed / norm_original;
        
        // 종합 품질 점수 (0-100)
        let psnr_component = (psnr.min(60.0) / 60.0 * 40.0).max(0.0);
        let cosine_component = cosine_similarity * 30.0;
        let frobenius_component = (1.0 - (frobenius_ratio - 1.0).abs()).max(0.0f32) * 30.0;
        
        let quality_score = (psnr_component + cosine_component + frobenius_component).min(100.0);
        
        Ok(QualityMetrics {
            mse,
            psnr,
            cosine_similarity,
            frobenius_ratio,
            quality_score,
        })
    }
    
    /// 변환 통계 업데이트
    fn update_conversion_stats(&mut self, original_size: usize, compressed_size: usize, time_ms: u128) {
        self.conversion_stats.converted_layers += 1;
        self.conversion_stats.total_time_ms += time_ms;
        self.conversion_stats.total_original_size += original_size;
        self.conversion_stats.total_compressed_size += compressed_size;
        
        // 평균 압축률 재계산
        if self.conversion_stats.total_original_size > 0 {
            self.conversion_stats.average_compression_ratio = 
                self.conversion_stats.total_original_size as f32 / 
                self.conversion_stats.total_compressed_size as f32;
        }
        
        // 평균 품질 점수 재계산
        if !self.converted_layers.is_empty() {
            self.conversion_stats.average_quality_score = 
                self.converted_layers.values()
                    .map(|layer| layer.quality_metrics.quality_score)
                    .sum::<f32>() / self.converted_layers.len() as f32;
        }
    }
    
    /// 변환 결과 리포트 출력
    pub fn print_conversion_report(&self) {
        println!("\n=== RBE 변환 결과 리포트 ===");
        println!("변환된 레이어 수: {}", self.conversion_stats.converted_layers);
        println!("총 변환 시간: {}ms", self.conversion_stats.total_time_ms);
        println!("원본 크기: {:.1}MB", self.conversion_stats.total_original_size as f32 / 1024.0 / 1024.0);
        println!("압축 후 크기: {:.1}MB", self.conversion_stats.total_compressed_size as f32 / 1024.0 / 1024.0);
        println!("평균 압축률: {:.1}:1", self.conversion_stats.average_compression_ratio);
        println!("평균 품질 점수: {:.1}/100", self.conversion_stats.average_quality_score);
        
        let savings_mb = (self.conversion_stats.total_original_size - 
                         self.conversion_stats.total_compressed_size) as f32 / 1024.0 / 1024.0;
        let savings_ratio = savings_mb / (self.conversion_stats.total_original_size as f32 / 1024.0 / 1024.0) * 100.0;
        
        println!("절약된 메모리: {:.1}MB ({:.1}%)", savings_mb, savings_ratio);
        
        println!("\n=== 레이어별 상세 정보 ===");
        for (layer_id, layer) in &self.converted_layers {
            println!("Layer {}: {:?}", layer_id, layer.layer_type);
            println!("  압축률: {:.1}:1", layer.metadata.actual_compression_ratio);
            println!("  품질: {:.1}/100", layer.quality_metrics.quality_score);
            println!("  변환시간: {}ms", layer.metadata.conversion_time_ms);
        }
    }
    
    /// 변환된 레이어 가져오기
    pub fn get_converted_layer(&self, layer_id: usize) -> Option<&ConvertedLayer> {
        self.converted_layers.get(&layer_id)
    }
    
    /// 모든 변환된 레이어 가져오기
    pub fn get_all_converted_layers(&self) -> &HashMap<usize, ConvertedLayer> {
        &self.converted_layers
    }
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_threads: num_cpus::get(),
            quality_priority: true,
            memory_limit: 1024 * 1024 * 1024, // 1GB
            precision_mode: PrecisionMode::Balanced,
        }
    }
}

impl Default for ConversionStatistics {
    fn default() -> Self {
        Self {
            total_time_ms: 0,
            converted_layers: 0,
            total_original_size: 0,
            total_compressed_size: 0,
            average_compression_ratio: 0.0,
            average_quality_score: 0.0,
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            mse: 0.0,
            psnr: 0.0,
            cosine_similarity: 0.0,
            frobenius_ratio: 1.0,
            quality_score: 0.0,
        }
    }
} 