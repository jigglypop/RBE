//! RBE 기반 가중치 생성기 - HybridEncodedBlock의 decode와 일치하는 방식

use crate::packed_params::{HybridEncodedBlock, TransformType};
use crate::core::decoder::caching_strategy::{CachingStrategy, DecoderCache, create_cache};
use std::sync::Arc;
use parking_lot::RwLock;

/// RBE 디코더 설정
#[derive(Debug, Clone)]
pub struct RBEDecoderConfig {
    /// 캐싱 전략
    pub caching_strategy: CachingStrategy,
    /// 병렬 처리 활성화
    pub enable_parallel: bool,
    /// SIMD 최적화 활성화
    pub enable_simd: bool,
}

impl RBEDecoderConfig {
    /// 최소 메모리 설정
    pub fn minimal_memory() -> Self {
        Self {
            caching_strategy: CachingStrategy::NoCache,
            enable_parallel: false,
            enable_simd: true,
        }
    }
    
    /// 균형잡힌 설정 (기본값)
    pub fn balanced() -> Self {
        Self {
            caching_strategy: CachingStrategy::PercentageBased { percentage: 0.1 },
            enable_parallel: true,
            enable_simd: true,
        }
    }
    
    /// 최대 성능 설정
    pub fn max_performance() -> Self {
        Self {
            caching_strategy: CachingStrategy::PrecomputeAll,
            enable_parallel: true,
            enable_simd: true,
        }
    }
    
    /// 적응형 설정
    pub fn adaptive() -> Self {
        Self {
            caching_strategy: CachingStrategy::Adaptive {
                min_size: 8,
                max_size: 1024,
                target_hit_rate: 0.9,
            },
            enable_parallel: true,
            enable_simd: true,
        }
    }
    
    /// 레거시 호환성을 위한 생성자
    pub fn legacy(cache_size: usize) -> Self {
        Self {
            caching_strategy: CachingStrategy::FixedLRU { size: cache_size },
            enable_parallel: true,
            enable_simd: true,
        }
    }
}

impl Default for RBEDecoderConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

/// RBE 기반 가중치 생성기
#[derive(Clone)]
pub struct WeightGenerator {
    config: RBEDecoderConfig,
    /// 디코딩 캐시 - 블록 해시를 키로 사용
    cache: Arc<Box<dyn DecoderCache>>,
    /// 성능 통계
    stats: Arc<RwLock<DecoderStats>>,
}

impl std::fmt::Debug for WeightGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightGenerator")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

#[derive(Debug, Default, Clone)]
pub struct DecoderStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_decodes: u64,
    pub total_decode_time_ns: u64,
}

impl WeightGenerator {
    /// 새로운 가중치 생성기 생성
    pub fn new() -> Self {
        Self::with_config(RBEDecoderConfig::default())
    }
    
    /// 설정 기반 생성
    pub fn with_config(config: RBEDecoderConfig) -> Self {
        let cache = create_cache(config.caching_strategy, None);
        Self {
            config,
            cache: Arc::new(cache),
            stats: Arc::new(RwLock::new(DecoderStats::default())),
        }
    }
    
    /// 총 블록 수를 알고 있을 때 생성
    pub fn with_config_and_blocks(config: RBEDecoderConfig, total_blocks: usize) -> Self {
        let cache = create_cache(config.caching_strategy, Some(total_blocks));
        Self {
            config,
            cache: Arc::new(cache),
            stats: Arc::new(RwLock::new(DecoderStats::default())),
        }
    }
    
    /// 블록 해시 계산 (캐시 키로 사용)
    fn compute_block_hash(block: &HybridEncodedBlock) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // RBE 파라미터 해시
        for &param in &block.rbe_params {
            param.to_bits().hash(&mut hasher);
        }
        
        // 블록 크기 정보
        block.rows.hash(&mut hasher);
        block.cols.hash(&mut hasher);
        
        // 잔차 계수 개수
        block.residuals.len().hash(&mut hasher);
        
        hasher.finish()
    }
    
    /// 블록 디코딩 (HybridEncodedBlock::decode와 동일한 로직)
    pub fn decode_block(&self, block: &HybridEncodedBlock) -> Arc<Vec<f32>> {
        let start_time = std::time::Instant::now();
        
        // 캐시 확인
        let block_hash = Self::compute_block_hash(block);
        
        if let Some(cached) = self.cache.get(block_hash) {
            let mut stats = self.stats.write();
            stats.cache_hits += 1;
            stats.total_decodes += 1;
            return cached;
        }
        
        // 캐시 미스 - 디코딩 수행
        let decoded = if self.config.enable_simd {
            self.decode_block_simd(block)
        } else {
            self.decode_block_scalar(block)
        };
        
        let decoded_arc = Arc::new(decoded);
        
        // 캐시에 저장
        self.cache.put(block_hash, decoded_arc.clone());
        
        // 통계 업데이트
        {
            let mut stats = self.stats.write();
            stats.cache_misses += 1;
            stats.total_decodes += 1;
            stats.total_decode_time_ns += start_time.elapsed().as_nanos() as u64;
        }
        
        decoded_arc
    }
    
    /// 스칼라 버전 디코딩
    fn decode_block_scalar(&self, block: &HybridEncodedBlock) -> Vec<f32> {
        // HybridEncodedBlock::decode 호출
        block.decode()
    }
    
    /// SIMD 버전 디코딩 (x86_64)
    #[cfg(target_arch = "x86_64")]
    fn decode_block_simd(&self, block: &HybridEncodedBlock) -> Vec<f32> {
        use std::arch::x86_64::*;
        
        let rows = block.rows;
        let cols = block.cols;
        let total_size = rows * cols;
        
        // SIMD 사용 불가능한 경우 스칼라 버전 사용
        if !is_x86_feature_detected!("avx2") || total_size < 8 {
            return self.decode_block_scalar(block);
        }
        
        let mut reconstruction = vec![0.0f32; total_size];
        
        unsafe {
            // RBE 파라미터를 SIMD 레지스터에 로드
            let rbe_params_ptr = block.rbe_params.as_ptr();
            let params_low = _mm256_loadu_ps(rbe_params_ptr);
            
            // 상수들
            let pi = std::f32::consts::PI;
            let two = _mm256_set1_ps(2.0);
            let one = _mm256_set1_ps(1.0);
            let pi_vec = _mm256_set1_ps(pi);
            let two_pi_vec = _mm256_set1_ps(2.0 * pi);
            
            // 8개씩 벡터화 처리
            let mut idx = 0;
            while idx + 8 <= total_size {
                // 좌표 계산
                let mut x_vals = [0.0f32; 8];
                let mut y_vals = [0.0f32; 8];
                
                for i in 0..8 {
                    let row = (idx + i) / cols;
                    let col = (idx + i) % cols;
                    x_vals[i] = if cols > 1 { (col as f32 / (cols - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                    y_vals[i] = if rows > 1 { (row as f32 / (rows - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                }
                
                let x_vec = _mm256_loadu_ps(x_vals.as_ptr());
                let y_vec = _mm256_loadu_ps(y_vals.as_ptr());
                
                // d = sqrt(x*x + y*y)
                let x_sq = _mm256_mul_ps(x_vec, x_vec);
                let y_sq = _mm256_mul_ps(y_vec, y_vec);
                let d_sq = _mm256_add_ps(x_sq, y_sq);
                let d_vec = _mm256_sqrt_ps(d_sq);
                
                // 기저 함수 계산 (처음 4개만 벡터화)
                let basis0 = one; // 1.0
                let basis1 = d_vec; // d
                let basis2 = _mm256_mul_ps(d_vec, d_vec); // d*d
                
                // cos 계산은 스칼라로 (AVX2에는 cos 명령어가 없음)
                let mut result = _mm256_setzero_ps();
                result = _mm256_fmadd_ps(_mm256_set1_ps(block.rbe_params[0]), basis0, result);
                result = _mm256_fmadd_ps(_mm256_set1_ps(block.rbe_params[1]), basis1, result);
                result = _mm256_fmadd_ps(_mm256_set1_ps(block.rbe_params[2]), basis2, result);
                
                // 결과 저장
                _mm256_storeu_ps(&mut reconstruction[idx], result);
                
                // 나머지 기저 함수들은 스칼라로 처리
                for i in 0..8 {
                    let x = x_vals[i];
                    let y = y_vals[i];
                    reconstruction[idx + i] += 
                        block.rbe_params[3] * (pi * x).cos() +
                        block.rbe_params[4] * (pi * y).cos() +
                        block.rbe_params[5] * (2.0 * pi * x).cos() +
                        block.rbe_params[6] * (2.0 * pi * y).cos() +
                        block.rbe_params[7] * (pi * x).cos() * (pi * y).cos();
                }
                
                idx += 8;
            }
            
            // 나머지 요소들은 스칼라로 처리
            while idx < total_size {
                let row = idx / cols;
                let col = idx % cols;
                let x = if cols > 1 { (col as f32 / (cols - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                let y = if rows > 1 { (row as f32 / (rows - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                let d = (x * x + y * y).sqrt();
                
                reconstruction[idx] = 
                    block.rbe_params[0] * 1.0 +
                    block.rbe_params[1] * d +
                    block.rbe_params[2] * d * d +
                    block.rbe_params[3] * (pi * x).cos() +
                    block.rbe_params[4] * (pi * y).cos() +
                    block.rbe_params[5] * (2.0 * pi * x).cos() +
                    block.rbe_params[6] * (2.0 * pi * y).cos() +
                    block.rbe_params[7] * (pi * x).cos() * (pi * y).cos();
                
                idx += 1;
            }
        }
        
        // DWT 역변환은 block.decode()에서 처리하도록 위임
        if !block.residuals.is_empty() && matches!(block.transform_type, crate::packed_params::TransformType::Dwt) {
            // 잔차가 있으면 전체 디코딩 수행
            return block.decode();
        }
        
        reconstruction
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn decode_block_simd(&self, block: &HybridEncodedBlock) -> Vec<f32> {
        // 다른 아키텍처에서는 스칼라 버전 사용
        self.decode_block_scalar(block)
    }
    
    /// 캐시 정리
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    /// 블록들을 미리 디코딩하여 캐시에 로드
    pub fn preload_blocks(&self, blocks: &[HybridEncodedBlock]) -> usize {
        let mut loaded = 0;
        for block in blocks {
            let _ = self.decode_block(block);
            loaded += 1;
        }
        loaded
    }
    
    /// 워밍업 - 캐시를 미리 채우기
    pub fn warmup(&self, blocks: &[HybridEncodedBlock], warmup_ratio: f32) {
        let num_to_load = (blocks.len() as f32 * warmup_ratio.min(1.0)) as usize;
        let start = std::time::Instant::now();
        
        println!("워밍업 시작: {} / {} 블록 사전 로딩", num_to_load, blocks.len());
        
        // 병렬로 블록 디코딩
        use rayon::prelude::*;
        blocks.par_iter()
            .take(num_to_load)
            .for_each(|block| {
                let _ = self.decode_block(block);
            });
            
        let elapsed = start.elapsed();
        println!("워밍업 완료: {:.2}ms 소요", elapsed.as_millis());
    }
    
    /// 캐시 크기 동적 조정
    pub fn resize_cache(&mut self, new_size: usize) -> bool {
        if new_size == 0 {
            return false;
        }
        
        self.cache.resize(new_size)
    }
    
    /// 통계 반환
    pub fn get_stats(&self) -> DecoderStats {
        let stats = self.stats.read();
        let cache_stats = self.cache.stats();
        
        DecoderStats {
            cache_hits: cache_stats.hits,
            cache_misses: cache_stats.misses,
            total_decodes: stats.total_decodes,
            total_decode_time_ns: stats.total_decode_time_ns,
        }
    }
    
    /// 캐시 히트율 계산
    pub fn get_cache_hit_rate(&self) -> f32 {
        let cache_stats = self.cache.stats();
        cache_stats.hit_rate()
    }
}

/// 전역 가중치 생성기 (후위 호환성)
pub fn generate_weight_global(
    block: &HybridEncodedBlock,
    _row: usize,
    _col: usize,
    _rows: usize,
    _cols: usize,
) -> f32 {
    // 단순히 블록을 디코딩하고 평균값 반환 (호환성용)
    let decoded = block.decode();
    decoded.iter().sum::<f32>() / decoded.len() as f32
}