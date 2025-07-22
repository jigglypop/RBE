//! RBE 기반 가중치 생성기 - HybridEncodedBlock의 decode와 일치하는 방식

use crate::packed_params::HybridEncodedBlock;
use std::sync::Arc;
use parking_lot::RwLock;
use lru::LruCache;
use std::num::NonZeroUsize;

/// RBE 디코더 설정
#[derive(Debug, Clone)]
pub struct RBEDecoderConfig {
    /// LRU 캐시 크기 (블록 수)
    pub cache_size: usize,
    /// 병렬 처리 활성화
    pub enable_parallel: bool,
    /// SIMD 최적화 활성화
    pub enable_simd: bool,
}

impl Default for RBEDecoderConfig {
    fn default() -> Self {
        Self {
            cache_size: 16,      // 16개 블록 캐시
            enable_parallel: true,
            enable_simd: true,
        }
    }
}

/// RBE 기반 가중치 생성기
#[derive(Clone, Debug)]
pub struct WeightGenerator {
    config: RBEDecoderConfig,
    /// 디코딩 캐시 - 블록 해시를 키로 사용
    cache: Arc<RwLock<LruCache<u64, Arc<Vec<f32>>>>>,
    /// 성능 통계
    stats: Arc<RwLock<DecoderStats>>,
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
        let cache_size = NonZeroUsize::new(config.cache_size).unwrap_or(NonZeroUsize::new(16).unwrap());
        Self {
            config,
            cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
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
        
        {
            let mut cache = self.cache.write();
            if let Some(cached) = cache.get(&block_hash) {
                let mut stats = self.stats.write();
                stats.cache_hits += 1;
                stats.total_decodes += 1;
                return cached.clone();
            }
        }
        
        // 캐시 미스 - 디코딩 수행
        let decoded = if self.config.enable_simd {
            self.decode_block_simd(block)
        } else {
            self.decode_block_scalar(block)
        };
        
        let decoded_arc = Arc::new(decoded);
        
        // 캐시에 저장
        {
            let mut cache = self.cache.write();
            cache.put(block_hash, decoded_arc.clone());
        }
        
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
        // 현재는 스칼라 버전과 동일하게 처리
        // TODO: 실제 SIMD 최적화 구현
        block.decode()
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn decode_block_simd(&self, block: &HybridEncodedBlock) -> Vec<f32> {
        // 다른 아키텍처에서는 스칼라 버전 사용
        self.decode_block_scalar(block)
    }
    
    /// 캐시 정리
    pub fn clear_cache(&mut self) {
        self.cache.write().clear();
    }
    
    /// 통계 반환
    pub fn get_stats(&self) -> DecoderStats {
        let stats = self.stats.read();
        DecoderStats {
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
            total_decodes: stats.total_decodes,
            total_decode_time_ns: stats.total_decode_time_ns,
        }
    }
    
    /// 캐시 히트율 계산
    pub fn get_cache_hit_rate(&self) -> f32 {
        let stats = self.stats.read();
        if stats.total_decodes == 0 {
            0.0
        } else {
            stats.cache_hits as f32 / stats.total_decodes as f32
        }
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