//! Updated encoder module with parallel grid compression and block caching
//! This extends the original compress_grid implementation to speed up encoding
//! and avoid recompressing identical blocks.

use crate::types::{Packed64, Packed128, PoincareMatrix};
use crate::math::compute_full_rmse;
use rand::Rng;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use rayon::prelude::*;

impl PoincareMatrix {
    /// Parallel grid-based compression with block caching
    ///
    /// This method splits the matrix into blocks of size `block_size` and
    /// compresses each block in parallel using Rayon. A simple hash of
    /// the block's contents is used to cache previously compressed blocks
    /// and reuse their seeds, which can dramatically reduce work when
    /// neighbouring blocks contain identical patterns.
    pub fn compress_grid_parallel(
        matrix: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> GridCompressedMatrix {
        let grid_rows = (rows + block_size - 1) / block_size;
        let grid_cols = (cols + block_size - 1) / block_size;
        // Shared cache for compressed blocks
        let cache = std::sync::Mutex::new(HashMap::<u64, PoincareMatrix>::new());

        // Helper to hash a slice of f32
        fn hash_block(data: &[f32]) -> u64 {
            let mut hasher = DefaultHasher::new();
            data.hash(&mut hasher);
            hasher.finish()
        }

        // Use parallel iterators to process blocks concurrently
        let blocks: Vec<PoincareMatrix> = (0..grid_rows)
            .into_par_iter()
            .flat_map(|grid_i| {
                (0..grid_cols)
                    .into_par_iter()
                    .map(move |grid_j| {
                        // Determine block boundaries
                        let start_i = grid_i * block_size;
                        let start_j = grid_j * block_size;
                        let end_i = ((grid_i + 1) * block_size).min(rows);
                        let end_j = ((grid_j + 1) * block_size).min(cols);
                        let block_rows = end_i - start_i;
                        let block_cols = end_j - start_j;
                        // Extract block data
                        let mut block_data = Vec::with_capacity(block_rows * block_cols);
                        for i in start_i..end_i {
                            let row_start = i * cols + start_j;
                            let row_end = i * cols + end_j;
                            block_data.extend_from_slice(&matrix[row_start..row_end]);
                        }
                        // Compute hash and check cache
                        let h = hash_block(&block_data);
                        if let Some(cached) = cache.lock().unwrap().get(&h) {
                            // Reuse cached compressed block
                            return cached.clone();
                        }
                        // Otherwise compress the block
                        let compressed = Self::compress(&block_data, block_rows, block_cols);
                        // Store in cache
                        cache.lock().unwrap().insert(h, compressed.clone());
                        compressed
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        GridCompressedMatrix {
            blocks,
            grid_rows,
            grid_cols,
            block_size,
            total_rows: rows,
            total_cols: cols,
        }
    }
}

/// Structure representing a grid-compressed matrix
#[derive(Clone)]
pub struct GridCompressedMatrix {
    pub blocks: Vec<PoincareMatrix>,
    pub grid_rows: usize,
    pub grid_cols: usize,
    pub block_size: usize,
    pub total_rows: usize,
    pub total_cols: usize,
}

impl GridCompressedMatrix {
    /// Compression ratio (without metadata)
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.total_rows * self.total_cols * 4; // f32 = 4 bytes
        let compressed_size = self.blocks.len() * 16; // each block is 128 bits = 16 bytes
        original_size as f32 / compressed_size as f32
    }

    /// Effective compression ratio including metadata
    pub fn effective_compression_ratio(&self) -> f32 {
        let original_size = self.total_rows * self.total_cols * 4;
        let compressed_size = self.blocks.len() * 16 + 24; // 24 bytes for grid metadata
        original_size as f32 / compressed_size as f32
    }
}
