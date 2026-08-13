use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn::{Module, VarBuilder, VarMap};
use crate::nlp::{
    embedding::RBEEmbedding,
    layernorm::RBELayerNorm,
    ffn::RBEFFN,
    attention::RBEAttention,
    dropout::RBEDropout,
};

/// Candle과 RBE 구현 비교 테스트
#[cfg(test)]
mod candle_comparison_tests {
    use super::*;

    #[test]
    fn 임베딩_레이어_candle_비교_테스트() -> Result<()> {
        // 설정
        let vocab_size = 1000;
        let embedding_dim = 128;
        let seq_len = 10;
        let device = Device::Cpu;
        
        // Candle 임베딩 생성
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let candle_embedding = candle_nn::embedding(vocab_size, embedding_dim, vs)?;
        
        // RBE 임베딩 생성 (같은 가중치로 초기화)
        let mut rbe_config = crate::nlp::embedding::RBEEmbeddingConfig {
            vocab_size,
            embedding_dim,
            max_position_embeddings: 512,
            block_size: 64,
            quality_grade: crate::QualityGrade::A,
            enable_parallel: true,
            cache_size: 100,
        };
        
        // 같은 가중치로 초기화
        let weights = candle_embedding.embeddings().to_vec2::<f32>()?;
        let flat_weights: Vec<f32> = weights.into_iter().flatten().collect();
        let rbe_embedding = RBEEmbedding::from_pretrained_weights(&flat_weights, None, rbe_config)?;
        
        // 테스트 입력
        let token_ids = vec![1u32, 5, 10, 100, 500];
        let input_tensor = Tensor::new(token_ids.clone(), &device)?;
        
        // Candle forward
        let candle_output = candle_embedding.forward(&input_tensor)?;
        let candle_result = candle_output.to_vec2::<f32>()?;
        
        // RBE forward
        let rbe_output = rbe_embedding.forward(&token_ids)?;
        let rbe_result: Vec<Vec<f32>> = token_ids.iter()
            .map(|&id| {
                rbe_output[id as usize * embedding_dim..(id as usize + 1) * embedding_dim].to_vec()
            })
            .collect();
        
        // 비교 (압축으로 인한 오차 허용)
        for (i, (candle_row, rbe_row)) in candle_result.iter().zip(&rbe_result).enumerate() {
            let mse: f32 = candle_row.iter()
                .zip(rbe_row)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>() / embedding_dim as f32;
            
            println!("Token {}: MSE = {}", token_ids[i], mse);
            assert!(mse < 0.001, "MSE too high: {}", mse);
        }
        
        // 압축률 확인
        let (compressed_size, compression_ratio) = rbe_embedding.memory_usage();
        println!("압축률: {:.2}:1, 압축 크기: {} bytes", compression_ratio, compressed_size);
        assert!(compression_ratio > 10.0, "압축률이 너무 낮음");
        
        Ok(())
    }

    #[test]
    fn 레이어놈_candle_비교_테스트() -> Result<()> {
        let hidden_size = 768;
        let device = Device::Cpu;
        
        // Candle LayerNorm 생성
        let eps = 1e-5;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let candle_ln = candle_nn::layer_norm(hidden_size, eps, vs)?;
        
        // RBE LayerNorm 생성
        let rbe_ln = RBELayerNorm::new(crate::nlp::layernorm::RBELayerNormConfig {
            normalized_shape: vec![hidden_size],
            eps,
            elementwise_affine: true,
            use_fused_ops: true,
        })?;
        
        // 테스트 입력 (배치 크기 4)
        let input_data: Vec<f32> = (0..4 * hidden_size)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        let input_tensor = Tensor::from_vec(input_data.clone(), (4, hidden_size), &device)?;
        
        // Candle forward
        let candle_output = candle_ln.forward(&input_tensor)?;
        let candle_result = candle_output.to_vec2::<f32>()?;
        
        // RBE forward (배치 처리)
        let mut rbe_result = Vec::new();
        for i in 0..4 {
            let start = i * hidden_size;
            let end = (i + 1) * hidden_size;
            let output = rbe_ln.forward(&input_data[start..end])?;
            rbe_result.push(output);
        }
        
        // 비교
        for (i, (candle_row, rbe_row)) in candle_result.iter().zip(&rbe_result).enumerate() {
            let max_diff: f32 = candle_row.iter()
                .zip(rbe_row)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            
            println!("Batch {}: Max diff = {}", i, max_diff);
            assert!(max_diff < 1e-4, "Difference too large: {}", max_diff);
        }
        
        Ok(())
    }

    #[test]
    fn FFN_레이어_candle_비교_테스트() -> Result<()> {
        let hidden_dim = 768;
        let intermediate_dim = 3072;
        let device = Device::Cpu;
        
        // Candle FFN (2개의 Linear + GELU)
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let linear1 = candle_nn::linear(hidden_dim, intermediate_dim, vs.pp("fc1"))?;
        let linear2 = candle_nn::linear(intermediate_dim, hidden_dim, vs.pp("fc2"))?;
        
        // RBE FFN 생성
        let ffn_config = crate::nlp::ffn::RBEFFNConfig {
            hidden_dim,
            intermediate_dim,
            activation: crate::nlp::ffn::ActivationType::Gelu,
            dropout: 0.0,
            block_size: 128,
            quality_grade: crate::QualityGrade::A,
            enable_parallel: true,
            cache_size: 100,
        };
        
        // Candle 가중치 추출 및 RBE 초기화
        let w1 = linear1.weight().to_vec2::<f32>()?;
        let w2 = linear2.weight().to_vec2::<f32>()?;
        let w1_flat: Vec<f32> = w1.into_iter().flatten().collect();
        let w2_flat: Vec<f32> = w2.into_iter().flatten().collect();
        
        let mut rbe_ffn = RBEFFN::from_pretrained_weights(&w1_flat, &w2_flat, ffn_config)?;
        
        // 테스트 입력
        let input_data: Vec<f32> = (0..hidden_dim)
            .map(|i| ((i as f32) * 0.1).cos())
            .collect();
        let input_tensor = Tensor::from_vec(input_data.clone(), hidden_dim, &device)?;
        
        // Candle forward
        let x1 = linear1.forward(&input_tensor)?;
        let x2 = x1.gelu()?; // gelu는 Tensor의 메서드
        let candle_output = linear2.forward(&x2)?;
        let candle_result = candle_output.to_vec1::<f32>()?;
        
        // RBE forward
        let rbe_result = rbe_ffn.forward(&input_data)?;
        
        // 비교
        let mse: f32 = candle_result.iter()
            .zip(&rbe_result)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / hidden_dim as f32;
        
        println!("FFN MSE: {}", mse);
        assert!(mse < 0.01, "MSE too high: {}", mse);
        
        // 압축률 확인
        let (compressed_size, compression_ratio) = rbe_ffn.memory_usage();
        println!("FFN 압축률: {:.2}:1", compression_ratio);
        assert!(compression_ratio > 50.0, "압축률이 너무 낮음");
        
        Ok(())
    }

    #[test]
    fn 드롭아웃_candle_비교_테스트() -> Result<()> {
        // Dropout은 확률적이므로 통계적 비교
        let dropout_prob = 0.5;
        let size = 10000;
        
        // RBE Dropout
        let mut rbe_dropout = RBEDropout::new(dropout_prob)?;
        rbe_dropout.set_training(true);
        
        let input: Vec<f32> = vec![1.0; size];
        
        // 여러 번 실행하여 통계 수집
        let mut rbe_zeros = 0;
        let num_trials = 100;
        
        for _ in 0..num_trials {
            let output = rbe_dropout.forward(&input);
            rbe_zeros += output.iter().filter(|&&x| x == 0.0).count();
        }
        
        let rbe_drop_rate = rbe_zeros as f32 / (size * num_trials) as f32;
        
        println!("RBE Dropout rate: {:.3}", rbe_drop_rate);
        
        // 드롭률이 목표값의 ±5% 이내
        assert!((rbe_drop_rate - dropout_prob).abs() < 0.05);
        
        // 푸앵카레 마스크 테스트
        let poincare_mask = rbe_dropout.generate_poincare_mask(1000);
        let center_rate = poincare_mask[0..100].iter().filter(|&&x| x).count() as f32 / 100.0;
        let boundary_rate = poincare_mask[900..1000].iter().filter(|&&x| x).count() as f32 / 100.0;
        
        println!("Center drop rate: {:.3}, Boundary drop rate: {:.3}", center_rate, boundary_rate);
        
        // 경계에서 더 많이 드롭되어야 함
        assert!(boundary_rate > center_rate * 1.2);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // L4 종단 (하네스 6절, CP-8): kogpt2 전모델 순전파 대조.
    // 전 트랜스포머 matmul 가중치(12층 x 4행렬)를 매칭 퍼슈트 + 2비트 하이브리드로
    // 압축하고, candle 그래프로 기준(원본 f32)과 압축본의 순전파를 대조한다.
    // - 게이트(유도 상계): 행렬별 f64 전파식 ||X dW||_F <= ||X||_F ||dW||_F
    // - 보고(비게이트, 하네스 L4-2 원칙): 로짓 RMSE, top-1/top-5 일치율, 층별 c
    // 실행: cargo test --release --lib -- --ignored 전모델 --nocapture
    // -----------------------------------------------------------------------

    const KOGPT2_DIR: &str = "models/skt-kogpt2-base-v2";
    const N_LAYER: usize = 12;
    const N_HEAD: usize = 12;
    const LN_EPS: f64 = 1e-5;

    /// 층별 matmul 가중치 이름 (Conv1D 규약: x [t,in] @ W [in,out])
    fn matmul_weight_names() -> Vec<String> {
        (0..N_LAYER)
            .flat_map(|l| {
                [
                    format!("transformer.h.{}.attn.c_attn.weight", l),
                    format!("transformer.h.{}.attn.c_proj.weight", l),
                    format!("transformer.h.{}.mlp.c_fc.weight", l),
                    format!("transformer.h.{}.mlp.c_proj.weight", l),
                ]
            })
            .collect()
    }

    fn layer_norm(
        wm: &std::collections::HashMap<String, Tensor>,
        name: &str,
        x: &Tensor,
    ) -> Result<Tensor> {
        let ln = candle_nn::LayerNorm::new(
            wm[&format!("{}.weight", name)].clone(),
            wm[&format!("{}.bias", name)].clone(),
            LN_EPS,
        );
        Ok(ln.forward(x)?)
    }

    /// Conv1D 선형: y = x @ W + b. taps 가 있으면 (이름, 입력) 을 기록한다.
    fn conv1d(
        wm: &std::collections::HashMap<String, Tensor>,
        name: &str,
        x: &Tensor,
        taps: &mut Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Tensor> {
        if let Some(t) = taps.as_mut() {
            t.push((format!("{}.weight", name), x.clone()));
        }
        let y = x.matmul(&wm[&format!("{}.weight", name)])?;
        Ok(y.broadcast_add(&wm[&format!("{}.bias", name)])?)
    }

    fn causal_attention(scores: &Tensor) -> Result<Tensor> {
        let (_, t, _) = scores.dims3()?;
        let mask: Vec<f32> = (0..t)
            .flat_map(|i| (0..t).map(move |j| if j <= i { 0.0f32 } else { -1e10 }))
            .collect();
        let mask = Tensor::from_vec(mask, (t, t), scores.device())?;
        let masked = scores.broadcast_add(&mask)?;
        Ok(candle_nn::ops::softmax(&masked, candle_core::D::Minus1)?)
    }

    /// GPT-2 순전파 (candle f32). wm 의 가중치를 그대로 사용 — 기준/압축 공용 경로.
    fn gpt2_forward(
        wm: &std::collections::HashMap<String, Tensor>,
        ids: &[u32],
        mut taps: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Tensor> {
        let dev = Device::Cpu;
        let t = ids.len();
        let ids_t = Tensor::new(ids, &dev)?;
        let wte = &wm["transformer.wte.weight"];
        let mut h = wte
            .index_select(&ids_t, 0)?
            .add(&wm["transformer.wpe.weight"].narrow(0, 0, t)?)?;

        for l in 0..N_LAYER {
            let p = format!("transformer.h.{}", l);
            let x = layer_norm(wm, &format!("{}.ln_1", p), &h)?;
            let qkv = conv1d(wm, &format!("{}.attn.c_attn", p), &x, &mut taps)?;
            let split = |o: usize| -> Result<Tensor> {
                Ok(qkv
                    .narrow(1, o * 768, 768)?
                    .reshape((t, N_HEAD, 64))?
                    .transpose(0, 1)?
                    .contiguous()?)
            };
            let (q, k, v) = (split(0)?, split(1)?, split(2)?);
            let scores = (q.matmul(&k.transpose(1, 2)?)? / 8.0)?;
            let probs = causal_attention(&scores)?;
            let ctx = probs
                .matmul(&v)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((t, 768))?;
            let attn_out = conv1d(wm, &format!("{}.attn.c_proj", p), &ctx, &mut taps)?;
            h = h.add(&attn_out)?;

            let x = layer_norm(wm, &format!("{}.ln_2", p), &h)?;
            let ff = conv1d(wm, &format!("{}.mlp.c_fc", p), &x, &mut taps)?.gelu()?;
            let ff = conv1d(wm, &format!("{}.mlp.c_proj", p), &ff, &mut taps)?;
            h = h.add(&ff)?;
        }
        let h = layer_norm(wm, "transformer.ln_f", &h)?;
        Ok(h.matmul(&wm["transformer.wte.weight"].t()?)?)
    }

    fn tensor_to_f64(t: &Tensor) -> Result<Vec<f64>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .map(|&v| v as f64)
            .collect())
    }

    fn frob(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    #[test]
    #[ignore = "모델 파일 필요 (하네스 8절 L4: 명시 실행)"] // lint-allow: L4 는 모델 존재 시 명시 실행이 규약
    fn kogpt2_전모델_압축_candle_순전파_대조() -> Result<()> {
        use crate::core::encoder::hybrid_codec::{hybrid_roundtrip, LloydMaxQuantizer};
        use crate::core::math::verification::{bounds, check};
        use crate::core::matrix::layer_codec::{fit_matching_pursuit, PursuitConfig};

        let dev = Device::Cpu;
        let wm_ref = candle_core::safetensors::load(
            format!("{}/model.safetensors", KOGPT2_DIR),
            &dev,
        )?;
        let wm_ref: std::collections::HashMap<String, Tensor> = wm_ref
            .into_iter()
            .map(|(k, v)| Ok((k, v.to_dtype(DType::F32)?)))
            .collect::<Result<_>>()?;

        // 실제 한국어 입력 (실토크나이저 — 난수 입력 금지)
        let tok = tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", KOGPT2_DIR))
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let text = "대한민국의 수도는 서울이다. 오늘 날씨가 맑아서 공원으로 산책을 나갔다.";
        let ids: Vec<u32> = tok
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .get_ids()
            .to_vec();
        println!("[보고] 입력 토큰 수 = {}", ids.len());

        // (1) c 사다리 적응 배분 (17.2절): 행렬별로 동일비트 저랭크 상한(프로토콜 7)을
        // 재고, 손익분기 c > 0.105 를 넘을 가능성이 있는 행렬만 좌표 학습 원자 부호화.
        // 적합 후 실제 달성 c 가 손익분기 미달이면 원자를 버린다 (달성치 기준 판정).
        // 잔차는 4비트 (17.3절 R2 목표 상대오차 ~0.1 = INT4 급).
        use crate::core::matrix::layer_codec::{fit_alternating, svd_energy_ceiling, LayerCodec, LearnConfig};
        const BREAK_EVEN_C: f64 = 0.105; // 17.2절 손익분기 (원자 0.08 bpw 의 구조 제거 이득 하한)
        let q4 = LloydMaxQuantizer::new_gaussian(4);
        let learn = LearnConfig {
            pursuit: PursuitConfig {
                n_theta: 64,
                n_lambda: 128,
                n_atoms: 512,
            },
            rounds: 4,
            sgd_steps: 200,
            batch: 96,
            lr: 0.05,
            seed: 0x5242_4590,
        };
        let mut wm_cmp = wm_ref.clone();
        let mut total_bits = 0u64;
        let mut total_params = 0u64;
        for name in matmul_weight_names() {
            let t0 = std::time::Instant::now();
            let (m, n) = wm_ref[&name].dims2()?;
            let w = tensor_to_f64(&wm_ref[&name])?;
            let mn = m * n;
            let ceiling = svd_energy_ceiling(&w, m, n, 5);
            let (k, c, j_used) = if ceiling > BREAK_EVEN_C {
                let fit = fit_alternating(&w, m, n, &learn);
                let c = fit.c_curve.last().copied().unwrap_or(0.0);
                if c > BREAK_EVEN_C {
                    let k: Vec<f64> = w.iter().zip(&fit.residual).map(|(a, r)| a - r).collect();
                    (k, c, fit.codec.atoms.len())
                } else {
                    (vec![0.0; mn], c, 0) // 달성 c 미달 -> 원자 폐기, 잔차 전용
                }
            } else {
                (vec![0.0; mn], 0.0, 0)
            };
            let (recon, rmse) = hybrid_roundtrip(&w, &k, &q4);
            let sigma = (w.iter().map(|x| x * x).sum::<f64>() / mn as f64).sqrt();
            let bits = if j_used > 0 {
                LayerCodec::code_bits_formula(m, n, j_used)
            } else {
                0
            } + 4 * mn as u64
                + 64;
            total_bits += bits;
            total_params += mn as u64;
            let recon32: Vec<f32> = recon.iter().map(|&v| v as f32).collect();
            wm_cmp.insert(name.clone(), Tensor::from_vec(recon32, (m, n), &dev)?);
            println!(
                "[보고] {} ({}x{}): 상한 = {:.3}, c = {:.4}, J = {}, 상대 RMSE = {:.4}, bpw = {:.2}, {:.0}s",
                name,
                m,
                n,
                ceiling,
                c,
                j_used,
                rmse / sigma,
                bits as f64 / mn as f64,
                t0.elapsed().as_secs_f64()
            );
        }

        // wpe: 진짜 장 (상한 0.90) — 좌표 학습 스케일업 + 잔차 3비트
        {
            let t0 = std::time::Instant::now();
            let (m, n) = wm_ref["transformer.wpe.weight"].dims2()?;
            let w = tensor_to_f64(&wm_ref["transformer.wpe.weight"])?;
            let wpe_learn = LearnConfig {
                rounds: 8,
                sgd_steps: 400,
                ..learn
            };
            let fit = fit_alternating(&w, m, n, &wpe_learn);
            let c = fit.c_curve.last().copied().unwrap_or(0.0);
            let k: Vec<f64> = w.iter().zip(&fit.residual).map(|(a, r)| a - r).collect();
            let q3 = LloydMaxQuantizer::new_gaussian(3);
            let (recon, rmse) = hybrid_roundtrip(&w, &k, &q3);
            let sigma = (w.iter().map(|x| x * x).sum::<f64>() / (m * n) as f64).sqrt();
            let j = fit.codec.atoms.len();
            let bits = LayerCodec::code_bits_formula(m, n, j) + 3 * (m * n) as u64 + 64;
            total_bits += bits;
            total_params += (m * n) as u64;
            let recon32: Vec<f32> = recon.iter().map(|&v| v as f32).collect();
            wm_cmp.insert(
                "transformer.wpe.weight".to_string(),
                Tensor::from_vec(recon32, (m, n), &dev)?,
            );
            println!(
                "[보고] transformer.wpe.weight ({}x{}): c = {:.4}, J = {}, 상대 RMSE = {:.4}, bpw = {:.2}, {:.0}s",
                m,
                n,
                c,
                j,
                rmse / sigma,
                bits as f64 / (m * n) as f64,
                t0.elapsed().as_secs_f64()
            );
        }
        println!(
            "[보고] 압축 대상 전체: {:.3} bpw, f32 대비 {:.1}:1 (wte·LN·bias 제외)",
            total_bits as f64 / total_params as f64,
            32.0 * total_params as f64 / total_bits as f64
        );

        // (2) 기준 순전파 (활성값 기록) + 압축 순전파
        let mut taps = Vec::new();
        let logits_ref = gpt2_forward(&wm_ref, &ids, Some(&mut taps))?;
        let logits_cmp = gpt2_forward(&wm_cmp, &ids, None)?;

        // (3) 게이트: 행렬별 f64 전파 상계 ||X dW||_F <= ||X||_F ||dW||_F
        for (name, x) in &taps {
            let (tt, nin) = x.dims2()?;
            let xv = tensor_to_f64(x)?;
            let wv = tensor_to_f64(&wm_ref[name])?;
            let cv = tensor_to_f64(&wm_cmp[name])?;
            let (_, nout) = wm_ref[name].dims2()?;
            let dw: Vec<f64> = wv.iter().zip(&cv).map(|(a, b)| a - b).collect();
            let mut dy2 = 0.0;
            for r in 0..tt {
                for o in 0..nout {
                    let mut acc = 0.0;
                    for i in 0..nin {
                        acc += xv[r * nin + i] * dw[i * nout + o];
                    }
                    dy2 += acc * acc;
                }
            }
            check(
                &format!("전파 상계 {}", name),
                dy2.sqrt(),
                frob(&xv) * frob(&dw) * (1.0 + bounds::f64_chain(nin as u32)),
            );
        }

        // (4) 종단 보고 (비게이트): 로짓 오차와 다음 토큰 예측 일치율
        let lr = tensor_to_f64(&logits_ref)?;
        let lc = tensor_to_f64(&logits_cmp)?;
        let rmse = (lr
            .iter()
            .zip(&lc)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / lr.len() as f64)
            .sqrt();
        let sigma_l = (lr.iter().map(|x| x * x).sum::<f64>() / lr.len() as f64).sqrt();

        let vocab = 51200;
        let top1 = |row: &[f64]| -> usize {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        let mut agree1 = 0;
        for p in 0..ids.len() {
            if top1(&lr[p * vocab..(p + 1) * vocab]) == top1(&lc[p * vocab..(p + 1) * vocab]) {
                agree1 += 1;
            }
        }
        println!(
            "[보고] 종단 로짓: RMSE = {:.4e} (로짓 RMS {:.4e} 대비 {:.4}), top-1 일치 = {}/{}",
            rmse,
            sigma_l,
            rmse / sigma_l,
            agree1,
            ids.len()
        );
        Ok(())
    }
} 