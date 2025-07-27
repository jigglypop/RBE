//! 한국어 토크나이저
//! 
//! 실제 Hugging Face 토크나이저와 연동하여 한국어 텍스트를 처리합니다.

use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// 한국어 토크나이저
#[derive(Debug)]
pub struct KoreanTokenizer {
    pub model_id: String,
    pub tokenizer: Option<Tokenizer>,
    pub vocab_size: usize,
    pub special_tokens: HashMap<String, u32>,
}

impl KoreanTokenizer {
    /// 새로운 토크나이저 생성
    pub fn new(model_id: &str) -> Self {
        let vocab_size = match model_id {
            "BM-K/KoMiniLM" => 32000,
            "skt/kogpt2-base-v2" => 51200,
            _ => 30000,
        };

        let mut special_tokens = HashMap::new();
        special_tokens.insert("[PAD]".to_string(), 0);
        special_tokens.insert("[UNK]".to_string(), 1);
        special_tokens.insert("[CLS]".to_string(), 2);
        special_tokens.insert("[SEP]".to_string(), 3);
        special_tokens.insert("[MASK]".to_string(), 4);

        Self {
            model_id: model_id.to_string(),
            tokenizer: None,
            vocab_size,
            special_tokens,
        }
    }

    /// 모델 경로에서 토크나이저 로딩
    pub async fn load_from_model_path(&mut self, model_path: &PathBuf) -> Result<()> {
        println!("🔤 토크나이저 로딩 중: {}", model_path.display());

        // 다양한 tokenizer 파일 형식 확인
        let tokenizer_path = model_path.join("tokenizer.json");
        let vocab_path = model_path.join("vocab.json");
        let merges_path = model_path.join("merges.txt");
        
        if tokenizer_path.exists() {
            // 표준 tokenizer.json 사용
            println!("  ✅ tokenizer.json 발견");
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("토크나이저 로딩 실패: {}", e))?;
            
            self.tokenizer = Some(tokenizer);
            
            // 실제 vocab 크기 업데이트
            if let Some(ref tokenizer) = self.tokenizer {
                self.vocab_size = tokenizer.get_vocab_size(true);
                println!("  📊 실제 vocab 크기: {}", self.vocab_size);
            }
        } else if vocab_path.exists() && merges_path.exists() {
            // GPT2 스타일 tokenizer 생성 (vocab.json + merges.txt)
            println!("  ✅ vocab.json + merges.txt 발견 (GPT2 스타일)");
            
            // vocab.json 로드
            let vocab_content = tokio::fs::read_to_string(&vocab_path).await?;
            let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_content)?;
            self.vocab_size = vocab.len();
            println!("  📊 실제 vocab 크기: {}", self.vocab_size);
            
            // merges.txt 로드
            let merges_content = tokio::fs::read_to_string(&merges_path).await?;
            let merges: Vec<(String, String)> = merges_content
                .lines()
                .skip(1)  // 첫 줄은 버전 정보
                .filter_map(|line| {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() == 2 {
                        Some((parts[0].to_string(), parts[1].to_string()))
                    } else {
                        None
                    }
                })
                .collect();
            println!("  📊 merges 개수: {}", merges.len());
            
            // BPE 모델 생성 - 실제로는 safetensors tokenizer.json 사용 권장
            println!("  ⚠️  GPT2 스타일 토크나이저는 제한적 지원입니다.");
            println!("  ℹ️  가능하면 tokenizer.json 형식을 사용하세요.");
            
            // 기본 토크나이저 설정
            let mut tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
            
            // GPT2 스타일 pre-tokenizer
            let pre_tokenizer = tokenizers::pre_tokenizers::byte_level::ByteLevel::default();
            tokenizer.with_pre_tokenizer(Some(pre_tokenizer));
            
            // GPT2 스타일 decoder
            let decoder = tokenizers::decoders::byte_level::ByteLevel::default();
            tokenizer.with_decoder(Some(decoder));
            
            // 특수 토큰 추가
            let special_tokens: Vec<tokenizers::AddedToken> = vec![
                tokenizers::AddedToken::from("<|endoftext|>", true),
                tokenizers::AddedToken::from("<|unk|>", true),
            ];
            
            tokenizer.add_special_tokens(&special_tokens);
            
            self.tokenizer = Some(tokenizer);
            
            // GPT2 특수 토큰 매핑
            for (token, id) in &vocab {
                if token.starts_with("<|") && token.ends_with("|>") {
                    self.special_tokens.insert(token.clone(), *id);
                    println!("  - 특수 토큰: {} -> {}", token, id);
                }
            }
        } else {
            return Err(anyhow::anyhow!("tokenizer 파일을 찾을 수 없습니다"));
        }

        // 특수 토큰 정보 업데이트
        self.update_special_tokens().await?;
        
        println!("✅ 토크나이저 로딩 완료");
        Ok(())
    }

    /// 특수 토큰 정보 업데이트
    async fn update_special_tokens(&mut self) -> Result<()> {
        if let Some(ref tokenizer) = self.tokenizer {
            // 주요 특수 토큰들 확인
            let common_special_tokens = vec![
                "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                "<pad>", "<unk>", "<s>", "</s>", "<mask>",
                "[BOS]", "[EOS]", "<bos>", "<eos>",
            ];

            for token in common_special_tokens {
                if let Some(id) = tokenizer.token_to_id(token) {
                    self.special_tokens.insert(token.to_string(), id);
                    println!("  - 특수 토큰: {} -> {}", token, id);
                }
            }
        }
        
        Ok(())
    }

    /// 텍스트를 토큰 ID로 인코딩
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if let Some(ref tokenizer) = self.tokenizer {
            let encoding = tokenizer.encode(text, false)
                .map_err(|e| anyhow::anyhow!("인코딩 실패: {}", e))?;
            
            Ok(encoding.get_ids().to_vec())
        } else {
            Err(anyhow::anyhow!("토크나이저가 로딩되지 않았습니다"))
        }
    }

    /// 토큰 ID를 텍스트로 디코딩
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        if let Some(ref tokenizer) = self.tokenizer {
            let text = tokenizer.decode(ids, false)
                .map_err(|e| anyhow::anyhow!("디코딩 실패: {}", e))?;
            
            Ok(text)
        } else {
            Err(anyhow::anyhow!("토크나이저가 로딩되지 않았습니다"))
        }
    }

    /// 배치 인코딩
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        if let Some(ref tokenizer) = self.tokenizer {
            let encodings = tokenizer.encode_batch(texts.to_vec(), false)
                .map_err(|e| anyhow::anyhow!("배치 인코딩 실패: {}", e))?;
            
            let ids_batch: Vec<Vec<u32>> = encodings
                .into_iter()
                .map(|encoding| encoding.get_ids().to_vec())
                .collect();
            
            Ok(ids_batch)
        } else {
            Err(anyhow::anyhow!("토크나이저가 로딩되지 않았습니다"))
        }
    }

    /// 배치 디코딩
    pub fn decode_batch(&self, ids_batch: &[Vec<u32>]) -> Result<Vec<String>> {
        if let Some(ref tokenizer) = self.tokenizer {
            let texts: Result<Vec<String>, _> = ids_batch
                .iter()
                .map(|ids| {
                    tokenizer.decode(ids, false)
                        .map_err(|e| anyhow::anyhow!("배치 디코딩 실패: {}", e))
                })
                .collect();
            
            texts
        } else {
            Err(anyhow::anyhow!("토크나이저가 로딩되지 않았습니다"))
        }
    }

    /// 텍스트를 토큰으로 분할 (디버깅용)
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if let Some(ref tokenizer) = self.tokenizer {
            let encoding = tokenizer.encode(text, false)
                .map_err(|e| anyhow::anyhow!("토크나이징 실패: {}", e))?;
            
            let tokens: Vec<String> = encoding.get_tokens().to_vec();
            Ok(tokens)
        } else {
            Err(anyhow::anyhow!("토크나이저가 로딩되지 않았습니다"))
        }
    }

    /// 한국어 텍스트 전처리
    pub fn preprocess_korean(&self, text: &str) -> String {
        text.trim()
            .replace("  ", " ")  // 중복 공백 제거
            .replace("ㅋㅋㅋ", "ㅋㅋ")  // 반복 자음 정리
            .replace("ㅎㅎㅎ", "ㅎㅎ")
            .replace("ㅠㅠㅠ", "ㅠㅠ")
            .replace("ㅜㅜㅜ", "ㅜㅜ")
    }

    /// 토큰 통계 분석
    pub fn analyze_text(&self, text: &str) -> Result<TokenAnalysis> {
        let preprocessed = self.preprocess_korean(text);
        let tokens = self.tokenize(&preprocessed)?;
        let ids = self.encode(&preprocessed)?;
        
        // 한국어 토큰 분석
        let korean_tokens = tokens.iter()
            .filter(|token| {
                token.chars().any(|c| c >= '가' && c <= '힣')
            })
            .count();
        
        // 영어 토큰 분석
        let english_tokens = tokens.iter()
            .filter(|token| {
                token.chars().any(|c| c.is_ascii_alphabetic())
            })
            .count();
        
        // 특수 토큰 분석
        let special_tokens = tokens.iter()
            .filter(|token| {
                token.starts_with('[') && token.ends_with(']') ||
                token.starts_with('<') && token.ends_with('>')
            })
            .count();
        
        // 서브워드 분석 (##로 시작하는 토큰들)
        let subword_tokens = tokens.iter()
            .filter(|token| token.starts_with("##"))
            .count();

        Ok(TokenAnalysis {
            original_text: text.to_string(),
            preprocessed_text: preprocessed,
            total_tokens: tokens.len(),
            total_ids: ids.len(),
            korean_tokens,
            english_tokens,
            special_tokens,
            subword_tokens,
            avg_token_length: tokens.iter().map(|t| t.len()).sum::<usize>() as f32 / tokens.len() as f32,
            tokens,
            ids,
        })
    }

    /// 토크나이저 정보
    pub fn get_info(&self) -> TokenizerInfo {
        TokenizerInfo {
            model_id: self.model_id.clone(),
            vocab_size: self.vocab_size,
            is_loaded: self.tokenizer.is_some(),
            special_tokens_count: self.special_tokens.len(),
            special_tokens: self.special_tokens.clone(),
        }
    }

    /// 토큰 ID를 토큰 문자열로 변환
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.id_to_token(id)
        } else {
            None
        }
    }

    /// 토큰 문자열을 ID로 변환
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(ref tokenizer) = self.tokenizer {
            tokenizer.token_to_id(token)
        } else {
            None
        }
    }

    /// 패딩 추가
    pub fn add_padding(&self, ids: &mut Vec<u32>, max_length: usize) {
        if let Some(pad_id) = self.special_tokens.get("[PAD]").or_else(|| self.special_tokens.get("<pad>")) {
            while ids.len() < max_length {
                ids.push(*pad_id);
            }
        } else {
            // 패딩 토큰이 없으면 0으로 패딩
            while ids.len() < max_length {
                ids.push(0);
            }
        }
    }

    /// 특별한 토큰 추가 (BOS, EOS 등)
    pub fn add_special_tokens(&self, ids: &mut Vec<u32>, add_bos: bool, add_eos: bool) {
        if add_bos {
            if let Some(bos_id) = self.special_tokens.get("[BOS]")
                .or_else(|| self.special_tokens.get("<s>"))
                .or_else(|| self.special_tokens.get("<bos>")) {
                ids.insert(0, *bos_id);
            }
        }
        
        if add_eos {
            if let Some(eos_id) = self.special_tokens.get("[EOS]")
                .or_else(|| self.special_tokens.get("</s>"))
                .or_else(|| self.special_tokens.get("<eos>")) {
                ids.push(*eos_id);
            }
        }
    }
}

/// 토큰 분석 결과
#[derive(Debug, Clone)]
pub struct TokenAnalysis {
    pub original_text: String,
    pub preprocessed_text: String,
    pub total_tokens: usize,
    pub total_ids: usize,
    pub korean_tokens: usize,
    pub english_tokens: usize,
    pub special_tokens: usize,
    pub subword_tokens: usize,
    pub avg_token_length: f32,
    pub tokens: Vec<String>,
    pub ids: Vec<u32>,
}

/// 토크나이저 정보
#[derive(Debug, Clone)]
pub struct TokenizerInfo {
    pub model_id: String,
    pub vocab_size: usize,
    pub is_loaded: bool,
    pub special_tokens_count: usize,
    pub special_tokens: HashMap<String, u32>,
}

/// 토큰 통계 (기존 호환성)
#[derive(Debug, Clone)]
pub struct TokenStats {
    pub total_tokens: usize,
    pub korean_tokens: usize,
    pub english_tokens: usize,
    pub avg_token_length: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = KoreanTokenizer::new("BM-K/KoMiniLM");
        assert_eq!(tokenizer.model_id, "BM-K/KoMiniLM");
        assert_eq!(tokenizer.vocab_size, 32000);
    }

    #[test]
    fn test_preprocess_korean() {
        let tokenizer = KoreanTokenizer::new("test");
        let processed = tokenizer.preprocess_korean("안녕하세요  ㅋㅋㅋㅋ  반갑습니다");
        assert_eq!(processed, "안녕하세요 ㅋㅋ 반갑습니다");
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = KoreanTokenizer::new("test");
        assert!(tokenizer.special_tokens.contains_key("[PAD]"));
        assert!(tokenizer.special_tokens.contains_key("[UNK]"));
    }
} 