/// 한국어 LLM 처리 모듈
use crate::types::*;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// 한국어 토큰 처리기
#[derive(Debug)]
pub struct KoreanTokenProcessor {
    /// 한글 음절 매핑
    hangul_map: HashMap<char, usize>,
    /// 자모 분해 캐시
    jamo_cache: HashMap<char, Vec<char>>,
}

impl KoreanTokenProcessor {
    /// 새로운 한국어 토큰 처리기 생성
    pub fn new() -> Self {
        let mut hangul_map = HashMap::new();
        let mut base_idx = 0;
        
        // 한글 음절 범위 (가-힣)
        for ch in '가'..='힣' {
            hangul_map.insert(ch, base_idx);
            base_idx += 1;
        }
        
        Self {
            hangul_map,
            jamo_cache: HashMap::new(),
        }
    }
    
    /// 한글 음절을 자모로 분해
    pub fn decompose_hangul(&mut self, syllable: char) -> Vec<char> {
        if let Some(cached) = self.jamo_cache.get(&syllable) {
            return cached.clone();
        }
        
        if syllable < '가' || syllable > '힣' {
            return vec![syllable];
        }
        
        let code = syllable as u32 - '가' as u32;
        let cho = code / (21 * 28);
        let jung = (code % (21 * 28)) / 28;
        let jong = code % 28;
        
        let cho_chars = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 
                        'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
        let jung_chars = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
                         'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'];
        let jong_chars = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 
                         'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 
                         'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'];
        
        let mut jamos = vec![cho_chars[cho as usize], jung_chars[jung as usize]];
        if jong > 0 {
            jamos.push(jong_chars[jong as usize]);
        }
        
        self.jamo_cache.insert(syllable, jamos.clone());
        jamos
    }
    
    /// 텍스트의 한글 비율 계산
    pub fn calculate_korean_ratio(text: &str) -> f32 {
        let total_chars = text.chars().count();
        if total_chars == 0 {
            return 0.0;
        }
        
        let korean_chars = text.chars()
            .filter(|&ch| (ch >= '가' && ch <= '힣') || 
                         (ch >= 'ㄱ' && ch <= 'ㅎ') || 
                         (ch >= 'ㅏ' && ch <= 'ㅣ'))
            .count();
        
        korean_chars as f32 / total_chars as f32
    }
    
    /// 한글 텍스트 정규화
    pub fn normalize_korean_text(&self, text: &str) -> String {
        text.chars()
            .map(|ch| {
                // 전각 문자를 반각으로 변환
                match ch {
                    '０'..='９' => ((ch as u32 - '０' as u32) + '0' as u32) as u8 as char,
                    'Ａ'..='Ｚ' => ((ch as u32 - 'Ａ' as u32) + 'A' as u32) as u8 as char,
                    'ａ'..='ｚ' => ((ch as u32 - 'ａ' as u32) + 'a' as u32) as u8 as char,
                    '　' => ' ',
                    _ => ch,
                }
            })
            .collect()
    }
}

/// 한국어 모델 전처리 설정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KoreanPreprocessConfig {
    /// 최대 시퀀스 길이
    pub max_sequence_length: usize,
    /// 자모 분해 사용 여부
    pub use_jamo_decomposition: bool,
    /// 특수 토큰 추가
    pub add_special_tokens: bool,
    /// 패딩 토큰 ID
    pub pad_token_id: u32,
}

impl Default for KoreanPreprocessConfig {
    fn default() -> Self {
        Self {
            max_sequence_length: 512,
            use_jamo_decomposition: false,
            add_special_tokens: true,
            pad_token_id: 0,
        }
    }
}

/// 한국어 생성 설정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KoreanGenerationConfig {
    /// 반복 페널티 (한글은 반복이 많아 높게 설정)
    pub repetition_penalty: f32,
    /// 길이 페널티
    pub length_penalty: f32,
    /// 금지 토큰 ID들
    pub bad_words_ids: Vec<Vec<u32>>,
    /// 최소 생성 길이
    pub min_length: usize,
}

impl Default for KoreanGenerationConfig {
    fn default() -> Self {
        Self {
            repetition_penalty: 1.2, // 한글 특성상 높게 설정
            length_penalty: 1.0,
            bad_words_ids: vec![],
            min_length: 5,
        }
    }
}

/// 한국어 품질 평가 메트릭
#[derive(Debug, Serialize, Deserialize)]
pub struct KoreanQualityMetrics {
    /// 문법 점수 (0.0 ~ 1.0)
    pub grammar_score: f32,
    /// 자연스러움 점수
    pub fluency_score: f32,
    /// 한글 비율
    pub korean_ratio: f32,
    /// 반복 비율
    pub repetition_ratio: f32,
}

impl KoreanQualityMetrics {
    /// 생성된 텍스트의 품질 평가
    pub fn evaluate(text: &str) -> Self {
        let korean_ratio = KoreanTokenProcessor::calculate_korean_ratio(text);
        
        // 간단한 반복 검사
        let words: Vec<&str> = text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        let repetition_ratio = if words.is_empty() {
            0.0
        } else {
            1.0 - (unique_words.len() as f32 / words.len() as f32)
        };
        
        // 문법과 자연스러움은 간단한 휴리스틱으로 평가
        let grammar_score = if korean_ratio > 0.5 { 0.8 } else { 0.5 };
        let fluency_score = if repetition_ratio < 0.3 { 0.9 } else { 0.6 };
        
        Self {
            grammar_score,
            fluency_score,
            korean_ratio,
            repetition_ratio,
        }
    }
    
    /// 전체 품질 점수 계산
    pub fn overall_score(&self) -> f32 {
        (self.grammar_score * 0.3 + 
         self.fluency_score * 0.3 + 
         self.korean_ratio * 0.2 + 
         (1.0 - self.repetition_ratio) * 0.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hangul_decomposition() {
        let mut processor = KoreanTokenProcessor::new();
        
        // 기본 음절 분해 테스트
        let jamos = processor.decompose_hangul('한');
        assert_eq!(jamos, vec!['ㅎ', 'ㅏ', 'ㄴ']);
        
        let jamos = processor.decompose_hangul('글');
        assert_eq!(jamos, vec!['ㄱ', 'ㅡ', 'ㄹ']);
        
        // 받침 없는 경우
        let jamos = processor.decompose_hangul('가');
        assert_eq!(jamos, vec!['ㄱ', 'ㅏ']);
    }
    
    #[test]
    fn test_korean_ratio() {
        assert_eq!(KoreanTokenProcessor::calculate_korean_ratio("한글"), 1.0);
        assert_eq!(KoreanTokenProcessor::calculate_korean_ratio("English"), 0.0);
        assert_eq!(KoreanTokenProcessor::calculate_korean_ratio("한글 English"), 2.0/8.0);
    }
    
    #[test]
    fn test_text_normalization() {
        let processor = KoreanTokenProcessor::new();
        assert_eq!(processor.normalize_korean_text("ＡＢＣ"), "ABC");
        assert_eq!(processor.normalize_korean_text("１２３"), "123");
        assert_eq!(processor.normalize_korean_text("　"), " ");
    }
    
    #[test]
    fn test_quality_metrics() {
        let metrics = KoreanQualityMetrics::evaluate("안녕하세요 반갑습니다");
        assert!(metrics.korean_ratio > 0.9);
        assert!(metrics.repetition_ratio < 0.1);
        assert!(metrics.overall_score() > 0.7);
    }
} 