use crate::sllm::korean_generator::*;

/// 🇰🇷 한국어 LLM 데모 실행
#[tokio::test]
async fn test_korean_llm_demo() {
    println!("\n🇰🇷 === 한국어 LLM 데모 시작 ===\n");
    
    // 전체 파이프라인 실행
    run_korean_llm_pipeline().await;
    
    println!("\n✅ 한국어 LLM 데모 완료!");
}

/// 🔧 간단한 한국어 생성 테스트
#[test]
fn test_simple_korean_generation() {
    println!("\n🔧 === 간단한 한국어 생성 테스트 ===");
    
    let generator = KoreanTextGenerator::new();
    
    let prompts = vec![
        ("안녕하세요!", "인사"),
        ("오늘 날씨 어때요?", "날씨"),
        ("리만 기저 인코딩이 뭔가요?", "기술"),
        ("한국어 AI 발전 속도가 빠르네요", "AI"),
        ("인공지능의 미래는?", "미래"),
    ];
    
    for (prompt, category) in prompts {
        println!("\n📌 카테고리: {}", category);
        println!("👤 입력: \"{}\"", prompt);
        
        let response = generator.generate(prompt, 50);
        
        println!("🤖 출력: \"{}\"", response);
        println!("✅ 한글 포함: {}", 
            if response.chars().any(|c| c >= '가' && c <= '힣') {
                "예"
            } else {
                "아니오"
            });
    }
    
    println!("\n✅ 간단한 한국어 생성 테스트 완료!");
} 