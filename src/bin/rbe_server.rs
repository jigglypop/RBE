use rbe_llm::sllm::{RBEApiServer, ServerConfig};
use clap::{Arg, Command};
use std::process;

#[tokio::main]
async fn main() {
    // 로거 초기화
    env_logger::init();
    
    println!("🚀 === RBE API 서버 시작 ===");
    
    // 명령행 인자 파싱
    let matches = Command::new("RBE API Server")
        .version("1.0.0")
        .about("푸앵카레 볼 기반 RBE 모델 API 서버")
        .arg(
            Arg::new("host")
                .long("host")
                .value_name("HOST")
                .help("서버 호스트 주소")
                .default_value("0.0.0.0")
        )
        .arg(
            Arg::new("port")
                .long("port")
                .short('p')
                .value_name("PORT")
                .help("서버 포트 번호")
                .default_value("8080")
        )
        .arg(
            Arg::new("max-tokens")
                .long("max-tokens")
                .value_name("TOKENS")
                .help("최대 생성 토큰 수")
                .default_value("100")
        )
        .arg(
            Arg::new("temperature")
                .long("temperature")
                .value_name("TEMP")
                .help("기본 temperature 값")
                .default_value("0.7")
        )
        .arg(
            Arg::new("top-p")
                .long("top-p")
                .value_name("TOP_P")
                .help("기본 top-p 값")
                .default_value("0.9")
        )
        .arg(
            Arg::new("no-cors")
                .long("no-cors")
                .help("CORS 비활성화")
                .action(clap::ArgAction::SetTrue)
        )
        .get_matches();
    
    // 서버 설정 구성
    let config = ServerConfig {
        host: matches.get_one::<String>("host").unwrap().clone(),
        port: matches.get_one::<String>("port").unwrap().parse().unwrap_or(8080),
        max_tokens: matches.get_one::<String>("max-tokens").unwrap().parse().unwrap_or(100),
        default_temperature: matches.get_one::<String>("temperature").unwrap().parse().unwrap_or(0.7),
        default_top_p: matches.get_one::<String>("top-p").unwrap().parse().unwrap_or(0.9),
        enable_cors: !matches.get_flag("no-cors"),
    };
    
    println!("⚙️ 서버 설정:");
    println!("   주소: {}:{}", config.host, config.port);
    println!("   최대 토큰: {}", config.max_tokens);
    println!("   Temperature: {}", config.default_temperature);
    println!("   Top-p: {}", config.default_top_p);
    println!("   CORS: {}", if config.enable_cors { "활성화" } else { "비활성화" });
    
    // API 서버 생성 및 시작
    let server = RBEApiServer::new(config);
    
    println!("\n🌟 API 엔드포인트:");
    println!("   GET  /health           - 헬스체크");
    println!("   GET  /model/info       - 모델 정보");
    println!("   POST /model/load       - 모델 로딩");
    println!("   POST /model/compress   - 모델 압축");
    println!("   POST /generate         - 텍스트 생성");
    println!("   POST /benchmark        - 벤치마크");
    println!("   GET  /docs/*           - API 문서");
    
    println!("\n🔗 사용 예시:");
    println!("   curl http://localhost:8080/health");
    println!("   curl -X POST http://localhost:8080/generate \\");
    println!("     -H 'Content-Type: application/json' \\");
    println!("     -d '{{\"prompt\": \"안녕하세요\", \"max_tokens\": 50}}'");
    
    // 서버 시작
    if let Err(e) = server.start().await {
        eprintln!("❌ 서버 시작 실패: {}", e);
        process::exit(1);
    }
} 