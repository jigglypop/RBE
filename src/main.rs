use rbe_llm::core::tensors::packed_types::*;

fn main() {
    println!("RBE 푸앵카레볼 비트도메인 성능 분석 시작...\n");
    
    // 종합 성능 리포트 실행
    generate_comprehensive_report();
    
    println!("\n개별 측정 함수들:");
    println!("1. Packed128::benchmark_speed(100_000) - 속도 측정");
    println!("2. Packed128::measure_accuracy(1_000) - 정확도 측정");  
    println!("3. Packed128::analyze_memory_efficiency() - 메모리 분석");
    println!("4. Packed128::realtime_performance_monitor(5) - 실시간 모니터링");
    
    // 추가 개별 측정 예시
    println!("\n=== 추가 개별 측정 ===");
    
    // 빠른 속도 측정
    println!("\n[빠른 속도 측정]");
    Packed128::benchmark_speed(10_000);
    
    // 정밀 정확도 측정  
    println!("\n[정밀 정확도 측정]");
    Packed128::measure_accuracy(1_000);
} 