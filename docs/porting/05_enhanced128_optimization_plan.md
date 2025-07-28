# Enhanced128 최적화 및 성능 개선 계획서

## 1. 프로젝트 개요

### 1.1 현황 분석
- **달성**: Enhanced128 기본 구조 완성, 8/12 기저함수 수렴 성공
- **문제**: 기저함수 1,5,8,11 수렴 실패, DifferentialSystem 0 고착
- **목표**: 모든 기저함수 RMSE < 0.001, 순환주기 최적화로 속도 향상

### 1.2 핵심 목표
1. **정확도**: 모든 12개 기저함수 완벽 수렴 (RMSE < 0.001)
2. **속도**: 순전파 > 5 Mop/s, 역전파 > 2 Mop/s
3. **압축**: 128비트/가중치 유지하면서 성능 개선
4. **주기성**: Periodic-Block 학습으로 80% 시간 절감

## 2. 기술적 문제 진단

### 2.1 그래디언트 정확도 문제
```
현재 상태:
- 수치미분 (h=1e-6) → 기울기 노이즈 발생
- 복잡한 기저함수 (Bessel, Morlet)에서 부정확
- 업데이트 후 파라미터 비트 포화 현상

해결책:
- Analytic Gradient 폐쇄형 수식 도입
- Q16.16 고정소수점 LUT 적용
- 기저함수별 특화된 미분 엔진
```

### 2.2 인코딩 정밀도 부족
```
현재 상태:
- log2_c: 3비트 (±4) → 특수기저에 비해 해상도 부족
- 복잡한 곡률 변화 표현 한계

해결책:
- log2_c 비트폭 3→5비트 (±16)
- hi 필드 재배치로 정밀도 확장
- 마이그레이션 도구 제공
```

### 2.3 옵티마이저 한계
```
현재 상태:
- 단일 β₁=0.9, β₂=0.999 → 기저별 최적값 상이
- Gradient Clipping 없음 → 오버슈트 발생

해결책:
- 기저함수별 적응적 하이퍼파라미터
- 자동 그래디언트 클리핑
- RMSE 추세 기반 스케줄링
```

### 2.4 순환주기 미활용
```
현재 상태:
- (i,j) 격자 전체 1-패스 학습
- 시드함수 주기성 f(i,j)=f(i+period_r,j+period_θ) 무시

해결책:
- 주기 블록 자동 탐지
- 블록 단위 학습 후 대칭 복사
- 계산량 80% 절감 가능
```

## 3. 세부 기술 방안

### 3.1 Analytic Gradient 모듈

#### 3.1.1 구조 설계
```rust
// src/core/tensors/analytic_grad.rs
pub struct AnalyticGradient {
    basis_lut: [[i16; 256]; 256],  // [r_idx][theta_idx] → grad_r
    theta_lut: [[i16; 256]; 256],  // [r_idx][theta_idx] → grad_theta
}

impl AnalyticGradient {
    fn grad_r_bessel_j0(r: f32, theta: f32) -> f32;
    fn grad_theta_morlet(r: f32, theta: f32, omega: f32) -> f32;
    // ... 12개 기저함수별 특화 구현
}
```

#### 3.1.2 기저함수별 폐쇄형 미분
```
기저 0-3: 삼각함수 × 쌍곡함수 → 곱의 법칙
기저 4:   J₀(x) → J₁(x) 관계 활용
기저 5:   I₀(x) → I₁(x) 베셀 변형
기저 6:   K₀(x) → K₁(x) 수정 베셀
기저 7:   Y₀(x) → Y₁(x) 뉴만 함수
기저 8:   tanh(x) → sech²(x) 쌍곡미분
기저 9:   sech(x) × triangle → 복합미분
기저 10:  exp(-x) × sin(θ) → 지수삼각미분
기저 11:  Morlet(r,θ,ω) → 가우시안×코사인 복합
```

#### 3.1.3 LUT 생성 알고리즘
```
1. r ∈ [0, 0.999999] → 256 샘플
2. θ ∈ [0, 2π) → 256 샘플
3. 각 (r,θ)에서 analytic grad 계산
4. Q16.16 고정소수점 변환
5. 256×256 테이블 사전 구축
```

### 3.2 인코딩 업그레이드

#### 3.2.1 비트필드 재설계
```
현재 Enhanced128:
lo[63-40]: r (24bit)
lo[39-12]: theta (28bit)
lo[11-8]:  basis_id (4bit)
lo[7-4]:   rot_code (4bit)
lo[3-1]:   log2_c (3bit) ← 확장 필요
lo[0]:     d_r (1bit)

신규 Enhanced128:
lo[63-40]: r (24bit)
lo[39-12]: theta (28bit)
lo[11-8]:  basis_id (4bit)
lo[7-4]:   rot_code (4bit)
lo[3-0]:   reserved (4bit)

hi[63-59]: log2_c (5bit) ← 확장됨 ±16
hi[58]:    d_r (1bit)
hi[57-0]:  metadata/cycle
```

#### 3.2.2 마이그레이션 도구
```rust
// src/core/tensors/migration.rs
impl Enhanced128 {
    fn migrate_from_v1(old: &Enhanced128V1) -> Enhanced128V2;
    fn batch_migrate(seeds: &[Enhanced128V1]) -> Vec<Enhanced128V2>;
}
```

### 3.3 Periodic-Block 학습 엔진

#### 3.3.1 주기 탐지 알고리즘
```rust
fn detect_periods(basis_id: u8, rows: usize, cols: usize) -> (usize, usize) {
    let period_r = match basis_id {
        0..=3 => gcd(rows, 16),      // 삼각함수 기본주기
        4..=7 => gcd(rows, 32),      // 베셀함수 진동주기
        8..=11 => gcd(rows, 64),     // 복합함수 장주기
        _ => rows,
    };
    let period_theta = gcd(cols, 2 * PI_SAMPLES);
    (period_r, period_theta)
}
```

#### 3.3.2 블록 학습 최적화
```rust
impl DifferentialSystem {
    fn forward_block<T: RBESeed>(
        &self,
        seed: &T,
        block_r: usize,
        block_theta: usize,
        period_r: usize,
        period_theta: usize,
    ) -> Vec<f32>;
    
    fn backward_block<T: RBESeed>(
        &mut self,
        seed: &mut T,
        block_values: &[f32],
        targets: &[f32],
        period_r: usize,
        period_theta: usize,
    );
}
```

### 3.4 Adaptive Optimizer Layer

#### 3.4.1 기저별 하이퍼파라미터 테이블
```rust
#[derive(Debug, Clone)]
pub struct BasisProfile {
    beta1: f32,        // 1차 모멘트 계수
    beta2: f32,        // 2차 모멘트 계수
    epsilon: f32,      // 수치 안정성
    grad_clip: f32,    // 그래디언트 클리핑
    lr_scale: f32,     // 학습률 스케일
}

pub struct BitAdamStateExt {
    profiles: [BasisProfile; 12],  // 기저별 프로파일
    rmse_history: VecDeque<f32>,   // RMSE 추세
    adaptive_schedule: bool,        // 적응적 스케줄링
}
```

#### 3.4.2 적응적 스케줄링
```
RMSE 증가 추세 감지:
- 연속 100 iter RMSE 상승 → β₂ *= 0.95, grad_clip *= 1.1
- 연속 500 iter 평탄 → lr_scale *= 0.8
- 목표 달성 시 → 모든 파라미터 고정

기저별 특화:
- 베셀(4-7): β₂=0.995, grad_clip=0.1
- 복합(8-11): β₂=0.99, grad_clip=0.05
- 기본(0-3): β₂=0.999, grad_clip=1.0
```

### 3.5 Hybrid 저장 시스템

#### 3.5.1 압축 파이프라인 개선
```rust
pub enum WeightSeed {
    Standard(Packed128),
    Enhanced(Enhanced128),
    Hybrid(Vec<(usize, SeedType)>),  // 레이어별 최적 선택
}

impl WeightCompressor {
    fn compress_adaptive(&self, weights: &[f32]) -> WeightSeed {
        // 1. Enhanced128 시도
        let (enhanced, rmse_e) = self.try_enhanced(weights);
        if rmse_e < 0.001 { return WeightSeed::Enhanced(enhanced); }
        
        // 2. Packed128 시도
        let (packed, rmse_p) = self.try_packed(weights);
        if rmse_p < 0.001 { return WeightSeed::Standard(packed); }
        
        // 3. Hybrid 구성
        WeightSeed::Hybrid(self.build_hybrid(weights))
    }
}
```

## 4. 개발 일정

### 4.1 1주차: 기반 기술 구현
```
Day 1-2: Analytic Gradient 모듈
- 12개 기저함수 폐쇄형 미분 구현
- LUT 생성 및 Q16.16 최적화
- 단위 테스트 (정확도 검증)

Day 3-4: 인코딩 업그레이드
- 비트필드 재설계 (log2_c 확장)
- Enhanced128V2 구조체 정의
- 마이그레이션 도구 구현

Day 5-6: Forward/Backward 통합
- AnalyticGradient → BitForward 라우팅
- BitBackward에 새 grad 엔진 적용
- 기존 시스템과 호환성 유지

Day 7: 1차 검증
- 기저별 수렴 테스트 (≤5000 iter)
- 성능 벤치마크 (속도 측정)
```

### 4.2 2주차: 최적화 및 통합
```
Day 8-9: Periodic-Block 엔진
- 주기 탐지 알고리즘 구현
- 블록 단위 forward/backward
- 성능 최적화 (SIMD, 캐시)

Day 10: Adaptive Optimizer
- 기저별 프로파일 테이블
- RMSE 추세 기반 스케줄링
- 자동 하이퍼파라미터 조정

Day 11: Hybrid 시스템
- WeightSeed::Hybrid 구현
- 적응적 압축 파이프라인
- 추론 시 분기 최적화

Day 12-13: 통합 테스트
- 10000 iter, 12 기저 모두 수렴 검증
- 메모리/속도/정확도 종합 평가
- 실제 NLP 모델 적용 테스트

Day 14: 문서화
- 성능 보고서 작성
- API 문서 업데이트
- 사용자 가이드 정리
```

## 5. 성공 지표

### 5.1 정량적 목표
```
정확도:
- 모든 12개 기저함수 RMSE < 0.001
- 5000 iter 내 수렴 달성률 100%
- DifferentialSystem 0 고착 해결

성능:
- 순전파 속도: >5 Mop/s (현재 대비 2배)
- 역전파 속도: >2 Mop/s (현재 대비 3배)
- 메모리 사용량: 128 bit/weight 유지

압축:
- 압축률: >150:1 (목표 달성)
- 모델 정확도 손실: <0.1%
- 추론 지연시간: <기존 시스템 110%
```

### 5.2 정성적 목표
```
안정성:
- 모든 기저함수에서 일관된 수렴
- 하이퍼파라미터 민감도 최소화
- 다양한 모델 크기에서 검증

확장성:
- 새로운 기저함수 추가 용이
- 다른 아키텍처 적용 가능
- 멀티 GPU 스케일링 지원
```

## 6. 리스크 관리

### 6.1 기술적 리스크
```
특수함수 구현 오차:
- 완화: 표준 라이브러리 대비 정확도 검증
- 백업: GSL, Boost Math 폴백 옵션

비트필드 호환성:
- 완화: 자동 마이그레이션 도구
- 백업: V1/V2 동시 지원 기간

성능 회귀:
- 완화: 각 단계별 벤치마크
- 백업: 기존 시스템 롤백 가능
```

### 6.2 일정 리스크
```
복잡도 증가:
- 완화: 단계별 검증점 설정
- 백업: 핵심 기능 우선 구현

통합 이슈:
- 완화: 지속적 통합 테스트
- 백업: 모듈별 독립 배포
```

## 7. 결론

본 계획서는 Enhanced128의 근본적 한계를 해결하여 RBE 시스템의 완전체를 구현하는 것을 목표로 합니다. 

**핵심 혁신 요소:**
1. **Analytic Gradient**: 수치미분 → 폐쇄형 해석적 미분
2. **Adaptive Optimization**: 기저별 특화 최적화
3. **Periodic Learning**: 주기성 활용한 계산량 혁신
4. **Hybrid Storage**: 요구 정확도별 최적 압축

이를 통해 **압축률 150:1, 정확도 RMSE<0.001, 속도 5Mop/s**를 동시에 달성하여 RBE의 3대 목표를 완전히 실현할 수 있을 것입니다. 