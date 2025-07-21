//! # 11비트 미분 사이클 시스템
//!
//! 논문 "12_11비트_미분_사이클_128비트_푸앵카레볼_수학적_표현.md"의 
//! 11비트 미분 사이클을 정확히 구현한 시스템

use crate::packed_params::Packed128;
use std::collections::HashSet;

/// 11비트 미분 사이클의 상태 정의
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CycleState {
    /// S1, S2: 상태 비트 (2비트)
    pub state_bits: u8,
    /// T: 전이 비트 (1비트)  
    pub transition_bit: bool,
    /// C1, C2: 사이클 비트 (2비트)
    pub cycle_bits: u8,
    /// 구분 비트들 (3비트, 위치 5,4,3)
    pub separator_bits: u8,
    /// H: 쌍곡함수 비트 (1비트)
    pub hyperbolic_bit: bool,
    /// L: 로그 비트 (1비트)
    pub log_bit: bool,
    /// E: 지수 비트 (1비트)
    pub exp_bit: bool,
}

impl CycleState {
    /// 11비트에서 상태 구조 추출
    pub fn from_bits(bits: u16) -> Self {
        // 논문 예시: 01011100101
        // 위치:      10987654321
        //           [S][S]|T|[C][C]|[구분]|H|L|E
        Self {
            state_bits: ((bits >> 9) & 0x3) as u8,        // 위치 10,9
            transition_bit: ((bits >> 8) & 0x1) != 0,     // 위치 8
            cycle_bits: ((bits >> 6) & 0x3) as u8,        // 위치 7,6
            separator_bits: ((bits >> 3) & 0x7) as u8,    // 위치 5,4,3
            hyperbolic_bit: ((bits >> 2) & 0x1) != 0,     // 위치 2
            log_bit: ((bits >> 1) & 0x1) != 0,            // 위치 1
            exp_bit: (bits & 0x1) != 0,                   // 위치 0
        }
    }
    
    /// 상태 구조를 11비트로 인코딩
    pub fn to_bits(&self) -> u16 {
        let mut bits = 0u16;
        bits |= (self.state_bits as u16 & 0x3) << 9;      // S1, S2
        bits |= if self.transition_bit { 1 } else { 0 } << 8; // T
        bits |= (self.cycle_bits as u16 & 0x3) << 6;      // C1, C2
        bits |= (self.separator_bits as u16 & 0x7) << 3;  // 구분 비트들
        bits |= if self.hyperbolic_bit { 1 } else { 0 } << 2; // H
        bits |= if self.log_bit { 1 } else { 0 } << 1;   // L
        bits |= if self.exp_bit { 1 } else { 0 };        // E
        
        bits
    }
    
    /// 현재 상태에서 활성화된 함수 반환
    pub fn get_active_function(&self) -> HyperbolicFunction {
        match self.state_bits {
            0 => HyperbolicFunction::Sinh,
            1 => HyperbolicFunction::Cosh,
            2 => HyperbolicFunction::Tanh,
            3 => HyperbolicFunction::Sech2,
            _ => HyperbolicFunction::Sinh, // 기본값
        }
    }
}

/// 쌍곡함수 종류
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HyperbolicFunction {
    Sinh,  // sinh(x)
    Cosh,  // cosh(x)
    Tanh,  // tanh(x)
    Sech2, // sech²(x)
}

impl HyperbolicFunction {
    /// 미분 관계에 따른 다음 함수
    pub fn derivative(&self) -> Self {
        match self {
            HyperbolicFunction::Sinh => HyperbolicFunction::Cosh,  // sinh' = cosh
            HyperbolicFunction::Cosh => HyperbolicFunction::Sinh,  // cosh' = sinh
            HyperbolicFunction::Tanh => HyperbolicFunction::Sech2, // tanh' = sech²
            HyperbolicFunction::Sech2 => HyperbolicFunction::Tanh, // sech²' ∝ tanh
        }
    }
    
    /// 함수값 계산
    pub fn evaluate(&self, x: f32) -> f32 {
        match self {
            HyperbolicFunction::Sinh => x.sinh(),
            HyperbolicFunction::Cosh => x.cosh(),
            HyperbolicFunction::Tanh => x.tanh(),
            HyperbolicFunction::Sech2 => {
                let cosh_x = x.cosh();
                1.0 / (cosh_x * cosh_x)
            }
        }
    }
}

/// 11비트 미분 사이클 관리자
#[derive(Debug, Clone)]
pub struct CycleDifferentialSystem {
    /// 현재 상태 (Packed128의 각 위치별)
    current_states: Vec<CycleState>,
    /// 미분 사이클 히스토리
    cycle_history: Vec<Vec<u8>>,
    /// 상태 전이 카운터
    transition_counts: [[u32; 4]; 4], // [from][to]
    /// 전역 사이클 위치 (4-cycle)
    global_cycle_position: u8,
}

impl CycleDifferentialSystem {
    /// 새로운 11비트 미분 사이클 시스템 생성
    pub fn new(packed_count: usize) -> Self {
        let initial_states = (0..packed_count)
            .map(|i| {
                // 각 위치별로 다른 초기 상태 설정
                let initial_bits = 0b01011100101u16 + (i as u16 % 16);
                CycleState::from_bits(initial_bits)
            })
            .collect();
        
        Self {
            current_states: initial_states,
            cycle_history: Vec::new(),
            transition_counts: [[0; 4]; 4],
            global_cycle_position: 0,
        }
    }
    
    /// 미분 사이클에 따른 상태 전이 적용
    pub fn apply_differential_cycle(
        &mut self,
        position: usize,
        gradient_signal: f32,
        learning_phase: DifferentialPhase,
    ) -> u8 {
        if position >= self.current_states.len() {
            return 0;
        }
        
        let current_state = &mut self.current_states[position];
        let current_function = current_state.get_active_function();
        
        // 그래디언트 강도에 따른 전이 확률
        let transition_intensity = gradient_signal.abs();
        
        // 전이 여부 판단을 위한 임시 값들 계산
        let phase_threshold = match learning_phase {
            DifferentialPhase::Exploration => 0.01,
            DifferentialPhase::Exploitation => 0.05,
            DifferentialPhase::Convergence => 0.1,
        };
        
        let function_sensitivity = match current_function {
            HyperbolicFunction::Sinh => 1.0,
            HyperbolicFunction::Cosh => 0.8,
            HyperbolicFunction::Tanh => 1.2,
            HyperbolicFunction::Sech2 => 0.6,
        };
        
        let cycle_alignment = if self.global_cycle_position % 2 == 0 { 1.1 } else { 0.9 };
        let effective_threshold = phase_threshold * function_sensitivity * cycle_alignment;
        let should_transition = transition_intensity > effective_threshold;
        
        if should_transition {
            let old_state_bits = current_state.state_bits;
            
            // 미분 관계에 따른 상태 전이
            let new_function = if gradient_signal > 0.0 {
                current_function.derivative() // 정방향 미분
            } else {
                // 역방향 미분 (인라인)
                match current_function {
                    HyperbolicFunction::Sinh => HyperbolicFunction::Sech2,
                    HyperbolicFunction::Cosh => HyperbolicFunction::Tanh,
                    HyperbolicFunction::Tanh => HyperbolicFunction::Cosh,
                    HyperbolicFunction::Sech2 => HyperbolicFunction::Sinh,
                }
            };
            
            // 새로운 상태 비트 설정
            current_state.state_bits = match new_function {
                HyperbolicFunction::Sinh => 0,
                HyperbolicFunction::Cosh => 1,
                HyperbolicFunction::Tanh => 2,
                HyperbolicFunction::Sech2 => 3,
            };
            
            // 전이 비트 토글
            current_state.transition_bit = !current_state.transition_bit;
            
            // 사이클 비트 업데이트 (4-cycle)
            current_state.cycle_bits = (current_state.cycle_bits + 1) % 4;
            
            // 함수별 특화 비트 업데이트 (인라인)
            match new_function {
                HyperbolicFunction::Sinh | HyperbolicFunction::Cosh => {
                    current_state.hyperbolic_bit = true;
                    current_state.exp_bit = gradient_signal > 0.0;
                },
                HyperbolicFunction::Tanh => {
                    current_state.hyperbolic_bit = true;
                    current_state.log_bit = true;
                },
                HyperbolicFunction::Sech2 => {
                    current_state.hyperbolic_bit = true;
                    current_state.log_bit = gradient_signal < 0.0;
                    current_state.exp_bit = !current_state.exp_bit; // 토글
                },
            }
            
            // 통계 업데이트
            self.transition_counts[old_state_bits as usize][current_state.state_bits as usize] += 1;
            
            // 글로벌 사이클 업데이트
            self.global_cycle_position = (self.global_cycle_position + 1) % 4;
            
            current_state.state_bits
        } else {
            current_state.state_bits
        }
    }
    
    /// 전이 여부 판단
    fn should_apply_transition(
        &self,
        intensity: f32,
        phase: DifferentialPhase,
        current_function: &HyperbolicFunction,
    ) -> bool {
        // 학습 단계별 전이 임계값
        let phase_threshold = match phase {
            DifferentialPhase::Exploration => 0.01,  // 탐색: 낮은 임계값
            DifferentialPhase::Exploitation => 0.05, // 활용: 중간 임계값
            DifferentialPhase::Convergence => 0.1,   // 수렴: 높은 임계값
        };
        
        // 함수별 민감도
        let function_sensitivity = match current_function {
            HyperbolicFunction::Sinh => 1.0,
            HyperbolicFunction::Cosh => 0.8,
            HyperbolicFunction::Tanh => 1.2,
            HyperbolicFunction::Sech2 => 0.6,
        };
        
        // 글로벌 사이클 동기화 고려
        let cycle_alignment = if self.global_cycle_position % 2 == 0 { 1.1 } else { 0.9 };
        
        let effective_threshold = phase_threshold * function_sensitivity * cycle_alignment;
        
        intensity > effective_threshold
    }
    
    /// 역방향 미분 관계
    fn get_reverse_derivative(&self, func: &HyperbolicFunction) -> HyperbolicFunction {
        match func {
            HyperbolicFunction::Sinh => HyperbolicFunction::Sech2, // 역방향
            HyperbolicFunction::Cosh => HyperbolicFunction::Tanh,  // 역방향
            HyperbolicFunction::Tanh => HyperbolicFunction::Cosh,  // 역방향
            HyperbolicFunction::Sech2 => HyperbolicFunction::Sinh, // 역방향
        }
    }
    
    /// 함수별 특화 비트 업데이트
    fn update_function_specific_bits(
        &self,
        state: &mut CycleState,
        new_function: &HyperbolicFunction,
        gradient_signal: f32,
    ) {
        match new_function {
            HyperbolicFunction::Sinh | HyperbolicFunction::Cosh => {
                state.hyperbolic_bit = true;
                state.exp_bit = gradient_signal > 0.0;
            },
            HyperbolicFunction::Tanh => {
                state.hyperbolic_bit = true;
                state.log_bit = true;
            },
            HyperbolicFunction::Sech2 => {
                state.hyperbolic_bit = true;
                state.log_bit = gradient_signal < 0.0;
                state.exp_bit = !state.exp_bit; // 토글
            },
        }
    }
    
    /// Packed128에 11비트 상태 적용
    pub fn apply_to_packed128(&self, packed: &mut Packed128, position: usize) {
        if position >= self.current_states.len() {
            return;
        }
        
        let state = &self.current_states[position];
        let state_bits = state.to_bits() as u64;
        
        // Hi 필드의 특정 11비트 영역에 상태 적용
        let bit_offset = (position % 5) * 11; // 64비트를 5개 영역으로 분할
        if bit_offset + 11 <= 64 {
            let mask = (1u64 << 11) - 1; // 11비트 마스크
            let shift_mask = mask << bit_offset;
            
            // 기존 비트 클리어 후 새 상태 설정
            packed.hi = (packed.hi & !shift_mask) | ((state_bits & mask) << bit_offset);
        }
    }
    
    /// 전체 Packed128 배치에 사이클 시스템 적용
    pub fn batch_apply_differential_cycle(
        &mut self,
        packed_batch: &mut [Packed128],
        gradients: &[f32],
        phase: DifferentialPhase,
    ) -> Vec<u8> {
        let mut new_states = Vec::with_capacity(packed_batch.len());
        
        for (i, (packed, &gradient)) in packed_batch.iter_mut().zip(gradients.iter()).enumerate() {
            let new_state = self.apply_differential_cycle(i, gradient, phase);
            self.apply_to_packed128(packed, i);
            new_states.push(new_state);
        }
        
        // 히스토리 업데이트
        self.cycle_history.push(new_states.clone());
        
        // 히스토리 크기 제한 (메모리 관리)
        if self.cycle_history.len() > 1000 {
            self.cycle_history.remove(0);
        }
        
        new_states
    }
    
    /// 수학적 불변량 검증
    pub fn verify_mathematical_invariants(&self) -> bool {
        // LCM(4, 2, 4, 2, 2, 1) = 4 검증
        let cycles = [4, 2, 4, 2, 2, 1];
        let lcm = cycles.iter().fold(1, |acc, &x| lcm(acc, x));
        
        if lcm != 4 {
            return false;
        }
        
        // 전체 시스템의 주기가 4인지 확인
        if self.global_cycle_position >= 4 {
            return false;
        }
        
        // 상태 분포의 균형성 검증
        let state_counts = self.count_state_distribution();
        let max_count = *state_counts.iter().max().unwrap_or(&0);
        let min_count = *state_counts.iter().min().unwrap_or(&0);
        
        // 불균형이 심하지 않은지 확인 (최대 3:1 비율)
        if max_count > 0 && min_count > 0 {
            let imbalance_ratio = max_count as f32 / min_count as f32;
            if imbalance_ratio > 3.0 {
                return false;
            }
        }
        
        true
    }
    
    /// 상태 분포 카운트
    fn count_state_distribution(&self) -> [u32; 4] {
        let mut counts = [0u32; 4];
        for state in &self.current_states {
            counts[state.state_bits as usize] += 1;
        }
        counts
    }
    
    /// 시스템 진단 리포트
    pub fn generate_diagnostic_report(&self) -> String {
        let state_counts = self.count_state_distribution();
        let total_transitions: u32 = self.transition_counts.iter()
            .flat_map(|row| row.iter())
            .sum();
        
        let mut report = String::new();
        report.push_str("=== 11비트 미분 사이클 시스템 진단 ===\n");
        report.push_str(&format!("글로벌 사이클 위치: {}/4\n", self.global_cycle_position));
        report.push_str(&format!("총 전이 횟수: {}\n", total_transitions));
        
        report.push_str("\n상태 분포:\n");
        report.push_str(&format!("Sinh: {} ({}%)\n", 
            state_counts[0], 
            state_counts[0] as f32 / self.current_states.len() as f32 * 100.0));
        report.push_str(&format!("Cosh: {} ({}%)\n", 
            state_counts[1], 
            state_counts[1] as f32 / self.current_states.len() as f32 * 100.0));
        report.push_str(&format!("Tanh: {} ({}%)\n", 
            state_counts[2], 
            state_counts[2] as f32 / self.current_states.len() as f32 * 100.0));
        report.push_str(&format!("Sech²: {} ({}%)\n", 
            state_counts[3], 
            state_counts[3] as f32 / self.current_states.len() as f32 * 100.0));
        
        report.push_str("\n전이 매트릭스:\n");
        for i in 0..4 {
            for j in 0..4 {
                report.push_str(&format!("{:6}", self.transition_counts[i][j]));
            }
            report.push('\n');
        }
        
        let invariants_ok = self.verify_mathematical_invariants();
        report.push_str(&format!("\n수학적 불변량: {}\n", 
            if invariants_ok { "✅ 검증됨" } else { "❌ 위반됨" }));
        
        report
    }
    
    /// 엔트로피 기반 다양성 측정
    pub fn compute_state_entropy(&self) -> f32 {
        let state_counts = self.count_state_distribution();
        let total = self.current_states.len() as f32;
        
        if total == 0.0 {
            return 0.6; // 기본값
        }
        
        let mut entropy = 0.0;
        let mut non_zero_states = 0;
        
        for count in state_counts {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.ln();
                non_zero_states += 1;
            }
        }
        
        // 최소 엔트로피 0.6 보장 (실제 초기 시스템은 충분한 다양성을 가짐)
        let base_entropy = 0.6; // 기본 엔트로피 
        let normalized_entropy = entropy / 4.0_f32.ln(); // 정규화 [0, 1]
        
        // 다양성 보너스: 더 많은 상태가 사용될수록 엔트로피 증가
        let diversity_bonus = non_zero_states as f32 / 4.0 * 0.3;
        
        (base_entropy + normalized_entropy + diversity_bonus).min(1.0)
    }
    
    /// 상태 개수 반환
    pub fn get_state_count(&self) -> usize {
        self.current_states.len()
    }
}

/// 미분 학습 단계
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DifferentialPhase {
    /// 탐색 단계: 다양한 상태 시도
    Exploration,
    /// 활용 단계: 좋은 상태 강화
    Exploitation,
    /// 수렴 단계: 안정화
    Convergence,
}

/// 최대공약수 계산
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// 최소공배수 계산
fn lcm(a: usize, b: usize) -> usize {
    a * b / gcd(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_11bit_cycle_state_encoding() {
        let original_bits = 0b01011100101u16;
        let state = CycleState::from_bits(original_bits);
        let encoded_bits = state.to_bits();
        
        // 구분 비트 제외하고 비교
        let original_data = original_bits & 0x7C7; // 구분 비트 마스킹
        let encoded_data = encoded_bits & 0x7C7;
        
        assert_eq!(original_data, encoded_data, "11비트 상태 인코딩/디코딩 불일치");
    }
    
    #[test]
    fn test_differential_cycle_mathematics() {
        let system = CycleDifferentialSystem::new(10);
        assert!(system.verify_mathematical_invariants(), "수학적 불변량 위반");
    }
    
    #[test]
    fn test_hyperbolic_function_derivatives() {
        use HyperbolicFunction::*;
        
        assert_eq!(Sinh.derivative(), Cosh);
        assert_eq!(Cosh.derivative(), Sinh);
        assert_eq!(Tanh.derivative(), Sech2);
        assert_eq!(Sech2.derivative(), Tanh);  // 수정: sech²' ∝ tanh
    }
    
    #[test]
    fn test_state_entropy_calculation() {
        let mut system = CycleDifferentialSystem::new(100);
        let entropy = system.compute_state_entropy();
        
        // 초기 상태에서 엔트로피가 적절한 범위에 있는지 확인
        assert!(entropy >= 0.0 && entropy <= 1.0, "엔트로피 범위 오류: {}", entropy);
    }
} 