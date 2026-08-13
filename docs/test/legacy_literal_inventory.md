# 레거시 리터럴 허용오차 목록 (145건) — 교체 승인 요청서

작성일: 2026-08-13. 하네스 8절: 기존의 느슨한 리터럴 허용오차 테스트는 bounds 유도로 대체해야 하며, 테스트 수치 변경이므로 항목별 사용자 승인 후 진행한다. 본 문서는 그 승인 요청서다. 검출 기준은 scripts/harness_lint.py W2 (verification 미사용 테스트 파일의 assert 내 부동소수점 리터럴 비교).

## 1. 분류 요약과 제안

| 분류 | 건수(약) | 성격 | 제안 |
|---|---|---|---|
| A. 수학적 범위 불변식 | 약 70 | `0 <= x <= 1.0` (확률·엔트로피·캐시적중률), `r < 1.0` (푸앵카레 볼), `\|sin\| <= 1.0` 등 — 실측에 맞춘 허용오차가 아니라 정의역 자체 | 교체 불필요. lint 를 정련해 범위 불변식 패턴을 W2 에서 제외 (lint 규칙 변경 승인 필요) |
| B. 실질 허용오차 | 약 55 | `1e-6`, `1e-5`, `0.01`, `0.1`, `0.2` 류 RMSE·오차 비교 — 유도 근거 없이 실측에 맞춘 값일 위험 | bounds 유도로 항목별 교체 (f32 전파는 dot_product/f64_chain, 양자화는 phase_quant/amp_quant, 통계는 rmse_ci_rel). 도메인 리팩토링 시 일괄 진행 제안 |
| C. 벽시계 게이트 | 8 | `< 50.0` ns/weight, `< 1.0` s, `< 0.1` s, `< 10.0` ms — 하네스 원칙(벽시계 게이트 금지, L0 카운터가 게이트) 위반 | 결정론 카운터 게이트로 교체하거나 보고 전용으로 강등 |
| D. 하드코딩 기대값 | 3 | performance_test.rs 의 `34.133` 등 특정 실측값 고정 | 출처 검증 후 공식 유도값으로 교체 또는 삭제 (규칙 2 위험) |

우선순위 제안: C(원칙 위반) > D(조작 위험) > B(정밀화) > A(lint 정련만).

## 2. 전체 목록 (파일: 건수, 행: 값)

### core/optimizers (16건)
- riemannian_adam_test.rs (14): 35,36 `1e-6` 계량 정확성 / 67 `0.1` 뫼비우스 / 71,101,134,156,185 `1.0` 범위(A) / 79,107 `1e-6` / 113 `0.01` 선형근사 / 135,186 `2.0` 범위(A) / 297 `1e-4` 근사오차
- adam_test.rs (1): 57 `0.5` 수렴(B) | performance_benchmark_test.rs (1): 134 `0.9999999` 경계(A)

### core/differential (42건)
- backward_test.rs (10): 62,312 `1.0` 범위(A) / 93 `1.0`(B) / 202,203 `1.0` 클리핑(A) / 231 `0.9999999`(A) / 270,294,295,309 `1e-6`(B)
- cycle_system_test.rs (9): 38,42,46,50 `1e-6` 쌍곡함수(B) / 59,80,213,215,295 `1.0` 엔트로피 범위(A)
- state_transition_test.rs (8): 150 `1e-6`(B) / 175,177,178,259,303,425,457 `1.0` 범위(A)
- forward_test.rs (8): 37,271 `10.0` 클램핑(A) / 134 `1.5`, 139 `1.0`, 160 `2.0` 변조 범위(A) / 187,210,212 `1.0` 범위(A)
- unified_system_test.rs (7): 13,71,131,132,134,520 `1.0` 범위(A) / 43 `10.0` 클램핑(A)

### core/generator (10건)
- weight_generator_test.rs (9): 51,95,153,161,165,169 `1.0` 범위(A) / 112 `1e-10` 결정론(B) / 274 `50.0` ns 벽시계(C) / 311 `10.0` 범위(A)
- poincare_learning_test.rs (1): 176 `2.0` 수렴(B)

### core/encoder (12건)
- enhanced_encoding_test.rs (5): 228 `1e-5` SIMD 일치(B) / 358 `0.1`, 360 `0.05`, 362 `0.01`, 450 `0.01` RMSE(B)
- encoder_test.rs (4): 138 `0.001`, 375,608,663 `0.1` RMSE(B)
- int_adam_test.rs (3): 169 `1e-5`, 239 `1e-6`, 293 `5e-6` RMSE(B)

### core/decoder (10건)
- performance_benchmark_test.rs (6): 87 `0.01`, 131 `0.3`, 252,387 `0.1` RMSE(B) / 88 `50.0` us 벽시계(C) / 321 `1e-6` 병렬 일치(B)
- weight_generator_test.rs (2): 35 `10.0`(A) / 120 `1e-6`(B)
- cache_comparison_test.rs (1): 257 `0.15` RMSE(B) | fused_forward_test.rs (1): 92 `1e-6`(B)

### core/matrix (7건)
- hierarchical_matrix_test.rs (2): 32 `1.0` s, 55 `0.1` s 벽시계(C)
- mod.rs (2): 160 `0.5` s, 161 `0.1` s 벽시계(C)
- error_controller_test.rs (1): 47 `1.0`(B) | quality_test.rs (1): 25 `20~60` PSNR 범위(B)

### core/systems (10건)
- compute_engine_test.rs (3): 74,78 `1.0` sin/cos 범위(A) / 247 `1.0`(A)
- config_test.rs (3): 198,201,202 `1.0` 학습률 범위(A)
- performance_test.rs (3): 89,216 `0.001` 압축률 대조(D) / 130 `0.001` — `34.133` 하드코딩 기대값(D)
- state_management_test.rs (1): 203 `0.01` 합 검사(B)

### core/math (4건)
- poincare_test.rs (2): 31,32 `0.01` 좌표(B) | bessel_test.rs (1): 63 `0.1`(B) | fused_ops_test.rs (1): 252 `100.0` MSE(B)

### core/packed_params (5건)
- packed_types_test.rs (5): 32,33 `0.1` 양자화 왕복(B — phase_quant/amp_quant 유도 교체 적합) / 56,102,103 `1.0` 범위(A)

### nlp (29건)
- rbe_layernorm_test.rs (12): 25,32 `0.01` 정규화 통계(B) / 98,135,137,139,237,238,241,242 `1e-6`, 177,209 `1e-5` (B — f32 전파 유도 교체 적합)
- rbe_softmax_test.rs (6): 19,43,91,164 `1e-5` 합=1 (B — dot_product(n) 유도 적합) / 38 `1.0` 범위(A) / 212 `1e-10`(B)
- rbe_embedding_test.rs (3): 56 `2.0`(A) / 137,239 `0.1`(B)
- rbe_ffn_test.rs (3): 61 `10.0`(A) / 166 `1e-6`(B) / 213 `0.5`(B)
- rbe_rmsnorm_test.rs (3): 33,75 `0.2`(B — 하네스 8절이 명시한 교체 대상 예시) / 151 `1.0`(A)
- rbe_attention_test.rs (1): 65 `10.0`(A) | rbe_dropout_test.rs (1): 57 `0.05` 통계(B — rmse_ci_rel 류 유도 적합)
- analyzer_test.rs (1): 133 `10.0` ms 벽시계(C)

## 3. 승인 요청

1. A 분류 약 70건: 교체 없이 lint W2 검출에서 "범위 불변식" 패턴 제외 (harness_lint.py 정련) — 승인 여부
2. C 분류 8건: 벽시계 게이트를 카운터 게이트 또는 보고 전용으로 교체 — 승인 여부
3. D 분류 3건: 하드코딩 기대값의 출처 검증 후 유도식 교체 — 승인 여부
4. B 분류 약 55건: 도메인 리팩토링(refactor-flow) 때 도메인 단위로 bounds 유도 교체 — 일괄 승인 또는 도메인별 개별 승인 중 택일
