# 하네스 루프 세션 인수인계 보고서

작성일: 2026-08-13. 목적: 다음 하네스 루프 세션이 맥락 손실 없이 이어받도록 경과·상태·다음 작업을 기록한다.

## 0. 한 줄 요약

논문(PAPER_POINCARE_RBE.md)을 수학적으로 전면 재유도한 뒤, 그 이론의 구현 정밀도를 보증하는 검증 하네스를 CP-1~CP-7 + CP-8 선행분까지 구축 완료했다. 테스트 44/44 통과, lint 통과. 유일한 블로커는 kogpt2 모델 파일(네트워크 차단)이며, 확보 즉시 CP-8 본체를 재개한다.

## 1. 전체 맥락 (왜 이 작업을 하는가)

1. 코드 전수 분석에서 기존 구현·논문의 수학적 결함 다수 발견 (치명 오류: CORDIC 게인/부호, sech2 순환, 무한 정보 주장, 잔차 소실, 재현 불가 실험 수치 등).
2. 사용자 지시로 논문을 공격적으로 재유도: 미검증 수치 삭제, 성립하는 수학만 남김.
   - 제1부: 위상-비트 미분 (양자화 위상 레지스터에서 미분 = 정수 덧셈 q += 2^(b-2), 무오차. 정리 3.1~3.4). sin 트랙 Z/4, sinh/cosh Z/2, tanh 는 유한 순환 불가 증명 후 1차 테이블 + 지름길 A - W^2/A.
   - 제2부: 대상 재정의 — "가중치 표본 압축"이 아니라 **평면 장(field) 부호화** (사용자: "평면 자체를 압축, 레이어 하나를 통째로"). ε-엔트로피 논거(12장), Helgason 평면파 원자 W = A e^{ρB_b} cos(λB_b + φ), B_b = log P(z,b), ρ = 1/2 강제(부록 C.1), 일반화 미분 δ = atan2(λ, ρ)(정리 13.2), CWT 위치-스케일 동일시(13.1), 레이어 단위 부호화 = 잠재 좌표(Aldous-Hoover) + Busemann 차 커널 = 묶인 랭크 2J 인수분해(15장), 이론 스펙 시트(17장: c 사다리, 손익분기 c > 10.5%, 배치-1 속도 S = 4I*/(2+C_g)).
   - 부록 A~F: 전 정리의 자기완결 증명 (Busemann 성질 직접 계산, Blaschke 공변성, Shannon 하한, Sterbenz 등).
3. 이론 정밀도를 코드에서 보증하기 위해 검증 하네스를 정의(docs/test/verification_harness.md, 문서 버전 2)하고 체크포인트(CP) 단위로 구현. **사용자 요구 워크플로우: CP 하나 완료할 때마다 결과 보고** (하네스 11절 양식).

## 2. 핵심 문서

| 문서 | 역할 |
|---|---|
| docs/paper/PAPER_POINCARE_RBE.md | 재유도판 논문 (1~17장 + 부록 A~F). 모든 테스트 상계의 유도 원천 |
| docs/test/verification_harness.md | 하네스 규범 (버전 2): 계층 L0~L4, 체크포인트 계획(10절), 보고 양식(11절), 실패 대응(12절), 상태표(13절) |
| docs/test/harness_session_handover.md | 본 문서 |
| 메모리: rbe-field-reframing.md | 프로젝트 방향 요약 (자동 메모리) |

## 3. 하네스 3대 불변 원칙 (반드시 유지)

1. **허용오차 리터럴 금지**: 모든 허용오차는 `src/core/math/verification.rs::bounds` 의 유도 함수에서만. 예외는 `// lint-allow: 사유` 명기.
2. **실패 대응**: 상계 완화·테스트 삭제·ignore 금지. 허용되는 대응은 (a) 구현 수정, (b) 유도 오류면 논문 부록 먼저 수정 후 bounds 갱신. 지금까지 실패 4건 전부 (b) 유형이었고 매번 유도가 정교해짐.
3. **L3 양측 판정**: 이론보다 좋아도 실패 (샤논 하한 가드 = 조작/버그 검출기).

## 4. 완료된 체크포인트 상세

### CP-1: 검증 기반 (src/core/math/verification.rs, #[cfg(test)])
- bounds 11종: U32=2^-24, U64=2^-53, lut_interp(k)=(π/2^(k+1))²/8, sterbenz_product()=2u+u², phase_quant(b)=π/2^b, amp_quant(fb)=ln2/2^(fb+1), lut_eval_f32, dot_product(n)=Higham γₙ, central_diff(+h_opt), rmse_ci_rel(n)=5/√(2n), shannon_floor, f64_chain, busemann_radial_oracle(r).
- oracle: busemann 직역(소박형 — 상쇄 있음, 검증 전용), central_diff.
- check(name, measured, bound): ratio>1 이면 panic, (measured, bound, ratio) 출력.
- **발견 1**: 오라클 소박형 1-|z|² 의 경계 상쇄 증폭을 f64_chain 이 누락 → busemann_radial_oracle(r) 로 유도 보정 (r 의존 상계).

### CP-2: 위상-비트 상태 (src/core/math/phase_state.rs)
- hi 64b 레이아웃 (규범, 하네스 9.1): [63:62] trk | [61:42] q(20b, 상위 2비트=사분면=미분 카운터) | [41:32] m_r(10) | [31:26] m_psi(6) | [25] s_h | [24] sign | [23:8] amp(16) | [7:0] 예약.
- 미분: differentiate_circular = q += 2^18, differentiate_hyperbolic = s_h 플립, advance_phase_n = 단일 덧셈 n계.
- one_minus_r_sq(r) = (1-r)(1+r) — 유일 허용 형태 (lint H3 강제).
- L0 테스트 8종: 양자화 가환 **2^20 전수**, D^4 복귀, 2-플립, 사분면 카운터, n계 단일덧셈, 적분 역원, **Sterbenz f32 [0.5,1] 전수(8.4M)**, 경계감산 상계.

### CP-3: 사분파 LUT (src/core/math/lut.rs)
- k=12, f64 코어(+f32 래퍼), q 비트 분할 [19:18]사분면 [17:6]인덱스 [5:0]분수. cos = sin(q + 2^18).
- 전수 스윕 max err 1.8384e-8 = **상계에 ratio 1.000 정확히 접함** (최악점 q=262112 = π/2 직전, 이론 예측 위치와 일치 → 상계 타이트 실증). L0 대칭성 비트 동일 2종.

### CP-4: Busemann 좌표 (src/core/math/busemann.rs)
- **프로덕션 형(상쇄 없음)**: P = (1-r)(1+r) / [(1-r)² + 4r sin²(Δ/2)]. busemann_xy 는 검증 전용.
- Mobius { ax, ay, rot }: apply / apply_boundary(단위원 재정규화) / deriv_boundary_abs = (1-|a|²)/|1-āb|².
- 6종 통과: 대수항등식(10만), 극좌표=직교좌표, 단위기울기(중심차분), 호로사이클 등위선, 뫼비우스 공변성 P(gz,gb)|g'(b)|=P (2만), 라디얼=2artanh r 을 **r=1-2^-30 까지 균일 상계**로.

### CP-5: 평면파 원자 (src/core/math/atom.rs)
- Atom { theta_b, lambda, phi_q, log2_amp }, RHO=0.5 (강제). eval_at_b / eval(극좌표).
- differentiated_n(n): 위상 += n*q_delta (단일 덧셈), log2_amp += n*step. q_delta = quantize(atan2(λ, ρ)).
- quantize_amp/dequantize_amp (16b, a = round(log2A*2048)+32768). tanh_derivative_shortcut(w, amp) = amp - w²/amp.
- 5종 통과: 비트 경로=해석식(δ 양자화 반스텝 상계), 중심차분 대조, n계 누적, tanh 지름길, 위상+진폭 양자화 바닥(7.3절).
- **발견 2**: 중심차분 상계의 함수값 오차에 cos 인자 조건수(u·envelope·|arg|) 누락 → m0 에 조건수 포함으로 유도 보정 (ratio 15.6 검출).

### CP-6: 레이어 코덱 (src/core/matrix/layer_codec.rs)
- LayerCodec { rows: Vec<LatentCoord>, cols, atoms: Vec<KernelAtom> }. 커널 = Σ A cos(λ[B(z)-B(w)] + φ).
- forward: y = F_out(F_in^T x), W 비실체화·특징 비캐시. OpCounter(FLOP/바이트) 내장, forward_flops_formula/bytes_formula (L0 정수 일치 게이트 — 벽시계 게이트 금지).
- gauge_transform: 좌표 z→g(z), 원자 b→g(b), **위상·진폭 무보정** (코사이클 log|g'(b)| 가 상대차에서 상쇄 — 실증됨).
- 4종 통과: 분리 동치(Higham γ_2J + cos 조건수), 게이지 불변성, 카운터 공식, 순전파 동치.
- 부수: matrix/__tests__/mod.rs 기존 이모지 제거 (훅 강제, 규칙 8).

### CP-7: 하이브리드 코덱 + L3 (src/core/encoder/hybrid_codec.rs)
- LloydMaxQuantizer::new_gaussian(bits): **런타임 Lloyd 수렴** (레벨 하드코딩 없음, libm::erf 무게중심 공식, 500회 반복). 해석 왜곡 D = 1 - Σ p l².
- encode/decode_residual (표본 σ 정규화), energy_capture(c 추정기, 17.1절), hybrid_roundtrip.
- 5종 통과: 왜곡 이론 vs 몬테카를로 400만 표본 **양측**, 문헌값 대조(Max 1960, lint-allow 급 유효자리 밴드), 샤논 가드 준수, **가드 조작 검출력 should_panic** ("2bpw 무손실" 조작 → 반드시 실패함을 검증), 분산축소 이론식 q(b)σ√(1-c) 양측 (c ∈ {0, .5, .9}) + c 추정기 보정.

### CP-8 선행분 (layer_codec.rs 확장)
- 직렬화: 원자 1개 = u64 (θ_b 12b | λ 10b(step 1/16) | φ 20b | amp 16b | 예약 6b), 좌표 1개 = u64 (r/θ f32 쌍). to_bits/from_bits, code_bits_formula = 64(M+N+J).
- 3종 통과: pack→unpack→pack **멱등성 비트동일**, 격자 원자 왕복 f64 비트 정확(1만), 부호길이/압축률 공식 (kogpt2 FFN 사례: 3072×768, J=512 → 34KB, 271:1).

### 하네스 lint (scripts/harness_lint.py)
- H1(하네스 테스트 리터럴 허용오차), H2(from_entropy), H3(소박한 1-r²) 하드 실패 + W1(#[ignore]) W2(레거시 리터럴) 경고. `lint-allow: 사유` 예외.
- 현재: **하드 위반 0 통과. 레거시 리터럴 145건 경고 집계** (교체는 테스트 수치 변경 = 사용자 승인 필요, 하네스 8절).

## 5. 테스트 실행 방법과 환경 특이사항 (중요)

- 실행: `cargo test -j 1 --lib -- atom_test phase_state_test lut_test busemann_test verification_test layer_codec_test hybrid_codec_test` → 44 passed 이어야 정상.
- **반드시 `-j 1`**: 이 머신은 페이징 파일 부족(os error 1455, STATUS_STACK_BUFFER_OVERRUN)으로 병렬 빌드가 간헐 실패한다. 실패해도 코드 문제 아님, 재시도.
- 훅 2종 작동 중: (a) 포매터가 저장 시 mod 선언을 알파벳 정렬함 — Edit 전 Read 권장. (b) post_edit_rs.py 가 **이모지 포함 파일 편집을 차단** — 기존 파일에 이모지가 있으면 먼저 제거해야 편집 가능.
- **네트워크 차단**: huggingface.co HTTP 000. kogpt2 다운로드 불가.
- 커밋 안 됨: 모든 작업이 워킹 트리에만 있음 (사용자가 커밋 지시한 적 없음 — 임의 커밋 금지).

## 6. 기지의 사전 존재 실패 (하네스와 무관, 건드리지 말 것)

- core::math::__tests__::gradient 계열 5건 실패: 기존 gradient.rs 의 "해석적 미분"이 손실 무관 0 반환 (P0 결함, 세션 시작 전부터 워킹 트리에 존재). **CP-5 의 위상이동 미분이 대체 대상이며, 대체 작업은 사용자 승인 후**.
- nlp 테스트 22/89 실패 (rbe_linear.rs:92 배치 입력 assert 등) — 별도 사전 존재 이슈, 코드 분석 보고서 참조.

## 7. 다음 작업 (우선순위 순)

1. **CP-8 본체 (블로킹: 모델 파일)**: `models/kogpt2/` 에 skt/kogpt2-base-v2 가중치가 나타나면:
   a. safetensors 로드 → FFN 층 1개 (768→3072) 추출.
   b. 원자 적합기 구현 (매칭 퍼슈트: b, λ 격자 탐색 + A, φ 최소자승; 논문 15.6절) — 아직 미구현.
   c. c 실측 → 17.3절 이론 예측 RMSE 와 실측 비교 (rmse_ci_rel 밴드, L4 게이트 4번).
   d. candle 대조 (nlp-verify 스킬 규칙: 실가중치만).
   e. docs/test 에 c 측정 보고서 작성 (perf-reporter 에이전트 사용 가능).
2. **좌표 학습 (리만 SGD)**: 논문 15.6절 — 아직 미구현. ∂K/∂z 는 정리 13.2 위상 이동으로 (atom::differentiated 재사용). 기존 riemannian_adam.rs 는 결함 다수(fast_sqrt, static mut) — 수리 필요, 사용자 확인 후.
3. **레거시 리터럴 145건 목록화 → 사용자 승인 요청** (bounds 유도로 교체 제안).
4. **f32 프로덕션 경로**: 현재 코어는 f64. f32 강하 시 7.3절 합성 상계로 L1 재검증 필요.
5. 논문 01~04, 12장 등 구판 문서들을 재유도판 체계로 정리 (선택, 사용자 지시 대기).

## 8. 루프 운영 상태

- 동적 페이싱 /loop 진행 중. 작업 있을 때 90s, 현재는 모델 대기라 1500s 간격.
- 사용자 지시: "스톱하지 말고 루프로", "CP 마다 결과 보고". 보고 양식은 하네스 11절.
- 다음 틱 판단 기준: models/ 하위 또는 HF 캐시에 kogpt2 파일 존재 여부 확인 → 있으면 CP-8 재개(90s 간격 복귀), 없으면 noop 유지.
