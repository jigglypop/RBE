# RBE (Riemannian Basis Encoding)

신경망 가중치를 128비트 푸앵카레 볼 공간에 극한 압축(최대 1000배)하고, 복원 없이 직접 추론하는 Rust 연구 프로젝트. 크레이트 이름은 `rbe_llm`.

## 빌드 / 테스트 명령

```bash
cargo check                                  # 빠른 컴파일 확인 (편집 후 기본)
cargo test <도메인>::__tests__ -- --nocapture # 도메인 단위 테스트
cargo test --lib                             # 전체 라이브러리 테스트
cargo run --example nlp_layers_demo          # NLP 레이어 데모
./run_nlp_tests.sh                           # NLP 레이어 테스트 일괄 실행
```

도메인 테스트 예: `cargo test encoder::__tests__`, `cargo test nlp::linear -- --nocapture`

## 아키텍처

- `src/core/` — 압축 코어. 9개 도메인: `encoder`(압축), `decoder`(가중치 생성/융합 순전파), `differential`(비트 상태 미분·11비트 사이클), `generator`(푸앵카레 학습), `math`(기저함수·베셀·그래디언트), `matrix`(블록·품질등급), `optimizers`(Adam·리만 Adam), `packed_params`(128비트 타입), `systems`(하이브리드 통합)
- `src/nlp/` — RBE 기반 NLP 레이어. `linear`, `layernorm`, `attention`, `ffn`, `embedding`, `softmax`, `rmsnorm`, `dropout`, `model_tools`(다운로드·압축·분석). core 구현체를 그대로 사용하며, skt-kogpt2 실제 가중치로 검증
- 테스트는 각 도메인 하위 `__tests__/` 디렉토리에 `<원본파일명>_test.rs` 로 병치. 새 테스트 파일을 늘리지 말고 이 규칙을 따를 것
- `docs/paper/` — 각 도메인의 이론 문서. 구현 완료 시 해당 챕터와 정합성 확인 필수
- `docs/test/` — 도메인별 성능 보고서. 리팩토링 완료 후 갱신

## 절대 규칙 (연구 무결성)

이 프로젝트는 연구 프로젝트로, 결과 조작에 해당하는 행위를 엄격히 금한다.

1. **시뮬레이션·더미 금지**: 실제 연산 없이 `println!` 으로 결과를 출력하거나, 테스트 외 코드에 더미 데이터를 생성하는 것은 기만 행위
2. **테스트 값 하드코딩 금지**: 기대값을 실측에 맞춰 끼워넣지 말 것. 테스트 수치(허용 오차 등) 변경은 반드시 사용자 승인 후
3. **NLP 검증 기준**: 실제 모델 가중치(skt-kogpt2)만 허용. candle 레이어와 순전파·역전파 결과를 비교해 정확도·압축률·속도를 검증
4. **로직 보존**: 로직 변경 요구가 없으면 기존 로직을 망가트리지 말 것. 리팩토링 후 테스트는 반드시 통과해야 함
5. **Cargo.toml 수정은 사전 허가 필수** (hook이 확인을 요구함)
6. **파일 증식 금지**: `compress_kogpt`, `compress_gpt` 식으로 유사 로직 파일을 새로 만들지 말고 기존 코드를 확장
7. **지시 외 방법 임의 선택 금지**: 명령된 방법이 막히면 임의로 우회하지 말고 확인을 받을 것
8. **이모지 금지** (코드·문서·논문 전부)

## 코드 스타일

- 테스트 함수명은 한글로, 무엇을 검증하는지 명확히 (예: `fn 인코딩_후_디코딩_RMSE_검증()`)
- 함수는 단일 책임, 20-30줄 이내, 매개변수 3개 이하 권장. 중첩 if는 early return으로
- 모든 미분·Adam·리만 Adam·순전파·역전파는 비트 상태에서 수행. 인코딩은 최초 가중치 압축 1회만
- 논문(docs/paper) 작성 시: 유도 과정부터 상세히, "압도적인" 류의 형용사 금지, 검증되지 않은 주장 금지

## 표준 워크플로우

리팩토링: 해당 도메인 테스트 실행 → 개선점 파악 → 리팩토링 → 테스트 통과 확인 → 사용자 컨펌 → `docs/test/<도메인>.md` 성능 보고서 작성. `/refactor-flow` 스킬 참조.
