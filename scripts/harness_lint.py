# -*- coding: utf-8 -*-
"""검증 하네스 정적 감사 (docs/test/verification_harness.md 7절)

규칙:
  H1 (하드 실패): verification 모듈을 사용하는 테스트 파일에서
      부동소수점 리터럴 허용오차 비교(assert 내 `< 1e-3` 류) 금지.
      허용오차는 bounds:: 함수에서만 나와야 한다.
      예외: 같은 줄 또는 직전 줄에 `lint-allow:` 주석 (사유 필수 기재).
  H2 (하드 실패): __tests__ 내 from_entropy 사용 금지 (결정론 원칙).
  H3 (하드 실패): 하네스 모듈에서 소박한 `1.0 - r * r` 형 금지 (9.3절 — Sterbenz 형 강제).
  W1 (경고): __tests__ 내 #[ignore] 목록 보고.
  W2 (경고): 레거시 테스트(verification 미사용)의 리터럴 허용오차 개수 보고
      (교체는 규칙상 사용자 승인 필요 — 하네스 8절).
"""
import re
import sys
from pathlib import Path

# Windows cp949 콘솔 대응
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# 하네스 모듈 (H3 적용 대상)
HARNESS_MODULES = [
    "core/math/phase_state.rs",
    "core/math/lut.rs",
    "core/math/busemann.rs",
    "core/math/atom.rs",
    "core/math/verification.rs",
    "core/matrix/layer_codec.rs",
    "core/encoder/hybrid_codec.rs",
]

# assert 안의 부동소수점 리터럴 비교: `< 0.5`, `<= 1e-3`, `< 5e-3` 등 (0.0 은 예외)
LITERAL_CMP = re.compile(r"<=?\s*(\d+\.\d+([eE]-?\d+)?|\d+[eE]-?\d+)")
NAIVE_ONE_MINUS_R2 = re.compile(r"1\.0\s*-\s*(\w+)\s*\*\s*\1|1\.0\s*-\s*\w+\.powi\(2\)")


def lines_of(path):
    return path.read_text(encoding="utf-8").splitlines()


def main():
    errors = []
    warnings = []

    test_files = sorted(SRC.glob("**/__tests__/*.rs"))
    for f in test_files:
        lines = lines_of(f)
        uses_verification = any(re.search(r"use .*math::verification", l) for l in lines)
        rel = f.relative_to(ROOT)
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("//"):
                continue
            allowed = "lint-allow:" in line or (i > 0 and "lint-allow:" in lines[i - 1])

            if "from_entropy" in line:
                errors.append(f"[H2] {rel}:{i+1} from_entropy 사용 (고정 시드 필수)")

            if "#[ignore" in line:
                warnings.append(f"[W1] {rel}:{i+1} ignore 된 테스트")

            m = LITERAL_CMP.search(line)
            if m and ("assert" in line or "check(" in line):
                lit = m.group(1)
                if float(lit.replace("e", "E").replace("E", "e") if "e" in lit.lower() else lit) == 0.0:
                    continue
                if uses_verification:
                    if not allowed:
                        errors.append(
                            f"[H1] {rel}:{i+1} 리터럴 허용오차 `{lit}` — bounds:: 유도로 교체하거나 lint-allow 사유 명기"
                        )
                else:
                    warnings.append(f"[W2] {rel}:{i+1} 레거시 리터럴 허용오차 `{lit}`")

    for mod in HARNESS_MODULES:
        p = SRC / mod
        if not p.exists():
            continue
        for i, line in enumerate(lines_of(p)):
            if line.strip().startswith("//"):
                continue
            if NAIVE_ONE_MINUS_R2.search(line) and "lint-allow:" not in line:
                errors.append(
                    f"[H3] {mod}:{i+1} 소박한 1-r^2 감산 — (1-r)(1+r) Sterbenz 형으로 교체 (하네스 9.3)"
                )

    print(f"하네스 lint: 테스트 파일 {len(test_files)}개 검사")
    for e in errors:
        print("오류  ", e)
    legacy = [w for w in warnings if w.startswith("[W2]")]
    ignored = [w for w in warnings if w.startswith("[W1]")]
    print(f"경고: 레거시 리터럴 허용오차 {len(legacy)}건 (교체는 사용자 승인 필요), ignore {len(ignored)}건")
    if errors:
        print(f"\n실패: 하드 규칙 위반 {len(errors)}건")
        return 1
    print("통과: 하드 규칙 위반 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
