"""PostToolUse hook: .rs 파일 편집 후 자동 rustfmt + 연구 무결성 패턴 검사.

Edit/Write 도구 사용 직후 실행된다.
- 대상이 .rs 파일이면 rustfmt로 포맷팅한다.
- 이모지, 테스트 외 더미 패턴 등 금지 패턴이 보이면 exit 2로
  stderr 메시지를 Claude에게 피드백한다 (편집 자체를 되돌리지는 않음).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF"
    "\U0001F000-\U0001F0FF\U00002B00-\U00002BFF✅❌✨]"
)


def main() -> int:
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    file_path = (payload.get("tool_input") or {}).get("file_path", "")
    if not file_path.endswith(".rs"):
        return 0

    path = Path(file_path)
    if not path.exists():
        return 0

    subprocess.run(
        ["rustfmt", "--edition", "2021", str(path)],
        capture_output=True,
        timeout=30,
    )

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0

    warnings = []

    if EMOJI_RE.search(text):
        warnings.append("이 파일에 이모지가 포함되어 있습니다. 프로젝트 규칙상 코드에서 이모지는 금지입니다. 제거하세요.")

    is_test_file = "__tests__" in str(path) or path.stem.endswith("_test")
    if not is_test_file:
        for pattern, msg in [
            (r"\bdummy\w*\s*[=(]", "테스트 외 코드에 dummy 데이터 생성 패턴이 보입니다. 연구 무결성 규칙 위반 여부를 확인하세요."),
            (r"//\s*(TODO|FIXME):?\s*(시뮬레이션|simulate)", "시뮬레이션 관련 미완성 주석이 있습니다. 실제 연산으로 구현되어야 합니다."),
        ]:
            if re.search(pattern, text, re.IGNORECASE):
                warnings.append(msg)

    if warnings:
        print("\n".join(warnings), file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
