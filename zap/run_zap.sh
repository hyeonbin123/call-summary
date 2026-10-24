#!/usr/bin/env bash
# OWASP ZAP 배포판(크로스플랫폼 패키지, Java 17 이상)으로 zap/automation.yaml을 모델 없는 오프라인 서비스에
# 돌린다 (README "실행"의 보안 스캔, docs/experiments.md "보안 스캔 기록").
#
#   ZAP_DIR=<zap-2.17.0.jar가 있는 폴더> zap/run_zap.sh <name>
#
# 스캔 대상은 TARGET(기본 http://127.0.0.1:8072)에서 이미 돌고 있어야 한다:
#   uv run --no-sync uvicorn call_summary.service:create_offline_app --factory --port 8072
# 보고서 <name>.json과 <name>.md, ZAP 로그(zap.log), 새 ZAP 홈 폴더(home/)는 work/zap/<name>/에 생긴다. 출력은
# 파일로만 보내고 파이프로는 보내지 않는다(스캐너 자식 프로세스가 파이프를 쥐고 남는 일을 막는다).
# [파일 역할] HawkScan 체험이 2026-10-05에 끝나 무료 ZAP으로 같은 스캔(OpenAPI 설명, 씨앗 요청, 스파이더, 패시브
# 스캔, 기본 정책의 액티브 스캔)을 하게 바꾼 실행 도우미다. 스캔 내용은 zap/automation.yaml 하나에 있다.
# [흐름] 인자·환경 변수 확인 → 결과 폴더 만들기(이미 있으면 중단) → 계획이 ${…}로 읽는 변수 내보내기 → java로
# ZAP을 화면 없이 띄워 계획 실행 → ZAP의 종료 코드를 그대로 돌려줌.
# [주의] 이 서비스에는 인증이 없어 토큰 같은 비밀값을 넘기지 않는다. 스캔 뒤 처리기에 닿은 /summarize 요청
# 수는 대상 서버의 로그(JSON 줄의 "request_id")로 센다. 서버를 멈출 때는 uvicorn이 찍은 PID와 실행기 PID를
# 둘 다 멈춘다.
set -euo pipefail

# 첫 인자(스캔 이름)는 필수다. 결과 폴더 이름이 된다. root는 이 스크립트 위치에서 구한 저장소 루트라 어느
# 폴더에서 실행해도 경로가 같다. ZAP_DIR이 없으면 무엇을 넣어야 하는지 알리고 멈춘다.
name=${1:?usage: zap/run_zap.sh <name>}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ZAP_DIR:?set ZAP_DIR to the folder that holds zap-<version>.jar}"

# 경로를 Java가 읽는 절대 경로로 바꾼다. Git Bash에서는 cygpath -m으로 C:/... 꼴(Windows 경로, 슬래시)로 바꾸고,
# 다른 환경에서는 절대 경로 그대로 쓴다. ZAP은 상대 경로를 작업 폴더가 아니라 계획 파일 폴더(zap/) 기준으로
# 풀기 때문에 절대 경로로 넘긴다.
native() {
  local abs
  abs=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
  if command -v cygpath > /dev/null; then cygpath -m "$abs"; else printf '%s\n' "$abs"; fi
}

# 같은 이름의 결과 폴더가 있으면 덮어쓰지 않고 멈춘다(이전 스캔 기록을 지우지 않으려는 것). 그다음 ZAP_DIR에서
# zap-*.jar를 찾는다.
out=$root/work/zap/$name
if [ -e "$out" ]; then
  echo "$out exists: pick another name" >&2
  exit 1
fi
jars=("$ZAP_DIR"/zap-*.jar)
[ -f "${jars[0]}" ] || { echo "no zap-*.jar in $ZAP_DIR" >&2; exit 1; }
mkdir -p "$out"

# 계획 파일(zap/automation.yaml)이 ${…}로 읽는 변수: 스캔 대상 주소(기본 127.0.0.1:8072의 오프라인 서비스),
# 보고서 폴더와 이름.
export TARGET=${TARGET:-http://127.0.0.1:8072}
export REPORT_DIR REPORT_NAME=$name
REPORT_DIR=$(native "$out")

# ZAP 실행: 힙 2 GB, 화면 없는 명령 모드(-cmd), 원격 측정 끔(-notel), ZAP 자신의 프록시 포트(-port. 기본
# 8097: 같은 PC에서 다른 프로젝트의 ZAP이 8080을 쥐고 있으면 "Address already in use"로 바로 끝나므로
# 프로젝트마다 다른 포트를 쓴다. 스캔 대상 주소와는 상관없다), 스캔마다 새 ZAP 홈 폴더(-dir), 업데이트 확인 끔,
# 계획 자동 실행(-autorun). 출력은 모두 zap.log 파일로 보낸다.
# || status=$? 로 실패 코드를 받아 두어, set -e로 바로 끝나지 않고 결과 위치를 알린 뒤 같은 코드로 끝난다.
status=0
java -Xmx2g -jar "$(native "${jars[0]}")" -cmd -notel -port "${ZAP_PORT:-8097}" -dir "$(native "$out/home")" \
  -config start.checkForUpdates=false -autorun "$(native "$root/zap/automation.yaml")" \
  > "$out/zap.log" 2>&1 || status=$?
echo "ZAP exit status $status, reports and log in $out"
exit "$status"
