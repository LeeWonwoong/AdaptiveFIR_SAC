#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Pilot-in-the-loop 수동 비행 세션 런처
#
#   bash datagen/launch_manual.sh <out_dir> <mode> [speed] [extra args...]
#
#   <out_dir>  데이터 저장 루트 (예: data_manual) — <out_dir>/heldout/ 에 저장됨
#   <mode>     nominal | wind | mass
#   [speed]    Isaac 물리 속도 (기본 1.0 — 수동 조종이므로 실시간 고정 권장)
#   [extra]    manual_session.py 로 그대로 전달 (예: --start enter --traj_id 3)
#
# 예:
#   bash datagen/launch_manual.sh data_manual wind
#   bash datagen/launch_manual.sh data_manual mass 1.0 --traj_id 5
#
# 절차 (엔진 부팅 후 안내 출력됨):
#   1. QGroundControl 실행 → PX4 SITL 자동 연결(UDP)
#   2. QGC 조이스틱 설정 (datagen/MANUAL_FLIGHT.md 참고)
#   3. Position 모드로 이륙 → 고도 0.5 m 넘으면 자동으로 기록 시작
#   4. 50초 비행 (외란 창 상태가 터미널에 실시간 표시됨)
#   5. 자동 저장 → y 입력 시 다음 궤적 반복
#
# 주의: 수동 조종이므로 --speed 는 1.0 을 강력 권장 (RTF>1 이면 사람 반응이
#       시뮬레이션 시간 기준으로 느려져 체감 조종성이 나빠짐).
# ─────────────────────────────────────────────────────────────────────────────
set -u
OUT="${1:?out_dir 필요 (예: data_manual)}"
MODE="${2:?mode 필요: nominal|wind|mass}"
SPEED="${3:-1.0}"
shift $(( $# >= 3 ? 3 : 2 )) || true

LOGDIR=$(mktemp -d /tmp/manual_XXXXXX)
echo "[manual] out=$OUT mode=$MODE speed=${SPEED}x"
echo "[manual] 엔진 로그: $LOGDIR/engine.log"

# ── 0) 잔재 엔진 감지: 이미 /gt/odometry 가 살아 있으면 그건 '이전' 세션 ──
#     (런처의 준비판정이 이 토픽 존재라서, 잔재가 있으면 새 Isaac 부팅이
#      끝나기도 전에 세션이 옛 엔진의 '비행 중' z 를 읽고 오동작한다)
if ros2 topic list 2>/dev/null | grep -q "^/gt/odometry$"; then
    echo "[manual] ✗ /gt/odometry 가 이미 존재 — 이전 엔진/commander 가 아직 실행 중입니다:"
    pgrep -af "run_datagen.py|commander.py" || true
    echo "[manual]   먼저 정리하세요:  pkill -f run_datagen.py; pkill -f commander.py; pkill -9 -f bin/px4"
    echo "[manual]   (토픽이 사라진 뒤 다시 실행)"
    exit 1
fi
# 고아 px4: run_datagen 을 죽여도 Pegasus 가 띄운 px4 는 살아남아 포트
# (TCP 4560 / UDP 18570)를 물고 있다 → 새 PX4 가 못 뜨거나 다른 인스턴스
# 번호로 떠서 QGC 가 Disconnected 로 남는다.
if pgrep -f bin/px4 >/dev/null 2>&1; then
    echo "[manual] ✗ 잔재 px4 프로세스 발견 — QGC 연결 실패의 주범입니다:"
    pgrep -af bin/px4
    echo "[manual]   정리:  pkill -9 -f bin/px4   (3초 후 재실행)"
    exit 1
fi

HEADLESS="${HEADLESS:-0}"   # SSH 등 화면 없는 환경이면 HEADLESS=1 bash datagen/launch_manual.sh ...

# ── 1) Isaac 엔진 (백그라운드) — 헤드리스 0: 수동 비행은 화면을 봐야 함 ──
"$HOME/isaacsim/python.sh" datagen/run_datagen.py \
    --headless "$HEADLESS" --speed "$SPEED" --out "$OUT" \
    > "$LOGDIR/engine.log" 2>&1 &
ENGINE_PID=$!
cleanup() {
    echo; echo "[manual] 정리 중..."
    kill "$ENGINE_PID" 2>/dev/null
    wait "$ENGINE_PID" 2>/dev/null
    pkill -9 -f bin/px4 2>/dev/null   # 이 세션의 PX4 고아 방지
    exit "${1:-0}"
}
trap 'cleanup 130' INT TERM

# ── 2) 부팅 대기: /gt/odometry = 엔진 준비 완료 ──
echo -n "[manual] Isaac+PX4 부팅 대기 (최대 ~5분)"
READY=0
for _ in $(seq 1 150); do
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo; echo "[manual] ✗ 엔진 조기 종료 — tail $LOGDIR/engine.log:"
        tail -20 "$LOGDIR/engine.log"; exit 1
    fi
    if ros2 topic list 2>/dev/null | grep -q "^/gt/odometry$"; then READY=1; break; fi
    echo -n "."; sleep 2
done
echo
if [ "$READY" -ne 1 ]; then
    echo "[manual] ✗ 300s 내 /gt/odometry 미검출 — tail $LOGDIR/engine.log:"
    tail -20 "$LOGDIR/engine.log"; cleanup 1
fi

cat <<'EOF'
[manual] ✓ 엔진 준비 완료.
[manual] ── 이제 QGroundControl 에서: ──────────────────────────────
[manual]   1. 좌상단 Q 아이콘 → Application Settings → 자동연결 확인
[manual]      (PX4 SITL 은 UDP 로 자동 연결됨)
[manual]   2. Vehicle Setup → Joystick → Enable joystick input
[manual]      → 스틱 캘리브레이션 (조종기/게임패드 공통)
[manual]   3. 비행모드 Position 선택 → 이륙
[manual] ────────────────────────────────────────────────────────────
EOF

# ── 3) 수동 세션 (포그라운드, 키 입력 사용) ──
python3 datagen/manual_session.py --mode "$MODE" "$@"
cleanup 0
