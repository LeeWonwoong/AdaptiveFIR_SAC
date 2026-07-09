#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# datagen/launch_datagen.sh — ONE-COMMAND Isaac datagen
#
#   ./datagen/launch_datagen.sh [n_train] [n_heldout] [out_dir] [speed]
#   예)  ./datagen/launch_datagen.sh 24 6 data_isaac 2.0
#
# 하는 일 (기존 터미널 2개를 하나로):
#   0. ROS2 humble + colcon_ws 소싱, MicroXRCEAgent 기동 (isim 함수와 동일)
#   1. Isaac 엔진(run_datagen.py)을 백그라운드로 — 물리 250Hz, 실물리 외란
#      (set_masses payload / apply_forces wind), 50Hz 로깅
#   2. /gt/odometry 토픽이 살아날 때까지 대기 (= Isaac+PX4 부팅 완료 신호)
#   3. commander.py 포그라운드 실행 — 패턴 비행 + 시나리오 샘플·퍼블리시
#   4. commander가 전 궤적 완료 시 엔진에 /datagen/control "shutdown"을
#      스스로 보내므로(commander.py L158) 엔진은 자동 종료 → 런처도 종료
#   Ctrl-C 시 양쪽 모두 정리.
#
# 두 프로세스인 이유(합치지 않는 이유): 엔진은 isaacsim 전용 파이썬의
# 블로킹 물리 루프이고 부팅에 수 분이 걸린다. commander는 가벼운 ROS2
# 노드라 파라미터를 바꿔 재시작하는 일이 잦다 — 합치면 commander를 고칠
# 때마다 Isaac을 재부팅해야 한다. PX4 SITL도 Pegasus가 띄우는 제3의
# 프로세스라 "한 프로세스"는 애초에 불가능하고, "한 명령"이 정답이다.
# ═══════════════════════════════════════════════════════════════════════
set -o pipefail   # -u 제거: ROS2 setup.bash가 미정의 변수(AMENT_TRACE_SETUP_FILES)를 참조해 set -u와 충돌

N_TRAIN=${1:-24}
N_HELDOUT=${2:-6}
OUT=${3:-data_isaac}
SPEED=${4:-2.0}

cd "$(dirname "$0")/.."                      # repo root

# ── 0) 환경 (isim 함수와 동일 절차; 이미 돼 있으면 무해) ──
source /opt/ros/humble/setup.bash
[ -f "$HOME/colcon_ws/install/setup.bash" ] && source "$HOME/colcon_ws/install/setup.bash"
if ! pgrep -x MicroXRCEAgent >/dev/null; then
    echo "[launch] MicroXRCEAgent 기동"
    nohup MicroXRCEAgent udp4 -p 8888 > "$HOME/.xrce_agent.log" 2>&1 &
    sleep 1
fi

LOGDIR=$(mktemp -d /tmp/datagen_XXXXXX)
echo "[launch] n_train=$N_TRAIN heldout=$N_HELDOUT out=$OUT speed=${SPEED}x"
echo "[launch] 로그: $LOGDIR/{engine,commander}.log"

# ── 1) Isaac 엔진 (백그라운드) ──
"$HOME/isaacsim/python.sh" datagen/run_datagen.py \
    --headless 1 --speed "$SPEED" --out "$OUT" \
    > "$LOGDIR/engine.log" 2>&1 &
ENGINE_PID=$!
cleanup() {
    echo; echo "[launch] 정리 중..."
    kill "$ENGINE_PID" 2>/dev/null
    wait "$ENGINE_PID" 2>/dev/null
    exit "${1:-0}"
}
trap 'cleanup 130' INT TERM

# ── 2) 부팅 대기: /gt/odometry가 곧 "엔진 준비 완료" ──
echo -n "[launch] Isaac+PX4 부팅 대기 (최대 ~5분)"
READY=0
for _ in $(seq 1 150); do
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo; echo "[launch] ✗ 엔진 조기 종료 — tail $LOGDIR/engine.log:"
        tail -20 "$LOGDIR/engine.log"; exit 1
    fi
    if ros2 topic list 2>/dev/null | grep -q "^/gt/odometry$"; then READY=1; break; fi
    echo -n "."; sleep 2
done
echo
if [ "$READY" -ne 1 ]; then
    echo "[launch] ✗ 300s 내 /gt/odometry 미검출 — tail $LOGDIR/engine.log:"
    tail -20 "$LOGDIR/engine.log"; cleanup 1
fi
echo "[launch] ✓ 엔진 준비 — commander 시작 (RTF 로그는 engine.log에서 확인)"

# ── 3) commander (포그라운드; 완료 시 엔진에 shutdown 자동 전송) ──
python3 datagen/commander.py \
    --n_train "$N_TRAIN" --n_heldout "$N_HELDOUT" \
    --alt 1.5 --cx 5 --cy 5 2>&1 | tee "$LOGDIR/commander.log"
CMD_RC=${PIPESTATUS[0]}

# ── 4) 엔진 자연 종료 대기 (shutdown 수신 후 스스로 닫힘) ──
echo "[launch] commander 종료(rc=$CMD_RC) — 엔진 마무리 대기 (최대 60s)"
for _ in $(seq 1 30); do
    kill -0 "$ENGINE_PID" 2>/dev/null || break
    sleep 2
done
kill "$ENGINE_PID" 2>/dev/null; wait "$ENGINE_PID" 2>/dev/null

echo "[launch] ✓ 완료 — 데이터: $OUT/{train,heldout}/  로그: $LOGDIR"
echo "[launch] 다음: STAGE B 게이트 → python3 -m tools.adapt_signal --data $OUT"
