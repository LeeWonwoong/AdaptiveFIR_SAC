# 수동 비행 (Pilot-in-the-Loop) 가이드

## ★ 확정 프로토콜 요약 (v13, 앵커 3 m)

| 항목 | 값 | 근거 |
|---|---|---|
| 비행 길이 | **wind 8 초 / mass·nominal 15 초** | wind=직선 1회 통과, 나머지=사각형 1바퀴 |
| 궤적 (nominal·mass) | **4 m 사각형** 한 바퀴 (하늘색 가이드, 3–7 m) | 실측 순항 1.2 m/s로 ≈13 s; UIFM-SLAC Fig.12 구도 |
| 궤적 (wind) | **바람 축 x 직선 왕복** (7,5)→(3,5)→복귀 | 바람과 나란히 → **좌우 보정 불필요**; UIFM-SLAC 시나리오1(직선) 선례 |
| 외란 창 | **wind 3.5–6.5 s / mass 6–11 s** | 평가창은 2 s 부터 시작(필터 성장) |
| wind | **9.5 m/s, dir 180°** (기본값) | v13 학습범위 [9,15] 하한부 = in-range (항력 12 m/s 대비 0.63배) |
| mass | +70 %, 6–11 s (기본값) | 스크립트 heldout과 동일 세기 |
| 창 중 조종 | **스틱 1/4 유지, 밀림 허용** (요 스틱 절대 금지) | 학습 데이터도 이동 중 피격; 압도되면 손 떼기(PX4 홀드) |
| 순항 속도 | 스틱 절반 ≈ **1.0–1.2 m/s** | 학습 실측 중앙값 1.0 |
| QGC 캡 | Horizontal **2.0~2.5**, Vertical **1.0**, **Responsiveness 0.6~0.8** | 풀스틱도 학습 최대(2.3) 이내 + 바람 복귀 여유 |
| 고도 | 순항 **1.5–2.0 m**, 변마다 변주 | 학습 대역 1.3–2.4 m |
| **고도 상한** | **2.5 m 엄수** | **앵커 평면 3 m 아래 유지 (z-기하 급락 방지)** |

**회차 채택 게이트** (기록 후 자동 검증):
- GT 속도: 수평 중앙값 0.7–1.3 / 최대 ≤ 2.5 / 수직 최대 ≤ 0.6 m/s
- 고도: 1.2 ≤ z ≤ 2.5 m 유지
- 경계: 앵커 사각형(1–9 m) 내
```bash
python3 - <<'EOF'
import numpy as np, glob
for f in sorted(glob.glob('data_manual15/heldout/traj_*.npz')):
    g=np.load(f)['gt']; v=g[:,3:6]; hp=np.linalg.norm(v[:,:2],axis=1)
    ok=(0.7<=np.median(hp)<=1.3) and hp.max()<=2.5 and abs(v[:,2]).max()<=0.6 \
       and 1.2<=g[:,2].min() and g[:,2].max()<=2.5 \
       and g[:,0].min()>=1 and g[:,0].max()<=9 and g[:,1].min()>=1 and g[:,1].max()<=9
    print(f, '채택가능 ✓' if ok else '게이트 탈락 ✗',
          f"(v중앙 {np.median(hp):.2f}, z {g[:,2].min():.2f}~{g[:,2].max():.2f})")
EOF
```


사람이 직접 조종해 비행하면서 AFME localization 데이터를 기록하는 절차.
필터는 전부 인과적(causal)이라 **기록 후 후처리 = 온라인 실행과 동일 결과**다.
논문 표기: "the estimator runs causally on the recorded stream; no future
information is used."

## 0. 구성 요소

```
사람 → 조이스틱/조종기 → QGroundControl → PX4 SITL → Isaac (run_datagen.py, 무수정)
                                                        ├─ /gt/odometry 50 Hz
                                                        ├─ 외란(wind/mass) 창 적용
                                                        └─ TrajLogger → traj_XXXX.npz
                        manual_session.py ──(/datagen/scenario, /datagen/control)──┘
```

`run_datagen.py`(엔진)는 **한 줄도 안 바뀜** — commander 대신 사람이 조종하고,
`manual_session.py`가 시나리오 퍼블리시 + 기록 시작/종료만 담당한다.

## 1. 입력장치 설정 — 두 경로 모두 QGC Joystick 으로 통일됨

### A. USB 게임패드 (Xbox/PS/로지텍 등)
1. PC에 USB 연결 (리눅스: `ls /dev/input/js*` 로 인식 확인, `jstest-gtk`로 축 확인 가능)
2. QGroundControl → **Vehicle Setup → Joystick**
3. `Enable joystick input` 체크 → **Calibrate** 진행 (스틱 4축: 스로틀/요/피치/롤)
4. 버튼 매핑(선택): Arm, 모드 전환 등을 버튼에 할당 가능

### B. 실제 조종기 (RadioMaster / FrSky Taranis / Jumper 등)
방법 B-1 (권장): **USB 조이스틱 모드**
1. 조종기를 USB로 PC에 연결 → 조종기 화면에서 **"USB Joystick (HID)"** 선택
   (EdgeTX/OpenTX 계열은 연결 시 모드 선택 팝업이 뜸)
2. 이후는 게임패드와 동일 — QGC Joystick 캘리브레이션

방법 B-2: 시뮬레이터 동글(트레이너 포트 → USB) 사용 시에도 OS에는 HID 조이스틱으로
잡히므로 동일.

> 어느 쪽이든 **코드 경로는 완전히 동일**하다. QGC가 joystick 입력을 MAVLink
> MANUAL_CONTROL 로 PX4에 전달하고, PX4 파라미터 `COM_RC_IN_MODE=1`
> (Joystick/무RC) 이면 그대로 RC 입력으로 처리된다. SITL 기본값이 보통 1이며,
> 아니면 QGC → Parameters 에서 변경.

### 비행 모드
- **Position** 모드 권장: 스틱 중립 = 호버 유지라 조종 부담이 적고, localization
  평가 목적에 적합. (Stabilized 는 난이도 높음)
- 이륙: 스로틀 올리거나 QGC Takeoff 슬라이더 사용.

## 2. 세션 실행

```bash
# 터미널 1 — 엔진+세션 한 번에 (headless 0: 화면 보면서 조종)
bash datagen/launch_manual.sh data_manual wind        # 바람 시나리오
bash datagen/launch_manual.sh data_manual mass        # 페이로드 시나리오
bash datagen/launch_manual.sh data_manual nominal     # 외란 없음
```

엔진 부팅(수 분) 후 QGC 안내가 출력되면:
1. QGC에서 조이스틱 확인 → Position 모드 → 이륙
2. **고도 0.5 m 를 2초 유지하면 자동으로 기록 시작** (또는 `--start enter`)
3. 터미널에 초 단위 진행 + 외란 창 상태 표시:
   `[log] t = 8/50s  z = 1.52 m  << WIND 12 m/s ACTIVE >>`
4. 50초 후 자동 저장 → `y` 입력하면 다음 궤적 연속 기록 (traj_id 자동 증가)

### 권장 비행 프로파일 (단일 waypoint / 직선 기반)
계획대로 "직선 왕복" 기동이면:
- 기록 시작 전 한쪽 끝(예: x≈2 m)에서 호버
- 기록 중 반대편(x≈8 m)까지 직선 순항 → 잠시 호버 → 복귀, 반복
- 앵커 사각형(1~9 m) 안쪽 유지, 고도 1~2 m 권장
- 바람 창(6–16 s, 26–36 s)과 직선 구간이 겹치도록 타이밍을 잡으면
  "순항 중 돌풍" 그림이 깔끔하게 나옴

### 자주 쓰는 옵션
```bash
# 기존 기록에 이어서 (traj_0005 부터)
bash datagen/launch_manual.sh data_manual wind 1.0 --traj_id 5

# 엔터로 수동 시작
bash datagen/launch_manual.sh data_manual mass 1.0 --start enter

# 바람 세기/창 바꾸기 (기본은 논문 heldout 과 동일: 12 m/s, 6-16 & 26-36 s)
... wind 1.0 --wind-speed 10 --windows "10:10,30:10"

# 페이로드 (기본: +70 %, 15 s 시작 18 s 지속 — 논문과 동일)
... mass 1.0 --mass-delta 0.5 --mass-onset 20
```

## 3. 후처리 (표 + 그림)

```bash
python3 tools/eval_manual.py --data_dir data_manual \
    --ckpt results/v12_50k/ckpt.pt --seed 13 --outdir figures_manual
```

- 궤적별 RMSE 표 (2–40 s, 논문과 동일 상수: Q=3e-3, FME N=10) + 외란 창내 RMSE
- 궤적별 그림 2장: RMSE 시계열(4필터) / N·λ 적응 — 창 음영·라벨 자동
- 측정 합성(UWB 4 + 자세 3 + 각속도 3, noise_seed = 1234+seed·101)도 본편과 동일

## 4. 논문 서술 포인트

- **일반화**: 정책은 스크립트 패턴 3종으로만 학습 — 사람 조종 궤적은 unseen
  trajectory distribution. 외란 세기는 학습 범위 내(12 m/s ∈ [10,15],
  +70 % ∈ [+60,+90 %]) → "in-range disturbances, out-of-distribution
  trajectories, no retraining"
- **인과성**: 후처리지만 모든 필터가 causal → 온라인 실행과 동일 (명시할 것)
- **정직성 각주**: PX4 제어기·사람 입력은 실제이나 UWB 측정은 GT 기반 합성
  (실기 UWB 라디오 아님) — 명시 필요

## 5. 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| QGC에 조이스틱 탭이 없음 | 기체 연결 전에는 안 뜸 — SITL 연결 후 Vehicle Setup |
| 스틱 움직여도 기체 반응 없음 | `COM_RC_IN_MODE` 확인(1), QGC Joystick "Enable" 체크, 모드 Position |
| 조종기가 PC에서 인식 안 됨 | 조종기 USB 모드를 "Joystick(HID)"로 (충전 모드 아님) |
| 기록이 자동 시작 안 됨 | 고도 트리거(0.5 m, 2 s) 미달 — 엔터로 강제 시작 가능 |
| RTF가 1보다 커서 조종 어색 | launch 인자 speed=1.0 확인 (수동 비행은 실시간 고정) |
| 저장 파일이 안 보임 | `<out>/heldout/traj_XXXX.npz` — 엔진 `--out` 경로 확인 |

## 6. 화면 보조 (자동)

`--headless 0` (launch_manual.sh 기본)으로 엔진을 켜면 자동으로:
- **초록 사각형**: 앵커 영역(1–9 m) 바닥 외곽선 — **이 안에서만 비행**
- **주황 기둥 4개**: 앵커 위치 (머리 높이 = 실제 앵커 z: 두 개 지면, 두 개 5 m)
- **탑다운 카메라**: 사각형 전체가 항상 화면에 (자동 전환; 실패 시 뷰포트
  카메라 메뉴에서 `/World/ManualAids/OverviewCam` 수동 선택)

전부 시각 전용(충돌 없음)이고 headless 데이터 생성에는 생성조차 안 됨.
고도는 세션 터미널(초당 z 출력) 또는 QGC로 확인 — 탑다운이라 화면엔 안 보임.

### 사각형 waypoint 가이드 (자동 표시)
- **하늘색 사각형**: 권장 주행 궤적 — 한 변 5 m (2.5–7.5 m), 앵커 경계에서 1.5 m 안쪽
- **꼭짓점 퍽 4개**: waypoint. **흰색 퍽 = 시작점 (2.5, 2.5)** — 기록 전 여기서 호버
- 한 바퀴 = 20 m ≈ 순항 1 m/s 기준 24–28 s (30 s 프로토콜에 맞춤)

## ★ 바람 회차 조종 요령 (좌우 보정 0회)

바람 dir 180° = 바람이 **−x(서쪽)로 민다**. 그러니 **x축 직선만** 타면 좌우 조작이 사라진다.

| 시각 | 위치 | 스틱 |
|---|---|---|
| 시작 전 | **(7, 5, 1.7)** 풍상 끝에서 호버 | — |
| 0–6 s | 서쪽으로 순항 (기준선) | 앞으로 절반 |
| **6–11 s** | **바람 창 — 밀리는 대로 둔다** | 1/4 유지, x<3 이면 살짝 뒤로 |
| 11–15 s | 동쪽으로 복귀 (정풍) | 좀 더 밀어야 함 |

- 풍상(동쪽 끝)에서 시작하므로 밀릴 여유가 4.5 m 있다.
- **자세 보정은 하지 않는다** — 기울기는 PX4 가 계산한다. 스틱은 "속도" 명령이다.
- **요(왼스틱 좌우) 절대 금지** — 기수가 돌면 스틱 방향과 지도 방향이 어긋난다.
- 그래도 힘들면 `--wind-speed 9` 까지 내려도 in-range 다.

## ★★ 모드별 프로토콜 (v13 최종)

**폴더를 반드시 분리한다** — 길이가 다른 궤적은 한 데이터셋에 못 섞는다(로더가 텐서로 스택).

```bash
# wind: 8 초, 바람 축 x 직선 1회 통과, 창 3.5-6.5 s
bash datagen/launch_manual.sh data_manual_wind wind      # 6회 이상 권장
# mass / nominal: 15 초, 4 m 사각형 1바퀴, 창 6-11 s
bash datagen/launch_manual.sh data_manual_sq   mass
bash datagen/launch_manual.sh data_manual_sq   nominal --traj_id 6
```
(길이·창은 모드에 따라 자동 설정된다. 바꾸려면 `--duration` / `--windows`.)

### wind 8초 안무 — 일방향 통과
| 시각 | 위치 | 스틱 |
|---|---|---|
| 시작 전 | **(7.5, 5, 1.7)** 풍상 끝 호버 | — |
| 0–1.5 s | 그대로 호버 (필터 워밍업) | 중립 |
| 1.5–6.5 s | **서쪽으로 등속 통과** (7.5→2.5, ~1.2 m/s) | 앞으로 절반 |
| **3.5–6.5 s** | **바람 창 — 속도를 유지한다** (배풍이라 빨라지니 스틱을 오히려 줄임) | 좌우 조작 없음 |
| 6.5–8 s | 도착 후 호버 | 중립 |

핵심: **창 동안 멈추지 않는다.** 정지하면 3D 플롯에서 그 구간이 점으로 뭉쳐 UIFM-SLAC Fig.10 같은 그림이 안 나온다. 외란의 증거는 GT 가 휘는 것이 아니라 **추정선이 흩어지는 것**이다.

### mass 15초 안무 — 사각형 1바퀴 (UIFM-SLAC Fig.12 구도)
창(6–11 s)이 2–3번째 변에 걸리게 페이싱. 6 s 픽업 순간 가라앉는 것은 **스로틀로 고도만 복구하고 수평 진행은 유지**한다.

### 후처리 (폴더별로 따로)
```bash
python3 tools/eval_manual.py  --data_dir data_manual_wind --ckpt $CK --seed $S --outdir figures_manual_wind
python3 tools/plot_traj3d.py --data_dir data_manual_wind --ckpt $CK --seed $S --outdir figures_manual_wind
python3 tools/eval_manual.py  --data_dir data_manual_sq   --ckpt $CK --seed $S --outdir figures_manual_sq
python3 tools/plot_traj3d.py --data_dir data_manual_sq   --ckpt $CK --seed $S --outdir figures_manual_sq
```
