# SAC-AFME — RL-Adaptive Time-Weighted Finite Memory Estimation for UAV UWB Localization

SAC(Soft Actor-Critic)이 시간가중 FME(Finite Memory Estimator)의 **(N_t, λ_t)** 를
매 스텝 적응 튜닝하여, 질량 변화·돌풍 등 모델 불일치 상황에서 UWB 기반
드론 위치추정의 과도응답·정상정확도 트레이드오프를 자동으로 넘나들게 한다.

```
sac-afme/
├── config.py               # 단일 Config dataclass (필터/MDP/SAC/시나리오/데이터 전부)
├── filter/
│   ├── uav_model.py        # 12상태 쿼드로터 f, UWB h, AD Jacobian (torch.func, 배치)
│   ├── wfme.py             # ★ 시간가중 FME: 링버퍼 동결선형화 + 마스크드 배치 QR solve
│   └── baselines.py        # EKF / UKF / FixedFME / DI-FME(재구현) / Rule-FME
├── rlenv/
│   ├── synth.py            # Tier-0 합성 궤적 생성기 (동일 npz 스키마 — 컨테이너 검증용)
│   ├── dataset.py          # npz → GPU 텐서, UWB clean range 계산(노이즈는 env에서 주입)
│   └── replay_env.py       # 벡터화 log-replay POMDP env (동기화 에피소드)
├── rl/                     # CleanRL 스타일 SAC (GPU 상주 버퍼, auto-α)
├── train.py                # 학습 엔트리 (--synth N 으로 Tier-0 데이터 자동생성 가능)
├── evaluate.py             # ★ 논문용 Online-play 검증: 전 방법 비교표 + money figure
├── datagen/                # Isaac Sim + Pegasus + PX4 데이터 생성 (2터미널)
│   ├── run_datagen.py      #   시뮬 엔진 (레포 run_sim.py 개조: 질량주입+바람+50Hz 로깅)
│   ├── commander.py        #   오케스트레이터 (시나리오 샘플→오프보드 비행→저장 제어)
│   ├── scenario.py         #   시나리오 샘플러 (synth와 공유 — config로 관리)
│   ├── wind.py             #   WindModel (half-cosine gust; 레포 이식)
│   ├── traj_logger.py      #   50Hz npz+meta 로거
│   └── select_dataset.py   #   ★ 오프라인 풀 → 학습셋 선별(품질 게이트+시나리오 균형)
└── tests/test_wfme.py      # T1~T5 필터 검증 스위트
```

## Quickstart (Tier-0: Isaac 없이 전체 파이프라인)

```bash
pip install -r requirements.txt
python -m tests.test_wfme                      # T1~T5 (전부 통과해야 정상)
python train.py --outdir results/run0 --synth 200   # 합성 200궤적 생성 + 학습
python evaluate.py --outdir results/run0       # online-play: 표 + money_fig.png
```

GPU 권장 기본값: `n_envs=64, total_steps=300k, updates_per_step=8`.
Ablation: `--fix_lambda` (N만 적응), `--fix_N` (λ만 적응).

## Isaac Sim 데이터 생성 (2터미널)

RL 학습은 Isaac과 **완전 분리**되어 있다: 시뮬은 아무 때나 돌려 RAW 풀만 쌓고,
학습은 선별된 npz만 읽는다.

```bash
# ① (언제든, 별도 시간대) RAW 풀 수집 — 필요량보다 넉넉히 (기각분 감안 1.5~2배)
#   T1 (Isaac Sim python; Pegasus+PX4+px4_msgs 필요)
ISAACSIM_PYTHON datagen/run_datagen.py --headless 1 --out data_raw
#   T2 (일반 ROS2 python env)
python datagen/commander.py --n_train 400 --n_heldout 100

# ② 풀에서 학습셋 선별 (품질 게이트 G1~G4 + 시나리오 균형; selection_report.csv 생성)
python -m datagen.select_dataset --pool data_raw --out data --n_train 200 --n_heldout 50

# ③ 학습/평가 (npz만 읽음 — Isaac 불필요)
python train.py --outdir results/run0 && python evaluate.py --outdir results/run0
```

선별 게이트: G1 무결성(길이/유한/이륙추력), G2 워크스페이스(프레임 오류 즉시 검출),
G3 정합성(nominal 1스텝 전파오차 중앙값 < 1 cm — u 캘리브레이션 파일별 검증),
G4 외란신호(외란형 시나리오에서 disturbed/nominal 전파오차비 ≥ 3 — 주입이 실제로
물렸는지). heldout 강외란(δ 0.45~0.6, gust 11~14 m/s)은 이탈 기각률이 높으므로 풀을
여유 있게 수집할 것.

### 수집 직후 검증 체크리스트 (필수)
1. **u 정합성**: nominal 구간 1스텝 전파오차 `‖f(gt_k,u_k)−gt_{k+1}‖` ≈ 1e-3 m 수준.
   (Tier-0 실측: nominal 0.0013 m vs 외란구간 0.036 m — 28배 대비)
   틀어지면 FRD→FLU 부호(`run_datagen.FRD_TO_FLU`)나 캘리브레이션부터 의심.
2. **프레임**: gt 위치가 앵커 워크스페이스 [0,10]² 내부인지 (ENU↔NED 매핑 검증).
3. **시나리오 메타**: meta_XXXX.json의 외란 구간이 gt 거동과 일치하는지 육안 확인.

## 핵심 설계 (동결)

- **필터**: lag j 가중치 w_j = λ_t^j · 1[j<N_t], N∈[8,20], λ∈[0.7,1], 링버퍼 W=N_max 고정
  shape. 매 스텝 도착 시 1차 테일러 동결(EFIR 관행), 의사입력/의사측정으로 어파인 항 처리.
  solve는 **활성 윈도우 시작(lag N_t−1) 앵커**의 가중 QR lstsq(fp64) — 정규방정식의
  조건수 제곱 문제를 원천 회피. 불편성 Lemma(임의 admissible (N,λ)에서 K H̄=Φ)는
  T2에서 수치 검증(deadbeat 3e-4 = fp32 버퍼 양자화 하한).
- **워밍업/핸드오버 (시간이 아닌 조건 기반)**: 보조 EKF는 게인 존재 조건
  `filled_valid ≥ N_min`까지만(≈8스텝) 서빙 → 이후 에이전트가 행동하고 필터는
  성장 윈도우 `N_eff(t)=min(N_agent, filled_valid)`로 서빙, N_max 스텝 내 전 범위 도달.
  Lemma가 모든 중간 윈도우를 커버하므로 램프가 정확성을 해치지 않음(T5 검증).
  게이팅으로 유효 슬롯이 부족해지면 같은 조건으로 warm restart.
- **순수 FME 계약**: 추정치는 윈도우 행들의 시간가중 LS **그 자체** — prior 없음,
  타 추정기와의 블렌딩 없음. 남는 것은 ① 보조 EKF의 **워밍업 전용** 서빙(핸드오버
  래치 이후 절대 개입 안 함), ② 혁신 게이트/랭크 부족 시 **예측 유지**(시간갱신만 —
  표준 관행, 추정기 전환 아님), ③ 1e-8 상대 Tikhonov 행(수치 랭크 안전용, FME 문헌의
  εI ridge). 옵션 A(자기앵커 선형화) 유지.
- **측정 스위트 [스펙 변경 — 확인 필요]**: z = [UWB 4거리; FCU 자세 η; 자이로 ω] ∈ R¹⁰.
  근거: UWB-only 12상태는 호버 근방에서 ψ가 **구조적으로 비관측**(추력이 수직 →
  위치응답에 ψ 부재)이고 (v,η) 부분공간도 짧은 윈도우(0.16~0.4 s)에서 v·t ↔ ½g·θ·t²
  공선으로 분산 폭발 → prior 없는 순수 추정기가 성립 불가. 자세/자이로는 기체 표준
  가용 신호(FCU/IMU)이며 이를 추가하면 전 윈도우 완전관측 → 순수 FME + 옵션 A가
  무정규화로 안정(실측: 랜덤액션 250스텝 worst 0.47 m, ψ 오차 0.002 rad).
  **UWB-only로 회귀하려면** 순수성 유지가 불가능하므로 선형화 앵커를 보조필터
  궤적으로 옮기는 절충(EFIR식)이 필요 — 결정 대기.
- **POMDP 관측**: per-step 특징 o_t = [ν₁,ν₂,ν₃,ν₄, N̂_{t−1}, λ̂_{t−1}]를 최근 L=6
  스텝 스택 → **36차원**(flatten). ν는 **anchor별 혁신 벡터**(스칼라 RMSE 아님 —
  [0.4,0,0,0] 단일앵커 이상과 [0.2,0.2,0.2,0.2] 전반저하를 구분해야 dropout 학습 가능),
  σ화 후 [−1,1] clip. N̂/λ̂는 min-max 정규화(N→2(N−N_min)/(N_max−N_min)−1,
  λ는 max=1 고정). **보상 r = −min(‖p_GT−p̂‖, 10)** — 순수 L2 위치오차 + 안전클립만
  (reward engineering 없음). done=0, 동기화 M=64 벡터 에피소드(워밍업 8 + RL 400),
  log-replay(σ∈[0.03,0.10] 온라인 주입 = 증강).
- **비교군**: EKF/UKF/FME-N{8,14,20}/DI-FME(N=14,γ=0.3)/Rule-FME/SAC-AFME/Greedy-GT.
  Greedy-GT: anchored 선형화(기본)에서는 커밋이 미래 solve를 오염시킬 수 없어
  스텝별 오라클 상한 근사 — 실측 0.052 m vs 최고 고정 FME 0.067 m → **적응 여지
  ~22%** (SAC가 본학습에서 노릴 갭). self_anchor ablation에서는 전 방법과 함께 붕괴.

## 문헌 파라미터
- DI-FME: N=14, γ=0.3 — **원문에서 확인된 값** (N_default와 그리드 [8,20]이 브래킷).
- λ 범위 [0.7,1]은 Ω≻0 조건에서의 본 연구 설계 선택(특정 문헌값 아님).
- `filter/baselines.py`의 DI-FME는 논문 수식 기반 재구현 — **카메라레디 전 원저자
  구현과 대조 필수** (코드 내 주석 표기).

## 컨테이너 검증 현황
| 항목 | 상태 |
|---|---|
| T1 Jacobian AD vs FD(fp64) | ✅ 2.5e-9 |
| T2 deadbeat/불편성 (임의 N,λ) | ✅ 7.4e-5 |
| T3 λ 트레이드오프 (과도 λ<1 우세 / 정상 λ=1 우세) | ✅ |
| T4 비선형 무노이즈 | ✅ 5.9e-7 m |
| T5 핸드오버/성장윈도우 (클리핑 동치성) | ✅ |
| Tier-0 데이터 sanity (u 정합/외란 대비) | ✅ 0.0013 vs 0.036 m |
| 순수 FME 안정성 (UWB-only, 랜덤액션 250스텝) | ✅ worst 0.62 m |
| 논문 불안정성 정량 재현 (--self_anchor) | ✅ 714 km 발산 |
| 고정 FME 정확도 (UWB-only, N=8/14/20) | ✅ 0.095/0.078/0.068 m |
| heldout 평가 (Greedy-GT 상한 갭) | ✅ 0.052 vs 0.067 m (~22%) |
| 선별 도구 end-to-end (풀→선별→로드) | ✅ |
| 스모크 학습 (CPU, UWB-only 구성) | ✅ rmse 0.095 m 동작 |
| 36D 벡터관측 + −‖e‖ 보상 파이프라인 | ✅ ret/ep −9.7 정합 |
| anchor dropout (강제 시나리오, per-anchor 배제) | ✅ NaN 1.8%, max 0.25 m 무발산 |
| 스모크 평가 (전 방법, 표+figure 산출) | ✅ |
| Isaac datagen (run_datagen/commander) | ⚠️ 라이브 심 검증 필요 ([VERIFY-IN-SIM] 주석) |
