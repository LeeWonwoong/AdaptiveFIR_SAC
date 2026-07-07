# CLAUDE CODE 작업지침 — 지속형 외란 3종(A/B/C) 구현 & 적응여지 검증

## 배경 (읽고 시작할 것)

이 프로젝트(`sac-afme`)는 **SAC가 시간가중 FIR/FME 필터의 (N, λ)를 매 스텝 적응**시켜
UWB 기반 UAV 위치추정을 개선하는 IEEE 논문용 코드다. 측정은 **UWB 4거리뿐**,
모델은 12상태, 산출물은 **위치 xyz만**.

**현재 막힌 지점**: 합성 데이터로 학습하면 reward가 정체하고 SAC가 고정 N을 못 이긴다.
진단 결과 원인은 명확하다 — **"모델 불일치가 지속되는 구간"이 없어서** 스텝별 최적 N이
시간에 따라 안 갈린다. 기존 mass_step은 질량 계단 + 1스텝 속도 임펄스라 필터가 1~2스텝에
흡수해버려 적응 이득이 사라진다.

**논문 근거(project 내 PDF)**:
- UIFM-SLAC(An_Unscent...): payload 결합 후 **"second spike during recovery oscillations"** —
  충격이 한 번에 안 끝나고 결합 후에도 진동하며 지속됨.
- FM-SMC: **impulse disturbances**를 구간에 연속 주입.
- 1_FIR_filter(Shmaliy): FIR 강점 = *"does not project 'old' errors beyond the horizon"*.
  즉 **불일치가 지속되는 동안 오래된(불일치 이전) 데이터가 편향을 만들고 작은 N이 그걸 빨리 버린다.**

## 목표

**지속형 외란 3종(A/B/C)을 구현**하고, 각각에서 **"스텝별 최적 (N,λ)가 시간적으로
갈리는지"를 데이터로 측정**해서, **적응이 실제로 이기는 외란만 선별**한다.
우선순위: **C(주력) > B > A**. 최종적으로 어느 조합을 논문 시나리오로 쓸지 결정한다.

---

## 구현할 외란 3종

기존 시나리오(`config.py`의 scenario_types, `datagen/scenario.py` 샘플러,
`rlenv/synth.py`의 plant 롤아웃)에 아래를 **추가**한다. 급기동(aggressive)은 이미 제외됨 —
비선형성이 커서 1차 테일러 선형화가 무너지는 영역이라 절대 넣지 말 것.

### C. UWB 측정 지속 편향 (NLOS 다중경로) — **주력**
가장 SAC 학습에 유리. 측정 편향은 **혁신(observation)에 직접 나타나서** SAC가
"이 앵커 혁신이 지속적으로 크다 → λ 낮춰 과거 빨리 잊자"를 배울 수 있다.

- **구현 위치**: `rlenv/dataset.py`의 `range_clean` 생성 직후 (dropout NaN 마스킹과 같은 자리).
  dropout이 range를 NaN으로 만들듯, NLOS는 **양(+)의 지속 편향**을 더한다.
- **모델**: 특정 앵커 1개에, 구간 [k0,k1] 동안 `range_clean[i, k0:k1, a] += bias(t)`.
  bias(t)는 0.3~1.5 m 크기의 **지속 편향**(다중경로는 거리를 늘림 → 항상 +).
  프로파일: 계단(step) 또는 완만한 진입/이탈(raised-cosine ramp). 스파이크 아님 — **지속**이 핵심.
- **시나리오 dict**: `nlos_bias: [{anchor, start_s, duration_s, bias_m, profile}]`
- **config 파라미터**: `nlos_count_range=(1,2)`, `nlos_duration_range=(2.0,5.0)`,
  `nlos_bias_range=(0.3,1.5)`, heldout는 `(1.5,2.5)`.
- **필터 처리**: 현재 innov_gate(2m)가 큰 편향을 아웃라이어로 배제할 수 있음 — 편향이
  게이트보다 작으면(0.3~1.5m < 2m) 통과되어 추정을 오염시킨다. 이게 의도된 동작
  (작은 지속 편향은 못 걸러냄 → λ로 과거 가중을 낮춰 대응). 게이트 조정 불필요.

### B. Payload 결합 후 지속 진동
UIFM-SLAC Scenario 2 직접 대응. 결합 순간 임펄스 + **결합 후 감쇠 진동**(모델이 예측 못 함).

- **구현 위치**: `rlenv/synth.py`의 plant 롤아웃 루프. 기존 mass 임펄스 코드 근처.
- **모델**: onset k0에서 질량 계단 변화(기존 유지) + **결합 후 [k0, k0+T_osc] 동안
  plant 가속도에 감쇠 사인 외력 주입**:
  `a_extra = A0 * exp(-(t-t0)/tau) * sin(2*pi*f*(t-t0)) * dir`,
  A0~2~5 m/s², f~2~4 Hz, tau~0.5~1.0 s, dir=랜덤 수평 단위벡터 + z성분.
  이 외력은 **필터 모델에 없으므로** 위치추정 편향으로 지속 발현된다.
- **시나리오 dict**: 기존 `mass` dict에 `osc_amp, osc_freq, osc_tau, osc_dir` 추가.
- **config**: `payload_osc_amp=(2.0,5.0)`, `payload_osc_freq=(2.0,4.0)`, `payload_osc_tau=(0.5,1.0)`.

### A. 지속 강풍/항력 (모델 없는 외력)
plant에 모델 없는 외력을 수 초간 지속. 기존 gust/sustained_wind를 **강화**.

- **구현 위치**: `datagen/wind.py`의 WindModel (이미 gust/sustained 지원).
- **변경**: gust를 짧은 half-cosine이 아니라 **지속 플래토**(2~4초 일정 강풍)로도 넣을 수 있게
  프로파일 옵션 추가. `wind.py`의 `wind_velocity`에 profile='plateau' 분기 추가.
- **config**: `wind_plateau_duration=(2.0,4.0)`, `wind_plateau_speed=(6.0,12.0)`.

---

## 작업 순서 (반복 루프)

### STEP 0 — 환경 확인
```bash
pip install -r requirements.txt
python -m tests.test_wfme          # T1~T5 전부 PASS 확인 (필터 안 건드렸으면 통과)
```

### STEP 1 — 외란 C부터 구현 (주력)
1. `config.py`에 C 파라미터 추가, `scenario_types`에 `"nlos_bias"` 추가(prob 배분).
2. `datagen/scenario.py`에 `_nlos()` 샘플러 + `disturbance_intervals()`에 nlos 구간 추가.
3. `rlenv/dataset.py`에 NLOS 편향 주입(dropout NaN 마스킹 코드 바로 아래, 같은 패턴).
4. 구문 확인: `python -c "from config import Config; from rlenv.dataset import *; print('ok')"`

### STEP 2 — 적응여지 측정 (C에 대해)
`tools/diagnose_recovery.py`가 이미 있다. 이걸 확장하거나 아래 새 스크립트를 만든다:
**`tools/adapt_signal.py`** — 각 외란 타입에서 다음을 측정:
- 외란 구간의 **스텝별 최적 N** 분포 (Greedy가 고르는 N) — 정상 구간 대비 **평균이 갈리는가**.
- **혁신으로 최적 N을 예측 가능한가**: 외란 구간에서 `corr(‖혁신‖, 최적N)` 또는
  `corr(anchor별 혁신, 최적λ)`. **상관이 높으면 SAC가 배울 수 있는 구조**(C의 핵심 가설).
- 외란 구간 **Greedy(상한) vs best-fixed RMSE 갭 %** (기존 진단과 동일).
- **회복시간**: 외란 종료 후 작은 N vs 큰 N.

합성 데이터는 소규모로: `python -m rlenv.synth --out data --n_train 30 --n_heldout 16`
(Greedy는 무거우니 heldout 12궤적, 앞 1200스텝으로 제한. 기존 diagnose_recovery.py 참고.)

**판정 기준(C가 성공인가)**:
- 외란 구간 최적 N/λ가 정상 대비 **명확히 갈림**(예: λ가 정상 1.0 → NLOS 0.7~0.85).
- `corr(혁신, 최적λ)` **> 0.3** (관측으로 예측 가능 = SAC 학습 가능).
- 적응 여지 **> 20%**.
→ 통과하면 C는 논문 시나리오로 채택.

### STEP 3 — B, A 동일 반복
B, A 각각 구현 후 STEP 2 측정. **통과한 것만** 최종 시나리오에 남긴다.
(A는 위치엔 나타나도 혁신 패턴이 약할 수 있음 — 측정으로 확인. 약하면 비중 낮추거나 제외.)

### STEP 4 — 최종 시나리오 확정 & 학습
통과한 외란들로 `scenario_probs` 재배분(C 비중 최대). 그 다음:
```bash
python -m rlenv.synth --out data --n_train 200 --n_heldout 50
python -m tools.adapt_signal          # 최종 데이터 적응여지 재확인
python train.py --outdir results/run0  # GPU 권장 (CPU는 매우 느림)
python evaluate.py --outdir results/run0
```
**학습 성공 판정**: `learning_curve.png`에서 ret/ep가 상승 추세 + evaluate 표에서
SAC-AFME의 **외란구간(disturb) RMSE**가 best-fixed FME보다 낮음. alpha가 0으로
붕괴하지 않는지 로그 확인(현재 alpha_min=0.02, target_entropy_scale=0.5 적용됨).

---

## 반드시 지킬 것 (제약)
1. **측정은 UWB 4거리만**. 자세/자이로 측정 추가 금지. h(s)는 4채널.
2. **급기동(high-G) 금지**. aggressive 패턴 넣지 말 것. 선형화 유효 영역 유지.
3. **필터 코어(filter/wfme.py) 수정 시 T1~T5 재실행**. 특히 handover/solve/게이트 로직은
   신중히 — 이미 검증된 상태다. 외란은 **데이터 생성부(synth/dataset/wind/scenario)에서만**
   주입하고 필터는 건드리지 않는 게 원칙.
4. **각 외란은 "지속" 구간을 만들어야 함**. 1~2스텝 임펄스는 필터가 흡수해버려 적응 이득이
   없음(이미 실패 확인). C=지속편향, B=감쇠진동(수 초), A=지속강풍(수 초).
5. **데이터로 판정**. "이 외란은 적응이 이긴다"를 adapt_signal.py로 확인한 것만 논문에 넣는다.
   직관으로 넣지 말 것.

## 현재 코드 상태 참고
- 시나리오 이미 구현됨: nominal, mass_step(급결합+1스텝임펄스), gust, sustained_wind,
  anchor_dropout, mixed. **anchor_dropout은 이미 잘 작동**(range NaN → 필터 per-anchor 배제).
- `filter/wfme.py`: 순수 FME(prior 없음), anchored 선형화(self_anchor=False),
  조건부 핸드오버(filled_valid>=N_min). self_anchor=True는 논문원형 ablation(발산 재현).
- `rlenv/replay_env.py`: 36D 관측 [ν1..4, N̂, λ̂]×L=6, 보상 −min(‖e‖,10), 외란 편향 샘플링
  (sample_segments의 disturb_frac=0.7).
- `config.py`: N∈[N_min,N_max], λ∈[lam_min,1.0]. N_min 관측성 하한은 논의 중
  (벡터측정 q=4 → 랭크상 N≥⌈12/4⌉=3, dropout 고려 시 4).
- `tools/diagnose_recovery.py`: 구간별 최적 N + Greedy 갭 + 회복시간 측정 (참고/확장용).

## 산출물
- 구현: config/scenario/synth/dataset/wind 수정본.
- `tools/adapt_signal.py`: 적응신호 측정 + 그림(외란별 최적N/λ 시계열, 혁신-최적λ 산점도).
- 짧은 결정 리포트: A/B/C 각각 "적응 이김/애매/못 이김" 판정과 근거 수치.
