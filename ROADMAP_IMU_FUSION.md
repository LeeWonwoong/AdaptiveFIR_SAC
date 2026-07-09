# SAC-AFME: IMU 융합 로드맵 & Isaac 실행 가이드

## 확정 구조 (컨테이너 실측 검증 완료)
- **필터**: 12차원 예측(50Hz, 제어입력) + 12차원 측정보정(10Hz, UWB+IMU 융합)
- **측정 z ∈ R^10**: [UWB 4거리, IMU 자세 3(roll/pitch/yaw), 자이로 3]
- **est_dim=12**: UWB+IMU로 관측성 rank 12 확보 → 발산 없음 (yaw오차 0.005rad)
- **N ∈ [4, 20]**: 관측성 하한 N=2, 노이즈여유 4, 정확도포화 20
- **앵커**: 4개 비평면, z 0~4로 수직분산 [(0,0,0)(10,0,4)(10,10,0)(0,10,4)]

## 실측 근거 (STAGE 2 게이트 PASS)
| 항목 | 값 |
|---|---|
| nominal 정확도 | 0.182m (DI-FME 0.140 근접) |
| nominal N_opt | 14 |
| 강풍 N_opt | 4 (이동 +10) |
| payload급변 N_opt | 6 (이동 +8) |
| 외란시 innovation | 2.1배 증가 (SAC 관측가능) |

핵심: **강한 외란**에서만 N_opt 이동. 약한 wind/1앵커 dropout은 고정 → 강한 외란 시나리오 필수.

---

## 로드맵 3단계

### STAGE A: Isaac 강한 외란 데이터 생성
기존 Isaac 파이프라인으로, **PX4가 못 잡는 강한 외란** 생성. GT가 실제로 교란되어야 N_opt 이동.

**시나리오별 config (commander.py 인자):**
```
# 공통: --est_dim 12 --meas_dim 10 --uwb_stride 5 --patterns helix,eight
#       --duration 40 --hz 50

# 1. nominal (기준선, 각 궤적 2 seed)
--force_type nominal --seeds 0,1

# 2. 강풍 (STRONG - 기존 6m/s는 PX4가 보상하니 15~20m/s 지속돌풍)
--force_type wind --wind_speed 18 --wind_sustained --window_s 15,25 --seeds 0,1

# 3. payload 급변 (질량 +100%, 기존 +50%보다 강하게)
--force_type payload --mass_scale 2.0 --window_s 15,25 --seeds 0,1

# (dropout은 IMU가 메꿔 약함 → 우선순위 낮음. 여유되면 2앵커:)
--force_type dropout --dropout_anchors 2 --window_s 15,22 --seeds 0,1
```

**IMU 측정 합성** (Isaac 재실행 불필요, GT에서):
- 자세측정 = gt[6:9] + N(0, 0.02²) rad
- 자이로측정 = gt[9:12] + N(0, 0.01²) rad/s
- UWB = ‖gt[0:3] - anchor‖ + N(0, 0.12²) m

**명령어:**
```bash
# Isaac 실행 (isim 워크플로우)
isim datagen/commander.py --force_type wind --wind_speed 18 --wind_sustained \
     --window_s 15,25 --patterns helix,eight --seeds 0,1 \
     --est_dim 12 --meas_dim 10 --uwb_stride 5 --out data_imu/

# 생성 확인
python -m tools.smoke_patterns --data data_imu/ --est_dim 12 --meas_dim 10
```

### STAGE B: N_opt 이동 게이트 (SAC 학습 전 필수)
```bash
python -m tools.stage_b_gate --data data_imu/ \
     --scenarios nominal,wind,payload --N_grid 4,6,8,10,12,14,16,18,20 \
     --est_dim 12 --meas_dim 10 --uwb_stride 5
```
**통과 기준**: 어느 강한 외란서든 N_opt 이동 ≥2 AND corr(UWB innovation, N_opt) 유의.
(컨테이너 실측: 강풍 +10, payload +8 → 통과 예상)

### STAGE C: SAC 학습
```bash
# 스모크 (50k, 발산·alpha붕괴 확인)
python -m rlenv.train --data data_imu/ --steps 50000 --smoke \
     --est_dim 12 --meas_dim 10 --N_min 4 --N_max 20 --uwb_stride 5 \
     --action N,lambda --obs channel_innov

# 풀 학습 (200k)
python -m rlenv.train --data data_imu/ --steps 200000 \
     --est_dim 12 --meas_dim 10 --N_min 4 --N_max 20 --uwb_stride 5 \
     --episode_len 400 --disturb_frac 0.7 \
     --action N,lambda --obs channel_innov --reward neg_pos_err

# 평가
python -m rlenv.evaluate --ckpt runs/latest --data data_imu/heldout/ \
     --baselines fixed_N,ekf,greedy,regime_oracle
```

---

## config.py 핵심값 (이미 수정됨)
```python
est_dim = 12          # 전체 상태 추정 (IMU 융합)
meas_dim = 10         # UWB 4 + IMU 자세 3 + 자이로 3
N_min = 4             # 관측성 rank12는 N=2, 여유 4
N_max = 20            # UIFM-SLAC 정합, 정확도 포화
N_default = 14        # nominal N_opt
uwb_stride = 5        # 측정 10Hz (예측은 50Hz)
warmup_steps = 20     # = N_max
anchors: z 0~4 수직분산
meas_sigma = (0.12×4, 0.02×3, 0.01×3)  # UWB / 자세 / 자이로
```

## SAC 인터페이스
- **액션**: (N, λ), N∈[4,20], λ∈[0.9,1.0]
- **관측**: [‖ν_UWB‖/σ, ‖ν_att‖/σ, ‖ν_gyro‖/σ, N_prev, λ_prev] × L스텝 (채널별 정규화)
- **보상**: -‖p_gt - p̂‖

## 남은 코드 작업 (Claude Code)
1. dataset 측정층: IMU 측정 합성 (자세·각속도 + 노이즈) 추가
2. SAC 관측: 채널별 innovation norm (UWB/att/gyro) 정규화 구현
3. commander.py: --wind_speed/--wind_sustained/--mass_scale 강한 외란 인자
4. tools/stage_b_gate.py: N_opt 이동 + corr 측정 스크립트
5. IMU 융합 단위테스트 (현재 test_wfme는 UWB-only 4차원 강제)
