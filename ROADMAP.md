# SAC-AFME 로드맵 — Isaac 검증까지

목표: 합성 데이터를 **Isaac Sim 주행/UWB 환경의 축소 모형**으로 만들어 SAC-AFME가
실제로 작동하는지 검증하고, Isaac 데이터로 최종 실험한다.

핵심 교훈(검증 루프에서 확정):
- (N,λ) 적응 레버 = 측정 열화 × 모델오차의 교차점. plant 킥 단독·지속편향은 무효.
- N_opt가 유한하려면 필터 모델과 plant 사이 **정직한 모델오차**가 필요 (Shmaliy:
  결정론적 세계는 full-horizon이 최적 → 지금까지 학습가능분 0%의 원인).
- 올바른 캘리브레이션 타깃 = **고정-N U커브 바닥 = 14** (DI-FME 실측값).
  greedy 중앙값은 스텝별 운의 분포라 타깃으로 부적합.
- turbulence_burst는 oracle-KF조차 FIR에 지는 유일 검증 시나리오 (Shmaliy Fig.10 재현).

불변 제약: 측정=UWB 4거리만 / 산출=위치 xyz / filter/wfme.py 무수정(T1~T5 유지) /
N∈[8,20], FIR 베이스라인 N=14 / 급기동(aggressive) 영구 금지.

---

## Phase 0 — 메커니즘 확인 (수 시간) ← 현재
q 노브로 U커브 재캘리브레이션. 질문: "모델오차가 충분하면 구간별 N_opt 이동이
물리적으로 존재하는가?"
- [ ] q 스윕 → nominal 고정-N U커브 바닥 = 14 (진짜 U자 확인)
- [ ] turbulence burst 구간 U커브 바닥 좌측 이동 (기대 ~7-9) — 논문 motivation figure
- [ ] 분해표: LEARNABLE% = regime-oracle − best-fixed
- [ ] nominal EKF vs FIR14 격차 축소 확인 / corr 재측정
**GO 기준: LEARNABLE% ≥ 10% → Phase 1. / ~0% → 프레이밍 전환 결정
(자동튜닝: DI-FME는 trial-and-error로 N 선택 + 회복속도 3~4× + 난류·NLoS 강건).**

## Phase 1 — Isaac-충실 합성 업그레이드 (1~2일)
plant에 Isaac(PhysX+PX4)에 실재하는 효과 주입, 필터 모델은 그대로 (격차=정직한 모델오차):
1. 미모델 공력 항력: F_drag = −c1·v − c2·|v|·v (plant만)
2. 로터/액추에이터 1차 지연 τ≈30~80ms (필터는 명령 u를 그대로 믿음)
3. 파라미터 오프셋: plant 질량/관성 ±3~8% 상시
4. 상시 미세 난류 (nominal 기본) + turbulence_burst = 강도 3~5×
5. UWB층 유지 (σ_LoS=0.12, NLoS burst, dropout) + 앵커별 σ 이질성
- 캘리브레이션: q 주입이 아니라 위 물리 강도로 U커브 바닥=14 재현
**통과: U자 + burst 바닥 좌측이동 + LEARNABLE% ≥ 10% + nominal 격차 축소.**

## Phase 2 — SAC 본학습 검증 (1~2일, RTX 6000)
- train.py 300k~ 스텝 (GPU), evaluate.py
- **통과: SAC ∈ [best-fixed, regime-oracle] 구간 안착.** regime-oracle 도달률 %로
  학습 품질 정량화 (분해표가 상한을 주므로 가능 — 논문 분석 포인트).
- 전이/회복 지표(피크·회복시간·ITAE) 병기. alpha 붕괴 없는지 로그 확인.

## Phase 3 — Isaac 파이프라인 (2~4일, Phase 2와 병렬 가능)
- Isaac 로깅: GT 12상태 + 제어입력 u + 타임스탬프, 50Hz, **합성과 동일 npz 스키마**
  → dataset/adapt_signal/train 전 스택 무수정 재사용
- UWB는 GT에서 오프라인 계산 + 측정층(노이즈/NLoS/dropout) 후처리 주입
  → 측정 시나리오 변경 시 Isaac 재실행 불필요
- 비행: hover/circle/figure8/waypoint + Isaac 바람 플러그인 난류. 급기동 금지.
- 최난점: PX4 로그 u → 모델 입력 [추력, 토크] 매핑 (단위/좌표계 검증 필수;
  매핑 오차 자체는 모델오차로 흡수됨)

## Phase 4 — Isaac 검증 & 논문 실험
- Isaac 데이터에 **adapt_signal 먼저** (SAC 아님): 자연 N_opt 위치(DI-FME 14와 비교),
  U커브 이동, LEARNABLE% 재확인
- 합성 학습 정책 zero-shot 평가 → 필요시 Isaac fine-tune
- 최종 표: 시나리오 × {EKF실무/EKF-oracle/UKF/FIR14/DI-FME/SAC-AFME},
  구간별 RMSE + 회복시간 + regime-oracle 도달률
