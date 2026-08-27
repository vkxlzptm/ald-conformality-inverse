# 인수인계 — ALD Conformality Inverse Problem

작성 2026-08-27. 설계 논의 세션이 길어져 새 세션으로 넘김.
**기술 내용은 `README.md` 가 정본이다. 이 문서는 "왜 그렇게 됐는지"와 "지금 어디까지 왔는지"만 담는다.**

---

## 1. 맥락

- 이동현. ASM International **Senior Engineer II, Product Intelligence** 서류 통과 → 인터뷰 예정.
  직무 관련 ML 미니 프로젝트를 만들어 어필하려 함. **인터뷰 날짜는 아직 미확인.**
- 부수 타깃: SK하이닉스 소자/TCAD, LG Display VDE 소자. LGD 적합도는 낮고, 억지로 맞추지 않기로 함.
- 기존 자산: SevenNet(7net-Nano-4.5) a-SiO₂ MLIP 벤치마크, Cu 나노입자 DFT(Br/DMF), 2019년 딥러닝
  역문제 프로젝트(MEM → CNN, 비음수 제약을 출력층에, 고노이즈에서 RMSE 75%↓).
- 계산 자원: MacBook Air(개발용) + 원격 데스크톱(`myDesktop_at_ELST`, **OS·코어 수 미확인**).
  DFT 돌릴 환경은 있으나 시간이 부족해 이번 프로젝트에서는 쓰지 않기로 함.

## 2. 최종 채택안

**고AR trench 의 step coverage 프로파일 한 장에서 표면반응 파라미터를 역추정하는 amortized inference.**
측정마다 반복되는 시뮬레이션 최적화 루프를 학습된 추론기가 대체한다. 상세는 README.

## 3. 폐기한 대안 — 다시 꺼내지 말 것 (이유 포함)

| 대안 | 폐기 이유 |
|---|---|
| MLIP → kMC → PINN 닫힌 루프 | 층이 4개라 미니 프로젝트 크기가 아님. ML 밀도도 낮음 |
| MLIP fine-tuning + active learning | ML 밀도는 최상이나 kMC/PINN/TCAD 가 전멸. 사용자가 그 계열 유지를 원함 |
| Neural operator (도펀트 확산 TCAD) | 하이닉스엔 정조준이나 **ASM 적합도가 낮음**. 원자 스케일 없음 |
| **Langmuir 포화 출력층** (dose→GPC) | **NN 필연성이 없음.** 파라미터 2개 + 스칼라 관측 3개 = 그냥 최소제곱. 면접에서 바로 찔림 |
| PINN drift-diffusion | ASM 무관 + 수렴 난이도 악명 |
| Δ-learning, UQ 툴, 문헌 LLM 파이프라인 | 단독으론 얇음. 부록으로만 |

**핵심 판정 기준**(재검토 시 이 기준을 쓸 것): amortized NN 이 최소제곱을 이기려면
① 순방향에 닫힌 해가 없음 ② 파라미터 다수 ③ 관측이 고차원 ④ 반복 추정 — 중 최소 2개.
채택안은 ①②③ 충족. Langmuir 안은 0/4 였다.

## 4. 대화 중 발견해서 정정한 것 (중요)

이것들은 **AI가 처음에 틀리게 말했다가 실측으로 뒤집힌 것들**이다. 다시 반복하지 말 것.

1. **`dose ∝ AR²` 를 코드 게이트로 쓰려 했음 → 철회.**
   Gordon 의 AR² 는 *큰 a · 완전 포화 기준 · 수송율속* 점근이지 보편 법칙이 아니다.
   실측 지수: s₀=0.05→0.39, 0.2→0.81, 0.6→1.21.
   **게이트는 대신 "무반응 투과확률"** 을 쓴다 (2D slit 은 ln(AR)/AR, 실측 −0.65, 고전 결과와 일치).
2. **Π₁ = s₀·AR 이라 했다가 → AR·√s₀ 로 정정.**
   1D 반응–확산에서 침투깊이 λ ∝ W/√s₀. 실측 형상 붕괴로 확인 (`src/pi_collapse.py`).
   AR·s₀ 묶음은 바닥/입구 비가 0.41/0.26/0.17 로 안 붕괴, AR·√s₀ 묶음은 0.41/0.42/0.45 로 붕괴.
3. **표기 β₀ → `s`, `s₀`** (Cremers 리뷰 표준). γ 는 플라즈마용, β 는 통용 표기 아님.
4. **"Cu deposition 을 ASM 지원서에 ongoing 으로 썼다"는 AI 의 잘못된 기억.**
   업로드된 CV·커버레터엔 없음. 포털 입력란은 미확인.
5. 코드의 `reemit` 은 **각분포 파라미터가 아니라 재방출 생존확률**이다. README 파라미터 표기와
   어긋났던 것을 바로잡음. 각분포를 추정 대상으로 하려면 정반사 비율/cos^m 지수를 새로 넣어야 함.

## 5. 현재 상태

**저장소**: `https://github.com/vkxlzptm/ald-conformality-inverse` (레포 생성 완료)
- 로컬 경로: 맥 `~/projects/PINN/ald-conformality-inverse` (**마스터 폴더 `PINN` 은 개명 예정**,
  후보 `semicon-ml-portfolio`). 데스크톱은 clone 예정.
- `git init` + remote 설정 완료. **최초 push 는 아직 미완료** (HTTPS→SSH 전환 및 이메일 noreply
  변경 안내한 시점에서 세션 종료).
- 커밋 이메일은 `41114918+vkxlzptm@users.noreply.github.com` 사용 (모든 주소가 Private 이라
  실주소로 push 하면 GH007 로 막힐 수 있음). SSH 키는 두 머신 모두 등록되어 있음.

**코드** (전부 동작 확인됨)
- `src/trench_mc.py` — 2D trench 탄도 MC. 단독 실행 시 설명 그림 4패널 생성
- `src/ar_scaling.py` — 투과확률 / 포화 dose vs AR 실측
- `src/fig_ar_scaling.py`, `src/pi_collapse.py` — 그림 생성
- `environment.yml` — conda env 이름 **`pinn`**
- `sync.sh` — 양방향 동기화. 저장소 체크 · 20MB 초과 경고 · 최초 push 처리 포함

**아직 없는 것**: `generate_dataset.py`, 학습 코드, baseline 최소제곱, 검증 스크립트 전부.

## 6. 다음 할 일 (순서대로)

1. **미결정 3개를 먼저 닫을 것** (아래 7절)
2. `generate_dataset.py` — shard 방식. 파일 있으면 skip = 이어하기.
   `seed = shard_id*100003 + i` (재현성보다 **워커 간 RNG 충돌 방지**가 이유).
   shard 수 ≈ 코어 수 × 5~10. 8코어면 500개×40shard.
   **소요시간 실측**: 1회 1.3~14.5초 (s₀ 작을수록 비쌈). 20,000개 ≈ 28 코어·시간.
3. 완전 포화 기준(99%)으로 AR 지수 재측정 → 2 에 접근하는지 확인
4. 1D CNN + Mixture Density head 학습 (Π₁~Π₄ 무차원 입력, log 공간)
5. 최소제곱 baseline 정면 대결 → 검증 4종 (README §5)

## 7. 미결정 — 새 세션에서 먼저 물을 것

- [ ] **기하**: 2D slit 유지 vs 3D 축대칭 원통. 원통이면 Clausing 인자 대조 가능 + Gordon 정의와 정합
- [ ] **하이브리드 kMC 도입 여부**. T 를 입력으로 쓰려면 필수이고, "kMC" 라는 표현도 정직해진다.
      (현재 코드는 탄도 MC 일 뿐 kMC 가 아니다 — 그대로면 kMC 라 부르면 안 됨)
- [ ] **인터뷰까지 남은 기간** → 범위 조정. 아직 안 물어봄
- [ ] 원격 데스크톱 **OS·코어 수** → shard 크기 확정
- [ ] 마스터 폴더 개명

## 8. 작업 방식 메모

- 사용자는 **과잉 주장·근거 약한 진술을 즉시 지적**한다. 실제로 이 세션에서 AI 의 주장 4건이
  사용자 질문으로 뒤집혔다. **모르면 모른다고 하고, 가능하면 돌려서 확인할 것.**
- 압축된 스캔 가능한 출력, 결론 먼저, 그 다음 근거.
- 그림 텍스트는 **영어**, 축 라벨은 문장형 대문자 시작.
- 문서·코드 산출물은 폴더에 바로 커밋. Claude 프로젝트 문서도 같이 갱신.
- 에이전트를 많이 쓰는 대규모 작업은 **사전 확인**을 받을 것.
- 정직성 원칙(README §8)을 발표 자료에도 유지 — 합성 데이터임을 첫 슬라이드에 명시,
  step coverage 역추출이 기존 방법임을 인정하고 기여를 "상각 + 불확실성 정량화" 로 한정.
