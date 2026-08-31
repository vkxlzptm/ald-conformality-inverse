# 인수인계 — ALD Conformality Inverse Problem

최초 2026-08-27. **최종 갱신 2026-08-31 (세션 3: 학습·평가 완료, 누설 게이트 통과, 전이 시험 코드 작성).**
**기술 내용은 `README.md` 가 정본이다. 이 문서는 "왜 그렇게 됐는지"와 "지금 어디까지 왔는지"만 담는다.**

---

## 1. 맥락

- 이동현. ASM International **Senior Engineer II, Product Intelligence** 서류 통과 → 인터뷰 예정.
  **인터뷰까지 2주 이내** (정확한 날짜는 여전히 미확정). 범위를 여기 맞춰 잘랐다.
- 부수 타깃: SK하이닉스 소자/TCAD, LG Display VDE. LGD 적합도는 낮고, 억지로 맞추지 않기로 함.
- 기존 자산: SevenNet(7net-Nano-4.5) a-SiO₂ MLIP 벤치마크, Cu 나노입자 DFT(Br/DMF), 2019년 딥러닝
  역문제 프로젝트(MEM → CNN, 비음수 제약을 출력층에, 고노이즈에서 RMSE 75%↓).
- 계산 자원: MacBook Air(개발용) + 원격 데스크톱 `myDesktop_at_ELST` — **Linux, 6코어 / 12스레드**.
  DFT 는 이번 프로젝트에서 쓰지 않는다.

## 2. 최종 채택안

**고AR 원통 via 의 step coverage 프로파일(온도 3점 세트)에서 표면반응 파라미터를 역추정하는
amortized inference.** 측정마다 반복되는 시뮬레이션 최적화 루프를 학습된 추론기가 대체한다.
상세는 README.

## 3. 폐기한 대안 — 다시 꺼내지 말 것 (이유 포함)

| 대안 | 폐기 이유 |
|---|---|
| MLIP → kMC → PINN 닫힌 루프 | 층이 4개라 미니 프로젝트 크기가 아님. ML 밀도도 낮음 |
| MLIP fine-tuning + active learning | ML 밀도는 최상이나 kMC/PINN/TCAD 가 전멸 |
| Neural operator (도펀트 확산 TCAD) | 하이닉스엔 정조준이나 **ASM 적합도가 낮음** |
| **Langmuir 포화 출력층** (dose→GPC) | **NN 필연성이 없음.** 파라미터 2개 + 스칼라 관측 3개 = 그냥 최소제곱 |
| PINN drift-diffusion | ASM 무관 + 수렴 난이도 악명 |
| **하이브리드 kMC** | 세션 2 에서 기각. 근거는 4절 5번 |
| Δ-learning, UQ 툴, 문헌 LLM 파이프라인 | 단독으론 얇음. 부록으로만 |

**핵심 판정 기준**: amortized NN 이 최소제곱을 이기려면
① 순방향에 닫힌 해가 없음 ② 파라미터 다수 ③ 관측이 고차원 ④ 반복 추정 — 중 최소 2개.
채택안은 ①②③ 충족 (온도 3점으로 ③ 이 더 강해짐). Langmuir 안은 0/4 였다.

## 4. 대화 중 발견해서 정정한 것 (중요)

**AI 가 처음에 틀리게 말했다가 실측·논증으로 뒤집힌 것들.** 다시 반복하지 말 것.

1. **`dose ∝ AR²` 를 코드 게이트로 쓰려 했음 → 철회.**
   Gordon 의 AR² 는 *큰 a · 완전 포화 기준 · 수송율속* 점근이지 보편 법칙이 아니다.
   2D 실측 지수: s₀=0.05→0.39, 0.2→0.81, 0.6→1.21. 게이트는 투과확률을 쓴다.
2. **Π₁ = s₀·AR 이라 했다가 → AR·√s₀ 로 정정.** 1D 반응–확산에서 λ ∝ W/√s₀.
   2D·3D 모두 실측 형상 붕괴로 확인.
3. **표기 β₀ → `s`, `s₀`** (Cremers 리뷰 표준). γ 는 플라즈마용, β 는 통용 표기 아님.
4. **"Cu deposition 을 ASM 지원서에 ongoing 으로 썼다"는 AI 의 잘못된 기억.**
5. 코드의 `reemit` 은 **각분포 파라미터가 아니라 재방출 생존확률**이다.
6. **(세션 2) "T 를 입력으로 쓰려면 하이브리드 kMC 가 필수" → 틀린 근거였다.**
   단일 프로파일에서 T 의 유일한 효과가 s₀(T) 뿐이면 T 는 정보를 추가하지 않는다
   (순수한 재매개변수화, (A,Ea) 는 1-parameter family 로 비식별). 필요한 것은 rate 이벤트가
   아니라 **온도를 바꾼 관측 여러 개**. → kMC 기각, 온도 3점 세트 채택. README §4-e.
7. **(세션 2) 문헌 Clausing 인자 표를 기억으로 인용하려다 폐기.**
   짧은 관(L/R=1,2)은 맞았지만 긴 관 값(L/R≥10)이 실제 계산과 3 % 어긋났다.
   기억으로 적은 값이 틀렸던 것. → **적분방정식을 직접 풀어 기준을 만들었다** (`clausing_ref.py`).
   문헌 수치를 기억에서 인용하지 말 것.

8. **(세션 2) 조건 입력으로 Π₂(사이트당 dose)를 주려던 설계 → 폐기.**
   Π₂ 의 분모가 추정 대상(사이트밀도)이라 순환이고, 실측 결과 사이트밀도가 **정확히** 비식별이
   된다 (n_sites 2배에 프로파일 변화 0.0000). 조건은 **절대 dose** 로 준다. README §4-f.
   baseline 첫 실행에서 5케이스 전부 같은 방향으로 틀린 것을 보고 파고들어 찾았다 —
   *오차가 크다*가 아니라 *오차의 방향이 일관된다*가 단서였다.

9. **(세션 2) 첫 학습 결과가 너무 좋아서 파봤더니 데이터 누설.**
   Π₂ 가 고정 6점 격자라 dose 에서 사이트밀도가 역산됐다 (오라클 오차 0.00 %).
   Π₂ 에 draw 단위 log 지터를 넣어 차단, 데이터셋 재생성. README §4-g.
   **결과가 기대보다 좋으면 먼저 누설을 의심할 것.**

10. **(세션 2) 관측 단위와 무차원화를 전면 재정의.** 사용자 질문으로 잡힌 것 세 개:
    (i) 온도 3점 세트를 "측정 1건"으로 부르는 게 부자연스럽다 → **웨이퍼 1장 단위**로 바꾸고
        Arrhenius 를 네트워크 밖 선형회귀로 분리. 임의 온도·개수에 재학습 없이 적용된다.
    (ii) Ea·사이트밀도·dose 가 단위를 남긴다 → **입출력을 전부 무차원으로**.
        출력이 (s₀, n, p_reemit, **Π₂**) 가 되고 Π₂ 가 입력에서 출력으로 자리를 옮겼다.
    (iii) 절대 dose 는 프로파일이 이미 담고 있어 **입력에서 완전히 뺐다**.
    → 데이터 재생성 불필요 (저장값에서 전부 유도). README §2 가 정본.
    **교훈: 코드 짜기 전에 "무엇을 정하고/재고/모르는가" 표를 먼저 못 박을 것.**
    이 표가 없어서 하루에 관측 모델을 세 번 고쳤다.

11. **(세션 3) "AR 이 높으면 바닥이 안 덮이는 게 당연하다" → 원인이 AR 이 아니었다.**
    사용자가 "표면확산을 무시해서 그런 것 아니냐"고 물었고, 실제로 재보니 둘 다 아니었다.
    가르는 것은 **바운스당 소실 (1−p_reemit) 대 s₀ 의 경쟁**이고, prior 에서 **44 % 의 draw 가
    소실 > s₀** 인 영역이다 = ALD window 밖. 표면확산 무시는 문헌 표준(Cremers, Gordon)이고
    실제 ALD 의 고AR conformality 도 흡착종 확산이 아니라 전구체 재방출이 만든다 (모델에 있음).
    → 목표치를 낮추는 대신 **전이 시험 케이스를 ALD 영역으로 필터**. README §4-h.
    **"당연하다"로 넘기려던 것을 사용자가 붙잡아서 잡힌 건이다.**

12. **(세션 3) leak_test 실측이 README 의 12.2 % 와 어긋났다 → 19.5 % 로 정정.**
    옛 수치는 더 작은 데이터셋에서 잰 것. 정본을 재생성본 기준으로 갱신했다.

13. **(세션 3) Π₁ 이 근거 없이 출력에서 사라져 있었다 — 사용자가 발견.**
    원안(§4-b)은 NN 이 Π₁·Π₂·Π₃·Π₄ 를 추정하는 것이었다. Π₂ 이동(§4-f·§4-g)과
    Π₄ 외부화(§2-4)는 근거가 기록돼 있는데, **Π₁ → s₀ 는 기록이 없었다.**
    §4-b 가 §2 와 모순인 채로 남아 있었고(“NN 은 Π 를 추정”, “dose·T 가 알려진 조건”),
    세션 3 에서 §4-b 를 현재 사양으로 고쳤다.
    AR 이 조건이라 표현력은 사실상 동일하고, 실제 차이는 손실의 z-정규화 기준뿐이다
    (실측: log s₀ 의 sd 1.51 vs log Π₁ 의 sd 0.88 — 무시할 차이는 아니다).
    → **추측 대신 소거 실험으로 결론낸다** (`src/ablation_param.py`, 6절 2번).
    **교훈: 사양을 바꿀 때 옛 절을 같이 고치지 않으면 정본이 둘이 된다.**

## 5. 현재 상태 (세션 3 종료 시점)

**저장소**: `https://github.com/vkxlzptm/ald-conformality-inverse`
- 맥 `~/projects/PINN/ald-conformality-inverse` (작업·편집·baseline 실행)
- 데스크톱 `~/projects/PINN/ald-conformality-inverse` (데이터 생성·학습). Linux 6코어/12스레드
- 커밋 이메일 `41114918+vkxlzptm@users.noreply.github.com`, SSH 키 양쪽 등록됨
- 동기화는 맥에서 `./sync.sh` → 데스크톱에서 `git pull`

> **브리지 주의**: AI 가 `device_bash` 로 git 을 돌리면 `.git/index.lock` 이 남고 지울 수 없다.
> **브리지에서 git 명령을 실행하지 말 것.** 커밋·push 는 사용자가 직접.

**코드 — 전부 커밋됨, 컨테이너에서 전 경로 스모크 테스트 통과**

| 파일 | 상태 |
|---|---|
| `src/cyl_mc.py` | 3D 축대칭 원통 탄도 MC ✅ Clausing 대조 ≤0.25 % |
| `src/clausing_ref.py` | Clausing 적분방정식 결정론적 해 ✅ 자체 검증 통과 |
| `src/cyl_run.py` | dose 체크포인트 커널 ✅ |
| `src/generate_dataset.py` | 데이터 생성. shard·이어하기·Π₂ 지터 ✅ |
| `src/data.py` | 로딩·온도 단위 분해·측정 모델·무차원 변환 ✅ |
| `src/model.py` | 1D CNN + MDN (in 3ch, cond 1, out 4, 649,512 params) ✅ |
| `src/train.py` | 학습 루프 ✅ |
| `src/evaluate.py` | 정확도·보정·baseline 대조 ✅ |
| `src/arrhenius_fit.py` | ln s₀ vs 1/T 회귀 → Ea(eV), η ✅ |
| `src/baseline_ls.py` | 최소제곱 baseline (무차원 4개 미지수) ✅ |
| `src/leak_test.py` | **전면 재작성 ✅ 게이트 통과** (4검사 + 지터제거 음성대조) |
| `src/transfer_ar.py` | **신규 — 검증 3 (AR 전이). 코드·비용 검증 완료, 본실행 미실시** |
| `src/fig_*.py` | 워크플로·구조·기하검증·NM 설명 그림 ✅ |

**데이터**
- `results/dataset/` — Π₂ 지터 적용본, **재생성 완료** (데스크톱). 600 shard,
  train 540 / val 30 / test 30 shard
- `results/dataset_v1_leaky/` — 지터 전 구본. 누설 있음. 쓰지 말 것
- 재생성 명령: `nohup python -u src/generate_dataset.py --shards 0-599 --workers 6 > gen3.log 2>&1 &`

**측정된 결과 (유효)**
- 투과확률 MC vs Clausing: AR 0.5~40 에서 ≤0.25 % 일치 → `figures/geometry_validation.png`
- Π₁ = AR√s₀ 형상 붕괴 (원통에서 ±8 %)
- **누설 게이트 통과** (`src/leak_test.py`, 600 shard): 6후보 열거 오차 19.5 %,
  조건만 kNN 의 n_sites 23.3 % (null 26.3 %). 지터 제거 음성대조는 각각 0.00 % / 1.85 %
  로 옛 누설을 재현 → 게이트에 이빨 있음. README §4-g
- **NN vs 최소제곱** (같은 벤치마크 8케이스) → `figures/evaluation.png`

  | | NN | 최소제곱 |
  |---|---|---|
  | s₀ | 24 % | 72 % |
  | n | 9 % | 23 % |
  | 재방출 | 0.3 pt | 3.4 pt |
  | Π₂ | 28 % | 39 % |

  보정 곡선 4개 모두 대각선에 일치. **단, in-distribution 보정은 쉬운 쪽이다** —
  학습·시험이 같은 시뮬레이터·같은 prior. 발표에서 여기에 무게를 싣지 말 것.
- **Arrhenius** → `figures/arrhenius.png`. 산포는 넓으나 보정 정확. 낮은 Ea 를 못 맞추는 것은
  §4-e 예측대로 (15 kJ/mol ≈ 155 meV 아래는 노이즈에 묻힘). 예시: 참 301 meV → 314 [286, 338]
- **재방출 prior 문제 발견** — README §4-h. 학습셋 44 % 가 ALD window 밖

**무효 — 다시 돌려야 하는 것**
- `docs/nelder_mead_explained.png` (옛 조건으로 계산된 지형. `--redo` 필요)

**아직 없는 것**: 검증 3 본실행, 검증 4(모델 오설정), 발표자료.

## 6. 다음 할 일

**즉시**
1. **검증 3 본실행** (데스크톱 — `results/model/best.pt` 가 거기에만 있다):
   ```
   python src/transfer_ar.py --dry-run                       # 예산 확인
   nohup python -u src/transfer_ar.py --cases 6 --samples 100 --workers 6 \
       > transfer.log 2>&1 &                                 # 10~15분 예상
   ```
   붙이려면 `--ls-budget 1500` (케이스당 1500 시뮬 콜 추가 — 이게 바로 NN 이 없앤 비용이다).
   실측 예비값 (정답 파라미터, 6케이스): 필요 Π₂ 46~138, **dose 배수 15~35배**, censoring 0.
2. **Π₁ vs s₀ 소거 실험** (§6, README §4-b 의 미기록 변경을 실측으로 결론내는 자리).
   **순서 중요 — 2-a 를 빼먹으면 대조군이 사라져 소거 실험이 성립하지 않는다.**
   ```
   cp results/model/best.pt results/model/best_s0.pt          # 2-a 대조군 백업
   nohup python -u src/train.py --target-param pi1 \
       --out results/model_pi1 --epochs <현행과 동일> > train_pi1.log 2>&1 &
   python src/ablation_param.py --models results/model/best.pt \
                                         results/model_pi1/best.pt
   ```
   에폭 수·seed·데이터는 현행 학습과 **똑같이** 맞춰야 공정한 대조다 (`results/model/history.json`
   에서 에폭 수 확인). 결과를 README §6 표에 숫자로 적는다.
3. `docs/nelder_mead_explained.png` 재생성 (`python src/fig_nelder_mead.py --redo`)
4. 발표자료

**시간 남으면**
5. `reemit` prior 를 `U(0.98, 1.0)` 등으로 좁혀 **데이터 재생성 + 재학습** (§4-h 근본 해결, 5시간)
6. 검증 4 (모델 오설정), full-covariance MDN 헤드, MLP vs CNN 소거 실험,
   완전 포화 기준 AR 지수 재측정

**세션 3 에서 끝난 것**: 데이터 재생성, 학습, `evaluate.py --cases 8`, `arrhenius_fit.py`,
`leak_test.py` 재작성 + 게이트 통과, `transfer_ar.py` 작성.

## 7. 미결정

- [ ] 인터뷰 **정확한 날짜** (2주 이내만 확인)
- [ ] 완전 포화 기준(99 %)으로 원통에서 AR 지수가 2 에 접근하는지 — 미측정
- [ ] 마스터 폴더 `PINN` 개명 (후보 `semicon-ml-portfolio`)
- [ ] 순방향 대리모델(surrogate) 도입 여부 — 데이터는 이미 있음. 지금은 **안 함** (검증은 진짜 MC 로)
- [ ] `reemit` prior 재설정 + 재생성 여부 (§4-h). 지금은 전이 시험 케이스 필터로 대응 중

## 8. 작업 방식 메모

- 사용자는 **과잉 주장·근거 약한 진술을 즉시 지적**한다. 세션 1 에서 AI 주장 4건, 세션 2 에서
  2건이 뒤집혔다 (4절). **모르면 모른다고 하고, 가능하면 돌려서 확인할 것.**
  특히 **문헌 수치를 기억에서 인용하지 말 것** — 4절 7번.
- 압축된 스캔 가능한 출력, 결론 먼저, 그 다음 근거.
- 그림 텍스트는 **영어**, 축 라벨은 문장형 대문자 시작.
- 문서·코드 산출물은 폴더에 바로 커밋. Claude 프로젝트 문서도 같이 갱신.
- 에이전트를 많이 쓰는 대규모 작업은 **사전 확인**을 받을 것.
- 발표 자료에도 정직성 원칙(README §8) 유지 — 합성 데이터 명시, **kMC 라고 부르지 않기**,
  기여를 "상각 + 불확실성 정량화" 로 한정.
