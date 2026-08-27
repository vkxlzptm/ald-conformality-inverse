#!/usr/bin/env bash
# 양방향 동기화. 노트북·원격 어디서든 그냥 ./sync.sh
#
#   ./sync.sh              → .commit_msg 내용을 커밋 메시지로 사용
#   ./sync.sh "메시지"      → 인자를 커밋 메시지로 사용
#
# 하는 일: 잔여 lock 제거 → 대용량 파일 경고 → add -A → commit → pull(merge) → push
#
# 최초 1회 설정 (이 폴더에서):
#   git init -b main
#   git config user.name  "DongHyun Lee"
#   git config user.email "you@example.com"
#   git remote add origin git@github.com:<계정>/ald-conformality-inverse.git
#   ./sync.sh "initial commit"
set -u
cd "$(dirname "$0")" || exit 1

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "!! 이 폴더는 git 저장소가 아님. 위 주석의 '최초 1회 설정' 참고."
  exit 1
fi

rm -f .git/HEAD.lock .git/index.lock .git/refs/heads/*.lock 2>/dev/null

MSG="${1:-}"
if [ -z "$MSG" ]; then
  if [ -s .commit_msg ]; then
    MSG=$(cat .commit_msg)
  else
    MSG="sync from $(hostname -s) $(date +%F_%H%M)"
  fi
fi

# 20MB 넘는 파일이 스테이징될 참이면 경고 (데이터셋 실수 커밋 방지)
BIG=$(git ls-files -o -c --exclude-standard -z 2>/dev/null \
      | xargs -0 -I{} sh -c '[ -f "{}" ] && [ $(wc -c <"{}") -gt 20971520 ] && echo "{}"' 2>/dev/null)
if [ -n "$BIG" ]; then
  echo "!! 20MB 초과 파일이 있음:"
  echo "$BIG" | sed 's/^/     /'
  echo "   .gitignore 에 넣을 것. 계속하려면 Enter, 중단하려면 Ctrl-C."
  read -r _
fi

git add -A
if git diff --cached --quiet; then
  echo "[sync] 커밋할 변경 없음"
else
  git commit -m "$MSG" || exit 1
  echo "[sync] 커밋: ${MSG%%$'\n'*}"
fi

git config pull.rebase false
if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
  if ! git pull --no-rebase --no-edit; then
    echo ""
    echo "!! 충돌 발생. 아래로 상태 확인 후 수동 해결:"
    echo "   git status --short"
    echo "   git checkout --ours <파일>   # 또는 --theirs"
    echo "   git add <파일> && git commit --no-edit && git push"
    exit 1
  fi
  git push || exit 1
else
  echo "[sync] upstream 없음 → 최초 push"
  git push -u origin HEAD || exit 1
fi

echo ""
git log --oneline -3
