#!/usr/bin/env bash
# Upload this package to GitHub.
# Usage:  bash upload_to_github.sh [repo-url] [branch]
# Requires git and push access (authenticate with `gh auth login` or a personal access token).
set -euo pipefail

REPO_URL="${1:-https://github.com/thanmayeethella/DNA-Methylation-Based-Biological-Age-Prediction-and-Epigenetic-Biomarker-Discovery.git}"
BRANCH="${2:-main}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"

git clone "$REPO_URL" "$WORK/repo" 2>/dev/null || { mkdir -p "$WORK/repo"; git -C "$WORK/repo" init -q; git -C "$WORK/repo" remote add origin "$REPO_URL"; }
cd "$WORK/repo"
git checkout -B "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"

rsync -a --exclude '.git' --exclude 'upload_to_github.sh' --exclude '*.aux' --exclude '*.log' --exclude '*.out' \
      --exclude '*.fls' --exclude '*.fdb_latexmk' "$SRC_DIR"/ ./

git add -A
git commit -m "Update code, figures and manuscript for ARRAY-D-26-07385" || echo "Nothing new to commit"
git push -u origin "$BRANCH"
echo "Uploaded to $REPO_URL (branch $BRANCH)"
