#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
latexmk -xelatex main.tex
echo "Built: $(pwd)/build/main.pdf"
