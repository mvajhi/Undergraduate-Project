#!/bin/bash
set -e
cd "$(dirname "$0")"
xelatex -synctex=1 -interaction=nonstopmode main
xelatex -synctex=1 -interaction=nonstopmode main
