#!/bin/bash
set -e
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode main
xelatex -interaction=nonstopmode main
