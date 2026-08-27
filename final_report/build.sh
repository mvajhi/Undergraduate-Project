#!/bin/bash

xelatex -synctex=1 -interaction=nonstopmode main &&
{ bibtex8 -W -c cp1256fa main || true; } &&
{ xindy -L english -C utf8 -I xindy -M main.xdy -t main.alg -o main.acr main.acn || true; } &&
xelatex -synctex=1 -interaction=nonstopmode main &&
xelatex -synctex=1 -interaction=nonstopmode main
