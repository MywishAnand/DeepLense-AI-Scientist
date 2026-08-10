# Build the ML4PS workshop paper.
#
#   make            -> paper/main.pdf via tectonic (brew install tectonic)
#   make latexmk    -> alternative via a full TeX distribution (MacTeX)
#   make clean
#
# The official neurips_2025.sty is vendored in this directory (from the NeurIPS
# Styles.zip). TODO: swap for the ML4PS 2026 template when it is released.

.PHONY: all latexmk clean

all: main.pdf

main.pdf: main.tex references.bib neurips_2025.sty
	tectonic main.tex

latexmk:
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.fls main.fdb_latexmk
