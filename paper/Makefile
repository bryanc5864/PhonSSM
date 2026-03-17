# PhonSSM Paper Makefile

.PHONY: all clean figures

all: main.pdf

main.pdf: main.tex references.bib figures
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

figures:
	python scripts/create_figures.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.synctex.gz *.pdf

watch:
	latexmk -pdf -pvc main.tex
