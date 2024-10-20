# Define default rule
all:
	pdflatex main.tex
	pdflatex main.tex
	biber main
	pdflatex main.tex
	pdflatex main.tex

# Define clean rule
clean:
	rm -rf *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *fff *gz *ttt

.PHONY: all clean
