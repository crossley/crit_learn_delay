# # Define default rule
all:
	pdflatex cover_letter.tex
	pdflatex cover_letter.tex
	pdflatex main.tex
	pdflatex main.tex
	biber main
	pdflatex main.tex
	pdflatex main.tex
	pdflatex response_to_reviews.tex
	pdflatex response_to_reviews.tex
	biber response_to_reviews
	pdflatex response_to_reviews.tex
	pdflatex response_to_reviews.tex
	pdflatex main_revision.tex
	pdflatex main_revision.tex
	biber main_revision
	pdflatex main_revision.tex
	pdflatex main_revision.tex
	latexdiff --flatten \
	  --exclude-textcmd=section \
	  --exclude-textcmd=subsection \
	  --exclude-textcmd=subsubsection \
	  main.tex main_revision.tex > main_diff.tex
	pdflatex main_diff.tex
	pdflatex main_diff.tex
	biber main_diff
	pdflatex main_diff.tex
	pdflatex main_diff.tex

# Define clean rule
clean:
	rm -rf *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *fff *gz *ttt

.PHONY: all clean
