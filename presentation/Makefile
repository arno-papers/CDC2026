.PHONY: all clean render

FIGS = ../paper/figures/monod_comparison.png \
       ../paper/figures/monod_posterior.png \
       ../paper/figures/haldane_comparison.png \
       ../paper/figures/pharma_nuisance.png \
       ../paper/figures/dcmotor_timing.png

all: slides.pdf

slides.pdf: slides.tex $(FIGS)
	pdflatex -interaction=nonstopmode -halt-on-error slides.tex
	pdflatex -interaction=nonstopmode -halt-on-error slides.tex

# PNGs at projector scale, for reviewing the deck page by page
render: slides.pdf
	mkdir -p build/rendered
	rm -f build/rendered/slide-*.png
	pdftoppm -png -scale-to-x 1920 -scale-to-y -1 slides.pdf build/rendered/slide

clean:
	rm -f slides.aux slides.log slides.nav slides.out slides.snm slides.toc slides.vrb
	rm -rf build
