LATEX_CMD?=pdflatex
MKIDX_CMD?=makeindex
BIBTEX_CMD?=bibtex
LATEX_COUNT?=8
MANUAL_FILE?=refman

all: $(MANUAL_FILE).pdf

pdf: $(MANUAL_FILE).pdf

$(MANUAL_FILE).pdf: clean $(MANUAL_FILE).tex
	$(LATEX_CMD) $(MANUAL_FILE) || \
	if [ $$? != 0 ] ; then \
	        \echo "Please consult $(MANUAL_FILE).log to see the error messages" ; \
	        false; \
	fi
	$(MKIDX_CMD) $(MANUAL_FILE).idx
	$(LATEX_CMD) $(MANUAL_FILE) || \
	if [ $$? != 0 ] ; then \
	        \echo "Please consult $(MANUAL_FILE).log to see the error messages" ; \
	        false; \
	fi
	latex_count=$(LATEX_COUNT) ; \
	while grep -E -s 'Rerun (LaTeX|to get cross-references right|to get bibliographical references right)' $(MANUAL_FILE).log && [ $$latex_count -gt 0 ] ;\
	    do \
	      echo "Rerunning latex...." ;\
	      $(LATEX_CMD) $(MANUAL_FILE) || \
	      if [ $$? != 0 ] ; then \
	              \echo "Please consult $(MANUAL_FILE).log to see the error messages" ; \
	              false; \
	      fi; \
	      latex_count=`expr $$latex_count - 1` ;\
	    done
	$(MKIDX_CMD) $(MANUAL_FILE).idx
	$(LATEX_CMD) $(MANUAL_FILE) || \
	if [ $$? != 0 ] ; then \
	        \echo "Please consult $(MANUAL_FILE).log to see the error messages" ; \
	        false; \
	fi

clean:
	rm -f *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *.brf *.blg *.bbl $(MANUAL_FILE).pdf
