SOURCEDIR := $(pwd)
notebooks_pypercent := $(shell find $(SOURCEDIR) -name '*.pct.py')
notebooks_markdown := $(shell find $(SOURCEDIR) -name '*.md')

notebooks_ipynb_py = $(notebooks_pypercent:.pct.py=.ipynb)
notebooks_ipynb_md = $(notebooks_markdown:.md=.ipynb)

.SUFFIXES: .ipynb .pct.py .md

.PHONY: all
all: $(notebooks_ipynb_md) $(notebooks_ipynb_py)

$(notebooks_ipynb_py): %.ipynb: %.pct.py
	jupytext --execute --to notebook -o $@ $<

$(notebooks_ipynb_md): %.ipynb: %.md
	jupytext --execute --to notebook -o $@ $<

.PHONY: pair-ipynb
pair-ipynb:
	jupytext --set-formats ipynb,md intro.ipynb 
	jupytext --set-formats ipynb,md gpflow2_upgrade_guide.ipynb
	jupytext --set-formats ipynb,.pct.py:percent intro_to_gpflow2.ipynb 
	jupytext --set-formats ipynb,.pct.py:percent advanced/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent basics/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent tailor/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent understanding/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent theory/FITCvsVFE.ipynb
	jupytext --set-formats ipynb,.pct.py:percent theory/Sanity_check.ipynb
	jupytext --set-formats ipynb,.pct.py:percent theory/upper_bound.ipynb
	jupytext --set-formats ipynb,md theory/*_notes.ipynb
