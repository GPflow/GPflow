.PHONY: pair-ipynb
pair-ipynb:
	jupytext --set-formats ipynb,md intro.ipynb 
	jupytext --set-formats ipynb,.pct.py:percent intro_to_gpflow2.ipynb 
	jupytext --set-formats ipynb,.pct.py:percent advanced/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent basics/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent tailor/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent understanding/*.ipynb
	jupytext --set-formats ipynb,.pct.py:percent theory/{FITC*,Sanity*,upper*}.ipynb
	jupytext --set-formats ipynb,md theory/*_notes.ipynb
	jupytext --set-formats ipynb,md gpflow_2_upgrade/gpflow2_upgrade_guide.ipynb 
