# Federalist Papers
Bayesian analysis to determine the authorship of the disputed Federalist Papers

This is a quick project to learn Python and to test my understanding of data mining and Bayesian statistics.
My goal is to write a program which can determine the author of a given piece of text.
The real challenge, and the bulk of my effort, goes into making an honest and quantiative appraisal of the odds that my determination is correct.
Even more difficult is then to test those odds and show that they are in fact accurate.

I think this is an important issue, since statistical models often over-state their certainty, sometimes by many orders of magnitude.
This leaves us unprepared for unpleasant suprises, such as financial crashes which sink the global economy or 100-year storms which occur twice in a year.
Many of these errors seem to stem from either an assumption of normality in the data, or from an assumption that events are uncorrelated; these are convenient assumptions, but in real-world problems, neither is typically correct and the results can be especialy terrible for rare events in the "tails" of the distribution.
In this project, I'll try to improve upon both assumptions; as a side effect, I'll obtain what I think is the most accurate determination of the authorship of the disputed Federalist papers to date!

I finished the calculation in a few steps, spread over a few notebooks.
You can either view the notebooks here, on github, or perhaps better on jupyter's official *nbviewer* site:

1. [Introduction](https://nbviewer.jupyter.org/github/mkmcc/FederalistPapers/blob/master/1%20-%20Introduction.ipynb)
2. [Support Vector Machine](https://nbviewer.jupyter.org/github/mkmcc/FederalistPapers/blob/master/2%20-%20Support%20Vector%20Machine%20%28Separating%20Hyperplane%29.ipynb)
3. [First Bayesian Calculation](https://nbviewer.jupyter.org/github/mkmcc/FederalistPapers/blob/master/3%20-%20Poisson%20Calculation.ipynb)
4. [Bayesian MCMC Calculation](https://nbviewer.jupyter.org/github/mkmcc/FederalistPapers/blob/master/4%20-%20NegativeBinomial%20MCMC%20Calculation.ipynb)
5. [Coupled Bayesian MCMC Calculation](https://nbviewer.jupyter.org/github/mkmcc/FederalistPapers/blob/master/5%20-%20Coupled%20MCMC%20Calculation.ipynb)
6. [Summary and Comparison](https://nbviewer.jupyter.org/github/mkmcc/FederalistPapers/blob/master/6%20-%20Summary.ipynb)

Since the calculations in notebooks #4 and #5 are fairly expensive, I've re-written them as parallized batch scripts which I ran on 32-core machines provided by Google's "Compute Engine."
These are in the gcloud/ directory.
If you choose to reproduce my results, building the author models will cost you a few dollars; checking the accuracy is more involved, however, and should cost between $15 and $20.
I've saved the results of these calculations in serialized Python "pickles," however, so you should be able to check my work on a laptop without re-running anything expensive.
