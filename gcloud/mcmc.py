# build-author-models.py: This script takes in a training set of text
#   for Madison and Hamilton and runs the MCMC code to build models
#   for their word choices.
#
# N.B. running both models took 9.2 cpu-hours on google compute engine,
#   costing $0.33 to run  (or only $0.07 if run as a preemptible job!)
#
from __future__ import division

import numpy as np
from scipy.stats import nbinom
from math import log, exp, log10, floor
import pickle
import emcee
import nltk
from collections import Counter


# load stopword definition
execfile("./stopwords.py")


def lnlike(theta, y):
    """log-likelihood function for the negative binomial model defined by
    theta = {log(lambda), log(f)}, given the counts in y.

    N.B. y must be sorted."""

    # start by making an empirical CDF (or EDF) to compare
    # against the theoretical CDF.  it's technically wasteful
    # to re-do this at every step, but it simplifies the code
    # and contributes only ~5% to the overall cost
    #
    num = y.shape[0]
    edf = np.arange(1,num+1).astype(float)/num

    ind = np.where(np.diff(y) != 0)[0]
    pos = y[ind]
    obs = edf[ind]

    # theoretical CDF.  the call to nbinom.cdf dominates the
    # cost of this entire calculation.
    #
    lam, f = np.exp(theta)
    n = lam / f
    p = 1.0 / (1.0 + f)
    model = nbinom.cdf(pos, n, p)

    # modified chi^2 statistic, with the error estimated as C*(1-C) ~ P
    #
    err    = (model - obs)**2
    weight = model*(1.0-model)

    # i don't trust the weights when they get too close to zero...
    weight = np.clip(weight, a_min=0.01, a_max=None)

    return -np.sum(err / weight)


def lnprior(theta, y):
    """Prior probability on the negative binomial model parameters
    theta = {log(lambda), log(f)}"""

    lnlam, lnf = theta
    lam,   f   = np.exp(theta)

    # very low values of lambda or f make no difference, so not
    # worth wasting computer time exploring them.
    # and it seems scary to let f become too large...
    #
    if lnlam <= -10 or lnf <= -2 or lnf >= 3:
        return -np.inf

    # otherwise, express a mild preference for the mean of the distribution
    # to match the mean observed count rate
    #
    va = lam * (1.0 + f)

    mu_1 = np.mean(y)
    va_1 = np.var(y)

    return -(lam-mu_1)**2 / (2*(va_1+va))


def lnprob(theta, y):
    lp = lnprior(theta, y)
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, y)


def make_sample(data):
    """Run the MCMC simulation and make a coupled model for Madison's and
    Hamilton's usage of a single word"""

    # data must be sorted
    data = np.sort(np.asarray(data))

    # MCMC parameters
    ndim, nwalkers = 2, 24

    # initial guess for the MCMC chain based on first two moments of the observed data
    mu = np.mean(data)
    va = np.var(data)

    # if the word *never* occurs in the data, we have a problem because
    # we don't know what to compare our models to!  i'll be conservative
    # here and treat a non-occurrence as if it had occurred once.  but this
    # is a dirty hack, and it shows in the results.  the proper thing to do
    # is to use information from the other authors' usage... we do this in
    # the coupled model in coupled-mcmc.py
    #
    if mu <= 0:
        data[-1] = 1
        mu = np.mean(data)

    f_0   = np.max([va/mu - 1.0, exp(-1)])

    def mkpos_helper():
        return [ log(mu)  + 0.5*np.random.randn(),
                 log(f_0) + 0.5*np.random.randn() ]

    def mkpos(y):
        ret  = mkpos_helper()
        prob = lnprior(ret, y)
        while not np.isfinite(prob):
            ret  = mkpos_helper()
            prob = lnprior(ret, y)
        return ret

    pos = [ mkpos(data) for i in range(nwalkers) ]

    # run the MCMC simulation for 1e4 steps, throw out the first half
    # as "burn-in," and return a random sample of 500 points
    #
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data])
    sampler.run_mcmc(pos, 10000)
    samples = sampler.chain[:, 5000:, :].reshape((-1, ndim))

    return samples[np.random.randint(len(samples), size=500)]


import multiprocessing

if __name__ == '__main__':
    pool = multiprocessing.Pool()

    def train(filename):
        """Import a file containing training data, split into chunks of 1000 words,
        and tally each of the stopwords for each chunk.  Use this as input data for
        the MCMC simulations."""
        with open(filename, 'r') as f:
            raw = f.read()

        all_words = nltk.word_tokenize(raw.lower())

        n=1000
        samples = [all_words[i:i + n] for i in xrange(0, len(all_words), n)]

        c = [ Counter(s) for s in samples ]
        x = map(lambda(cc): [ cc[w] for w in stopwords ], c)

        return map(list, zip(*x))


    def run_sim(corpus_filename, out_filename):
        with open('progress.txt', 'a') as f:
            f.write(corpus_filename)
            f.write('\n')

        x = train(corpus_filename)

        data = pool.map(make_sample, x)

        with open(out_filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print 'separate model, 25k word training set, for testing:'
    run_sim('text/madison-train.txt',
            'author-models/mad.pickle')

    run_sim('text/hamilton-train.txt',
            'author-models/ham.pickle')


    print 'separate model, 40k word training set, for testing:'
    run_sim('text/madison-train2.txt',
            'author-models/mad2.pickle')

    run_sim('text/hamilton-train2.txt',
            'author-models/ham2.pickle')


    print 'separate model, trained on all federalist papers of known authorship:'
    run_sim('text/madison-corpus-small.txt',
            'author-models/mad-all.pickle')

    run_sim('text/hamilton-corpus-small.txt',
            'author-models/ham-all.pickle')
