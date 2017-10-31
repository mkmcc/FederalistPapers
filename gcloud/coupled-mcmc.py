# build-author-models.py: This script takes in a training set of text
#   for Madison and Hamilton and runs the MCMC code to build models
#   for their word choices.
#
# N.B. This takes about 30 cpu-hours on the google compute engine,
#   costing about $0.80 to $1.00.
#
from __future__ import division

import numpy as np
from scipy.stats import nbinom, beta
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
    """log-probability function for the coupled model.  This is the sum
    of log-probabilities for each individual model, along with a
    'coupled prior' which prevents the two authors from being too dissimilar.

    This coupled prior protects against selection effects: if the authors
    are different and there's a lot of evidence for it, the evidence wins.
    But if the evidence is weak, it may just be noise and this prior thus
    favors models in which the authors are more similar."""

    # first compute priors and return if we're invalid
    #
    lp1 = lnprior(theta[:2], y[0])
    lp2 = lnprior(theta[2:], y[1])
    if not np.isfinite(lp1) and np.isfinite(lp2):
        return -np.inf

    # next compute the coupler prior.  follow Mosteller & Wallace
    # and assume a beta distribution for the probability
    #
    lam1, f1, lam2, f2 = np.exp(theta)
    coupled_prior = beta.logpdf(lam1/(lam1+lam2), 10, 10)

    return lp1 + lp2 + coupled_prior \
           + lnlike(theta[:2], y[0]) + lnlike(theta[2:], y[1])


def make_sample(args):
    """Run the MCMC simulation and make a coupled model for Madison's and
    Hamilton's usage of a single word"""

    # take arguments as a single list for easier parallelization
    m_data, h_data, word = args

    # occurrence counts must be sorted
    m_data = np.sort(m_data)
    h_data = np.sort(h_data)

    # MCMC parameters
    ndim, nwalkers = 4, 48

    # initial guess for the MCMC chain based on first two moments of the observed data
    mu1, mu2 = np.mean(m_data), np.mean(h_data)
    va1, va2 = np.var(m_data),  np.var(h_data)

    # if the observed rate is zero, infer an upper limit
    mu1, mu2 = np.max([ mu1, 1.0/len(m_data) ]), np.max([ mu2, 1.0/len(h_data) ])

    f1, f2 = np.max([va1/mu1 - 1.0, exp(-1)]), np.max([va2/mu2 - 1.0, exp(-1)])

    def mkpos_helper():
        return [log(mu1) + 1.0*np.random.randn(),  log(f1)  + 1.0*np.random.randn(),
                log(mu2) + 1.0*np.random.randn(),  log(f2)  + 1.0*np.random.randn()]

    def mkpos(y):
        ret  = mkpos_helper()
        prob = lnprob(ret, y)
        while not np.isfinite(prob):
            ret  = mkpos_helper()
            prob = lnprob(ret, y)
        return ret

    p0 = [ mkpos([m_data, h_data]) for i in range(nwalkers) ]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[[m_data, h_data]])

    # burn in for 4000 steps... this problem seems to have slow mixing in some cases
    pos, prob, state = sampler.run_mcmc(p0, 4000)
    sampler.reset()

    # there's an interesting debate over whether to thin MCMC chains based
    # on the autocorrelation... after a bit of reading, i came to the conclusion
    # that it's best not to thin in most cases since the benefit from removing
    # correlations is overwhelmed by the loss of data.  however in this case, it's
    # not practical to use the full MCMC chain anyways... so i may as well thin it
    # and remove correlations so we can all sleep better at night.  the conservative
    # estimate for the correlation length seems to be about 30-40, so i thin the
    # chain to every 50th cell
    #
    for pos, prob, state in sampler.sample(pos, iterations=6000, thin=50, storechain=True):
        pass;
    samples = sampler.flatchain

    # return a random sample of 500 points
    #
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

        c = [Counter(s) for s in samples]
        x = map(lambda(cc): [cc[w] for w in stopwords], c)

        return map(list, zip(*x))


    def run_sim(mad_training_filename, ham_training_filename, out_filename):
        xm = train(mad_training_filename)
        xh = train(ham_training_filename)

        data = pool.map(make_sample, zip(xm, xh, stopwords))

        with open(out_filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # finally, produce the models!
    #
    print 'coupled model, 25k word training set, for testing:'
    run_sim('text/madison-train.txt',
            'text/hamilton-train.txt',
            'author-models/coupled.pickle')

    print 'coupled model, 40k word training set, for testing:'
    run_sim('text/madison-train2.txt',
            'text/hamilton-train2.txt',
            'author-models/coupled2.pickle')

    print 'coupled model, trained on all federalist papers of known authorship:'
    run_sim('text/madison-corpus-small.txt',
            'text/hamilton-corpus-small.txt',
            'author-models/coupled-all.pickle')
