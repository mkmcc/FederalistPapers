# check-errors.py: this script repeatedly runs the Madison and
#   Hamilton author models on fragments of text of different sizes.
#   The fragments are intentionally small to obtain a high error rate.
#   Since the fragments all have known authorship, this should enable
#   me to quantify the true accuracy of the code as a function of its
#   reported accuracy.  Hopefully they're similar!
#
# N.B. Each test takes 48 cpu-hours on google compute engine, costing
#   $1.70 to run.  Running all six tests as I do here will cost about
#   $10.20.  I've serialized the results using Python's pickle()
#   function and stored them in the errors/ directory.  Feel free to
#   use them if you don't want to re-run the code!
#
from __future__ import division

import numpy as np
from scipy.stats import nbinom
from math import log, exp, log10

import nltk
from collections import Counter

import pickle
from functools import partial

# load stopword definition
execfile("./stopwords.py")


# functions to calculate probability -- these happen on worker processes
#
def prob(x, num, sample):
    """log-probability for a single word to occur x times in a text
    of length num words, given author model defined by the MCMC
    chain in sample"""

    ps =  np.zeros(len(sample))

    for i, theta in enumerate(sample):
        lam, f = np.exp(theta)
        lam = lam * (num/1000) # scale predicted rate to text length

        n = lam / f
        p = 1.0 / (1.0 + f)
        ps[i] = nbinom.pmf(x, n, p)

    return log(np.mean(ps))


def measure(text):
    """tally the occurrcences of each of the stopwords in text"""
    c = Counter(text)
    x = [ c[w] for w in stopwords ]

    return (x, len(text))


def all_prob(text, model1, model2):
    """given a sample of text, along with two author models,
    return the difference in log-probabilities for each of the
    authors.  (equivalently, this is the log of the ratio of
    probabilities that each author wrote it.)"""
    xs, num = measure(text)

    m1 = np.sum([ prob(x, num, sample) for x, sample in zip(xs, model1) ])
    m2 = np.sum([ prob(x, num, sample) for x, sample in zip(xs, model2) ])

    return m1-m2


# main program -- this only happens on the root process
#
import multiprocessing

if __name__ == '__main__':
    pool = multiprocessing.Pool()

    def split(all_words, n):
        num = len(all_words)

        samples = [ all_words[i:i + n] for i in xrange(0, len(all_words), n) ]
        if len(samples) > 250:
            ind = np.random.randint(len(samples), size=250)
            samples = [ samples[i] for i in ind ]

        return samples

    def read_coupled_model(filename):
        """read in the MCMC chain for a coupled model and split into
        separate chains for madison and for hamilton."""
        with open(filename, 'rb') as handle:
            model = pickle.load(handle)

        model = np.asarray(model)
        m_model = model[:,:,:2]     # shape: words, samples, parameters
        h_model = model[:,:,2:]

        return (m_model, h_model)

    def read_single_model(filename):
        with open(filename, 'rb') as handle:
            model = pickle.load(handle)

        return model

    def prepare_sample_data(file):
        """read in writing sample in split into fragments of different sizes
        return as one long list so the program parallelizes efficiently."""

        fragment_lengths = np.logspace(log10(50), log10(10000), 20)
        fragment_lengths = np.round(fragment_lengths).astype(int)

        with open(file, 'r') as f:
            raw = f.read()
            all_words = nltk.word_tokenize(raw.lower())

        data = []
        for size in fragment_lengths:
            data = data + split(all_words, size)

        lens = np.asarray(map(len, data))

        return (lens, data)

    def run_simulation(mdata, m_model, hdata, h_model):
        fun = partial(all_prob, model1 = m_model, model2 = h_model)
        m_results  = pool.map(fun, mdata)
        m_results = np.asarray(m_results)

        fun = partial(all_prob, model1 = h_model, model2 = m_model)
        h_results  = pool.map(fun, hdata)
        h_results = np.asarray(h_results)

        return (m_results, h_results)


    # start producing results!
    print 'beginning with coupled models:'
    print 'coupled model, small training set, no overlap'
    #
    m_model, h_model = read_coupled_model('author-models/coupled.pickle')
    mlens, mdata = prepare_sample_data('text/madison-check.txt')
    hlens, hdata = prepare_sample_data('text/hamilton-check.txt')
    m_results, h_results =  run_simulation(mdata, m_model, hdata, h_model)
    #
    with open('errors/small-training-set-no-overlap_coupled.pickle', 'wb') as handle:
        pickle.dump((mlens, m_results, hlens, h_results),
                    handle, protocol=pickle.HIGHEST_PROTOCOL)


    print 'coupled model, large training set, no overlap'
    #
    m_model, h_model = read_coupled_model('author-models/coupled2.pickle')
    mlens, mdata = prepare_sample_data('text/madison-check2.txt')
    hlens, hdata = prepare_sample_data('text/hamilton-check2.txt')
    m_results, h_results =  run_simulation(mdata, m_model, hdata, h_model)
    #
    with open('errors/large-training-set-no-overlap_coupled.pickle', 'wb') as handle:
        pickle.dump((mlens, m_results, hlens, h_results),
                    handle, protocol=pickle.HIGHEST_PROTOCOL)


    print 'coupled model, full training set, overlaps with test data'
    #
    m_model, h_model = read_coupled_model('author-models/coupled-all.pickle')
    mlens, mdata = prepare_sample_data('text/madison-corpus-small.txt')
    hlens, hdata = prepare_sample_data('text/hamilton-corpus-small.txt')
    m_results, h_results =  run_simulation(mdata, m_model, hdata, h_model)
    #
    with open('errors/full-set-overlap_coupled.pickle', 'wb') as handle:
        pickle.dump((mlens, m_results, hlens, h_results),
                    handle, protocol=pickle.HIGHEST_PROTOCOL)



    print 'coupled models finished.  running de-coupled models for comparison:'
    print 'separate model, small training set, no overlap'
    #
    m_model = read_single_model('author-models/mad.pickle')
    h_model = read_single_model('author-models/ham.pickle')
    mlens, mdata = prepare_sample_data('text/madison-check.txt')
    hlens, hdata = prepare_sample_data('text/hamilton-check.txt')
    m_results, h_results =  run_simulation(mdata, m_model, hdata, h_model)
    #
    with open('errors/small-training-set-no-overlap.pickle', 'wb') as handle:
        pickle.dump((mlens, m_results, hlens, h_results),
                    handle, protocol=pickle.HIGHEST_PROTOCOL)


    print 'separate model, large training set, no overlap'
    #
    m_model = read_single_model('author-models/mad2.pickle')
    h_model = read_single_model('author-models/ham2.pickle')
    mlens, mdata = prepare_sample_data('text/madison-check2.txt')
    hlens, hdata = prepare_sample_data('text/hamilton-check2.txt')
    m_results, h_results =  run_simulation(mdata, m_model, hdata, h_model)
    #
    with open('errors/large-training-set-no-overlap.pickle', 'wb') as handle:
        pickle.dump((mlens, m_results, hlens, h_results),
                    handle, protocol=pickle.HIGHEST_PROTOCOL)


    print 'separate model, full training set, overlaps with test data'
    #
    m_model = read_single_model('author-models/mad-all.pickle')
    h_model = read_single_model('author-models/ham-all.pickle')
    mlens, mdata = prepare_sample_data('text/madison-corpus-small.txt')
    hlens, hdata = prepare_sample_data('text/hamilton-corpus-small.txt')
    m_results, h_results =  run_simulation(mdata, m_model, hdata, h_model)
    #
    with open('errors/full-set-overlap.pickle', 'wb') as handle:
        pickle.dump((mlens, m_results, hlens, h_results),
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
