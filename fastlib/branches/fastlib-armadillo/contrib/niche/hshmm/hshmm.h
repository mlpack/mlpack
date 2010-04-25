/*

Given: parameters of an HMM lambda = pi, A, M_1, ..., M_n

Goal: sample pairs x_t, x_t+1 for t = 1->compute \mu[P_x]


to sample the points, we need to be able to compute the probability of being in each state at a particular time

P(



*/

#ifndef HSHMM_H
#define HSHMM_H


#include "fastlib/fastlib.h"

#include "hmm.h"

const fx_entry_doc hshmm_entries[] = {
  FX_ENTRY_DOC_DONE
};

const fx_module_doc hshmm_doc = {
  hshmm_entries, NULL,
  "Hilbert-Schmidt embedding of an HMM.\n"

};


#endif /* HSHMM_H */
