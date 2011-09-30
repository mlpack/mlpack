/** @file sparse_censorship.h
 *
 *  This file implements a Sparse Coding model for documents, with censorship. It largely extends from SAGE (Sparse Additive Generative Models of Text), by Eisenstein et al.
 *
 *  @author Nishant Mehta (niche)
 *  @bug No known bugs, but no known completed code
 */

#ifndef SPARSE_CENSORSHIP_H
#define SPARSE_CENSORSHIP_H

#define INSIDE_SPARSE_CENSORSHIP_H


using namespace arma;
using namespace std;


class SparseCensorship {
 private:

  // the data
  X - // word counts - stored as (# docs) x (# vocab words)
      //   since X is sparse, storing the transpose instead may not matter

  theta
  beta
  eta
  
  nu
  xi
  tau

  sigma
  iota
  gamma
  

};

#include "sparse_censorship_impl.h"
#undef INSIDE_SPARSE_CENSORSHIP_H

#endif
