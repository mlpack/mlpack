/**
 * @file kdtree_impl.cc
 *
 * Specialized functions.
 */
#include "../fastlib.h"
#include "../la/matrix.h"
#include "kdtree.h"

#include <armadillo>

/***
 * Specialized MakeBoundVector function written for arma::vec instead of
 * GenVector.
 */
void tree_kdtree_private::MakeBoundVector(const arma::vec& point,
                     const arma::uvec& bound_dimensions,
                     arma::vec& bound_vector) {
  for(int i = 0; i < bound_dimensions.n_elem; i++)
    bound_vector[i] = point[(int) bound_dimensions[i]];
}
