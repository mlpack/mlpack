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
 * GenVector; or, well, at least partially written for arma::vec.
 */
void tree_kdtree_private::MakeBoundVector(const arma::vec& point,
                     const Vector& bound_dimensions,
                     arma::vec& bound_vector) {
  for(int i = 0; i < bound_dimensions.length(); i++)
    bound_vector[i] = point[(int) bound_dimensions[i]];
}
