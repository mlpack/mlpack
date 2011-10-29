/***
 * @file mvu.h
 * @author Ryan Curtin
 *
 * An implementation of Maximum Variance Unfolding.  This file defines an MVU
 * class as well as a class representing the objective function (a semidefinite
 * program) which MVU seeks to minimize.  Minimization is performed by the
 * Augmented Lagrangian optimizer (which in turn uses the L-BFGS optimizer).
 */

#ifndef __MLPACK_MVU_H
#define __MLPACK_MVU_H

#include <mlpack/core.h>

namespace mlpack {
namespace mvu {

/***
 * The MVU class is meant to provide a good abstraction for users.  The dataset
 * needs to be provided, as well as several parameters.
 *
 * - dataset
 * - new dimensionality
 */
template<typename LagrangianFunction>
class MVU {
 public:
  MVU(arma::mat& data_in); // probably needs arguments

  bool Unfold(arma::mat& output_coordinates); // probably needs arguments

 private:
  arma::mat& data_;
  LagrangianFunction f_;
};

}; // namespace mvu
}; // namespace mlpack

#include "mvu_impl.h"

#endif
