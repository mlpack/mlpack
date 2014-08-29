/**
 * @file lcc.hpp
 * @author Nishant Mehta
 *
 * Definition of the LocalCoordinateCoding class, which performs the Local
 * Coordinate Coding algorithm.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_LOCAL_COORDINATE_CODING_LCC_HPP
#define __MLPACK_METHODS_LOCAL_COORDINATE_CODING_LCC_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/lars/lars.hpp>

// Include three simple dictionary initializers from sparse coding.
#include "../sparse_coding/nothing_initializer.hpp"
#include "../sparse_coding/data_dependent_random_initializer.hpp"
#include "../sparse_coding/random_initializer.hpp"

namespace mlpack {
namespace lcc {

/**
 * An implementation of Local Coordinate Coding (LCC) that codes data which
 * approximately lives on a manifold using a variation of l1-norm regularized
 * sparse coding; in LCC, the penalty on the absolute value of each point's
 * coefficient for each atom is weighted by the squared distance of that point
 * to that atom.
 *
 * Let d be the number of dimensions in the original space, m the number of
 * training points, and k the number of atoms in the dictionary (the dimension
 * of the learned feature space). The training data X is a d-by-m matrix where
 * each column is a point and each row is a dimension. The dictionary D is a
 * d-by-k matrix, and the sparse codes matrix Z is a k-by-m matrix.
 * This program seeks to minimize the objective:
 * min_{D,Z} ||X - D Z||_{Fro}^2
 *            + lambda sum_{i=1}^m sum_{j=1}^k dist(X_i,D_j)^2 Z_i^j
 * where lambda > 0.
 *
 * This problem is solved by an algorithm that alternates between a dictionary
 * learning step and a sparse coding step. The dictionary learning step updates
 * the dictionary D by solving a linear system (note that the objective is a
 * positive definite quadratic program). The sparse coding step involves
 * solving a large number of weighted l1-norm regularized linear regression
 * problems problems; this can be done efficiently using LARS, an algorithm
 * that can solve the LASSO (paper below).
 *
 * The papers are listed below.
 *
 * @code
 * @incollection{NIPS2009_0719,
 *   title = {Nonlinear Learning using Local Coordinate Coding},
 *   author = {Kai Yu and Tong Zhang and Yihong Gong},
 *   booktitle = {Advances in Neural Information Processing Systems 22},
 *   editor = {Y. Bengio and D. Schuurmans and J. Lafferty and C. K. I. Williams
 *       and A. Culotta},
 *   pages = {2223--2231},
 *   year = {2009}
 * }
 * @endcode
 *
 * @code
 * @article{efron2004least,
 *   title={Least angle regression},
 *   author={Efron, B. and Hastie, T. and Johnstone, I. and Tibshirani, R.},
 *   journal={The Annals of statistics},
 *   volume={32},
 *   number={2},
 *   pages={407--499},
 *   year={2004},
 *   publisher={Institute of Mathematical Statistics}
 * }
 * @endcode
 */
template<typename DictionaryInitializer =
    sparse_coding::DataDependentRandomInitializer>
class LocalCoordinateCoding
{
 public:
  /**
   * Set the parameters to LocalCoordinateCoding.
   *
   * @param data Data matrix.
   * @param atoms Number of atoms in dictionary.
   * @param lambda Regularization parameter for weighted l1-norm penalty.
   */
  LocalCoordinateCoding(const arma::mat& data,
                        const size_t atoms,
                        const double lambda);

  /**
   * Run local coordinate coding.
   *
   * @param nIterations Maximum number of iterations to run algorithm.
   * @param objTolerance Tolerance of objective function.  When the objective
   *     function changes by a value lower than this tolerance, the optimization
   *     terminates.
   */
  void Encode(const size_t maxIterations = 0,
              const double objTolerance = 0.01);

  /**
   * Code each point via distance-weighted LARS.
   */
  void OptimizeCode();

  /**
   * Learn dictionary by solving linear system.
   *
   * @param adjacencies Indices of entries (unrolled column by column) of
   *    the coding matrix Z that are non-zero (the adjacency matrix for the
   *    bipartite graph of points and atoms)
   */
  void OptimizeDictionary(arma::uvec adjacencies);

  /**
   * Compute objective function given the list of adjacencies.
   */
  double Objective(arma::uvec adjacencies) const;

  //! Access the data.
  const arma::mat& Data() const { return data; }

  //! Accessor for dictionary.
  const arma::mat& Dictionary() const { return dictionary; }
  //! Mutator for dictionary.
  arma::mat& Dictionary() { return dictionary; }

  //! Accessor the codes.
  const arma::mat& Codes() const { return codes; }
  //! Modify the codes.
  arma::mat& Codes() { return codes; }

  // Returns a string representation of this object. 
  std::string ToString() const;

 private:
  //! Number of atoms in dictionary.
  size_t atoms;

  //! Data matrix (columns are points).
  const arma::mat& data;

  //! Dictionary (columns are atoms).
  arma::mat dictionary;

  //! Codes (columns are points).
  arma::mat codes;

  //! l1 regularization term.
  double lambda;
};

}; // namespace lcc
}; // namespace mlpack

// Include implementation.
#include "lcc_impl.hpp"

#endif
