/**
 * @file sparse_coding.hpp
 * @author Nishant Mehta
 *
 * Definition of the SparseCoding class, which performs L1 (LASSO) or
 * L1+L2 (Elastic Net)-regularized sparse coding with dictionary learning
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_HPP
#define __MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/lars/lars.hpp>

// Include our three simple dictionary initializers.
#include "nothing_initializer.hpp"
#include "data_dependent_random_initializer.hpp"
#include "random_initializer.hpp"

namespace mlpack {
namespace sparse_coding {

/**
 * An implementation of Sparse Coding with Dictionary Learning that achieves
 * sparsity via an l1-norm regularizer on the codes (LASSO) or an (l1+l2)-norm
 * regularizer on the codes (the Elastic Net).
 *
 * Let d be the number of dimensions in the original space, m the number of
 * training points, and k the number of atoms in the dictionary (the dimension
 * of the learned feature space). The training data X is a d-by-m matrix where
 * each column is a point and each row is a dimension. The dictionary D is a
 * d-by-k matrix, and the sparse codes matrix Z is a k-by-m matrix.
 * This program seeks to minimize the objective:
 *
 * \f[
 * \min_{D,Z} 0.5 ||X - D Z||_{F}^2\ + \lambda_1 \sum_{i=1}^m ||Z_i||_1
 *                                    + 0.5 \lambda_2 \sum_{i=1}^m ||Z_i||_2^2
 * \f]
 *
 * subject to \f$ ||D_j||_2 <= 1 \f$ for \f$ 1 <= j <= k \f$
 * where typically \f$ lambda_1 > 0 \f$ and \f$ lambda_2 = 0 \f$.
 *
 * This problem is solved by an algorithm that alternates between a dictionary
 * learning step and a sparse coding step. The dictionary learning step updates
 * the dictionary D using a Newton method based on the Lagrange dual (see the
 * paper below for details). The sparse coding step involves solving a large
 * number of sparse linear regression problems; this can be done efficiently
 * using LARS, an algorithm that can solve the LASSO or the Elastic Net (papers
 * below).
 *
 * Here are those papers:
 *
 * @code
 * @incollection{lee2007efficient,
 *   title = {Efficient sparse coding algorithms},
 *   author = {Honglak Lee and Alexis Battle and Rajat Raina and Andrew Y. Ng},
 *   booktitle = {Advances in Neural Information Processing Systems 19},
 *   editor = {B. Sch\"{o}lkopf and J. Platt and T. Hoffman},
 *   publisher = {MIT Press},
 *   address = {Cambridge, MA},
 *   pages = {801--808},
 *   year = {2007}
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
 *
 * @code
 * @article{zou2005regularization,
 *   title={Regularization and variable selection via the elastic net},
 *   author={Zou, H. and Hastie, T.},
 *   journal={Journal of the Royal Statistical Society Series B},
 *   volume={67},
 *   number={2},
 *   pages={301--320},
 *   year={2005},
 *   publisher={Royal Statistical Society}
 * }
 * @endcode
 *
 * Before the method is run, the dictionary is initialized using the
 * DictionaryInitializationPolicy class.  Possible choices include the
 * RandomInitializer, which provides an entirely random dictionary, the
 * DataDependentRandomInitializer, which provides a random dictionary based
 * loosely on characteristics of the dataset, and the NothingInitializer, which
 * does not initialize the dictionary -- instead, the user should set the
 * dictionary using the Dictionary() mutator method.
 *
 * @tparam DictionaryInitializationPolicy The class to use to initialize the
 *     dictionary; must have 'void Initialize(const arma::mat& data, arma::mat&
 *     dictionary)' function.
 */
template<typename DictionaryInitializer = DataDependentRandomInitializer>
class SparseCoding
{
 public:
  /**
   * Set the parameters to SparseCoding. lambda2 defaults to 0.
   *
   * @param data Data matrix
   * @param atoms Number of atoms in dictionary
   * @param lambda1 Regularization parameter for l1-norm penalty
   * @param lambda2 Regularization parameter for l2-norm penalty
   */
  SparseCoding(const arma::mat& data,
               const size_t atoms,
               const double lambda1,
               const double lambda2 = 0);

  /**
   * Run Sparse Coding with Dictionary Learning.
   *
   * @param maxIterations Maximum number of iterations to run algorithm.  If 0,
   *     the algorithm will run until convergence (or forever).
   * @param objTolerance Tolerance for objective function.  When an iteration of
   *     the algorithm produces an improvement smaller than this, the algorithm
   *     will terminate.
   * @param newtonTolerance Tolerance for the Newton's method dictionary
   *     optimization step.
   */
  void Encode(const size_t maxIterations = 0,
              const double objTolerance = 0.01,
              const double newtonTolerance = 1e-6);

  /**
   * Sparse code each point via LARS.
   */
  void OptimizeCode();

  /**
   * Learn dictionary via Newton method based on Lagrange dual.
   *
   * @param adjacencies Indices of entries (unrolled column by column) of
   *    the coding matrix Z that are non-zero (the adjacency matrix for the
   *    bipartite graph of points and atoms).
   * @param newtonTolerance Tolerance of the Newton's method optimizer.
   * @return the norm of the gradient of the Lagrange dual with respect to
   *    the dual variables
   */
  double OptimizeDictionary(const arma::uvec& adjacencies,
                            const double newtonTolerance = 1e-6);

  /**
   * Project each atom of the dictionary back onto the unit ball, if necessary.
   */
  void ProjectDictionary();

  /**
   * Compute the objective function.
   */
  double Objective() const;

  //! Access the data.
  const arma::mat& Data() const { return data; }

  //! Access the dictionary.
  const arma::mat& Dictionary() const { return dictionary; }
  //! Modify the dictionary.
  arma::mat& Dictionary() { return dictionary; }

  //! Access the sparse codes.
  const arma::mat& Codes() const { return codes; }
  //! Modify the sparse codes.
  arma::mat& Codes() { return codes; }

  // Returns a string representation of this object. 
  std::string ToString() const;

 private:
  //! Number of atoms.
  size_t atoms;

  //! Data matrix (columns are points).
  const arma::mat& data;

  //! Dictionary (columns are atoms).
  arma::mat dictionary;

  //! Sparse codes (columns are points).
  arma::mat codes;

  //! l1 regularization term.
  double lambda1;

  //! l2 regularization term.
  double lambda2;
};

}; // namespace sparse_coding
}; // namespace mlpack

// Include implementation.
#include "sparse_coding_impl.hpp"

#endif
