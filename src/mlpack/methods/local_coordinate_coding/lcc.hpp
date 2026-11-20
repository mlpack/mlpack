/**
 * @file methods/local_coordinate_coding/lcc.hpp
 * @author Nishant Mehta
 *
 * Definition of the LocalCoordinateCoding class, which performs the Local
 * Coordinate Coding algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LOCAL_COORDINATE_CODING_LCC_HPP
#define MLPACK_METHODS_LOCAL_COORDINATE_CODING_LCC_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/lars/lars.hpp>

// Include three simple dictionary initializers from sparse coding.
#include <mlpack/methods/sparse_coding/nothing_initializer.hpp>
#include <mlpack/methods/sparse_coding/data_dependent_random_initializer.hpp>
#include <mlpack/methods/sparse_coding/random_initializer.hpp>

namespace mlpack {

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
template<typename MatType = arma::mat>
class LocalCoordinateCoding
{
 public:
  using ColType = typename GetColType<MatType>::type;
  using RowType = typename GetRowType<MatType>::type;

  /**
   * Set the parameters to LocalCoordinateCoding, and train the dictionary.
   * This constructor will also initialize the dictionary using the given
   * DictionaryInitializer before training.
   *
   * If you want to initialize the dictionary to a custom matrix, consider
   * either writing your own DictionaryInitializer class (with void
   * Initialize(const MatType& data, MatType& dictionary) function), or call
   * the constructor that does not take a data matrix, then call Dictionary() to
   * set the dictionary matrix to a matrix of your choosing, and then call
   * Train() with NothingInitializer (i.e.  Train<NothingInitializer>(data)).
   *
   * @param data Data matrix.
   * @param atoms Number of atoms in dictionary.
   * @param lambda Regularization parameter for weighted l1-norm penalty.
   * @param maxIterations Maximum number of iterations for training (0 runs
   *      until convergence).
   * @param tolerance Tolerance for the objective function.
   * @param initializer Intializer to use.
   */
  template<typename DictionaryInitializer = DataDependentRandomInitializer>
  LocalCoordinateCoding(const MatType& data,
                        const size_t atoms,
                        const double lambda,
                        const size_t maxIterations = 0,
                        const double tolerance = 0.01,
                        const DictionaryInitializer& initializer =
                            DictionaryInitializer());

  /**
   * Set the parameters to LocalCoordinateCoding.  This constructor will not
   * train the model, and a subsequent call to Train() will be required before
   * the model can encode points with Encode().  The default values for atoms
   * and lambda should be changed if you intend to train the model!
   *
   * @param atoms Number of atoms in dictionary.
   * @param lambda Regularization parameter for weighted l1-norm penalty.
   * @param maxIterations Maximum number of iterations for training (0 runs
   *      until convergence).
   * @param tolerance Tolerance for the objective function.
   */
  LocalCoordinateCoding(const size_t atoms = 0,
                        const double lambda = 0.0,
                        const size_t maxIterations = 0,
                        const double tolerance = 0.01);

  /**
   * Run local coordinate coding.
   *
   * @param data Data matrix.
   * @param initializer Intializer to use.
   * @return The final objective value.
   */
  template<typename DictionaryInitializer = DataDependentRandomInitializer>
  double Train(const MatType& data,
               const DictionaryInitializer& initializer =
                   DictionaryInitializer());

  /**
   * Code each point via distance-weighted LARS.
   *
   * @param data Matrix containing points to encode.
   * @param codes Output matrix to store codes in.
   */
  void Encode(const MatType& data, MatType& codes);

  /**
   * Learn dictionary by solving linear system.
   *
   * @param data Matrix containing points to encode.
   * @param codes Output matrix to store codes in.
   * @param adjacencies Indices of entries (unrolled column by column) of
   *    the coding matrix Z that are non-zero (the adjacency matrix for the
   *    bipartite graph of points and atoms)
   */
  void OptimizeDictionary(const MatType& data,
                          const MatType& codes,
                          const arma::uvec& adjacencies);

  /**
   * Compute objective function given the list of adjacencies.
   *
   * @param data Matrix containing points to encode.
   * @param codes Output matrix to store codes in.
   */
  double Objective(const MatType& data,
                   const MatType& codes) const;

  /**
   * Compute objective function given the list of adjacencies.
   *
   * @param data Matrix containing points to encode.
   * @param codes Output matrix to store codes in.
   * @param adjacencies Indices of entries (unrolled column by column) of
   *    the coding matrix Z that are non-zero (the adjacency matrix for the
   *    bipartite graph of points and atoms)
   */
  double Objective(const MatType& data,
                   const MatType& codes,
                   const arma::uvec& adjacencies) const;

  //! Get the number of atoms.
  size_t Atoms() const { return atoms; }
  //! Modify the number of atoms.
  size_t& Atoms() { return atoms; }

  //! Accessor for dictionary.
  const MatType& Dictionary() const { return dictionary; }
  //! Mutator for dictionary.
  MatType& Dictionary() { return dictionary; }

  //! Get the L1 regularization parameter.
  double Lambda() const { return lambda; }
  //! Modify the L1 regularization parameter.
  double& Lambda() { return lambda; }

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the objective tolerance.
  double Tolerance() const { return tolerance; }
  //! Modify the objective tolerance.
  double& Tolerance() { return tolerance; }

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Number of atoms in dictionary.
  size_t atoms;

  //! Dictionary (columns are atoms).
  MatType dictionary;

  //! l1 regularization term.
  double lambda;

  //! Maximum number of iterations during training.
  size_t maxIterations;
  //! Tolerance for main objective.
  double tolerance;
};

} // namespace mlpack

CEREAL_TEMPLATE_CLASS_VERSION((typename MatType),
    (mlpack::LocalCoordinateCoding<MatType>), (1));

// Include implementation.
#include "lcc_impl.hpp"

#endif
