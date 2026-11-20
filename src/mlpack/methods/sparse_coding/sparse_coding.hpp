/**
 * @file methods/sparse_coding/sparse_coding.hpp
 * @author Nishant Mehta
 *
 * Definition of the SparseCoding class, which performs L1 (LASSO) or
 * L1+L2 (Elastic Net)-regularized sparse coding with dictionary learning
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_HPP
#define MLPACK_METHODS_SPARSE_CODING_SPARSE_CODING_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/lars/lars.hpp>

// Include our three simple dictionary initializers.
#include "nothing_initializer.hpp"
#include "data_dependent_random_initializer.hpp"
#include "random_initializer.hpp"

namespace mlpack {

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
 * Note that the implementation here does not use the feature-sign search
 * algorithm from Honglak Lee's paper, but instead the LARS algorithm suggested
 * in that paper.
 *
 * When Train() is called, the dictionary is initialized using the
 * DictionaryInitializationPolicy class.  Possible choices include the
 * RandomInitializer, which provides an entirely random dictionary, the
 * DataDependentRandomInitializer, which provides a random dictionary based
 * loosely on characteristics of the dataset, and the NothingInitializer, which
 * does not initialize the dictionary -- instead, the user should set the
 * dictionary using the Dictionary() mutator method.
 *
 * Once a dictionary is trained with Train(), another matrix may be encoded with
 * the Encode() function.
 *
 * @tparam DictionaryInitializationPolicy The class to use to initialize the
 *     dictionary; must have 'void Initialize(const MatType& data, MatType&
 *     dictionary)' function.
 */
template<typename MatType = arma::mat>
class SparseCoding
{
 public:
  using ColType = typename GetColType<MatType>::type;
  using RowType = typename GetRowType<MatType>::type;

  /**
   * Set the parameters to SparseCoding.  lambda2 defaults to 0.  This
   * constructor will train the model.  If that is not desired, call the other
   * constructor that does not take a data matrix.  This constructor will also
   * initialize the dictionary using the given DictionaryInitializer before
   * training.
   *
   * If you want to initialize the dictionary to a custom matrix, consider
   * either writing your own DictionaryInitializer class (with void
   * Initialize(const MatType& data, MatType& dictionary) function), or call
   * the constructor that does not take a data matrix, then call Dictionary() to
   * set the dictionary matrix to a matrix of your choosing, and then call
   * Train() with NothingInitializer (i.e. Train<NothingInitializer>(data)).
   *
   * @param data Data matrix.
   * @param atoms Number of atoms in dictionary.
   * @param lambda1 Regularization parameter for l1-norm penalty.
   * @param lambda2 Regularization parameter for l2-norm penalty.
   * @param maxIterations Maximum number of iterations to run algorithm.  If 0,
   *     the algorithm will run until convergence (or forever).
   * @param objTolerance Tolerance for objective function.  When an iteration of
   *     the algorithm produces an improvement smaller than this, the algorithm
   *     will terminate.
   * @param newtonTolerance Tolerance for the Newton's method dictionary
   *     optimization step.
   * @param initializer The initializer to use.
   */
  template<typename DictionaryInitializer = DataDependentRandomInitializer>
  SparseCoding(const MatType& data,
               const size_t atoms,
               const double lambda1,
               const double lambda2 = 0,
               const size_t maxIterations = 0,
               const double objTolerance = 0.01,
               const double newtonTolerance = 1e-6,
               const DictionaryInitializer& initializer =
                   DictionaryInitializer());

  /**
   * Set the parameters to SparseCoding.  lambda2 defaults to 0.  This
   * constructor will not train the model, and a subsequent call to Train() will
   * be required before the model can encode points with Encode().
   *
   * @param atoms Number of atoms in dictionary.
   * @param lambda1 Regularization parameter for l1-norm penalty.
   * @param lambda2 Regularization parameter for l2-norm penalty.
   * @param maxIterations Maximum number of iterations to run algorithm.  If 0,
   *     the algorithm will run until convergence (or forever).
   * @param objTolerance Tolerance for objective function.  When an iteration of
   *     the algorithm produces an improvement smaller than this, the algorithm
   *     will terminate.
   * @param newtonTolerance Tolerance for the Newton's method dictionary
   *     optimization step.
   */
  SparseCoding(const size_t atoms = 0,
               const double lambda1 = 0,
               const double lambda2 = 0,
               const size_t maxIterations = 0,
               const double objTolerance = 0.01,
               const double newtonTolerance = 1e-6);

  /**
   * Train the sparse coding model on the given dataset.
   * @return The final objective value.
   */
  template<typename DictionaryInitializer = DataDependentRandomInitializer>
  double Train(const MatType& data,
               const DictionaryInitializer& initializer =
                   DictionaryInitializer());

  /**
   * Sparse code each point in the given dataset via LARS, using the current
   * dictionary and store the encoded data in the codes matrix.
   *
   * @param data Input data matrix to be encoded.
   * @param codes Output codes matrix.
   */
  void Encode(const MatType& data, MatType& codes);

  /**
   * Learn dictionary via Newton method based on Lagrange dual.
   *
   * @param data Data matrix.
   * @param codes Matrix of codes.
   * @param adjacencies Indices of entries (unrolled column by column) of
   *    the coding matrix Z that are non-zero (the adjacency matrix for the
   *    bipartite graph of points and atoms).
   * @return the norm of the gradient of the Lagrange dual with respect to
   *    the dual variables
   */
  double OptimizeDictionary(const MatType& data,
                            const MatType& codes,
                            const arma::uvec& adjacencies);

  /**
   * Project each atom of the dictionary back onto the unit ball, if necessary.
   */
  void ProjectDictionary();

  /**
   * Compute the objective function.
   */
  double Objective(const MatType& data, const MatType& codes) const;

  //! Access the dictionary.
  const MatType& Dictionary() const { return dictionary; }
  //! Modify the dictionary.
  MatType& Dictionary() { return dictionary; }

  //! Access the number of atoms.
  size_t Atoms() const { return atoms; }
  //! Modify the number of atoms.
  size_t& Atoms() { return atoms; }

  //! Access the L1 regularization term.
  double Lambda1() const { return lambda1; }
  //! Modify the L1 regularization term.
  double& Lambda1() { return lambda1; }

  //! Access the L2 regularization term.
  double Lambda2() const { return lambda2; }
  //! Modify the L2 regularization term.
  double& Lambda2() { return lambda2; }

  //! Get the maximum number of iterations.
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations.
  size_t& MaxIterations() { return maxIterations; }

  //! Get the objective tolerance.
  double ObjTolerance() const { return objTolerance; }
  //! Modify the objective tolerance.
  double& ObjTolerance() { return objTolerance; }

  //! Get the tolerance for Newton's method (dictionary optimization step).
  double NewtonTolerance() const { return newtonTolerance; }
  //! Modify the tolerance for Newton's method (dictionary optimization step).
  double& NewtonTolerance() { return newtonTolerance; }

  //! Serialize the sparse coding model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Number of atoms.
  size_t atoms;

  //! Dictionary (columns are atoms).
  MatType dictionary;

  //! l1 regularization term.
  double lambda1;
  //! l2 regularization term.
  double lambda2;

  //! Maximum number of iterations during training.
  size_t maxIterations;
  //! Tolerance for main objective.
  double objTolerance;
  //! Tolerance for Newton's method (dictionary training).
  double newtonTolerance;
};

} // namespace mlpack

CEREAL_TEMPLATE_CLASS_VERSION((typename MatType),
    (mlpack::SparseCoding<MatType>), (1));

// Include implementation.
#include "sparse_coding_impl.hpp"

#endif
