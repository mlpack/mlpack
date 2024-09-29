/**
 * @file methods/svdplusplus/svdplusplus.hpp
 * @author Siddharth Agrawal
 * @author Wenhao Huang
 *
 * An implementation of SVD++.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_SVDPLUSPLUS_SVDPLUSPLUS_HPP
#define MLPACK_METHODS_SVDPLUSPLUS_SVDPLUSPLUS_HPP

#include <mlpack/core.hpp>

#include "svdplusplus_function.hpp"

namespace mlpack {

/**
 * SVD++ is a matrix decomposition technique used in collaborative filtering.
 * SVD++ is similar to BiasSVD, but it is a more expressive model because
 * SVD++ also models implicit feedback. SVD++ outputs user/item latent
 * vectors, user/item bias, and item vectors with regard to implicit feedback.
 * Parameters are optmized by Stochastic Gradient Desent(SGD). The updates also
 * penalize the learning of large feature values by means of regularization.
 *
 * For more information, see the following paper:
 *
 * @code
 * @inproceedings{koren2008factorization,
 * title={Factorization meets the neighborhood: a multifaceted collaborative
 *        filtering model},
 * author={Koren, Yehuda},
 * booktitle={Proceedings of the 14th ACM SIGKDD international conference on
 *            Knowledge discovery and data mining},
 * pages={426--434},
 * year={2008},
 * organization={ACM}
 * }
 * @endcode
 *
 * An example of how to use the interface is shown below:
 *
 * @code
 * arma::mat data; // Rating data in the form of coordinate list.
 *
 * // Implicit feedback data in the form of coordinate list.
 * arma::mat implicitData;
 *
 * const size_t rank = 10; // Rank used for the decomposition.
 * const size_t iterations = 10; // Number of iterations used for optimization.
 *
 * const double alpha = 0.001 // Learning rate for the SGD optimizer.
 * const double lambda = 0.1 // Regularization parameter for the optimization.
 *
 * // Make a SVD++ object.
 * SVDPlusPlus<> svdPP(iterations, alpha, lambda);
 *
 * arma::mat u, v; // Item and User matrices.
 * arma::vec p, q; // Item and User bias.
 * arma::mat y;    // Item matrix with respect to implicit feedback.
 *
 * // Use the Apply() method to get a factorization.
 * svdPP.Apply(data, implicitData, rank, u, v, p, q, y);
 * @endcode
 */
template<typename OptimizerType = ens::StandardSGD>
class SVDPlusPlus
{
 public:
  /**
   * Constructor of SVDPlusPlus. By default SGD optimizer is used in
   * SVDPlusPlus. The optimizer uses a template specialization of Optimize().
   *
   * @param iterations Number of optimization iterations.
   * @param alpha Learning rate for the SGD optimizer.
   * @param lambda Regularization parameter for the optimization.
   */
  SVDPlusPlus(const size_t iterations = 10,
              const double alpha = 0.001,
              const double lambda = 0.1);

  /**
   * Trains the model and obtains user/item matrices, user/item bias, and
   * item implicit matrix.
   *
   * @param data Rating data matrix.
   * @param implicitData Implicit feedback.
   * @param rank Rank parameter to be used for optimization.
   * @param u Item matrix obtained on decomposition.
   * @param v User matrix obtained on decomposition.
   * @param p Item bias.
   * @param q User bias.
   * @param y Item matrix with respect to implicit feedback.
   */
  void Apply(const arma::mat& data,
             const arma::mat& implicitData,
             const size_t rank,
             arma::mat& u,
             arma::mat& v,
             arma::vec& p,
             arma::vec& q,
             arma::mat& y);

  /**
   * Trains the model and obtains user/item matrices, user/item bias, and
   * item implicit matrix. Whether a user rates an item is used as implicit
   * feedback.
   *
   * @param data Rating data matrix.
   * @param rank Rank parameter to be used for optimization.
   * @param u Item matrix obtained on decomposition.
   * @param v User matrix obtained on decomposition.
   * @param p Item bias.
   * @param q User bias.
   * @param y Item matrix with respect to implicit feedback. Each column is a
   *     latent vector of an item with respect to implicit feedback.
   */
  void Apply(const arma::mat& data,
             const size_t rank,
             arma::mat& u,
             arma::mat& v,
             arma::vec& p,
             arma::vec& q,
             arma::mat& y);

  /**
   * Converts the User, Item matrix of implicit data to Item-User Table.
   */
  static void CleanData(const arma::mat& implicitData,
                        arma::sp_mat& cleanedData,
                        const arma::mat& data);

 private:
  //! Number of optimization iterations.
  size_t iterations;
  //! Learning rate for the SGD optimizer.
  double alpha;
  //! Regularization parameter for the optimization.
  double lambda;
};

} // namespace mlpack

// Include implementation.
#include "svdplusplus_impl.hpp"

#endif
