/**
 * @file regularized_svd.hpp
 * @author Siddharth Agrawal
 *
 * An implementation of Regularized SVD.
 */

#ifndef __MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_SVD_HPP
#define __MLPACK_METHODS_REGULARIZED_SVD_REGULARIZED_SVD_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/methods/cf/cf.hpp>

#include "regularized_svd_function.hpp"

namespace mlpack {
namespace svd {

template<
  template<typename> class OptimizerType = mlpack::optimization::SGD
>
class RegularizedSVD
{
 public:
 
  /**
   * Constructor for Regularized SVD. Obtains the user and item matrices after
   * training on the passed data. The constructor initiates an object of class
   * RegularizedSVDFunction for optimization. It uses the SGD optimizer by
   * default. The optimizer uses a template specialization of Optimize().
   *
   * @param iterations Number of optimization iterations.
   * @param alpha Learning rate for the SGD optimizer.
   * @param lambda Regularization parameter for the optimization.
   */
  RegularizedSVD(const size_t iterations = 10,
                 const double alpha = 0.01,
                 const double lambda = 0.02);
  
  /**
   * Obtains the user and item matrices using the provided data and rank.
   *
   * @param data Rating data matrix.
   * @param rank Rank parameter to be used for optimization.
   * @param u Item matrix obtained on decomposition.
   * @param v User matrix obtained on decomposition.
   */
  void Apply(const arma::mat& data,
             const size_t rank,
             arma::mat& u,
             arma::mat& v);
                 
 private:
  //! Number of optimization iterations.
  size_t iterations;
  //! Learning rate for the SGD optimizer.
  double alpha;
  //! Regularization parameter for the optimization.
  double lambda;
};

}; // namespace svd
}; // namespace mlpack

namespace mlpack {
namespace cf {

//! Factorizer traits of Regularized SVD.
template<>
class FactorizerTraits<mlpack::svd::RegularizedSVD<> >
{
 public:
  //! Data provided to RegularizedSVD need not be cleaned.
  static const bool IsCleaned = true;
};

}; // namespace cf
}; // namespace mlpack

// Include implementation.
#include "regularized_svd_impl.hpp"

#endif
