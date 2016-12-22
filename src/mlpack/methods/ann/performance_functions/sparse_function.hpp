/**
 * @file sparse_function.hpp
 * @author Siddharth Agrawal
 * @author Tham Ngap Wei
 *
 * Definition and implementation of the sparse performance function.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_SPARSE_FUNCTION_HPP
#define MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_SPARSE_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The cost function design for the sparse autoencoder.
 */
template<typename DataType = arma::mat>
class SparseErrorFunction
{
 public:
  /**
   * Computes the cost of sparse autoencoder.
   *
   * @param lambda L2-regularization parameter.
   * @param beta KL divergence parameter.
   * @param rho Sparsity parameter.
   */
  SparseErrorFunction(const double lambda = 0.0001,
                      const double beta = 3,
                      const double rho = 0.01) :
    lambda(lambda), beta(beta), rho(rho)
  {
    // Nothing to do here.
  }

  SparseErrorFunction(SparseErrorFunction &&layer) noexcept
  {
    *this = std::move(layer);
  }

  SparseErrorFunction& operator=(SparseErrorFunction &&layer) noexcept
  {
    lambda = layer.lambda;
    beta = layer.beta;
    rho = layer.rho;

    return *this;
  }

  //! Get the KL divergence parameter.
  double Beta() const { return beta; }
  //! Modify the KL divergence parameter.
  void Beta(double value) { beta = value;}

  //! Get the L2-regularization parameter.
  double Lambda() const { return lambda; }
  //! Modify the L2-regularization parameter.
  void Lambda(double value) { lambda = value;}

  //! Get the sparsity parameter.
  double Rho() const { return rho; }
  //! Modify the sparsity parameter.
  void Rho(double value) { rho = value;}

  /**
   * Computes the cost of sparse autoencoder.
   *
   * @param network Network type of FFN, CNN or RNN
   * @param target Target data.
   * @param error different between output and the input
   * @return sum of squared errors.
   */
  template<typename InType, typename Tp>
  double Error(const Tp& network,
               const InType& target, const InType &error)
  {
    return Error(std::get<0>(network).Weights(), std::get<3>(network).Weights(),
        std::get<3>(network).RhoCap(), target, error);
  }

  /**
   * Computes the cost of sparse autoencoder.
   *
   * @param w1 weights of hidden layer
   * @param w2 weights of output layer
   * @param rhoCap Average activations of the hidden layer
   * @param target Target data.
   * @param error different between output and the input
   * @return sum of squared errors.
   */
  template<typename InType>
  double Error(const InType& w1, const InType& w2,
               const InType& rhoCap, const InType& target,
               const InType& error)
  {
    // Calculate squared L2-norms of w1 and w2.
    const double wL2SquaredNorm =
        arma::accu(w1 % w1) + arma::accu(w2 % w2);

    // Calculate the reconstruction error, the regularization cost and the KL
    // divergence cost terms. 'sumOfSquaresError' is the average squared l2-norm
    // of the reconstructed data difference. 'weightDecay' is the squared l2-norm
    // of the weights w1 and w2. 'klDivergence' is the cost of the hidden layer
    // activations not being low. It is given by the following formula:
    // KL = sum_over_hSize(rho*log(rho/rhoCaq) + (1-rho)*log((1-rho)/(1-rhoCap)))
    const double sumOfSquaresError =
        0.5 * arma::accu(error % error) / target.n_cols;

    const double weightDecay = 0.5 * lambda * wL2SquaredNorm;
    const double klDivergence =
        beta * arma::accu(rho * arma::trunc_log(rho / rhoCap) + (1 - rho) *
                          arma::trunc_log((1 - rho) / (1 - rhoCap)));

    // The cost is the sum of the terms calculated above.
    return sumOfSquaresError + weightDecay + klDivergence;
  }

 private:
  //! Locally stored L2-regularization parameter.
  double lambda;

  //! Locally stored KL divergence parameter.
  double beta;

  //! Locally stored sparsity parameter.
  double rho;

}; // class SparseErrorFunction

} // namespace ann
} // namespace mlpack

#endif
