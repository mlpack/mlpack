#ifndef __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_SPARSE_FUNCTION_HPP
#define __MLPACK_METHODS_ANN_PERFORMANCE_FUNCTIONS_SPARSE_FUNCTION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The cost function design for sparse autoencoder
 */
template<typename DataType = arma::mat>
class SparseErrorFunction
{
 public:
  /**
   * @param w1 weight parameter of hidden layer
   * @param w2 weight parameter of output layer
   * @param rhoCap Average activations of the hidden layer
   * @param lambda L2-regularization parameter
   * @param beta KL divergence parameter
   * @param rho Sparsity parameter
   */
  SparseErrorFunction(const double lambda = 0.0001,
                      const double beta = 3,
                      const double rho = 0.01) :    
    lambda(lambda), beta(beta), rho(rho)
  {}

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

  void Beta(double value) { beta = value;}
  double Beta() const { return beta; }

  void Lambda(double value) { lambda = value;}
  double Lambda() const { return lambda; }

  void Rho(double value) { rho = value;}
  double Rho() const { return rho; }

  /**
   * Computes the cost of sparse autoencoder
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
    return Error(std::get<0>(network).Weights(),
                 std::get<3>(network).Weights(),
                 std::get<3>(network).RhoCap(),
                 target, error);
  }

  /**
   * Computes the cost of sparse autoencoder
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
  double lambda;
  double beta;
  double rho;

}; // class SparseErrorFunction

}; // namespace ann
}; // namespace mlpack

#endif
