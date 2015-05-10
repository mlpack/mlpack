/**
 * @file irpropm.hpp
 * @author Marcus Edel
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a random matrix to the weight matrix.
 */
#ifndef __MLPACK_METHODS_ANN_OPTIMIZER_IRPROPM_HPP
#define __MLPACK_METHODS_ANN_OPTIMIZER_IRPROPM_HPP

#include <mlpack/core.hpp>
#include <boost/math/special_functions/sign.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize randomly the weight matrix.
 *
 * @tparam MatType Type of matrix (should be arma::mat or arma::spmat).
 */
template<typename MatType = arma::mat, typename VecType = arma::rowvec>
class iRPROPm
{
 public:
  /**
   * Initialize the random initialization rule with the given lower bound and
   * upper bound.
   *
   * @param lowerBound The number used as lower bound.
   * @param upperBound The number used as upper bound.
   */
  iRPROPm(const size_t cols,
          const size_t rows,
          const double etaMin = 0.5,
          const double etaPlus = 1.2,
          const double minDelta = 1e-9,
          const double maxDelta = 50) :
      etaMin(etaMin), etaPlus(etaPlus), minDelta(minDelta), maxDelta(maxDelta)
  {
    prevDerivs = arma::zeros<MatType>(rows, cols);
    prevDelta = arma::zeros<MatType>(rows, cols);

    prevError = arma::datum::inf;
  }

  void UpdateWeights(MatType& weights,
                     const MatType& gradient,
                     const double /* unused */)
  {
    MatType derivs = gradient % prevDerivs;

    for (size_t i(0); i < derivs.n_cols; i++)
    {
      for (size_t j(0); j < derivs.n_rows; j++)
      {
        if (derivs(j, i) >= 0)
        {
          prevDelta(j, i) = std::min(prevDelta(j, i) * etaPlus, maxDelta);
          prevDerivs(j, i) = gradient(j, i);
        }
        else
        {
          prevDelta(j, i) = std::max(prevDelta(j, i) * etaMin, minDelta);
          prevDerivs(j, i) = 0;
        }
      }
    }

    weights -= arma::sign(gradient) % prevDelta;
  }

 private:
  //! The number used as learning rate.
  const double etaMin;

  const double etaPlus;

  const double minDelta;

  const double maxDelta;

  double prevError;

  MatType prevDelta;

  //! weight momentum
  MatType prevDerivs;
}; // class iRPROPm

}; // namespace ann
}; // namespace mlpack

#endif
