/**
 * @file rpropm.hpp
 * @author Marcus Edel
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a random matrix to the weight matrix.
 */
#ifndef __MLPACK_METHOS_ANN_OPTIMIZER_RPROPM_HPP
#define __MLPACK_METHOS_ANN_OPTIMIZER_RPROPM_HPP

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
class RPROPm
{
 public:
  /**
   * Initialize the random initialization rule with the given lower bound and
   * upper bound.
   *
   * @param lowerBound The number used as lower bound.
   * @param upperBound The number used as upper bound.
   */
  RPROPm(const size_t cols,
         const size_t rows,
         const double etaMin = 0.5,
         const double etaPlus = 1.2,
         const double minDelta = 1e-9,
         const double maxDelta = 50) :
      etaMin(etaMin), etaPlus(etaPlus), minDelta(minDelta), maxDelta(maxDelta)
  {
    prevDerivs = arma::zeros<MatType>(rows, cols);
    prevDelta = arma::zeros<MatType>(rows, cols);
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
        if (derivs(j, i) > 0)
          prevDelta(j, i) = std::min(prevDelta(j, i) * etaPlus, maxDelta);
        else
          prevDelta(j, i) = std::max(prevDelta(j, i) * etaMin, minDelta);
      }
    }

    weights -= arma::sign(gradient) % prevDelta;
    prevDerivs = gradient;
  }


 private:
  //! The number used as learning rate.
  const double etaMin;

  const double etaPlus;

  const double minDelta;

  const double maxDelta;

  MatType prevDelta;

  //! weight momentum
  MatType prevDerivs;
}; // class RPROPm

}; // namespace ann
}; // namespace mlpack

#endif


