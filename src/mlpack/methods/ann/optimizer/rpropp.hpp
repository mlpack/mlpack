/**
 * @file rpropp.hpp
 * @author Marcus Edel
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a random matrix to the weight matrix. 
 */
#ifndef __MLPACK_METHOS_ANN_OPTIMIZER_RPROPP_HPP
#define __MLPACK_METHOS_ANN_OPTIMIZER_RPROPP_HPP

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
class RPROPp
{
 public:
  /**
   * Initialize the random initialization rule with the given lower bound and
   * upper bound.
   *
   * @param lowerBound The number used as lower bound.
   * @param upperBound The number used as upper bound.
   */
  RPROPp(const size_t cols,
         const size_t rows,
         const double etaMin = 0.5,
         const double etaPlus = 1.2,
         const double minDelta = 1e-9,
         const double maxDelta = 50,
         const double initialUpdate = 0.1) :
      etaMin(etaMin), etaPlus(etaPlus), minDelta(minDelta), maxDelta(maxDelta)
  {
    prevDerivs = arma::zeros<MatType>(rows, cols);
    prevWeightChange = arma::zeros<MatType>(rows, cols);

    updateValues = arma::ones<MatType>(rows, cols);
    updateValues.fill(initialUpdate);
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
        {
          updateValues(j, i) = std::min(updateValues(j, i) * etaPlus, maxDelta);
          prevWeightChange(j, i) = boost::math::sign(gradient(j, i)) * updateValues(j, i);
          prevDerivs(j, i) = gradient(j, i);
        }
        else if (derivs(j, i) < 0)
        {
          updateValues(j, i) = std::max(updateValues(j, i) * etaMin, minDelta);
          prevDerivs(j, i) = 0;
        }
        else
        {
          prevWeightChange(j, i) = boost::math::sign(gradient(j, i)) * updateValues(j, i);
          prevDerivs(j, i) = gradient(j, i);
        }

        weights(j, i) -= prevWeightChange(j, i);
      }
    }
  }


 private:
  //! The number used as learning rate.
  const double etaMin;

  const double etaPlus;

  const double minDelta;

  const double maxDelta;

  MatType updateValues;

  MatType prevWeightChange;

  //! weight momentum
  MatType prevDerivs;
}; // class RPROPp

}; // namespace ann
}; // namespace mlpack

#endif
