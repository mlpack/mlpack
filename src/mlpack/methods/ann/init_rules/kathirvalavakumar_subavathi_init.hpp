/**
 * @file kathirvalavakumar_subavathi_init.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the initialization method by T.
 * Kathirvalavakumar and S. Subavathi. This initialization rule is based on
 * sensitivity analysis using cauchy’s inequality.
 *
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings{KathirvalavakumarJILSA2011,
 *   title={A New Weight Initialization Method Using Cauchy’s Inequality Based
 *   on Sensitivity Analysis},
 *   author={T. Kathirvalavakumar and S. Subavathi},
 *   booktitle={Journal of Intelligent Learning Systems and Applications,
 *   Vol. 3 No. 4},
 *   year={2011}
 * }
 * @endcode
 */
#ifndef __MLPACK_METHOS_ANN_INIT_RULES_KATHIRVALAVAKUMAR_SUBAVATHI_INIT_HPP
#define __MLPACK_METHOS_ANN_INIT_RULES_KATHIRVALAVAKUMAR_SUBAVATHI_INIT_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

#include "random_init.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * This class is used to initialize the weight matrix with the method proposed
 * by T. Kathirvalavakumar and S. Subavathi. The method is based on sensitivity
 * analysis using using cauchy’s inequality. The method is defined by
 *
 * @f[
 * \={s} &=& f^{-1}(\={t})
 * \Theta^{1}_{p} \le \={s} \sqrt{\frac{3}{I\sum_{i=1}^{I}(x_{ip}^2)}}
 * \Theta^1 = min(\Theta_{p}^{1}); p=1,2,..,P
 * -\Theta^{1} \le w_{i}^{1} \le \Theta^{1} \\
 * @f]
 *
 * Where I is the number of inputs including the bias, p refers the pattern
 * considered in training, f is the transfer function and \={s} is the active
 * region in which the derivative of the activation function is greater than 4%
 * of the maximum derivatives.
 *
 * @tparam MatType Type of matrix (should be arma::mat or arma::spmat).
 */
template<typename MatType = arma::mat>
class KathirvalavakumarSubavathiInitialization
{
 public:
  /**
   * Initialize the random initialization rule with the given values.
   *
   * @param data The input patterns.
   * @param s Parameter that defines the active region.
   */
  KathirvalavakumarSubavathiInitialization(const MatType& data, const double s)
      : data(data), s(s) { }

  /**
   * Initialize the elements of the specified weight matrix with the
   * Kathirvalavakumar-Subavathi method.
   *
   * @param W Weight matrix to initialize.
   * @param n_rows Number of rows.
   * @return n_cols Number of columns.
   */
  void Initialize(MatType& W, const size_t n_rows, const size_t n_cols)
  {
    arma::rowvec b = s * arma::sqrt(3 / (n_rows * sum(data + data)));
    double theta = b.min();

    RandomInitialization<MatType> randomInit(-theta, theta);
    randomInit.Initialize(W, n_rows, n_cols);
  }

 private:
  //! The input patterns.
  MatType data;

  //! Parameter that defines the active region.
  const double s;
}; // class KathirvalavakumarSubavathiInitialization


}; // namespace ann
}; // namespace mlpack

#endif
