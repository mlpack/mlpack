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
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_ANN_INIT_RULES_KATHIRVALAVAKUMAR_SUBAVATHI_INIT_HPP
#define __MLPACK_METHODS_ANN_INIT_RULES_KATHIRVALAVAKUMAR_SUBAVATHI_INIT_HPP

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
 * @f{eqnarray*}{
 * \overline{s} &=& f^{-1}(\overline{t}) \\
 * \Theta^{1}_{p} &\le& \overline{s}
 *     \sqrt{\frac{3}{I \sum_{i = 1}^{I} (x_{ip}^2)}} \\
 * \Theta^1 &=& min(\Theta_{p}^{1}); p=1,2,..,P \\
 * -\Theta^{1} \le w_{i}^{1} &\le& \Theta^{1}
 * @f}
 *
 * where I is the number of inputs including the bias, p refers the pattern
 * considered in training, f is the transfer function and \={s} is the active
 * region in which the derivative of the activation function is greater than 4%
 * of the maximum derivatives.
 */
class KathirvalavakumarSubavathiInitialization
{
 public:
  /**
   * Initialize the random initialization rule with the given values.
   *
   * @param data The input patterns.
   * @param s Parameter that defines the active region.
   */
  template<typename eT>
  KathirvalavakumarSubavathiInitialization(const arma::Mat<eT>& data,
                                           const double s) : s(s)
  {
    dataSum = arma::sum(data + data);
  }

  /**
   * Initialize the elements of the specified weight matrix with the
   * Kathirvalavakumar-Subavathi method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename eT>
  void Initialize(arma::Mat<eT>& W, const size_t rows, const size_t cols)
  {
    arma::Row<eT> b = s * arma::sqrt(3 / (rows * dataSum));
    const double theta = b.min();

    RandomInitialization randomInit(-theta, theta);
    randomInit.Initialize(W, rows, cols);
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor with the
   * Kathirvalavakumar-Subavathi method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename eT>
  void Initialize(arma::Cube<eT>& W,
                  const size_t rows,
                  const size_t cols,
                  const size_t slices)
  {
    W = arma::Cube<eT>(rows, cols, slices);

    for (size_t i = 0; i < slices; i++)
      Initialize(W.slice(i), rows, cols);
  }

 private:
  //! Parameter that defines the sum of elements in each column.
  arma::colvec dataSum;

  //! Parameter that defines the active region.
  const double s;
}; // class KathirvalavakumarSubavathiInitialization


} // namespace ann
} // namespace mlpack

#endif
