/**
 * @file methods/ann/init_rules/kathirvalavakumar_subavathi_init.hpp
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
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_KATHIRVALAVAKUMAR_SUBAVATHI_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_KATHIRVALAVAKUMAR_SUBAVATHI_INIT_HPP

#include <mlpack/prereqs.hpp>

#include "init_rules_traits.hpp"
#include "random_init.hpp"

#include <mlpack/methods/ann/activation_functions/logistic_function.hpp>

#include <iostream>

namespace mlpack {

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
  template<typename MatType>
  KathirvalavakumarSubavathiInitialization(const MatType& data,
                                           const double s) : s(s)
  {
    dataSum = sum(data % data);
  }

  /**
   * Initialize the elements of the specified weight matrix with the
   * Kathirvalavakumar-Subavathi method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename MatType>
  void Initialize(MatType& W, const size_t rows, const size_t cols)
  {
    using RowType = typename GetRowType<MatType>::type;
    RowType b = s * sqrt(3 / (rows * dataSum));
    const double theta = b.min();
    RandomInitialization randomInit(-theta, theta);
    randomInit.Initialize(W, rows, cols);
  }

  /**
   * Initialize the elements of the specified weight matrix with the
   * Kathirvalavakumar-Subavathi method.
   *
   * @param W Weight matrix to initialize.
   */
  template<typename MatType>
  void Initialize(MatType& W,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    using RowType = typename GetRowType<MatType>::type;
    RowType b = s * sqrt(3 / (W.n_rows * dataSum));
    const double theta = b.min();
    RandomInitialization randomInit(-theta, theta);
    randomInit.Initialize(W);
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor with the
   * Kathirvalavakumar-Subavathi method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param slices Number of slices
   */
  template<typename CubeType>
  void Initialize(CubeType& W,
                  const size_t rows,
                  const size_t cols,
                  const size_t slices)
  {
    if (W.is_empty())
      W.set_size(rows, cols, slices);

    for (size_t i = 0; i < slices; ++i)
      Initialize(W.slice(i), rows, cols);
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor with the
   * Kathirvalavakumar-Subavathi method.
   *
   * @param W Weight matrix to initialize.
   */
  template<typename CubeType>
  void Initialize(CubeType& W,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    if (W.is_empty())
      Log::Fatal << "Cannot initialize an empty cube." << std::endl;

    for (size_t i = 0; i < W.n_slices; ++i)
      Initialize(W.slice(i));
  }

  /**
   * Serialize the initialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(dataSum));
    ar(CEREAL_NVP(s));
  }

 private:
  //! Parameter that defines the sum of elements in each column.
  arma::rowvec dataSum;

  //! Parameter that defines the active region.
  double s;
}; // class KathirvalavakumarSubavathiInitialization

//! Initialization traits of the kathirvalavakumar subavath initialization rule.
template<>
class InitTraits<KathirvalavakumarSubavathiInitialization>
{
 public:
  //! The kathirvalavakumar subavath initialization rule is applied over the
  //! entire network.
  static const bool UseLayer = false;
};


} // namespace mlpack

#endif
