/**
 * @file methods/ann/init_rules/nguyen_widrow_init.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the Nguyen-Widrow method. This
 * initialization rule initialize the weights so that the active regions of the
 * neurons are approximately evenly distributed over the input space.
 *
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings{NguyenIJCNN1990,
 *   title={Improving the learning speed of 2-layer neural networks by choosing
 *   initial values of the adaptive weights},
 *   booktitle={Neural Networks, 1990., 1990 IJCNN International Joint
 *   Conference on},
 *   year={1990}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_NGUYEN_WIDROW_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_NGUYEN_WIDROW_INIT_HPP

#include <mlpack/prereqs.hpp>

#include "init_rules_traits.hpp"
#include "random_init.hpp"

namespace mlpack {

/**
 * This class is used to initialize the weight matrix with the Nguyen-Widrow
 * method. The method is defined by
 *
 * @f{eqnarray*}{
 * \gamma &\le& w_i \le \gamma \\
 * \beta &=& 0.7H^{\frac{1}{I}} \\
 * n &=& \sqrt{\sum_{i=0}{I}w_{i}^{2}} \\
 * w_i &=& \frac{\beta w_i}{n}
 * @f}
 *
 * Where H is the number of neurons in the outgoing layer, I represents the
 * number of neurons in the ingoing layer and gamma defines the random interval
 * that is used to initialize the weights with a random value in a specific
 * range.
 */
class NguyenWidrowInitialization
{
 public:
  /**
   * Initialize the random initialization rule with the given lower bound and
   * upper bound.
   *
   * @param lowerBound The number used as lower bound.
   * @param upperBound The number used as upper bound.
   */
  NguyenWidrowInitialization(const double lowerBound = -0.5,
                             const double upperBound = 0.5) :
      lowerBound(lowerBound), upperBound(upperBound) { }

  /**
   * Initialize the elements of the specified weight matrix with the
   * Nguyen-Widrow method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename MatType>
  void Initialize(MatType& W, const size_t rows, const size_t cols)
  {
    RandomInitialization randomInit(lowerBound, upperBound);
    randomInit.Initialize(W, rows, cols);

    double beta = 0.7 * std::pow(cols, 1.0 / rows);
    W *= (beta / norm(W));
  }

  /**
   * Initialize the elements of the specified weight matrix with the
   * Nguyen-Widrow method.
   *
   * @param W Weight matrix to initialize.
   */
  template<typename MatType>
  void Initialize(MatType& W,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    RandomInitialization randomInit(lowerBound, upperBound);
    randomInit.Initialize(W);

    double beta = 0.7 * std::pow(W.n_cols, 1.0 / W.n_rows);
    W *= (beta / norm(W));
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor with the
   * Nguyen-Widrow method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param slices Number of slices.
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
   * Nguyen-Widrow method.
   *
   * @param W Weight matrix to initialize.
   */
  template<typename CubeType>
  void Initialize(CubeType& W,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    if (W.is_empty())
      Log::Fatal << "Cannot initialize an empty matrix." << std::endl;

    for (size_t i = 0; i < W.n_slices; ++i)
      Initialize(W.slice(i));
  }

  /**
   * Serialize the initialization.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(lowerBound));
    ar(CEREAL_NVP(upperBound));
  }

 private:
  //! The number used as lower bound.
  double lowerBound;

  //! The number used as upper bound.
  double upperBound;
}; // class NguyenWidrowInitialization

//! Initialization traits of the Nguyen-Widrow initialization rule.
template<>
class InitTraits<NguyenWidrowInitialization>
{
 public:
  //! The Nguyen-Widrow initialization rule is applied over the entire network.
  static const bool UseLayer = false;
};

} // namespace mlpack

#endif
