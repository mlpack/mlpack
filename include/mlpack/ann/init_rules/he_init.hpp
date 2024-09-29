/**
 * @file methods/ann/init_rules/he_init.hpp
 * @author Dakshit Agrawal
 * @author Prabhat Sharma
 *
 * Intialization rule given by He et. al. for neural networks. The He
 * initialization initializes weights of the neural network to better
 * suit the rectified activation units.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_INIT_RULES_HE_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_HE_INIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

namespace mlpack {

/**
 * This class is used to initialize weight matrix with the He
 * initialization rule given by He et. al. for neural networks. The He
 * initialization initializes weights of the neural network to better
 * suit the rectified activation units.
 *
 * For more information, the following paper can be referred to:
 *
 * @code
 * @article{Delving2015,
 *   title   = {Delving Deep into Rectifiers: Surpassing Human-Level Performance
 *              on ImageNet Classification},
 *   author  = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
 *   journal = {2015 IEEE International Conference on Computer Vision (ICCV)},
 *   year    = {2015},
 *   pages   = {1026-1034}
 * }
 * @endcode
 *
 */
class HeInitialization
{
 public:
  /**
   * Initialize the HeInitialization object.
   */
  HeInitialization()
  {
    // Nothing to do here.
  }

  /**
   * Initialize the elements of the weight matrix with the He initialization
   * rule.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template <typename MatType>
  void Initialize(MatType& W, const size_t rows, const size_t cols)
  {
    // He initialization rule says to initialize weights with random
    // values taken from a gaussian distribution with mean = 0 and
    // standard deviation = sqrt(2/rows), i.e. variance = (2/rows).
    const double variance = 2.0 / (double) rows;

    if (W.is_empty())
      W.set_size(rows, cols);

    // Multipling a random variable X with variance V(X) by some factor c,
    // then the variance V(cX) = (c^2) * V(X).
    W = randn<MatType>(rows, cols) * std::sqrt(variance);
  }

  /**
   * Initialize the elements of the weight matrix with the He initialization
   * rule.
   *
   * @param W Weight matrix to initialize.
   */
  template <typename MatType>
  void Initialize(MatType& W,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    // He initialization rule says to initialize weights with random
    // values taken from a gaussian distribution with mean = 0 and
    // standard deviation = sqrt(2 / rows), i.e. variance = (2 / rows).
    const double variance = 2.0 / (double) W.n_rows;

    if (W.is_empty())
      Log::Fatal << "Cannot initialize an empty matrix." << std::endl;

    // Multipling a random variable X with variance V(X) by some factor c,
    // then the variance V(cX) = (c^2) * V(X).
    W = randn<MatType>(W.n_rows, W.n_cols) * std::sqrt(variance);
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor
   * with He initialization rule.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param slices Number of slices.
   */
  template <typename CubeType>
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
   * Initialize the elements of the specified weight 3rd order tensor
   * with He initialization rule.
   *
   * @param W Weight matrix to initialize.
   */
  template <typename CubeType>
  void Initialize(CubeType& W,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    if (W.is_empty())
      Log::Fatal << "Cannot initialize an empty matrix" << std::endl;

    for (size_t i = 0; i < W.n_slices; ++i)
      Initialize(W.slice(i));
  }

  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */)
  {
    // Nothing to do.
  }
}; // class HeInitialization

} // namespace mlpack

#endif
