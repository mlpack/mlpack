/**
 * @file methods/ann/init_rules/lecun_normal_init.hpp
 * @author Dakshit Agrawal
 * @author Prabhat Sharma
 *
 * Intialization rule given by Lecun et. al. for neural networks and
 * also mentioned in Self Normalizing Networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_ANN_INIT_RULES_LECUN_NORMAL_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_LECUN_NORMAL_INIT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/math/random.hpp>

namespace mlpack {

/**
 * This class is used to initialize weight matrix with the Lecun Normalization
 * initialization rule.
 *
 * For more information, the following papers can be referred to:
 *
 * @code
 * @inproceedings{Klambauer2017,
 *   itle  = {Self-Normalizing Neural Networks.},
 *   author = {Klambauer, Günter and Unterthiner, Thomas
 *             and Mayr, Andreas and Hochreiter, Sepp},
 *   pages  = {972-981},
 *   year   = {2017}
 * }
 *
 * @inproceedings{LeCun1998,
 *   title  = {Efficient BackProp},
 *   author = {LeCun, Yann and Bottou, L{\'e}on and Orr, Genevieve B.
 *             and M\"{u}ller, Klaus-Robert},
 *   year   = {1998},
 *   pages  = {9--50}
 * }
 * @endcode
 *
*/
class LecunNormalInitialization
{
 public:
  /**
   * Initialize the LecunNormalInitialization object.
   */
  LecunNormalInitialization()
  {
    // Nothing to do here.
  }

  /**
   * Initialize the elements of the weight matrix with the Lecun
   * Normal initialization rule.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template <typename MatType>
  void Initialize(MatType& W,
                  const size_t rows,
                  const size_t cols)
  {
    // Lecun initialization rule says to initialize weights with random
    // values taken from a gaussian distribution with mean = 0 and
    // standard deviation = sqrt(1 / rows), i.e. variance = (1 / rows).
    const double variance = 1.0 / ((double) rows);

    if (W.is_empty())
      W.set_size(rows, cols);

    // Multipling a random variable X with variance V(X) by some factor c,
    // then the variance V(cX) = (c ^ 2) * V(X).
    W = randn<MatType>(rows, cols) * std::sqrt(variance);
  }

  /**
   * Initialize the elements of the weight matrix with the Lecun
   * Normal initialization rule.
   *
   * @param W Weight matrix to initialize.
   */
  template <typename MatType>
  void Initialize(MatType& W,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    // Lecun initialization rule says to initialize weights with random
    // values taken from a gaussian distribution with mean = 0 and
    // standard deviation = sqrt(1 / rows), i.e. variance = (1 / rows).
    const double variance = 1.0 / (double) W.n_rows;

    if (W.is_empty())
      Log::Fatal << "Cannot initialize an empty matrix." << std::endl;

    // Multipling a random variable X with variance V(X) by some factor c,
    // then the variance V(cX) = (c ^ 2) * V(X).
    W = randn<MatType>(W.n_rows, W.n_cols) * std::sqrt(variance);
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor
   * with Lecun Normal initialization rule.
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
   * with Lecun Normal initialization rule.
   *
   * @param W Weight matrix to initialize.
   */
  template <typename CubeType>
  void Initialize(CubeType& W,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0)
  {
    if (W.is_empty())
      Log::Fatal << "Cannot initialize an empty cube." << std::endl;

    for (size_t i = 0; i < W.n_slices; ++i)
      Initialize(W.slice(i));
  }

  /**
   * Serialize the initialization.  (Nothing to serialize for this one.)
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */) { }
}; // class LecunNormalInitialization

} // namespace mlpack

#endif
