/**
 * @file methods/ann/init_rules/glorot_init.hpp
 * @author Prabhat Sharma
 *
 * Definition and implementation of the Glorot initialization method. This
 * initialization rule initialize the weights to maintain activation variances
 * and back-propagated gradients variance as one moves up or down the network.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_GLOROT_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_GLOROT_INIT_HPP

#include <mlpack/prereqs.hpp>
#include "random_init.hpp"
#include "gaussian_init.hpp"

namespace mlpack {

/**
 * This class is used to initialize the weight matrix with the Glorot
 * Initialization method. The method is defined by
 *
 * @f{eqnarray*}{
 * \mathrm{Var}[w_i] &=& \frac{2}{n_i + n_{i+1}} \\
 * w_i \sim \mathrm{U}[-\frac{\sqrt{6}}{\sqrt{n_i + n_{i+1}}},
 * \frac{\sqrt{6}}{\sqrt{n_i + n_{i+1}}}]
 * @f}
 *
 * where @f$ n_{i+1} @f$ is the number of neurons in the outgoing layer, @f$ n_i
 * @f$ represents the number of neurons in the ingoing layer. Here Normal
 * Distribution may also be used if needed
 *
 * For more information, see the following paper.
 *
 * @code
 * @inproceedings {pmlr-v9-glorot10a,
 *  title     = {Understanding the difficulty of training
 *               deep feedforward neural networks},
 *  author    = {Xavier Glorot and Yoshua Bengio},
 *  booktitle = {Proceedings of the Thirteenth International Conference
 *               on Artificial Intelligence and Statistics},
 *  year      = {2010}
 * }
 * @endcode
 *
 */
template<bool Uniform = true>
class GlorotInitializationType
{
 public:
  /**
   * Initialize the Glorot initialization object.
   */
  GlorotInitializationType()
  {
    // Nothing to do here.
  }

  /**
   * Initialize the elements weight matrix with glorot initialization method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename MatType>
  void Initialize(MatType& W,
                  const size_t rows,
                  const size_t cols);

  /**
   * Initialize the elements weight matrix with glorot initialization method.
   *
   * @param W Weight matrix to initialize.
   */
  template<typename MatType>
  void Initialize(MatType& W,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0);

  /**
   * Initialize the elements of the specified weight 3rd order tensor with
   * glorot initialization method.
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
                  const size_t slices);

  /**
   * Initialize the elements of the specified weight 3rd order tensor with
   * glorot initialization method.
   *
   * @param W Weight matrix to initialize.
   */
  template<typename CubeType>
  void Initialize(CubeType& W,
      const typename std::enable_if_t<IsCube<CubeType>::value>* = 0);

  /**
   * Serialize the initialization.  (Nothing to serialize for this one.)
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */) { }
}; // class GlorotInitializationType

template<>
template<typename MatType>
inline void GlorotInitializationType<false>::Initialize(MatType& W,
                                                        const size_t rows,
                                                        const size_t cols)
{
  if (W.is_empty())
    W.set_size(rows, cols);

  double stddev = std::sqrt(2.0 / double(rows + cols));
  GaussianInitialization normalInit(0.0, stddev);
  normalInit.Initialize(W, rows, cols);
}

template<>
template<typename MatType>
inline void GlorotInitializationType<true>::Initialize(MatType& W,
                                                       const size_t rows,
                                                       const size_t cols)
{
  if (W.is_empty())
    W.set_size(rows, cols);

  // Limit of distribution.
  double a = std::sqrt(6) / std::sqrt(rows + cols);
  RandomInitialization randomInit(-a, a);
  randomInit.Initialize(W, rows, cols);
}

template<>
template<typename MatType>
inline void GlorotInitializationType<false>::Initialize(MatType& W,
    const typename std::enable_if_t<IsMatrix<MatType>::value>*)
{
  if (W.is_empty())
    Log::Fatal << "Cannot initialize an empty matrix." << std::endl;

  double stddev = std::sqrt(2.0 / double(W.n_rows + W.n_cols));
  GaussianInitialization normalInit(0.0, stddev);
  normalInit.Initialize(W);
}

template<>
template<typename MatType>
inline void GlorotInitializationType<true>::Initialize(MatType& W,
    const typename std::enable_if_t<IsMatrix<MatType>::value>*)
{
  if (W.is_empty())
    Log::Fatal << "Cannot initialize an empty matrix." << std::endl;

  // Limit of distribution.
  double a = std::sqrt(6) / std::sqrt(W.n_rows + W.n_cols);
  RandomInitialization randomInit(-a, a);
  randomInit.Initialize(W);
}

template<bool Uniform>
template<typename CubeType>
inline void GlorotInitializationType<Uniform>::Initialize(CubeType& W,
                                                          const size_t rows,
                                                          const size_t cols,
                                                          const size_t slices)
{
  if (W.is_empty())
    W.set_size(rows, cols, slices);

  for (size_t i = 0; i < slices; ++i)
    Initialize(W.slice(i), rows, cols);
}

template <bool Uniform>
template<typename CubeType>
inline void GlorotInitializationType<Uniform>::Initialize(CubeType& W,
    const typename std::enable_if_t<IsCube<CubeType>::value>*)
{
  if (W.is_empty())
    Log::Fatal << "Cannot initialize an empty matrix." << std::endl;

  for (size_t i = 0; i < W.n_slices; ++i)
    Initialize(W.slice(i));
}

// Convenience typedefs.

/**
 * XavierInitilization is the popular name for this method.
 */
using XavierInitialization = GlorotInitializationType<true>;

/**
 * GlorotInitialization uses uniform distribution.
 */
using GlorotInitialization = GlorotInitializationType<false>;
// Uses normal distribution
} // namespace mlpack

#endif
