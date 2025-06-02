/**
 * @file methods/ann/init_rules/orthogonal_init.hpp
 * @author Marcus Edel
 *
 * Definition and implementation of the orthogonal matrix initialization method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_ORTHOGONAL_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_ORTHOGONAL_INIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This class is used to initialize the weight matrix with the orthogonal
 * matrix initialization
 */
class OrthogonalInitialization
{
 public:
  /**
   * Initialize the orthogonal matrix initialization rule with the given gain.
   *
   * @param gain The gain value.
   */
  OrthogonalInitialization(const double gain = 1.0) : gain(gain) { }

  /**
   * Initialize the elements of the specified weight matrix with the orthogonal
   * matrix initialization method.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename MatType>
  void Initialize(MatType& W, const size_t rows, const size_t cols)
  {
    MatType V;
    using ColType = typename GetColType<MatType>::type;
    ColType s;

    svd_econ(W, s, V, randu<MatType>(rows, cols));
    W *= gain;
  }

  /**
   * Initialize the elements of the specified weight matrix with the orthogonal
   * matrix initialization method.
   *
   * @param W Weight matrix to initialize.
   */
  template<typename MatType>
  void Initialize(MatType& W,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    MatType V;
    using ColType = typename GetColType<MatType>::type;
    ColType s;

    svd_econ(W, s, V, randu<MatType>(W.n_rows, W.n_cols));
    W *= gain;
  }

  /**
   * Initialize the elements of the specified weight 3rd order tensor with the
   * orthogonal matrix initialization method.
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
   * orthogonal matrix initialization method.
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
    ar(CEREAL_NVP(gain));
  }

 private:
  //! The number used as gain.
  double gain;
}; // class OrthogonalInitialization


} // namespace mlpack

#endif
