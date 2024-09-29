/**
 * @file methods/ann/init_rules/random_init.hpp
 * @author Marcus Edel
 *
 * Intialization rule for the neural networks. This simple initialization is
 * performed by assigning a random matrix to the weight matrix.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_INIT_RULES_RANDOM_INIT_HPP
#define MLPACK_METHODS_ANN_INIT_RULES_RANDOM_INIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This class is used to initialize randomly the weight matrix.
 */
class RandomInitialization
{
 public:
  /**
   * Initialize the random initialization rule with the given lower bound and
   * upper bound.
   *
   * @param lowerBound The number used as lower bound.
   * @param upperBound The number used as upper bound.
   */
  RandomInitialization(const double lowerBound = -1,
                       const double upperBound = 1) :
      lowerBound(lowerBound), upperBound(upperBound) { }

  /**
   * Initialize the random initialization rule with the given bound.
   * Using the negative of the bound as lower bound and the positive bound as
   * upper bound.
   *
   * @param bound The number used as lower bound
   */
  RandomInitialization(const double bound) :
      lowerBound(-std::abs(bound)), upperBound(std::abs(bound)) { }

  /**
   * Initialize randomly the elements of the specified weight matrix.
   *
   * @param W Weight matrix to initialize.
   * @param rows Number of rows.
   * @param cols Number of columns.
   */
  template<typename MatType>
  void Initialize(MatType& W, const size_t rows, const size_t cols)
  {
    if (W.is_empty())
      W.set_size(rows, cols);

    W.randu();
    W *= (upperBound - lowerBound);
    W += lowerBound;
  }

  /**
   * Initialize randomly the elements of the specified weight matrix.
   *
   * @param W Weight matrix to initialize.
   */
  template<typename MatType>
  void Initialize(MatType& W,
      const typename std::enable_if_t<IsMatrix<MatType>::value>* = 0)
  {
    if (W.is_empty())
      Log::Fatal << "Cannot initialize an empty matrix." << std::endl;

    W.randu();
    W *= (upperBound - lowerBound);
    W += lowerBound;
  }

  /**
   * Initialize randomly the elements of the specified weight 3rd order tensor.
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
   * Initialize randomly the elements of the specified weight 3rd order tensor.
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
}; // class RandomInitialization

} // namespace mlpack

#endif
