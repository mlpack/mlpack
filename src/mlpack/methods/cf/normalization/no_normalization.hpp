/**
 * @file methods/cf/normalization/no_normalization.hpp
 * @author Wenhao Huang
 *
 * This class performs no normalization. It is used as default type of
 * normalization for CF class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_NORMALIZATION_NO_NORMALIZATION_HPP
#define MLPACK_METHODS_CF_NORMALIZATION_NO_NORMALIZATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This normalization class doesn't perform any normalization. It is the default
 * normalization type for CF class.
 */
class NoNormalization
{
 public:
  // Empty constructor.
  NoNormalization() { }

  /**
   * Do nothing.
   *
   * @param * (data) Input dataset.
   */
  template<typename MatType>
  inline void Normalize(const MatType& /* data */) const { }

  /**
   * Do nothing.
   *
   * @param * (user) User ID.
   * @param * (item) Item ID.
   * @param rating Computed rating before denormalization.
   */
  inline double Denormalize(const size_t /* user */,
                            const size_t /* item */,
                            const double rating) const
  {
    return rating;
  }

  /**
   * Do nothing.
   *
   * @param * (combinations) User/Item combinations.
   * @param * (predictions) Predicted ratings for each user/item combination.
   */
  inline void Denormalize(const arma::Mat<size_t>& /* combinations */,
                          const arma::vec& /* predictions */) const
  { }

  /**
   * Serialization.
   */
  template<typename Archive>
  void serialize(Archive& /* ar */, const uint32_t /* version */) { }
};

} // namespace mlpack

#endif
