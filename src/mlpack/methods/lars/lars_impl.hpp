/**
 * @file methods/lars/lars_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of templated LARS functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LARS_LARS_IMPL_HPP
#define MLPACK_METHODS_LARS_LARS_IMPL_HPP

//! In case it hasn't been included yet.
#include "lars.hpp"

namespace mlpack {
namespace regression {

/**
 * Serialize the LARS model.
 */
template<typename Archive>
void LARS::serialize(Archive& ar)
{
  uint8_t version = 1;
  ar & CEREAL_NVP(version);

  // If we're loading, we have to use the internal storage.
  if (Archive::is_loading::value)
  {
    matGram = &matGramInternal;
    ar(CEREAL_NVP(matGramInternal));
  }
  else
  {
    ar & cereal::make_nvp("matGramInternal",
        (const_cast<arma::mat&>(*matGram)));
  }

  ar(CEREAL_NVP(matUtriCholFactor));
  ar(CEREAL_NVP(useCholesky));
  ar(CEREAL_NVP(lasso));
  ar(CEREAL_NVP(lambda1));
  ar(CEREAL_NVP(elasticNet));
  ar(CEREAL_NVP(lambda2));
  ar(CEREAL_NVP(tolerance));
  ar(CEREAL_NVP(betaPath));
  ar(CEREAL_NVP(lambdaPath));
  ar(CEREAL_NVP(activeSet));
  ar(CEREAL_NVP(isActive));
  ar(CEREAL_NVP(ignoreSet));
  ar(CEREAL_NVP(isIgnored));
}

} // namespace regression
} // namespace mlpack

#endif
