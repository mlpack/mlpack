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
void LARS::serialize(Archive& ar, const unsigned int /* version */)
{
  // If we're loading, we have to use the internal storage.
  if (Archive::is_loading::value)
  {
    matGram = &matGramInternal;
    ar & BOOST_SERIALIZATION_NVP(matGramInternal);
  }
  else
  {
    ar & boost::serialization::make_nvp("matGramInternal",
        (const_cast<arma::mat&>(*matGram)));
  }

  ar & BOOST_SERIALIZATION_NVP(matUtriCholFactor);
  ar & BOOST_SERIALIZATION_NVP(useCholesky);
  ar & BOOST_SERIALIZATION_NVP(lasso);
  ar & BOOST_SERIALIZATION_NVP(lambda1);
  ar & BOOST_SERIALIZATION_NVP(elasticNet);
  ar & BOOST_SERIALIZATION_NVP(lambda2);
  ar & BOOST_SERIALIZATION_NVP(tolerance);
  ar & BOOST_SERIALIZATION_NVP(betaPath);
  ar & BOOST_SERIALIZATION_NVP(lambdaPath);
  ar & BOOST_SERIALIZATION_NVP(activeSet);
  ar & BOOST_SERIALIZATION_NVP(isActive);
  ar & BOOST_SERIALIZATION_NVP(ignoreSet);
  ar & BOOST_SERIALIZATION_NVP(isIgnored);
}

} // namespace regression
} // namespace mlpack

#endif
