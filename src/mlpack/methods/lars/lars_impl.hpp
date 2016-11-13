/**
 * @file lars_impl.hpp
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
void LARS::Serialize(Archive& ar, const unsigned int /* version */)
{
  using data::CreateNVP;

  // If we're loading, we have to use the internal storage.
  if (Archive::is_loading::value)
  {
    matGram = &matGramInternal;
    ar & CreateNVP(matGramInternal, "matGramInternal");
  }
  else
  {
    ar & CreateNVP(const_cast<arma::mat&>(*matGram), "matGramInternal");
  }

  ar & CreateNVP(matUtriCholFactor, "matUtriCholFactor");
  ar & CreateNVP(useCholesky, "useCholesky");
  ar & CreateNVP(lasso, "lasso");
  ar & CreateNVP(lambda1, "lambda1");
  ar & CreateNVP(elasticNet, "elasticNet");
  ar & CreateNVP(lambda2, "lambda2");
  ar & CreateNVP(tolerance, "tolerance");
  ar & CreateNVP(betaPath, "betaPath");
  ar & CreateNVP(lambdaPath, "lambdaPath");
  ar & CreateNVP(activeSet, "activeSet");
  ar & CreateNVP(isActive, "isActive");
  ar & CreateNVP(ignoreSet, "ignoreSet");
  ar & CreateNVP(isIgnored, "isIgnored");
}

} // namespace regression
} // namespace mlpack

#endif
