/***
 * @file mahalanobis_distance.cc
 * @author Ryan Curtin
 *
 * Implementation of the Mahalanobis distance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_MAHALANOBIS_DISTANCE_IMPL_HPP
#define MLPACK_CORE_METRICS_MAHALANOBIS_DISTANCE_IMPL_HPP

#include "mahalanobis_distance.hpp"

namespace mlpack {
namespace metric {

/**
 * Specialization for non-rooted case.
 */
template<>
template<typename VecTypeA, typename VecTypeB>
double MahalanobisDistance<false>::Evaluate(const VecTypeA& a,
                                            const VecTypeB& b)
{
  arma::vec m = (a - b);
  arma::mat out = trans(m) * covariance * m; // 1x1
  return out[0];
}
/**
 * Specialization for rooted case.  This requires one extra evaluation of
 * sqrt().
 */
template<>
template<typename VecTypeA, typename VecTypeB>
double MahalanobisDistance<true>::Evaluate(const VecTypeA& a,
                                           const VecTypeB& b)
{
  // Check if covariance matrix has been initialized.
  if (covariance.n_rows == 0)
    covariance = arma::eye<arma::mat>(a.n_elem, a.n_elem);

  arma::vec m = (a - b);
  arma::mat out = trans(m) * covariance * m; // 1x1;
  return sqrt(out[0]);
}

// Serialize the Mahalanobis distance.
template<bool TakeRoot>
template<typename Archive>
void MahalanobisDistance<TakeRoot>::Serialize(Archive& ar,
                                              const unsigned int /* version */)
{
  ar & data::CreateNVP(covariance, "covariance");
}

} // namespace metric
} // namespace mlpack

#endif
