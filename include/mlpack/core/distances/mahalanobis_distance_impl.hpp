/**
 * @file core/distances/mahalanobis_distance_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of the Mahalanobis distance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTANCES_MAHALANOBIS_DISTANCE_IMPL_HPP
#define MLPACK_CORE_DISTANCES_MAHALANOBIS_DISTANCE_IMPL_HPP

#include "mahalanobis_distance.hpp"

namespace mlpack {

/**
 * Specialization for non-rooted case.
 */
template<bool TakeRoot, typename MatType>
template<typename VecTypeA, typename VecTypeB>
double MahalanobisDistance<TakeRoot, MatType>::Evaluate(
    const VecTypeA& a, const VecTypeB& b)
{
  // Check if Q matrix has been initialized.
  if (q.n_rows != a.n_elem)
  {
    std::ostringstream oss;
    oss << "MahalanobisDistance::Evaluate(): given vector dimensionality ("
        << a.n_elem << ") does not match Q dimensionality (" << q.n_rows
        << ")!";
    throw std::runtime_error(oss.str());
  }

  VecType m = (a - b);
  if (TakeRoot == true)
    return std::sqrt(as_scalar(m.t() * q * m));
  else
    return as_scalar(m.t() * q * m); // 1x1
}

// Serialize the Mahalanobis distance.
template<bool TakeRoot, typename MatType>
template<typename Archive>
void MahalanobisDistance<TakeRoot, MatType>::serialize(
    Archive& ar, const uint32_t version)
{
  if (Archive::is_loading::value && version == 0)
  {
    // Older versions of MahalanobisDistance always serialized as an arma::mat
    // named "covariance".
    arma::mat qTmp;
    ar(cereal::make_nvp("covariance", qTmp));
    q = arma::conv_to<MatType>::from(qTmp);
  }
  else
  {
    ar(CEREAL_NVP(q));
  }
}

} // namespace mlpack

#endif
