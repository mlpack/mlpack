/***
 * @file mahalanobis_distance.cc
 * @author Ryan Curtin
 *
 * Implementation of the Mahalanobis distance.
 */
#ifndef __MLPACK_CORE_METRICS_MAHALANOBIS_DISTANCE_IMPL_HPP
#define __MLPACK_CORE_METRICS_MAHALANOBIS_DISTANCE_IMPL_HPP

#include "mahalanobis_distance.hpp"

namespace mlpack {
namespace metric {

/**
 * Specialization for non-rooted case.
 */
template<>
template<typename VecType1, typename VecType2>
double MahalanobisDistance<false>::Evaluate(const VecType1& a,
                                            const VecType2& b)
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
template<typename VecType1, typename VecType2>
double MahalanobisDistance<true>::Evaluate(const VecType1& a,
                                           const VecType2& b)
{
  // Check if covariance matrix has been initialized.
  if (covariance.n_rows == 0)
    covariance = arma::eye<arma::mat>(a.n_elem, a.n_elem);

  arma::vec m = (a - b);
  arma::mat out = trans(m) * covariance * m; // 1x1;
  return sqrt(out[0]);
}

// Convert object into string.
template<bool TakeRoot>
std::string MahalanobisDistance<TakeRoot>::ToString() const
{
  std::ostringstream convert;
  std::ostringstream convertb;
  convert << "MahalanobisDistance [" << this << "]" << std::endl;
  if (TakeRoot)
    convert << "  TakeRoot: TRUE" << std::endl;
  if (covariance.size() < 65)
  {
    convert << "  Covariance: " << std::endl;
    convertb << covariance << std::endl;
    convert << mlpack::util::Indent(convertb.str(),2);
  }
  else
  {
    convert << "  Covariance matrix: " << covariance.n_rows << "x" ;
    convert << covariance.n_cols << std::endl << " Range: [" ;
    convert << covariance.min() << "," << covariance.max() << "]" << std::endl;
  }
  return convert.str();
}

}; // namespace metric
}; // namespace mlpack

#endif
