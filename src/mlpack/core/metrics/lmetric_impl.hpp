/**
 * @file lmetric_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of template specializations of LMetric class.
 */
#ifndef __MLPACK_CORE_METRICS_LMETRIC_IMPL_HPP
#define __MLPACK_CORE_METRICS_LMETRIC_IMPL_HPP

// In case it hasn't been included.
#include "lmetric.hpp"

namespace mlpack {
namespace metric {

// Unspecialized implementation.  This should almost never be used...
template<int t_pow, bool t_take_root>
template<typename VecType>
double LMetric<t_pow, t_take_root>::Evaluate(const VecType& a,
                                             const VecType& b)
{
  double sum = 0;
  for (size_t i = 0; i < a.n_elem; i++)
    sum += pow(fabs(a[i] - b[i]), t_pow);

  if (!t_take_root) // Suboptimal to have this here.
    return sum;

  return pow(sum, (1.0 / t_pow));
}

// L1-metric specializations; the root doesn't matter.
template<>
template<typename VecType>
double LMetric<1, true>::Evaluate(const VecType& a, const VecType& b)
{
  return accu(abs(a - b));
}

template<>
template<typename VecType>
double LMetric<1, false>::Evaluate(const VecType& a, const VecType& b)
{
  return accu(abs(a - b));
}

// L2-metric specializations.
template<>
template<typename VecType>
double LMetric<2, true>::Evaluate(const VecType& a, const VecType& b)
{
  return sqrt(accu(square(a - b)));
}

template<>
template<typename VecType>
double LMetric<2, false>::Evaluate(const VecType& a, const VecType& b)
{
  return accu(square(a - b));
}

// L3-metric specialization (not very likely to be used, but just in case).
template<>
template<typename VecType>
double LMetric<3, true>::Evaluate(const VecType& a, const VecType& b)
{
  double sum = 0;
  for (size_t i = 0; i < a.n_elem; i++)
    sum += pow(fabs(a[i] - b[i]), 3.0);

  return pow(accu(pow(abs(a - b), 3.0)), 1.0 / 3.0);
}

template<>
template<typename VecType>
double LMetric<3, false>::Evaluate(const VecType& a, const VecType& b)
{
  return accu(pow(abs(a - b), 3.0));
}

}; // namespace metric
}; // namespace mlpack

#endif
