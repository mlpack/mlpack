/**
 * @file core/distances/lmetric_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of template specializations of LMetric class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTANCES_LMETRIC_IMPL_HPP
#define MLPACK_CORE_DISTANCES_LMETRIC_IMPL_HPP

// In case it hasn't been included.
#include "lmetric.hpp"

namespace mlpack {

// Unspecialized implementation.  This should almost never be used...
template<int Power, bool TakeRoot>
template<typename VecTypeA, typename VecTypeB>
typename VecTypeA::elem_type LMetric<Power, TakeRoot>::Evaluate(
    const VecTypeA& a,
    const VecTypeB& b)
{
  typename VecTypeA::elem_type sum = 0;
  for (size_t i = 0; i < a.n_elem; ++i)
    sum += std::pow(fabs(a[i] - b[i]), Power);

  if (!TakeRoot) // The compiler should optimize this correctly at compile-time.
    return sum;

  return std::pow(sum, (1.0 / Power));
}

// L1-metric specializations; the root doesn't matter.
template<>
template<typename VecTypeA, typename VecTypeB>
typename VecTypeA::elem_type LMetric<1, true>::Evaluate(
    const VecTypeA& a,
    const VecTypeB& b)
{
  return accu(abs(a - b));
}

template<>
template<typename VecTypeA, typename VecTypeB>
typename VecTypeA::elem_type LMetric<1, false>::Evaluate(
    const VecTypeA& a,
    const VecTypeB& b)
{
  return accu(abs(a - b));
}

// L2-metric specializations.
template<>
template<typename VecTypeA, typename VecTypeB>
typename VecTypeA::elem_type LMetric<2, true>::Evaluate(
    const VecTypeA& a,
    const VecTypeB& b)
{
  return arma::norm(a - b, 2);
}

template<>
template<typename VecTypeA, typename VecTypeB>
typename VecTypeA::elem_type LMetric<2, false>::Evaluate(
    const VecTypeA& a,
    const VecTypeB& b)
{
  return accu(arma::square(a - b));
}

// L3-metric specialization (not very likely to be used, but just in case).
template<>
template<typename VecTypeA, typename VecTypeB>
typename VecTypeA::elem_type LMetric<3, true>::Evaluate(
    const VecTypeA& a,
    const VecTypeB& b)
{
  typename VecTypeA::elem_type sum = 0;
  for (size_t i = 0; i < a.n_elem; ++i)
    sum += std::pow(fabs(a[i] - b[i]), 3.0);

  return std::pow(accu(pow(arma::abs(a - b), 3.0)), 1.0 / 3.0);
}

template<>
template<typename VecTypeA, typename VecTypeB>
typename VecTypeA::elem_type LMetric<3, false>::Evaluate(
    const VecTypeA& a,
    const VecTypeB& b)
{
  return accu(pow(arma::abs(a - b), 3.0));
}

// L-infinity (Chebyshev distance) specialization
template<>
template<typename VecTypeA, typename VecTypeB>
typename VecTypeA::elem_type LMetric<INT_MAX, false>::Evaluate(
    const VecTypeA& a,
    const VecTypeB& b)
{
  return arma::as_scalar(arma::max(arma::abs(a - b)));
}

} // namespace mlpack

#endif
