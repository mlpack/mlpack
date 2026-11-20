/**
 * @file core/transforms/spline_envelope.hpp
 * @author Mohammad Mundiwala
 *
 * Implementation of a spline envelope builder for EMD
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_TRANSFORMS_SPLINE_ENVELOPE_HPP
#define MLPACK_CORE_TRANSFORMS_SPLINE_ENVELOPE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

template<typename ColType, typename UColType>
inline void BuildSplineEnvelope(const ColType& h,
                                const UColType& idx,
                                ColType& env)
{
  // Build a natural cubic spline through extrema (idx) and evaluate it on the
  // integer grid of the signal to obtain the envelope. This follows the
  // standard tridiagonal solve for natural splines (zero second-derivative at
  // endpoints), as in classical EMD
  // see ref: Burden & Faires, Numerical Analysis, Algorithm 3.4. pp 149
  using eT = typename ColType::elem_type;
  const size_t N = h.n_elem;
  env.set_size(N);

  const size_t m = idx.n_elem;
  if (m < 2)
  {
    env = h;
    return;
  }

  // x: knot positions (indices) and y: knot values.
  ColType y = h.elem(idx);
  ColType c(m);
  if (m > 2)
  {
    // Step 1: assemble h_i segment lengths and rhs alpha.
    ColType hSeg = conv_to<ColType>::from(
      idx.subvec(1, m - 1) - idx.subvec(0, m - 2));
    ColType alpha(m - 1);
    alpha.subvec(1, m - 2) = 3 * (
      (y.subvec(2, m - 1) - y.subvec(1, m - 2)) / hSeg.subvec(1, m - 2) -
      (y.subvec(1, m - 2) - y.subvec(0, m - 3)) / hSeg.subvec(0, m - 3));

    // Step 2: solve tridiagonal system for spline second deriv (c).
    ColType l(m), mu(m), z(m);
    l[0]  = eT(1);
    mu[0] = eT(0);
    z[0]  = eT(0);

    for (size_t i = 1; i < m - 1; ++i)
    {
      l[i]  = eT(2) * (idx[i + 1] - idx[i - 1]) - hSeg[i - 1] * mu[i - 1];
      mu[i] = hSeg[i] / l[i];
      z[i]  = (alpha[i] - hSeg[i - 1] * z[i - 1]) / l[i];
    }

    l[m - 1] = eT(1);
    z[m - 1] = eT(0);
    c[m - 1] = eT(0);

    for (arma::sword j = static_cast<arma::sword>(m) - 2; j >= 0; --j)
    {
      c[j] = z[j] - mu[j] * c[j + 1];
    }
  }
  // Step 3: evaluate spline on each segment and fill envelope values
  for (size_t seg = 0; seg < m - 1; ++seg)
  {
    const size_t i0 = idx[seg];
    const size_t i1 = idx[seg + 1];

    const eT x0 = eT(idx[seg]);
    const eT x1 = eT(idx[seg + 1]);
    const eT hSegLen = x1 - x0;

    if (hSegLen == 0)
      continue;

    const ColType xv = linspace<ColType>(i0, i1, i1 - i0 + 1);

    const ColType A = (x1 - xv) / hSegLen;
    const ColType B = (xv - x0) / hSegLen;

    const ColType Ai3 = A % A % A;
    const ColType Bi3 = B % B % B;

    env.subvec(i0, i1) =
        A * y[seg] +
        B * y[seg + 1] +
        ((Ai3 - A) * c[seg] + (Bi3 - B) * c[seg + 1]) *
          (hSegLen * hSegLen) / eT(6);
    }
  }

} // namespace mlpack

#endif
