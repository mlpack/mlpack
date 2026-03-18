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
    alpha.zeros();
    alpha.subvec(1, m - 2) = 3 * (
      (y.subvec(2, m - 1) - y.subvec(1, m - 2)) / hSeg.subvec(1, m - 2) -
      (y.subvec(1, m - 2) - y.subvec(0, m - 3)) / hSeg.subvec(0, m - 3));

    // Step 2: solve tridiagonal system for spline second deriv (c).
    typename GetDenseMatType<ColType>::type M(m, m);
    M(0, 0) = 1;
    M(m - 1, m - 1) = 1;
    // Use ColType of diagonal for sub vector slicing 
    // M.diag(0).subvec(1, m - 2) 
    ColType d0 = M.diag(0);
    d0.subvec(1, m - 2) = 2 * (hSeg.subvec(0, m - 3) + hSeg.subvec(1, m - 2));
    M.diag(0) = d0;
    // M.diag(-1).subvec(0, m - 3)
    ColType d_m1 = M.diag(-1);
    d_m1.subvec(0, m - 3) = hSeg.subvec(0, m - 3);
    M.diag(-1) = d_m1;
    // M.diag(+1).subvec(1, m - 2)
    ColType d_p1 = M.diag(+1);
    d_p1.subvec(1, m - 2) = hSeg.subvec(1, m - 2);
    M.diag(+1) = d_p1;

    ColType rhs(m);
    rhs.subvec(1, m - 2) = alpha.subvec(1, m - 2);

    const bool ok = solve(c, M, rhs);
    if (!ok) c.zeros();  // fallback
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
