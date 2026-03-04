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

// Generic to support Armadillo compatible column types
template<typename ColType>
inline void BuildSplineEnvelope(const ColType& h,
                                const arma::uvec& idx,
                                ColType& env)
{
  using eT = typename ColType::elem_type;

  arma::Col<eT> envTemp;
  BuildSplineEnvelope(arma::Col<eT>(h), idx, envTemp);
  env = envTemp;
  }

template<typename eT>
inline void BuildSplineEnvelope(const arma::Col<eT>& h,
                                const arma::uvec& idx,
                                arma::Col<eT>& env)
{
  // Build a natural cubic spline through extrema (idx) and evaluate it on the
  // integer grid of the signal to obtain the envelope. This follows the
  // standard tridiagonal solve for natural splines (zero second-derivative at
  // endpoints), as in classical EMD
  // see ref: Burden & Faires, Numerical Analysis, Algorithm 3.4. pp 149
  const size_t N = h.n_elem;
  env.set_size(N);

  const size_t m = idx.n_elem;
  if (m < 2)
  {
    env = h;
    return;
  }

  // x: knot positions (indices) and y: knot values.
  arma::Col<eT> y = h.elem(idx);

  arma::Col<eT> c(m, arma::fill::zeros);

  if (m > 2)
  {
    // Step 1: assemble h_i segment lengths and rhs alpha.
    arma::Col<eT> hSeg = idx.subvec(1, m - 1) - idx.subvec(0, m - 2);
    arma::Col<eT> alpha(m - 1, arma::fill::zeros);
    alpha.subvec(1, m - 2) = eT(3) * (
      (y.subvec(2, m - 1) - y.subvec(1, m - 2)) / hSeg.subvec(1, m - 2) -
      (y.subvec(1, m - 2) - y.subvec(0, m - 3)) / hSeg.subvec(0, m - 3));

    // Step 2: solve tridiagonal system for spline second deriv (c).
    arma::Mat<eT> M(m, m, arma::fill::zeros);
    M(0, 0) = 1; M(m - 1, m - 1) = 1;
    M.diag(0).subvec(1, m - 2) = eT(2) *
      (hSeg.subvec(0, m - 3) + hSeg.subvec(1, m - 2));
    M.diag(-1).subvec(0, m - 3) = hSeg.subvec(0, m - 3);
    M.diag(+1).subvec(1, m - 2) = hSeg.subvec(1, m - 2);

    arma::Col<eT> rhs(m, arma::fill::zeros);
    rhs.subvec(1, m - 2) = alpha.subvec(1, m - 2);

    const bool ok = arma::solve(c, M, rhs);
    if (!ok) c.zeros();  // fallback
  }
  // Step 3: evaluate spline on each segment and fill envelope values
  env.zeros();
  for (size_t seg = 0; seg < m - 1; ++seg)
  {
    const arma::uword i0 = idx[seg];
    const arma::uword i1 = idx[seg + 1];

    const eT x0 = eT(idx[seg]);
    const eT x1 = eT(idx[seg + 1]);
    const eT hSegLen = x1 - x0;

    if (hSegLen == eT(0))
      continue;

    for (arma::uword i = i0; i <= i1; ++i)
    {
      const eT xv = static_cast<eT>(i);
      const eT A  = (x1 - xv) / hSegLen;
      const eT B  = (xv - x0) / hSegLen;

      //natural cubic spline formula:
      // s(x) = A*y_i + B*y_{i+1} +
      //        ((A^3 - A)*c_i + (B^3 - B)*c_{i+1}) * h_i^2 / 6
      const eT Ai3 = A * A * A;
      const eT Bi3 = B * B * B;
      const eT C = ((Ai3 - A) * c[seg] +
                        (Bi3 - B) * c[seg + 1]) * (hSegLen * hSegLen) / eT(6);
      env[i] = A * y[seg] + B * y[seg + 1] + C;
    }
  }
}

} // namespace mlpack

#endif
