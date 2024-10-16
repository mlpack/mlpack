/**
 * @file methods/rann/ra_util_impl.hpp
 * @author Parikshit Ram
 * @author Ryan Curtin
 *
 * Utilities for rank-approximate neighbor search.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_RANN_RA_UTIL_IMPL_HPP
#define MLPACK_METHODS_RANN_RA_UTIL_IMPL_HPP

#include "ra_util.hpp"

#include <mlpack/core/util/log.hpp>

namespace mlpack {

inline size_t RAUtil::MinimumSamplesReqd(const size_t n,
                                         const size_t k,
                                         const double tau,
                                         const double alpha)
{
  size_t ub = n; // The upper bound on the binary search.
  size_t lb = k; // The lower bound on the binary search.
  size_t  m = lb; // The minimum number of random samples.

  // The rank-approximation.
  const size_t t = (size_t) std::ceil(tau * (double) n / 100.0);

  double prob;
  Log::Assert(alpha <= 1.0);

  // This performs a binary search on the integer values between 'lb = k'
  // and 'ub = n' to find the minimum number of samples 'm' required to obtain
  // the desired success probability 'alpha'.
  do
  {
    prob = SuccessProbability(n, k, m, t);

    if (prob > alpha)
    {
      if (prob - alpha < 0.001 || ub < lb + 2)
      {
        break;
      }
      else
        ub = m;
    }
    else
    {
      if (prob < alpha)
      {
        if (m == lb)
        {
          m++;
          continue;
        }
        else
          lb = m;
      }
      else
      {
        break;
      }
    }
    m = (ub + lb) / 2;
  } while (true);

  return (std::min(m + 1, n));
}

inline double RAUtil::SuccessProbability(const size_t n,
                                         const size_t k,
                                         const size_t m,
                                         const size_t t)
{
  if (k == 1)
  {
    if (m > n - t)
      return 1.0;

    double eps = (double) t / (double) n;

    return 1.0 - std::pow(1.0 - eps, (double) m);
  } // Faster implementation for topK = 1.
  else
  {
    if (m < k)
      return 0.0;

    if (m > n - t + k - 1)
      return 1.0;

    double eps = (double) t / (double) n;
    double sum = 0.0;

    // The probability that 'k' of the 'm' samples lie within the top 't'
    // of the neighbors is given by:
    // sum_{j = k}^m Choose(m, j) (t/n)^j (1 - t/n)^{m - j}
    // which is also equal to
    // 1 - sum_{j = 0}^{k - 1} Choose(m, j) (t/n)^j (1 - t/n)^{m - j}
    //
    // So this is a m - k term summation or a k term summation. So if
    // m > 2k, do the k term summation, otherwise do the m term summation.

    size_t lb;
    size_t ub;
    bool topHalf;

    if (2 * k < m)
    {
      // Compute 1 - sum_{j = 0}^{k - 1} Choose(m, j) eps^j (1 - eps)^{m - j}
      // eps = t/n.
      //
      // Choosing 'lb' as 1 and 'ub' as k so as to sum from 1 to (k - 1), and
      // add the term (1 - eps)^m term separately.
      lb = 1;
      ub = k;
      topHalf = true;
      sum = std::pow(1 - eps, (double) m);
    }
    else
    {
      // Compute sum_{j = k}^m Choose(m, j) eps^j (1 - eps)^{m - j}
      // eps = t/n.
      //
      // Choosing 'lb' as k and 'ub' as m so as to sum from k to (m - 1), and
      // add the term eps^m term separately.
      lb = k;
      ub = m;
      topHalf = false;
      sum = std::pow(eps, (double) m);
    }

    for (size_t j = lb; j < ub; ++j)
    {
      // Compute Choose(m, j).
      double mCj = (double) m;
      size_t jTrans;

      // If j < m - j, compute Choose(m, j).
      // If j > m - j, compute Choose(m, m - j).
      if (topHalf)
        jTrans = j;
      else
        jTrans = m - j;

      for (size_t i = 2; i <= jTrans; ++i)
      {
        mCj *= (double) (m - (i - 1));
        mCj /= (double) i;
      }

      sum += (mCj * std::pow(eps, (double) j)
              * std::pow(1.0 - eps, (double) (m - j)));
    }

    if (topHalf)
      sum = 1.0 - sum;

    return sum;
  } // For k > 1.
}

} // namespace mlpack

#endif
