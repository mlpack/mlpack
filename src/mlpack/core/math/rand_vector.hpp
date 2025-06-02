/**
 * @file core/math/rand_vector.hpp
 * @author Nishant Mehta
 *
 * Utility to generate a random vector on the unit sphere.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_MATH_RAND_VECTOR_HPP
#define MLPACK_CORE_MATH_RAND_VECTOR_HPP

namespace mlpack {

/**
 * Overwrites a dimension-N vector to a random vector on the unit sphere in R^N.
 */
template<typename eT>
inline void RandVector(arma::Col<eT>& v)
{
  for (size_t i = 0; i + 1 < v.n_elem; i += 2)
  {
    double a = Random();
    double b = Random();
    double first_term = std::sqrt(-2 * std::log(a));
    double second_term = 2 * M_PI * b;
    v[i]     = first_term * cos(second_term);
    v[i + 1] = first_term * sin(second_term);
  }

  if ((v.n_elem % 2) == 1)
  {
    v[v.n_elem - 1] = std::sqrt(-2 * std::log(Random())) *
        cos(2 * M_PI * Random());
  }

  v /= std::sqrt(dot(v, v));
}

} // namespace mlpack

#endif
