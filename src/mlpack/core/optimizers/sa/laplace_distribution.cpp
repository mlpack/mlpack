/*
 * @file laplace_distribution.cpp
 * @author Zhihao Lou
 *
 * Implementation of Laplace distribution
 */

#include <mlpack/core.hpp>
#include "laplace_distribution.hpp"
using namespace mlpack;
using namespace mlpack::optimization;
double LaplaceDistribution::operator () (const double param)
{
  // uniform [-1, 1]
  double unif = 2.0 * math::Random() - 1.0;
  // Laplace Distribution with mean 0
  // x = - param * sign(unif) * log(1 - |unif|)
  if (unif < 0) // why oh why we don't have a sign function in c++?
      return (param * std::log(1 + unif));
  else
      return (-1.0 * param * std::log(1 - unif));
}
