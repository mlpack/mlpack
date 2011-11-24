/**
 * @file cosine_distance.cpp
 * @author Ryan Curtin
 *
 * This implements the cosine distance.
 */
#include "cosine_distance.hpp"

using namespace mlpack;
using namespace mlpack::kernel;
using namespace arma;

double CosineDistance::Evaluate(const arma::vec& a, const arma::vec& b)
{
  // Since we are using the L2 inner product, this is easy.
  return 1 - dot(a, b) / (norm(a, 2) * norm(b, 2));
}
