/***
 * @file cosine_distance.cc
 * @author Ryan Curtin
 *
 * This implements the cosine distance.
 */
#include "cosine_distance.h"

using namespace mlpack;
using namespace mlpack::kernel;
using namespace arma;

double CosineDistance::Evaluate(const arma::vec& a, const arma::vec& b) {
  // Since we are using the L2 inner product, this is easy.
  return dot(a, b) / (norm(a, 2) * norm(b, 2));
}
