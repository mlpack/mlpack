/***
 * @file l2metric.cc
 * @author Ryan Curtin
 *
 * Simple L2 squared metric.
 */

#include "l2_squared_metric.h"

#include <fastlib/fastlib.h>
#include <fastlib/fx/io.h>

using namespace mlpack;
using namespace mlpack::kernel;

double L2SquaredMetric::Evaluate(arma::vec& point_a, arma::vec& point_b) {
  return dot((point_a - point_b), (point_a - point_b));
}
