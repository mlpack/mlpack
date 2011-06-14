/***
 * @file l2metric.h
 * @author Ryan Curtin
 *
 * Simple L2 squared metric.  Returns the squared Euclidean distance.
 */
#ifndef __MLPACK_KERNELS_L2_SQUARED_METRIC_H
#define __MLPACK_KERNELS_L2_SQUARED_METRIC_H

#include <armadillo>

namespace mlpack {
namespace kernel {

class L2SquaredMetric {
 public:
  /**
   * Returns the squared Euclidean distance between point_a and point_b.
   */
  static double Evaluate(arma::vec& point_a, arma::vec& point_b);
};

}; // namespace kernel
}; // namespace mlpack

#endif
