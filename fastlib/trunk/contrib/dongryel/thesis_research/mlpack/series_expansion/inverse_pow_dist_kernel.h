/** @file inverse_pow_dist_kernel.h
 *
 *  Defines the kernel of inverse distance power of the form $1 /
 *  r^{\alpha}$.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_INVERSE_POW_DIST_KERNEL_H
#define MLPACK_SERIES_EXPANSION_INVERSE_POW_DIST_KERNEL_H

#include <armadillo>
#include "core/math/range.h"
#include "core/table/dense_point.h"

namespace mlpack {
namespace series_expansion {

class InversePowDistKernel {

  private:
    double lambda_;

    int dimension_;

  public:

    double lambda() const {
      return lambda_;
    }

    void Init(double lambda_in, int dimension_in) {
      lambda_ = lambda_in;
      dimension_ = dimension_in;
    }

    double EvalUnnorm(const core::table::DensePoint &point) const {
      arma::vec point_alias;
      core::table::DensePointToArmaVec(point, &point_alias);
      double sqdist = arma::dot(point_alias, point_alias);
      if(lambda_ > 0) {
        return 1.0 / pow(sqdist, lambda_ / 2.0);
      }
      else {
        return pow(sqdist, -lambda_ / 2.0);
      }
    }

    double EvalUnnorm(double dist) const {
      return EvalUnnormOnSq(dist * dist);
    }

    double EvalUnnormOnSq(double sqdist) const {
      if(lambda_ > 0) {
        return 1.0 / pow(sqdist, lambda_ / 2.0);
      }
      else {
        return pow(sqdist, -lambda_ / 2.0);
      }
    }

    static inline double EvalUnnormOnSq(double lambda_in, double sqdist) {
      if(lambda_in > 0) {
        return 1.0 / pow(sqdist, lambda_in / 2.0);
      }
      else {
        return pow(sqdist, -lambda_in / 2.0);
      }
    }

    core::math::Range RangeUnnormOnSq(const core::math::Range &range) const {
      return core::math::Range(
               EvalUnnormOnSq(range.hi), EvalUnnormOnSq(range.lo));
    }
};
}
}

#endif
