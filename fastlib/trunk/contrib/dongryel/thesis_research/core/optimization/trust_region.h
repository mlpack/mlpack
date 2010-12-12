/** @file trust_region.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_OPTIMIZATION_TRUST_REGION_H
#define CORE_OPTIMIZATION_TRUST_REGION_H

#include <iostream>
#include <algorithm>
#include "core/table/dense_point.h"
#include "core/table/dense_matrix.h"

namespace core {
namespace optimization {
class TrustRegion {
  private:
    double max_radius_;

  private:

    void ComputeCauchyPoint_();

    void ComputeDoglegDirection_(
      double radius,
      core::table::DensePoint &gradient,
      core::table::DenseMatrix &hessian,
      core::table::DensePoint *p,
      double *delta_m);

    void ComputeSteihaugDirection_(
      double radius,
      core::table::DensePoint &gradient,
      core::table::DenseMatrix &hessian,
      core::table::DensePoint *p,
      double *delta_m);

    void TrustRadiusUpdate_(
      double rho, double p_norm, double *current_radius) {

      if(rho < 0.25) {
        std::cout << "Shrinking trust region radius..." << endl;
        (*current_radius) = p_norm / 4.0;
      }
      else if((rho > 0.75) && (p_norm > (0.99 *(*current_radius)))) {
        std::cout << "Expanding trust region radius..." << endl;
        (*current_radius) = std::min(2.0 * (*current_radius), max_radius_);
      }
    }

  public:

    double get_max_radius() const {
      return max_radius_;
    }

    void set_max_radius(double max_radius_in) {
      max_radius_ = max_radius_in;
    }

    TrustRegion() {
      max_radius_ = 10.0;
    }

    void Init() {
    }

    void Optimize() {

    }
};
};
};

#endif
