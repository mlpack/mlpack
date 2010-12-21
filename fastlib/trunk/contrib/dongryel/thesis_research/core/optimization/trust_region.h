/** @file trust_region.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_OPTIMIZATION_TRUST_REGION_H
#define CORE_OPTIMIZATION_TRUST_REGION_H

#include <iostream>
#include <algorithm>
#include <armadillo>

namespace core {
namespace optimization {
class TrustRegion {
  private:
    double max_radius_;

  private:

    void ComputeCauchyPoint_(
      double radius, const arma::vec &gradient,
      const arma::mat &hessian, arma::vec *p);

    void ComputeDoglegDirection_(
      double radius, const arma::vec &gradient, const arma::mat &hessian,
      arma::vec *p, double *delta_m);

    void ComputeSteihaugDirection_(
      double radius, const arma::vec &gradient, const arma::mat &hessian,
      arma::vec *p, double *delta_m);

    void TrustRadiusUpdate_(
      double rho, double p_norm, double *current_radius);

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

    void Optimize(int num_iterations, arma::vec *iterate);
};
};
};

#endif
