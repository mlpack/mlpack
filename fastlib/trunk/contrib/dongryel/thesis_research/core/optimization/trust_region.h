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
class TrustRegionSearchMethod {
  public:
    enum SearchType {CAUCHY, DOGLEG, STEIHAUG};
};

class TrustRegionUtil {
  public:

    static void ObtainStepDirection(
      core::optimization::TrustRegionSearchMethod::SearchType search_method_in,
      double radius, const arma::vec &gradient,
      const arma::mat &hessian, arma::vec *step_direction,
      double *step_direction_norm);

    static void ComputeCauchyPoint(
      double radius, const arma::vec &gradient,
      const arma::mat &hessian, arma::vec *p);

    static void ComputeDoglegDirection(
      double radius, const arma::vec &gradient, const arma::mat &hessian,
      arma::vec *p);

    static void ComputeSteihaugDirection(
      double radius, const arma::vec &gradient, const arma::mat &hessian,
      arma::vec *p);

    static void TrustRadiusUpdate(
      double rho, double p_norm, double max_radius, double *current_radius);

    static double ReductionRatio(
      const arma::vec &step,
      double iterate_function_value, double next_iterate_function_value,
      const arma::vec &gradient, const arma::mat &hessian);
};

template<typename FunctionType>
class TrustRegion {
  private:
    double max_radius_;

    FunctionType *function_;

    core::optimization::TrustRegionSearchMethod::SearchType search_method_;

  private:

    bool GradientNormTooSmall_(const arma::vec &gradient) const;

    double Evaluate_(const arma::vec &iterate) const;

  public:

    TrustRegion();

    double get_max_radius() const {
      return max_radius_;
    }

    void set_max_radius(double max_radius_in) {
      max_radius_ = max_radius_in;
    }

    void Init(
      FunctionType &function_in,
      core::optimization::TrustRegionSearchMethod::SearchType search_method_in);

    void Optimize(int num_iterations, arma::vec *iterate);
};
};
};

#endif
