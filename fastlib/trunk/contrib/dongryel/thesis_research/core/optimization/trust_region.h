/** @file trust_region.h
 *
 *  An implementation of trust region optimizer.
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

/** @brief The set of different options for doing the search within
 *         the trust region.
 */
class TrustRegionSearchMethod {
  public:
    enum SearchType {CAUCHY, DOGLEG, STEIHAUG};
};

/** @brief The set of utilities for performing a trust region
 *         optimization.
 */
class TrustRegionUtil {
  public:

    /** @brief Obtains the step direction using the given search
     *         method (one of Cauchy Point, Dogleg, or Steihaug
     *         methods).
     */
    static void ObtainStepDirection(
      core::optimization::TrustRegionSearchMethod::SearchType search_method_in,
      double radius, const arma::vec &gradient,
      const arma::mat &hessian, arma::vec *step_direction,
      double *step_direction_norm);

    /** @brief Computes the Cauchy Point iterate.
     */
    static void ComputeCauchyPoint(
      double radius, const arma::vec &gradient,
      const arma::mat &hessian, arma::vec *p);

    /** @brief Computes the step direction using the Dogleg method.
     */
    static void ComputeDoglegDirection(
      double radius, const arma::vec &gradient, const arma::mat &hessian,
      arma::vec *p);

    /** @brief Computes the step direction using the Steihaug method.
     */
    static void ComputeSteihaugDirection(
      double radius, const arma::vec &gradient, const arma::mat &hessian,
      arma::vec *p);

    /** @brief Given the reduction ratio $\rho$ and the norm of the
     *         step p_norm, updates the trust region radius.
     */
    static void TrustRadiusUpdate(
      double rho, double p_norm, double max_radius, double *current_radius);

    /** @brief Computes the reduction ratio rho (Equation 4.4).
     */
    static double ReductionRatio(
      const arma::vec &step,
      double iterate_function_value, double next_iterate_function_value,
      const arma::vec &gradient, const arma::mat &hessian);
};

template<typename FunctionType>
class TrustRegion {
  private:

    /** @brief The maximum radius of the trust region search.
     */
    double max_radius_;

    /** @brief The function to optimize: provides the gradient and the
     *         Hessian.
     */
    FunctionType *function_;

    /** @brief The search method to use.
     */
    core::optimization::TrustRegionSearchMethod::SearchType search_method_;

  private:

    /** @brief Determines whether the gradient norm is too small.
     */
    bool GradientNormTooSmall_(const arma::vec &gradient) const;

    /** @brief Evaluate the function.
     */
    double Evaluate_(const arma::vec &iterate) const;

  public:

    /** @brief The default constructor.
     */
    TrustRegion();

    /** @brief Gets the maximum radius.
     */
    double get_max_radius() const {
      return max_radius_;
    }

    /** @brief Sets the maximum radius.
     */
    void set_max_radius(double max_radius_in) {
      max_radius_ = max_radius_in;
    }

    /** @brief Initializes the trust region optimizer.
     */
    void Init(
      FunctionType &function_in,
      core::optimization::TrustRegionSearchMethod::SearchType search_method_in);

    /** @brief Optimize for a fixed number of iterations.
     */
    void Optimize(int num_iterations, arma::vec *iterate);
};
}
}

#endif
