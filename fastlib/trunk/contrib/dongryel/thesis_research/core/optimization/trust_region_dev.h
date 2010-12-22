/** @file trust_region_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_OPTIMIZATION_TRUST_REGION_DEV_H
#define CORE_OPTIMIZATION_TRUST_REGION_DEV_H

#include "core/optimization/trust_region.h"
#include "core/math/linear_algebra.h"

namespace core {
namespace optimization {

template<typename FunctionType>
TrustRegion<FunctionType>::TrustRegion() {
  max_radius_ = 10.0;
  function_ = NULL;
  search_method_ = CAUCHY;
}

template<typename FunctionType>
double TrustRegion<FunctionType>::ReductionRatio_(const arma::vec &iterate, const arma::vec &step,) {

}

template<typename FunctionType>
void TrustRegion<FunctionType>::ComputeSteihaugDirection_(
  double radius, const arma::vec &gradient, const arma::mat &hessian,
  arma::vec *p, double *delta_m) {

  // "Numerical optimization" p.171 CG-Steihaug

  // The maximum number of Steihaug iterations.
  const int max_num_steihaug_iterations = 150;

  // z_0 = 0
  arma::vec z(gradient.n_elem);
  z.fill(0);

  // r_0 = gradient
  arma::vec r = gradient;

  // d_0 = -r_0
  arma::vec d = -1.0 * r;

  // The epsilon tolerance (from Algorithm 7.1 on page 169)
  double r0_norm = arma::norm(r, 2);
  double epsilon = std::min(0.5, sqrt(r0_norm)) * r0_norm;

  // If the norm of the initial residual is too small, then return the
  // zero direction.
  if(r0_norm < epsilon) {
    p->set_size(gradient.n_elem);
    p->fill(0);
    *delta_m = 0;
    return;
  }

  // The temporary variables for storing the computed vectors for next
  // iterations.
  arma::vec z_next(gradient.n_elem);
  arma::vec d_next(gradient.n_elem);

  for(int i = 0; i < max_num_steihaug_iterations; i++) {

    // Compute d_j^T B_k d_j.
    double quadratic_form = arma::as_scalar(arma::trans(d) * hessian * d);

    // If the current search direction is a direction of non-positive
    // curvature along the currently approximated Hessian,
    if(quadratic_form <= 0) {

      // Solve ||z_j + \tau d_j|| = radius for \tau.
      double a = arma::dot(d, d);
      double b = 2 * arma::dot(z, d);
      double c = arma::dot(z, z) - core::math::Sqr(radius);
      double sqrt_discriminant = sqrt(b * b - 4 * a * c);
      double tau = (- b + sqrt_discriminant) / (2 * a);

      // p = z + \tau * d
      (*p) = z + tau * d;
      break;
    }

    // alpha_j
    double alpha = arma::dot(r, r) / quadratic_form;

    // z_{j + 1} = z_j + \alpha_j d_j
    z_next = z + alpha * d;

    // If the z_{j + 1} violates the trust region bound,
    if(arma::norm(z_next, 2) >= radius) {

      // Solve ||z_j + \tau d_j|| = radius for \tau.
      double a = arma::dot(d, d);
      double b = 2 * arma::dot(z, d);
      double c = arma::dot(z, z) - core::math::Sqr(radius);
      double sqrt_discriminant = sqrt(b * b - 4 * a * c);
      double tau = (- b + sqrt_discriminant) / (2 * a);

      // p = z + \tau * d
      (*p) = z + tau * d;
      break;
    }

    // r_{j + 1} = r_j + \alpha_j B_k d_j
    arma::vec r_next = r + alpha * hessian * d;

    if(arma::norm(r_next, 2) < epsilon) {
      (*p) = z_next;
      break;
    }

    // beta_{j + 1} = r_{j + 1}^T r_{j + 1} / r_j^T r_j
    double beta_next = arma::dot(r_next, r_next) / arma::dot(r, r);

    // d_{j + 1} = -r_{j + 1} + \beta_{j + 1} d_j
    d_next = beta_next * d - r_next;

    // Set the variables for the next iteration.
    core::math::CopyValues(r_next, &r);
    core::math::CopyValues(d_next, &d);
    core::math::CopyValues(z_next, &z);

  } // end of the main loop.

  // delta_m calculation -g'p-0.5*p'Hp
  double quadratic_form2 = arma::as_scalar(arma::trans(*p) * hessian * (*p));
  (*delta_m) = - arma::dot(gradient, *p) - 0.5 * quadratic_form2;
}

template<typename FunctionType>
void TrustRegion<FunctionType>::ComputeCauchyPoint_(
  double radius, const arma::vec &gradient,
  const arma::mat &hessian, arma::vec *p) {

  // Nocedal, J. and Wright, S. Numerical Optimization. page 72.

  // Check g_k^T B_k g_k <= 0.
  double quadratic_form =
    arma::as_scalar(arma::trans(gradient) * hessian * gradient);

  // Scaling value.
  double tau = 1.0;

  // The gradient norm.
  double gradient_norm = arma::norm(gradient, 2);

  if(quadratic_form > 0) {

    // Equation 4.12 on Page 72.
    tau = std::min(
            1.0, core::math::Pow<3, 1>(gradient_norm) / (
              radius * quadratic_form));
  }
  (*p) = (- tau * radius / gradient_norm)  *  gradient;
}

template<typename FunctionType>
void TrustRegion<FunctionType>::ComputeDoglegDirection_(
  double radius, const arma::vec &gradient, const arma::mat &hessian,
  arma::vec *p, double *delta_m) {

  // Eigendecompose the Hessian to find the eigenvalues. Here is a
  // point where the Hessian might want to exploit special block
  // structures to avoid dense matrices.
  arma::vec eigenvalues;
  arma::mat eigenvectors;
  arma::eig_sym(eigenvalues, eigenvectors, hessian);
  bool hessian_is_positive_definite = true;
  for(int i = 0; hessian_is_positive_definite && i < eigenvalues.n_elem; i++) {
    if(eigenvalues[i] < std::numeric_limits<double>::epsilon()) {
      hessian_is_positive_definite = false;
    }
  }

  // If the Hessian matrix is not positive-definite (up to small
  // epsilon tolerance), we have to use the Cauchy Point method.
  if(! hessian_is_positive_definite) {

    std::cerr << "The Hessian matrix is not positive-definite, so we "
              "have to use the Cauchy Point method..." << endl;

    ComputeCauchyPoint_(radius, gradient, hessian, p);
  } // end of the Cauchy Point case.
  else {

    // Solve for the full step, p_k^B = -B_k^{-1} g_k
    arma::vec p_b;
    arma::solve(p_b, hessian, - gradient);

    // The norm of the full step.
    double p_b_norm = arma::norm(p_b, 2);

    // If the full step is a feasible point, then it is obviously a
    // solution, so we are done.
    if(radius >= p_b_norm) {
      (*p) = p_b;
    }

    // If the full step is not a feasible point,
    else {

      // Compute g_k^T B_k g_k.
      double quadratic_form =
        arma::as_scalar(arma::trans(gradient) * hessian * gradient);

      // p_u= -(g'g/ g'B_k g)*g
      arma::vec p_u = - (
                        arma::dot(gradient, gradient) / quadratic_form) *
                      gradient;
      double p_u_norm = arma::norm(p_u, 2);

      // If the norm of p_u is beyond the trust radius, then the
      // solution lies on the boundary along p_u.
      if(p_u_norm >= radius) {
        (*p) = (radius / p_u_norm) * p_u;
      }

      // Otherwise the quadratic equation composed of the gradient and
      // the full step paths (a x^2 + b x + c = 0)
      else {

        // The equation we want to solve is (in terms of \tau)
        //
        // (\tau - 1)^2 || p_b - p_u ||^2 + 2 (\tau - 1) p_u^T (p_b - p_u)
        // + || p_u ||^2 - radius^2 = 0
        //
        // Let \alpha = \tau - 1, and solve for \alpha then add back in
        // 1 to get what you want in terms of \tau. It is probably true that
        // ( -b + sqrt_discriminant ) / (2 * a) is the only root that
        // falls between [0, 1] for \alpha.
        //
        arma::vec diff = p_b - p_u;
        double a = arma::dot(diff, diff);
        double b = 2.0 * arma::dot(p_u, diff);
        double c = arma::dot(p_u, p_u) - core::math::Sqr(radius);
        double sqrt_discriminant = sqrt(b * b - 4 * a * c);
        double alpha = (-b + sqrt_discriminant) / (2 * a);
        double tau = alpha + 1;

        // Compute the direction. (The second case of Equation 4.16).
        (*p) = p_u + (tau - 1) * diff;

      } // end of the gradient direction being inside the trust radius.

    } // end of the full step not being feasible case.

  } // end of the positive definite Hessian case.

  // delta_m calculation -g'p-0.5*p'Hp
  double quadratic_form2 = arma::as_scalar(arma::trans(*p) * hessian * (*p));
  (*delta_m) = - arma::dot(gradient, *p) - 0.5 * quadratic_form2;
}

template<typename FunctionType>
void TrustRegion<FunctionType>::TrustRadiusUpdate_(
  double rho, double p_norm, double *current_radius) {

  if(rho < 0.25) {
    std::cerr << "Shrinking trust region radius..." << endl;
    (*current_radius) = p_norm / 4.0;
  }
  else if((rho > 0.75) && (p_norm > (0.99 *(*current_radius)))) {
    std::cerr << "Expanding trust region radius..." << endl;
    (*current_radius) = std::min(2.0 * (*current_radius), max_radius_);
  }
}

template<typename FunctionType>
void TrustRegion<FunctionType>::ObtainStepDirection_(
  arma::vec *step_direction, double *step_direction_norm) {

  switch(search_method_) {

  }
}

template<typename FunctionType>
void TrustRegion<FunctionType>::Init(
  FunctionType &function_in, TrustRegionSearchMethod search_method_in) {
  function_ = &function_in;
  search_method_ = search_method_in;
}

template<typename FunctionType>
void TrustRegion<FunctionType>::Optimize(
  int num_iterations, arma::vec *iterate) {

  // eta: The threshold for determining whether to take the step or
  // not.
  const double eta = 0.05;

  // Initialize the starting iterate.
  iterate->set_size(function_->num_dimensions());
  iterate->fill(0.0);

  // Whether to optimize until convergence.
  bool optimize_until_convergence = (num_iterations <= 0);

  double rho = 0;
  double p_norm = 0;
  double current_radius = 0.1;
  int it_num;

  // Step direction.
  arma::vec p;

  // The main optimization loop.
  for(
    it_num = 0; optimize_until_convergence ||
    it_num < num_iterations; it_num++) {

    // Obtain the step direction approximately.
    ObtainStepDirection_(&p, &p_norm);

    TrustRadiusUpdate_(rho, p_norm, &current_radius);

    // If the decrease in the objective is sufficient enough, then
    // accept the step. Otherwise, don't move.
    if(rho > eta) {

    }
  }
}
};
};

#endif
