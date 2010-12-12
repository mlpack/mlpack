/** @file trust_region_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_OPTIMIZATION_TRUST_REGION_DEV_H
#define CORE_OPTIMIZATION_TRUST_REGION_DEV_H

#include "core/optimization/trust_region.h"
#include "core/math/armadillo_wrapper.h"

namespace core {
namespace optimization {

void TrustRegion::ComputeCauchyPoint_(
  double radius, const arma::vec &gradient,
  const arma::mat &hessian, arma::vec *p) {

  // Nocedal, J. and Wright, S. Numerical Optimization. page 72.

  // Check g_k^T B_k g_k <= 0.
  arma::mat quadratic_form = arma::trans(gradient) * hessian * gradient;

  // Scaling value.
  double tau = 1.0;

  // The gradient norm.
  double gradient_norm = arma::norm(gradient, 2);

  if(quadratic_form.at(0, 0) > 0) {

    // Equation 4.12 on Page 72.
    tau = std::min(
            1.0, core::math::Pow<3, 1>(gradient_norm) / (
              radius * quadratic_form.at(0, 0)));
  }
  core::math::ScaleInit(- tau * radius / gradient_norm , gradient, p);
}

void TrustRegion::ComputeDoglegDirection(
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
    else {

      // Compute g_k^T B_k g_k.
      arma::mat quadratic_form = arma::trans(gradient) * hessian * gradient;

      // p_u= -(g'g/g'Hg)*g
      arma::vec p_u = - (
                        arma::dot(gradient, gradient) / quadratic_form) * gradient;
      double p_u_norm = arma::norm(p_u, 2);

      // If the norm of p_u is beyond the trust radius, then the
      // solution lies on the boundary.
      if(p_u_norm >= radius) {
        core::math::ScaleInit(radius / p_u_norm, p_u, p);
      }

      // Otherwise the quadratic equation composed of the gradient and
      // the full step paths (a x^2 + b x + c = 0)
      else {

        // p_b - p_u
        arma::vec diff = p_b - p_u;
        double a = ;

      }

    } // end of the full step not being feasible case.

  } // end of the positive definite Hessian case.

  // delta_m calculation -g'p-0.5*p'Hp
  arma::mat quadratic_form2 = arma::trans(*p) * hessian * (*p);
  (*delta_m) = - arma::dot(gradient, *p) - 0.5 * quadratic_form2;
}
};
};

#endif
