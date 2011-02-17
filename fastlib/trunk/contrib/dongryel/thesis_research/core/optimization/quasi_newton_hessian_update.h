/** @file quasi_newton_hessian_update.h
 *
 *  A collection of routines for updating the Hessians for
 *  quasi-Newton methods.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_OPTIMIZATION_QUASI_NEWTON_HESSIAN_UPDATE_H
#define CORE_OPTIMIZATION_QUASI_NEWTON_HESSIAN_UPDATE_H

#include <armadillo>

namespace core {
namespace optimization {
class QuasiNewtonHessianUpdate {

  public:

    static void BFGSUpdate(
      const arma::mat &current_hessian, const arma::vec &old_iterate,
      const arma::vec &new_iterate, const arma::vec &old_gradient,
      const arma::vec &new_gradient, arma::mat *new_hessian) {

      arma::vec iterate_difference = new_iterate - old_iterate;
      arma::vec gradient_difference = new_gradient - old_gradient;

      // Compute the temporaries.
      double first_factor =
        1.0 / arma::dot(gradient_difference, iterate_difference);
      arma::tmp_vector = current_hessian * iterate_difference;
      double second_factor = 1.0 / arma::dot(iterate_difference, tmp_vector);

      // Update the Hessian.
      (*new_hessian) = current_hessian +
                       first_factor * gradient_difference * arma::trans(gradient_difference) -
                       second_factor * tmp_vector * arma::trans(tmp_vector);
    }

    static void SymmetricRank1Update(
      const arma::mat &current_hessian, const arma::vec &old_iterate,
      const arma::vec &new_iterate, const arma::vec &old_gradient,
      const arma::vec &new_gradient, arma::mat *new_hessian) {

      arma::vec iterate_difference = new_iterate - old_iterate;
      arma::vec gradient_difference = new_gradient - old_gradient;
      arma::vec tmp_vector =
        iterate_difference - current_hessian * gradient_difference;
      double scaling_factor = 1.0 / arma::dot(tmp_vector, gradient_difference);

      // Update the Hessian.
      (*new_hessian) = current_hessian +
                       scaling_factor * tmp_vector * arma::trans(tmp_vector);
    }
};
}
}

#endif
