/**
 * @file constr_lpball.hpp
 * @author Chenzhe Diao
 *
 * Lp ball constrained for FrankWolfe algorithm. Used as LinearConstrSolverType.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_CONSTR_LPBALL_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_CONSTR_LPBALL_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * LinearConstrSolver for FrankWolfe algorithm. Constraint domain given in the
 * form of lp ball. That is, given \f$ v \f$, solve
 * \f[
 * s:=arg\min_{s\in D} <s, v>
 * \f]
 * when \f$ D \f$ is an lp ball.
 *
 * For \f$ p=1 \f$: take (one) \f$ k = arg\max_j |v_j|\f$, then the solution is:
 * \f[
 * s_k = -sign(v_k), \qquad s_j = 0, j\neq k.
 * \f]
 *
 * For \f$ 1<p<\infty \f$: the solution is
 * \f[
 * s_j = -sign(v_j) |v_j|^{p-1}
 * \f]
 *
 * For \f$ p=\infty \f$: the solution is
 * \f[
 * s_j = -sign(v_j)
 * \f]
 *
 * where \f$ \alpha \f$ is a parameter which specifies the step size.  \f$ i \f$
 * is chosen according to \f$ j \f$ (the iteration number).
 */
class ConstrLpBallSolver
{
 public:
  /**
   * Construct the solver of constrained problem. The constrained domain should
   * be unit lp ball for this class.
   *
   * @param p The constraint is unit lp ball.
   */
  ConstrLpBallSolver(const double p) : p(p)
  { /* Do nothing. */ }

  /**
   * Optimizer of Linear Constrained Problem for FrankWolfe.
   *
   * @param v Input local gradient.
   * @param s Output optimal solution in the constrained domain (lp ball).
   */
  void Optimize(const arma::mat& v,
                arma::mat& s)
  {
    if (p == std::numeric_limits<double>::infinity())
    {
      // l-inf ball.
      s = -sign(v);
    }
    else if (p > 1.0)
    {
      // lp ball with 1<p<inf.
      s = -sign(v) % pow(abs(v), p-1);
    }
    else if (p == 1.0)
    {
      // l1 ball, also used in OMP.
      arma::mat tmp = arma::abs(v);
      arma::uword k;
      tmp.max(k);  // k is the linear index of the largest element.
      s.zeros(v.n_rows, v.n_cols);
      s(k) = - mlpack::math::Sign(v(k));
    }
    else
    {
      Log::Fatal << "Wrong norm p!" << std::endl;
    }

    return;
  }

 private:
  //! lp norm, 1<=p<=inf;
  //! use std::numeric_limits<double>::infinity() for inf norm.
  double p;
};

} // namespace optimization
} // namespace mlpack

#endif
