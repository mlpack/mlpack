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
      if (p == -1.0)
      {
      // l-inf ball
      s = -sign(v);
      return;
      }
      else if (p > 1.0)
      {
      // lp ball with 1<p<inf
      s = -sign(v) % pow(abs(v), p-1);
      return;
      }
      else if (p == 1.0)
      {
      // l1 ball, used in OMP
      arma::mat tmp = abs(v);
      arma::uword k = tmp.index_max();  // linear index of matrix
      tmp = 0 * tmp;
      tmp(k) = v(k);
      s = -sign(tmp);
      return;
      }
      else
      {
      Log::Fatal << "Wrong norm p!" << std::endl;
      return;
      }
  }

 private:
  // lp norm, take 1<p<inf, use -1 for inf norm.
  double p;
};

} // namespace optimization
} // namespace mlpack

#endif
