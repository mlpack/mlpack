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
 * \f$
 * s:=arg\min_{s\in D} <s, v>
 * \f$
 * when \f$ D \f$ is a regularized lp ball. That is,
 * \f[
 * D = \{ x: (\sum_j|\lambda_j x_j|^p)^{1/p}\leq 1 \}.
 * \f]
 * If \f$ \lambda \f$ is not given in the constructor, default is using all
 * \f$ \lambda_j = 1 \f$ for all \f$ j \f$.
 *
 * In applications such as Orthogonal Matching Pursuit (OMP), \f$ \lambda \f$
 * could be ideally set to the norm of the elements in the dictionary.
 *
 * For \f$ p=1 \f$: take (one) \f$ k = arg\max_j |v_j/\lambda_j|\f$, then the
 * solution is:
 * \f[
 * s_k = -sign(v_k)/\lambda_k, \qquad s_j = 0, \quad j\neq k.
 * \f]
 *
 * For \f$ 1<p<\infty \f$: the solution is
 * \f[
 * t_j = -sign(v_j) |v_j/\lambda_j|^{q-1}, \qquad
 * s_j = \frac{t_j}{||t||_p\cdot\lambda_j}, \quad
 * 1/p + 1/q = 1.
 * \f]
 *
 * For \f$ p=\infty \f$: the solution is
 * \f[
 * s_j = -sign(v_j)/\lambda_j
 * \f]
 *
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
   * Construct the solver of constrained problem, with regularization parameter
   * lambda here.
   *
   * @param p The constraint is unit lp ball.
   * @param lambda Regularization parameter.
   */
  ConstrLpBallSolver(const double p, const arma::vec lambda) :
      p(p), regFlag(true), lambda(lambda)
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
      if (regFlag)
        s = s / lambda;   // element-wise division.
    }
    else if (p > 1.0)
    {
      // lp ball with 1<p<inf.
      if (regFlag)
        s = v / lambda;
      else
        s = v;

      double q = 1 / (1.0 - 1.0 / p);
      s = - sign(v) % pow(abs(s), q - 1);  // element-wise multiplication.
      s = arma::normalise(s, p);

      if (regFlag)
        s = s / lambda;
    }
    else if (p == 1.0)
    {
      // l1 ball, also used in OMP.
      if (regFlag)
        s = arma::abs(v / lambda);
      else
        s = arma::abs(v);

      arma::uword k = 0;
      s.max(k);  // k is the linear index of the largest element.
      s.zeros();
      s(k) = - mlpack::math::Sign(v(k));

      if (regFlag)
        s = s / lambda;
    }
    else
    {
      Log::Fatal << "Wrong norm p!" << std::endl;
    }

    return;
  }

  //! Get the p-norm.
  double P() const { return p; }
  //! Modify the p-norm.
  double& P() { return p;}

  //! Get regularization flag.
  bool RegFlag() const {return regFlag;}
  //! Modify regularization flag.
  bool& RegFlag() {return regFlag;}

  //! Get the regularization parameter.
  arma::vec Lambda() const {return lambda;}
  //! Modify the regularization parameter.
  arma::vec& Lambda() {return lambda;}

 private:
  //! lp norm, 1<=p<=inf;
  //! use std::numeric_limits<double>::infinity() for inf norm.
  double p;

  //! Regularization flag.
  bool regFlag = false;

  //! Regularization parameter.
  arma::vec lambda;
};

} // namespace optimization
} // namespace mlpack

#endif
