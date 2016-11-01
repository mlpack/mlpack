/**
 * @file primal_dual_impl.hpp
 * @author Stephen Tu
 *
 * Contains an implementation of the "XZ+ZX" primal-dual infeasible interior
 * point method with a Mehrotra predictor-corrector update step presented and
 * analyzed in:
 *
 *   Primal-dual interior-point methods for semidefinite programming:
 *   Convergence rates, stability and numerical results.
 *   Farid Alizadeh, Jean-Pierre Haeberly, and Michael Overton.
 *   SIAM J. Optim. 1998.
 *   https://www.cs.nyu.edu/overton/papers/pdffiles/pdsdp.pdf
 *
 * We will refer to this paper as [AHO98] in this file.
 *
 * Note there are many optimizations that still need to be implemented. See the
 * code comments for more details.
 *
 * Also note the current implementation assumes the SDP problem has a strictly
 * feasible primal/dual point (and therefore the duality gap is zero), and
 * that the constraint matrices are linearly independent.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SDP_PRIMAL_DUAL_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_SDP_PRIMAL_DUAL_IMPL_HPP

#include "primal_dual.hpp"

namespace mlpack {
namespace optimization {

template <typename SDPType>
PrimalDualSolver<SDPType>::PrimalDualSolver(const SDPType& sdp)
  : sdp(sdp),
    initialX(arma::eye<arma::mat>(sdp.N(), sdp.N())),
    initialYsparse(arma::ones<arma::vec>(sdp.NumSparseConstraints())),
    initialYdense(arma::ones<arma::vec>(sdp.NumDenseConstraints())),
    initialZ(arma::eye<arma::mat>(sdp.N(), sdp.N())),
    tau(0.99),
    normXzTol(1e-7),
    primalInfeasTol(1e-7),
    dualInfeasTol(1e-7),
    maxIterations(1000)
{

}

template <typename SDPType>
PrimalDualSolver<SDPType>::PrimalDualSolver(const SDPType& sdp,
                                            const arma::mat& initialX,
                                            const arma::vec& initialYsparse,
                                            const arma::vec& initialYdense,
                                            const arma::mat& initialZ)
  : sdp(sdp),
    initialX(initialX),
    initialYsparse(initialYsparse),
    initialYdense(initialYdense),
    initialZ(initialZ),
    tau(0.99),
    normXzTol(1e-7),
    primalInfeasTol(1e-7),
    dualInfeasTol(1e-7),
    maxIterations(1000)
{
  arma::mat tmp;

  // Note that the algorithm we implement requires primal iterate X and
  // dual multiplier Z to be positive definite (but not feasible).

  if (initialX.n_rows != sdp.N() || initialX.n_cols != sdp.N())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialX needs to be square n x n matrix." << std::endl;

  if (!arma::chol(tmp, initialX))
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialX needs to be symmetric positive definite." << std::endl;

  if (initialYsparse.n_elem != sdp.NumSparseConstraints())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialYsparse needs to have the same length as the number of sparse "
      << "constraints." << std::endl;

  if (initialYdense.n_elem != sdp.NumDenseConstraints())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialYdense needs to have the same length as the number of dense "
      << "constraints." << std::endl;

  if (initialZ.n_rows != sdp.N() || initialZ.n_cols != sdp.N())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialZ needs to be square n x n matrix."  << std::endl;

  if (!arma::chol(tmp, initialZ))
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialZ needs to be symmetric positive definite." << std::endl;
}

/**
 * Compute
 *
 *     alpha = min(1, tau * alphahat(A, dA))
 *
 * where
 *
 *     alphahat = sup{ alphahat : A + dA is psd }
 *
 * See (2.18) of [AHO98] for more details.
 */
static inline double
Alpha(const arma::mat& A, const arma::mat& dA, double tau)
{
  // On Armadillo < 4.500, the "lower" option isn't available.
#if (ARMA_VERSION_MAJOR < 4) || \
    ((ARMA_VERSION_MAJOR == 4) && (ARMA_VERSION_MINOR < 500))
  const arma::mat L = arma::chol(A).t(); // This is less efficient.
#else
  const arma::mat L = arma::chol(A, "lower");
#endif
  const arma::mat Linv = arma::inv(arma::trimatl(L));
  // TODO(stephentu): We only want the top eigenvalue, we should
  // be able to do better than full eigen-decomposition.
  const arma::vec evals = arma::eig_sym(-Linv * dA * Linv.t());
  const double alphahatinv = evals(evals.n_elem - 1);
  double alphahat = 1. / alphahatinv;
  if (alphahat < 0.)
    // dA is PSD already
    alphahat = 1.;
  return std::min(1., tau * alphahat);
}

/**
 * Solve the following Lyapunov equation (for X)
 *
 *   AX + XA = H
 *
 * where A, H are symmetric matrices.
 *
 * TODO(stephentu): Note this method current uses arma's builtin arma::syl
 * method, which is overkill for this situation. See Lemma 7.2 of [AHO98] for
 * how to solve this Lyapunov equation using an eigenvalue decomposition of A.
 *
 */
static inline void
SolveLyapunov(arma::mat& X, const arma::mat& A, const arma::mat& H)
{
  arma::syl(X, A, A, -H);
}

/**
 * Solve the following KKT system (2.10) of [AHO98]:
 *
 *     [ 0  A^T  I ] [ dsx ] = [ rd ]
 *     [ A   0   0 ] [  dy ] = [ rp ]
 *     [ E   0   F ] [ dsz ] = [ rc ]
 *     \---- M ----/
 *
 * where
 *
 *     A  = [ Asparse ]
 *          [ Adense  ]
 *     dy = [ dysparse  dydense ]
 *     E  = Z sym I
 *     F  = X sym I
 *
 */
static inline void
SolveKKTSystem(const arma::sp_mat& Asparse,
               const arma::mat& Adense,
               const arma::mat& Z,
               const arma::mat& M,
               const arma::mat& F,
               const arma::vec& rp,
               const arma::vec& rd,
               const arma::vec& rc,
               arma::vec& dsx,
               arma::vec& dysparse,
               arma::vec& dydense,
               arma::vec& dsz)
{
  arma::mat Frd_rc_Mat, Einv_Frd_rc_Mat,
            Einv_Frd_ATdy_rc_Mat, Frd_ATdy_rc_Mat;
  arma::vec Einv_Frd_rc, Einv_Frd_ATdy_rc, dy;

  // Note: Whenever a formula calls for E^(-1) v for some v, we solve Lyapunov
  // equations instead of forming an explicit inverse.

  // Compute the RHS of (2.12)
  math::Smat(F * rd - rc, Frd_rc_Mat);
  SolveLyapunov(Einv_Frd_rc_Mat, Z, 2. * Frd_rc_Mat);
  math::Svec(Einv_Frd_rc_Mat, Einv_Frd_rc);

  arma::vec rhs = rp;
  const size_t numConstraints = Asparse.n_rows + Adense.n_rows;
  if (Asparse.n_rows)
    rhs(arma::span(0, Asparse.n_rows - 1)) += Asparse * Einv_Frd_rc;
  if (Adense.n_rows)
    rhs(arma::span(Asparse.n_rows, numConstraints - 1)) += Adense * Einv_Frd_rc;

  // TODO(stephentu): use a more efficient method (e.g. LU decomposition)
  if (!arma::solve(dy, M, rhs))
    Log::Fatal << "PrimalDualSolver::SolveKKTSystem(): Could not solve KKT "
        << "system." << std::endl;

  if (Asparse.n_rows)
    dysparse = dy(arma::span(0, Asparse.n_rows - 1));
  if (Adense.n_rows)
    dydense = dy(arma::span(Asparse.n_rows, numConstraints - 1));

  // Compute dx from (2.13)
  math::Smat(F * (rd - Asparse.t() * dysparse - Adense.t() * dydense) - rc,
      Frd_ATdy_rc_Mat);
  SolveLyapunov(Einv_Frd_ATdy_rc_Mat, Z, 2. * Frd_ATdy_rc_Mat);
  math::Svec(Einv_Frd_ATdy_rc_Mat, Einv_Frd_ATdy_rc);
  dsx = -Einv_Frd_ATdy_rc;

  // Compute dz from (2.14)
  dsz = rd - Asparse.t() * dysparse - Adense.t() * dydense;
}

namespace private_ {

// TODO(stephentu): should we move this somewhere more general?
template <typename T> struct vectype { };
template <typename eT> struct vectype<arma::Mat<eT>>
{ typedef arma::Col<eT> type; };
template <typename eT> struct vectype<arma::SpMat<eT>>
{ typedef arma::SpCol<eT> type; };

} // namespace private_

template <typename SDPType>
double
PrimalDualSolver<SDPType>::Optimize(arma::mat& X,
                                    arma::vec& ysparse,
                                    arma::vec& ydense,
                                    arma::mat& Z)
{
  // TODO(stephentu): We need a method which deals with the case when the Ais
  // are not linearly independent.

  const size_t n = sdp.N();
  const size_t n2bar = sdp.N2bar();

  // Form the A matrix in (2.7). Note we explicitly handle
  // sparse and dense constraints separately.

  arma::sp_mat Asparse(sdp.NumSparseConstraints(), n2bar);
  arma::sp_vec Aisparse;

  for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
  {
    math::Svec(sdp.SparseA()[i], Aisparse);
    Asparse.row(i) = Aisparse.t();
  }

  arma::mat Adense(sdp.NumDenseConstraints(), n2bar);
  arma::vec Aidense;
  for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
  {
    math::Svec(sdp.DenseA()[i], Aidense);
    Adense.row(i) = Aidense.t();
  }

  typename private_::vectype<typename SDPType::objective_matrix_type>::type sc;
  math::Svec(sdp.C(), sc);

  X = initialX;
  ysparse = initialYsparse;
  ydense = initialYdense;
  Z = initialZ;

  arma::vec sx, sz, dysparse, dydense, dsx, dsz;
  arma::mat dX, dZ;

  math::Svec(X, sx);
  math::Svec(Z, sz);

  arma::vec rp, rd, rc, gk;

  arma::mat Rc, F, Einv_F_AsparseT, Einv_F_AdenseT, Gk,
            M, DualCheck;

  rp.set_size(sdp.NumConstraints());

  Einv_F_AsparseT.set_size(n2bar, sdp.NumSparseConstraints());
  Einv_F_AdenseT.set_size(n2bar, sdp.NumDenseConstraints());
  M.set_size(sdp.NumConstraints(), sdp.NumConstraints());

  double primalObj = 0., alpha, beta;
  for (size_t iteration = 1; iteration != maxIterations; iteration++)
  {
    // Note: The Mehrotra PC algorithm works like this at a high level.
    // We first solve a KKT system with mu=0. Then, we use the results
    // of this KKT system to get a better estimate of mu and solve
    // the KKT system again. Empirically, this PC step has been shown to
    // significantly reduce the number of required iterations (and is used
    // by most practical solver implementations).

    if (sdp.NumSparseConstraints())
      rp(arma::span(0, sdp.NumSparseConstraints() - 1)) =
        sdp.SparseB() - Asparse * sx;
    if (sdp.NumDenseConstraints())
      rp(arma::span(sdp.NumSparseConstraints(), sdp.NumConstraints() - 1)) =
          sdp.DenseB() - Adense * sx;

    // Rd = C - Z - smat A^T y
    rd = sc - sz - Asparse.t() * ysparse - Adense.t() * ydense;

    math::SymKronId(X, F);

    // We compute E^(-1) F A^T by solving Lyapunov equations.
    // See (2.16).
    for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
    {
      SolveLyapunov(Gk, Z, X * sdp.SparseA()[i] + sdp.SparseA()[i] * X);
      math::Svec(Gk, gk);
      Einv_F_AsparseT.col(i) = gk;
    }

    for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
    {
      SolveLyapunov(Gk, Z, X * sdp.DenseA()[i] + sdp.DenseA()[i] * X);
      math::Svec(Gk, gk);
      Einv_F_AdenseT.col(i) = gk;
    }

    // Form the M = A E^(-1) F A^T matrix (2.15)
    //
    // Since we split A up into its sparse and dense components,
    // we have to handle each block separately.
    if (sdp.NumSparseConstraints())
    {
      M.submat(arma::span(0, sdp.NumSparseConstraints() - 1),
               arma::span(0, sdp.NumSparseConstraints() - 1)) =
          Asparse * Einv_F_AsparseT;
      if (sdp.NumDenseConstraints())
      {
        M.submat(arma::span(0, sdp.NumSparseConstraints() - 1),
                 arma::span(sdp.NumSparseConstraints(),
                            sdp.NumConstraints() - 1)) =
            Asparse * Einv_F_AdenseT;
      }
    }
    if (sdp.NumDenseConstraints())
    {
      if (sdp.NumSparseConstraints())
      {
        M.submat(arma::span(sdp.NumSparseConstraints(),
                            sdp.NumConstraints() - 1),
                 arma::span(0,
                            sdp.NumSparseConstraints() - 1)) =
            Adense * Einv_F_AsparseT;
      }
      M.submat(arma::span(sdp.NumSparseConstraints(),
                          sdp.NumConstraints() - 1),
               arma::span(sdp.NumSparseConstraints(),
                          sdp.NumConstraints() - 1)) =
          Adense * Einv_F_AdenseT;
    }

    const double sxdotsz = arma::dot(sx, sz);

    // TODO(stephentu): computing these alphahats should take advantage of
    // the cholesky decomposition of X and Z which we should have available
    // when we use more efficient methods above.

    // This solves step (1) of Section 7, the "predictor" step.
    Rc = -0.5*(X*Z + Z*X);
    math::Svec(Rc, rc);
    SolveKKTSystem(Asparse, Adense, Z, M, F, rp, rd, rc, dsx, dysparse, dydense,
        dsz);
    math::Smat(dsx, dX);
    math::Smat(dsz, dZ);

    // Step (2), determine step size lengths (alpha, beta)
    alpha = Alpha(X, dX, tau);
    beta = Alpha(Z, dZ, tau);

    // See (7.1)
    const double sigma =
      std::pow(arma::dot(X + alpha * dX, Z + beta * dZ) / sxdotsz, 3);
    const double mu = sigma * sxdotsz / n;

    // Step (3), the "corrector" step.
    Rc = mu*arma::eye<arma::mat>(n, n) - 0.5*(X*Z + Z*X + dX*dZ + dZ*dX);
    math::Svec(Rc, rc);
    SolveKKTSystem(Asparse, Adense, Z, M, F, rp, rd, rc, dsx, dysparse, dydense,
        dsz);
    math::Smat(dsx, dX);
    math::Smat(dsz, dZ);
    alpha = Alpha(X, dX, tau);
    beta = Alpha(Z, dZ, tau);

    // Iterate update
    X += alpha * dX;
    math::Svec(X, sx);
    ysparse += beta * dysparse;
    ydense += beta * dydense;
    Z += beta * dZ;
    math::Svec(Z, sz);

    // Below, we check the KKT conditions. Recall the KKT conditions are
    //
    // (1) Primal feasibility
    // (2) Dual feasibility
    // (3) XZ = 0 (slackness condition)
    //
    // If the KKT conditions are satisfied to a certain degree of precision,
    // then we consider this a valid certificate of optimality and terminate.
    // Otherwise, we proceed onwards.

    const double normXZ = arma::norm(X * Z, "fro");

    const double sparsePrimalInfeas = arma::norm(sdp.SparseB() - Asparse * sx,
        2);
    const double densePrimalInfeas = arma::norm(sdp.DenseB() - Adense * sx, 2);
    const double primalInfeas = sqrt(sparsePrimalInfeas * sparsePrimalInfeas +
        densePrimalInfeas * densePrimalInfeas);

    primalObj = arma::dot(sdp.C(), X);

    const double dualObj = arma::dot(sdp.SparseB(), ysparse) +
        arma::dot(sdp.DenseB(), ydense);
    const double dualityGap = primalObj - dualObj;

    // TODO(stephentu): this dual check is quite expensive,
    // maybe make it optional?
    DualCheck = Z - sdp.C();
    for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
      DualCheck += ysparse(i) * sdp.SparseA()[i];
    for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
      DualCheck += ydense(i) * sdp.DenseA()[i];
    const double dualInfeas = arma::norm(DualCheck, "fro");

    Log::Debug
        << "iter=" << iteration << ", "
        << "primal=" << primalObj << ", "
        << "dual=" << dualObj << ", "
        << "gap=" << dualityGap << ", "
        << "||XZ||=" << normXZ << ", "
        << "primalInfeas=" << primalInfeas << ", "
        << "dualInfeas=" << dualInfeas << ", "
        << "mu=" << mu
        << std::endl;

    if (normXZ <= normXzTol && primalInfeas <= primalInfeasTol &&
        dualInfeas <= dualInfeasTol)
      return primalObj;
  }

  Log::Warn << "PrimalDualSolver::Optimizer(): Did not converge after "
      << maxIterations << " iterations!" << std::endl;
  return primalObj;
}

} // namespace optimization
} // namespace mlpack

#endif
