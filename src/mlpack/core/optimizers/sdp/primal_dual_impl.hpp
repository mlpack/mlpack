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
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SDP_PRIMAL_DUAL_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_SDP_PRIMAL_DUAL_IMPL_HPP

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

  if (initialX.n_rows != sdp.N() || initialX.n_cols != sdp.N())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialX needs to be square n x n matrix"
      << std::endl;

  if (!arma::chol(tmp, initialX))
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialX needs to be symmetric positive definite"
      << std::endl;

  if (initialYsparse.n_elem != sdp.NumSparseConstraints())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialYsparse needs to have the same length as the number of sparse constraints"
      << std::endl;

  if (initialYdense.n_elem != sdp.NumDenseConstraints())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialYdense needs to have the same length as the number of dense constraints"
      << std::endl;

  if (initialZ.n_rows != sdp.N() || initialZ.n_cols != sdp.N())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialZ needs to be square n x n matrix"
      << std::endl;

  if (!arma::chol(tmp, initialZ))
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "initialZ needs to be symmetric positive definite"
      << std::endl;
}

static inline double
AlphaHat(const arma::mat& A, const arma::mat& dA)
{
  // note: arma::chol(A) returns an upper triangular matrix (instead of the
  // usual lower triangular)
  const arma::mat L = arma::chol(A).t();
  const arma::mat Linv = arma::inv(arma::trimatl(L));
  const arma::vec evals = arma::eig_sym(-Linv * dA * Linv.t());
  const double alphahatinv = evals(evals.n_elem - 1);
  return 1. / alphahatinv;
}

static inline double
Alpha(const arma::mat& A, const arma::mat& dA, double tau)
{
  double alphahat = AlphaHat(A, dA);
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
    Log::Fatal << "PrimalDualSolver::Optimize(): Could not solve KKT system" << std::endl;

  if (Asparse.n_rows)
    dysparse = dy(arma::span(0, Asparse.n_rows - 1));
  if (Adense.n_rows)
    dydense = dy(arma::span(Asparse.n_rows, numConstraints - 1));

  math::Smat(F * (rd - Asparse.t() * dysparse - Adense.t() * dydense) - rc, Frd_ATdy_rc_Mat);
  SolveLyapunov(Einv_Frd_ATdy_rc_Mat, Z, 2. * Frd_ATdy_rc_Mat);
  math::Svec(Einv_Frd_ATdy_rc_Mat, Einv_Frd_ATdy_rc);
  dsx = -Einv_Frd_ATdy_rc;
  dsz = rd - Asparse.t() * dysparse - Adense.t() * dydense;
}

namespace private_ {

// TODO(stephentu): should we move this somewhere more general
template <typename T> struct vectype { };
template <typename eT> struct vectype<arma::Mat<eT>> { typedef arma::Col<eT> type; };
template <typename eT> struct vectype<arma::SpMat<eT>> { typedef arma::SpCol<eT> type; };

} // namespace private_

template <typename SDPType>
std::pair<bool, double>
PrimalDualSolver<SDPType>::Optimize(arma::mat& X,
                                    arma::vec& ysparse,
                                    arma::vec& ydense,
                                    arma::mat& Z)
{
  // TODO(stephentu): We need a method which deals with the case when the Ais
  // are not linearly independent.

  const size_t n = sdp.N();
  const size_t n2bar = sdp.N2bar();

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

  double primal_obj = 0., alpha, beta;
  for (size_t iteration = 0; iteration != maxIterations; iteration++)
  {
    if (sdp.NumSparseConstraints())
      rp(arma::span(0, sdp.NumSparseConstraints() - 1)) =
        sdp.SparseB() - Asparse * sx;
    if (sdp.NumDenseConstraints())
      rp(arma::span(sdp.NumSparseConstraints(), sdp.NumConstraints() - 1)) =
          sdp.DenseB() - Adense * sx;

    rd = sc - sz - Asparse.t() * ysparse - Adense.t() * ydense;

    math::SymKronId(X, F);

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

    if (sdp.NumSparseConstraints())
    {
      M.submat(arma::span(0,
                          sdp.NumSparseConstraints() - 1),
               arma::span(0,
                          sdp.NumSparseConstraints() - 1)) =
        Asparse * Einv_F_AsparseT;
      if (sdp.NumDenseConstraints())
      {
        M.submat(arma::span(0,
                            sdp.NumSparseConstraints() - 1),
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

    Rc = -0.5*(X*Z + Z*X);
    math::Svec(Rc, rc);
    SolveKKTSystem(Asparse, Adense, Z, M, F, rp, rd, rc, dsx, dysparse, dydense, dsz);
    math::Smat(dsx, dX);
    math::Smat(dsz, dZ);
    alpha = Alpha(X, dX, tau);
    beta = Alpha(Z, dZ, tau);

    const double sigma =
      std::pow(arma::dot(X + alpha * dX, Z + beta * dZ) / sxdotsz, 3);
    const double mu = sigma * sxdotsz / n;

    Rc = mu*arma::eye<arma::mat>(n, n) - 0.5*(X*Z + Z*X + dX*dZ + dZ*dX);
    math::Svec(Rc, rc);
    SolveKKTSystem(Asparse, Adense, Z, M, F, rp, rd, rc, dsx, dysparse, dydense, dsz);
    math::Smat(dsx, dX);
    math::Smat(dsz, dZ);
    alpha = Alpha(X, dX, tau);
    beta = Alpha(Z, dZ, tau);

    X += alpha * dX;
    math::Svec(X, sx);
    ysparse += beta * dysparse;
    ydense += beta * dydense;
    Z += beta * dZ;
    math::Svec(Z, sz);

    const double norm_XZ = arma::norm(X * Z, "fro");

    const double sparse_primal_infeas = arma::norm(sdp.SparseB() - Asparse * sx, 2);
    const double dense_primal_infeas = arma::norm(sdp.DenseB() - Adense * sx, 2);
    const double primal_infeas = sqrt(
        sparse_primal_infeas * sparse_primal_infeas +
        dense_primal_infeas * dense_primal_infeas);

    primal_obj = arma::dot(sdp.C(), X);

    const double dual_obj =
      arma::dot(sdp.SparseB(), ysparse) +
      arma::dot(sdp.DenseB(), ydense);

    const double duality_gap = primal_obj - dual_obj;

    // TODO(stephentu): this dual check is quite expensive,
    // maybe make it optional?
    DualCheck = Z - sdp.C();
    for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
      DualCheck += ysparse(i) * sdp.SparseA()[i];
    for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
      DualCheck += ydense(i) * sdp.DenseA()[i];
    const double dual_infeas = arma::norm(DualCheck, "fro");

    Log::Debug
      << "iter=" << iteration + 1 << ", "
      << "primal=" << primal_obj << ", "
      << "dual=" << dual_obj << ", "
      << "gap=" << duality_gap << ", "
      << "||XZ||=" << norm_XZ << ", "
      << "primal_infeas=" << primal_infeas << ", "
      << "dual_infeas=" << dual_infeas << ", "
      << "mu=" << mu
      << std::endl;

    if (norm_XZ <= normXzTol &&
        primal_infeas <= primalInfeasTol &&
        dual_infeas <= dualInfeasTol)
      return std::make_pair(true, primal_obj);
  }

  Log::Warn << "Did not converge!" << std::endl;
  return std::make_pair(false, primal_obj);
}

} // namespace optimization
} // namespace mlpack

#endif
