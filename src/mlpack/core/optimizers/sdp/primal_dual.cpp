#include "primal_dual.hpp"

namespace mlpack {
namespace optimization {

PrimalDualSolver::PrimalDualSolver(const SDP& sdp)
  : sdp(sdp),
    X0(arma::eye<arma::mat>(sdp.N(), sdp.N())),
    ysparse0(arma::ones<arma::vec>(sdp.NumSparseConstraints())),
    ydense0(arma::ones<arma::vec>(sdp.NumDenseConstraints())),
    Z0(arma::eye<arma::mat>(sdp.N(), sdp.N())),
    sigma(0.5),
    tau(0.5),
    normXzTol(1e-7),
    primalInfeasTol(1e-7),
    dualInfeasTol(1e-7)
{

}

PrimalDualSolver::PrimalDualSolver(const SDP& sdp,
                                   const arma::mat& X0,
                                   const arma::vec& ysparse0,
                                   const arma::vec& ydense0,
                                   const arma::mat& Z0)
  : sdp(sdp),
    X0(X0),
    ysparse0(ysparse0),
    ydense0(ydense0),
    Z0(Z0),
    sigma(0.5),
    tau(0.5),
    normXzTol(1e-7),
    primalInfeasTol(1e-7),
    dualInfeasTol(1e-7)
{
  if (X0.n_rows != sdp.N() || X0.n_cols != sdp.N())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "X0 needs to be square n x n matrix"
      << std::endl;

  if (ysparse0.n_elem != sdp.NumSparseConstraints())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "ysparse0 needs to have the same length as the number of sparse constraints"
      << std::endl;

  if (ydense0.n_elem != sdp.NumDenseConstraints())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "ydense0 needs to have the same length as the number of dense constraints"
      << std::endl;

  if (Z0.n_rows != sdp.N() || Z0.n_cols != sdp.N())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "Z0 needs to be square n x n matrix"
      << std::endl;
}

static inline arma::mat
DenseFromSparse(const arma::sp_mat& input)
{
  return arma::mat(input);
}

static inline double
AlphaHat(const arma::mat& A, const arma::mat& dA)
{
  // note: arma::chol(A) returns an upper triangular matrix (instead of the
  // usual lower triangular)
  const arma::mat L = arma::trimatl(arma::chol(A).t());
  const arma::mat Linv = L.i();
  const arma::vec evals = arma::eig_sym(-Linv * dA * Linv.t());
  const double alphahatinv = evals(evals.n_elem - 1);
  return 1. / alphahatinv;
}

/**
 * Solve the following Lyapunov equation (for X)
 *
 *   AX + XA = H
 *
 * where A, H are symmetric matrices
 *
 */
static inline void
SolveLyapunov(arma::mat& X, const arma::mat& A, const arma::mat& H)
{
  arma::syl(X, A, A, -H);
}

double PrimalDualSolver::Optimize(arma::mat& X,
                                  arma::vec& ysparse,
                                  arma::vec& ydense,
                                  arma::mat& Z)
{
  const size_t n = sdp.N();
  const size_t n2bar = sdp.N2bar();

  // TODO: implementation does not take adv of sparsity yet

  arma::mat Asparse(sdp.NumSparseConstraints(), n2bar);
  arma::vec Ai;

  for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
  {
    math::Svec(DenseFromSparse(sdp.SparseA()[i]), Ai);
    Asparse.row(i) = Ai.t();
  }

  arma::mat Adense(sdp.NumDenseConstraints(), n2bar);
  for (size_t i = 0; i < sdp.NumDenseConstraints(); i++)
  {
    math::Svec(sdp.DenseA()[i], Ai);
    Adense.row(i) = Ai.t();
  }

  arma::vec scsparse;
  if (sdp.HasSparseObjective())
    math::Svec(DenseFromSparse(sdp.SparseC()), scsparse);

  arma::vec scdense;
  if (sdp.HasDenseObjective())
    math::Svec(sdp.DenseC(), scdense);


  X = X0;
  ysparse = ysparse0;
  ydense = ydense0;
  Z = Z0;

  arma::vec sx, sz, dy, dysparse, dydense, dsx, dsz;
  arma::mat dX, dZ;

  math::Svec(X, sx);
  math::Svec(Z, sz);

  arma::vec rp, rd, rc, gk, Einv_Frd_rc, Einv_Frd_ATdy_rc, rhs;

  arma::mat Rc, E, F, Einv_F_AsparseT, Einv_F_AdenseT, Gk,
            M, Einv_Frd_rc_Mat, Frd_rc_Mat, Frd_ATdy_Mat,
            Einv_Frd_ATdy_rc_Mat;

  rp.set_size(sdp.NumConstraints());
  rhs.set_size(sdp.NumConstraints());

  Einv_F_AsparseT.set_size(n2bar, sdp.NumSparseConstraints());
  Einv_F_AdenseT.set_size(n2bar, sdp.NumDenseConstraints());
  M.set_size(sdp.NumConstraints(), sdp.NumConstraints());

  for (;;)
  {

    const double mu = sigma * arma::dot(sx, sz) / n;

    if (sdp.NumSparseConstraints())
      rp(arma::span(0, sdp.NumSparseConstraints() - 1)) =
        sdp.SparseB() - Asparse * sx;
    if (sdp.NumDenseConstraints())
      rp(arma::span(sdp.NumSparseConstraints(), sdp.NumConstraints() - 1)) =
          sdp.DenseB() - Adense * sx;

    rd = - sz - Asparse.t() * ysparse - Adense.t() * ydense;
    if (sdp.HasSparseObjective())
      rd += scsparse;
    if (sdp.HasDenseObjective())
      rd += scdense;

    Rc = mu*arma::eye<arma::mat>(n, n) - 0.5*(X*Z + Z*X);
    math::Svec(Rc, rc);

    math::SymKronId(Z, E);
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


    math::Smat(F * rd - rc, Frd_rc_Mat);
    SolveLyapunov(Einv_Frd_rc_Mat, Z, 2. * Frd_rc_Mat);
    math::Svec(Einv_Frd_rc_Mat, Einv_Frd_rc);

    rhs = rp;
    if (sdp.NumSparseConstraints())
      rhs(arma::span(0, sdp.NumSparseConstraints() - 1)) += Asparse * Einv_Frd_rc;
    if (sdp.NumDenseConstraints())
      rhs(arma::span(sdp.NumSparseConstraints(), sdp.NumConstraints() - 1)) += Adense * Einv_Frd_rc;

    // TODO(stephentu): use a more efficient method (e.g. LU decomposition)
    arma::solve(dy, M, rhs);
    if (sdp.NumSparseConstraints())
      dysparse = dy(arma::span(0, sdp.NumSparseConstraints() - 1));
    if (sdp.NumDenseConstraints())
      dydense = dy(arma::span(sdp.NumSparseConstraints(), sdp.NumConstraints() - 1));

    math::Smat(F * (rd - Asparse.t() * dysparse - Adense.t() * dydense), Frd_ATdy_Mat);
    SolveLyapunov(Einv_Frd_ATdy_rc_Mat, Z, 2. * Frd_ATdy_Mat);
    math::Svec(Einv_Frd_ATdy_rc_Mat, Einv_Frd_ATdy_rc);
    dsx = -Einv_Frd_ATdy_rc;
    dsz = rd - Asparse.t() * dysparse - Adense.t() * dydense;

    math::Smat(dsx, dX);
    math::Smat(dsz, dZ);

    double alphahatX = AlphaHat(X, dX);
    if (alphahatX < 0.)
      // dX is PSD
      alphahatX = 1.;

    double alphahatZ = AlphaHat(Z, dZ);
    if (alphahatZ < 0.)
      // dZ is PSD
      alphahatZ = 1.;

    const double alpha = std::min(1., tau * alphahatX);
    const double beta = std::min(1., tau * alphahatZ);

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

    double primal_obj = 0.;
    if (sdp.HasSparseObjective())
      primal_obj += arma::dot(sdp.SparseC(), X);
    if (sdp.HasDenseObjective())
      primal_obj += arma::dot(sdp.DenseC(), X);

    const double dual_obj =
      arma::dot(sdp.SparseB(), ysparse) +
      arma::dot(sdp.DenseB(), ydense);

    const double duality_gap = dual_obj - primal_obj;

    Log::Debug
      << "primal=" << primal_obj << ", "
      << "dual=" << dual_obj << ", "
      << "gap=" << duality_gap << ", "
      << "||XZ||=" << norm_XZ << ", "
      << "primal_infeas=" << primal_infeas << ", "
      << "mu=" << mu
      << std::endl;

    if (norm_XZ <= normXzTol &&
        primal_infeas <= primalInfeasTol)
      return primal_obj;
  }
}

} // namespace optimization
} // namespace mlpack
