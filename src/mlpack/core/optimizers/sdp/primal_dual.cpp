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
    dualInfeasTol(1e-7),
    maxIterations(1000)
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
    dualInfeasTol(1e-7),
    maxIterations(1000)
{
  arma::mat tmp;

  if (X0.n_rows != sdp.N() || X0.n_cols != sdp.N())
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "X0 needs to be square n x n matrix"
      << std::endl;

  if (!arma::chol(tmp, X0))
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "X0 needs to be symmetric positive definite"
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

  if (!arma::chol(tmp, Z0))
    Log::Fatal << "PrimalDualSolver::PrimalDualSolver(): "
      << "Z0 needs to be symmetric positive definite"
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

std::pair<bool, double>
PrimalDualSolver::Optimize(arma::mat& X,
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
            M, Einv_Frd_rc_Mat, Frd_rc_Mat, Frd_ATdy_rc_Mat,
            Einv_Frd_ATdy_rc_Mat, DualCheck;

  rp.set_size(sdp.NumConstraints());
  rhs.set_size(sdp.NumConstraints());

  Einv_F_AsparseT.set_size(n2bar, sdp.NumSparseConstraints());
  Einv_F_AdenseT.set_size(n2bar, sdp.NumDenseConstraints());
  M.set_size(sdp.NumConstraints(), sdp.NumConstraints());

  double primal_obj = 0.;
  for (size_t iteration = 0; iteration < maxIterations; iteration++)
  {

    const double mu = sigma * arma::dot(sx, sz) / n;

    if (sdp.NumSparseConstraints())
      rp(arma::span(0, sdp.NumSparseConstraints() - 1)) =
        sdp.SparseB() - Asparse * sx;
    if (sdp.NumDenseConstraints())
      rp(arma::span(sdp.NumSparseConstraints(), sdp.NumConstraints() - 1)) =
          sdp.DenseB() - Adense * sx;

    //std::cout << "rp" << std::endl;
    //std::cout << rp << std::endl;

    rd = - sz - Asparse.t() * ysparse - Adense.t() * ydense;
    if (sdp.HasSparseObjective())
      rd += scsparse;
    if (sdp.HasDenseObjective())
      rd += scdense;

    //std::cout << "rd" << std::endl;
    //std::cout << rd << std::endl;

    Rc = mu*arma::eye<arma::mat>(n, n) - 0.5*(X*Z + Z*X);
    math::Svec(Rc, rc);

    //std::cout << "rc" << std::endl;
    //std::cout << rc << std::endl;

    math::SymKronId(Z, E);
    math::SymKronId(X, F);

    //std::cout << "E" << std::endl;
    //std::cout << E << std::endl;

    //std::cout << "F" << std::endl;
    //std::cout << F << std::endl;

    for (size_t i = 0; i < sdp.NumSparseConstraints(); i++)
    {
      SolveLyapunov(Gk, Z, X * sdp.SparseA()[i] + sdp.SparseA()[i] * X);
      math::Svec(Gk, gk);
      Einv_F_AsparseT.col(i) = gk;
    }

    //std::cout << "Einv_F_AsparseT" << std::endl;
    //std::cout << Einv_F_AsparseT << std::endl;

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

    //std::cout << "M" << std::endl;
    //std::cout << M << std::endl;

    math::Smat(F * rd - rc, Frd_rc_Mat);
    SolveLyapunov(Einv_Frd_rc_Mat, Z, 2. * Frd_rc_Mat);
    math::Svec(Einv_Frd_rc_Mat, Einv_Frd_rc);

    rhs = rp;
    if (sdp.NumSparseConstraints())
      rhs(arma::span(0, sdp.NumSparseConstraints() - 1)) += Asparse * Einv_Frd_rc;
    if (sdp.NumDenseConstraints())
      rhs(arma::span(sdp.NumSparseConstraints(), sdp.NumConstraints() - 1)) += Adense * Einv_Frd_rc;

    // TODO(stephentu): use a more efficient method (e.g. LU decomposition)
    //std::cout << "rhs" << std::endl;
    //std::cout << rhs << std::endl;

    if (!arma::solve(dy, M, rhs))
      Log::Fatal << "PrimalDualSolver::Optimize(): Could not solve KKT system" << std::endl;

    if (sdp.NumSparseConstraints())
      dysparse = dy(arma::span(0, sdp.NumSparseConstraints() - 1));
    if (sdp.NumDenseConstraints())
      dydense = dy(arma::span(sdp.NumSparseConstraints(), sdp.NumConstraints() - 1));
    //std::cout << "dy" << std::endl;
    //std::cout << dy << std::endl;

    math::Smat(F * (rd - Asparse.t() * dysparse - Adense.t() * dydense) - rc, Frd_ATdy_rc_Mat);
    SolveLyapunov(Einv_Frd_ATdy_rc_Mat, Z, 2. * Frd_ATdy_rc_Mat);
    math::Svec(Einv_Frd_ATdy_rc_Mat, Einv_Frd_ATdy_rc);
    dsx = -Einv_Frd_ATdy_rc;

    //std::cout << "dsx" << std::endl;
    //std::cout << dsx << std::endl;

    dsz = rd - Asparse.t() * dysparse - Adense.t() * dydense;

    //std::cout << "dsz" << std::endl;
    //std::cout << dsz << std::endl;

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

    primal_obj = 0.;
    if (sdp.HasSparseObjective())
      primal_obj += arma::dot(sdp.SparseC(), X);
    if (sdp.HasDenseObjective())
      primal_obj += arma::dot(sdp.DenseC(), X);

    const double dual_obj =
      arma::dot(sdp.SparseB(), ysparse) +
      arma::dot(sdp.DenseB(), ydense);

    const double duality_gap = primal_obj - dual_obj;

    DualCheck = Z;
    if (sdp.HasSparseObjective())
      DualCheck -= sdp.SparseC();
    if (sdp.HasDenseObjective())
      DualCheck -= sdp.DenseC();
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
