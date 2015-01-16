/**
 * @file primal_dual.hpp
 * @author Stephen Tu
 *
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_SDP_PRIMAL_DUAL_HPP
#define __MLPACK_CORE_OPTIMIZERS_SDP_PRIMAL_DUAL_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sdp/sdp.hpp>

namespace mlpack {
namespace optimization {

class PrimalDualSolver {
 public:

  PrimalDualSolver(const SDP& sdp);

  PrimalDualSolver(const SDP& sdp,
                   const arma::mat& X0,
                   const arma::vec& ysparse0,
                   const arma::vec& ydense0,
                   const arma::mat& Z0);

  double Optimize(arma::mat& X,
                  arma::vec& ysparse,
                  arma::vec& ydense,
                  arma::mat &Z);

  double Optimize(arma::mat& X) {
    arma::vec ysparse, ydense;
    arma::mat Z;
    return Optimize(X, ysparse, ydense, Z);
  }

  double& Sigma() { return sigma; }
  double& Tau() { return tau; }
  double& NormXzTol() { return normXzTol; }
  double& PrimalInfeasTol() { return primalInfeasTol; }
  double& DualInfeasTol() { return dualInfeasTol; }
  size_t& MaxIterations() { return maxIterations; }

 private:
  SDP sdp;

  arma::mat X0;
  arma::vec ysparse0;
  arma::vec ydense0;
  arma::mat Z0;

  double sigma;
  double tau;
  double normXzTol;
  double primalInfeasTol;
  double dualInfeasTol;

  size_t maxIterations;

};

} // namespace optimization
} // namespace mlpack

#endif
