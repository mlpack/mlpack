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

template <typename SDPType>
class PrimalDualSolver {
 public:

  PrimalDualSolver(const SDPType& sdp);

  PrimalDualSolver(const SDPType& sdp,
                   const arma::mat& initialX,
                   const arma::vec& initialYsparse,
                   const arma::vec& initialYdense,
                   const arma::mat& initialZ);

  std::pair<bool, double>
  Optimize(arma::mat& X,
           arma::vec& ysparse,
           arma::vec& ydense,
           arma::mat &Z);

  std::pair<bool, double>
  Optimize(arma::mat& X)
  {
    arma::vec ysparse, ydense;
    arma::mat Z;
    return Optimize(X, ysparse, ydense, Z);
  }

  const SDPType& Sdp() const { return sdp; }

  double& Tau() { return tau; }

  double& NormXzTol() { return normXzTol; }

  double& PrimalInfeasTol() { return primalInfeasTol; }

  double& DualInfeasTol() { return dualInfeasTol; }

  size_t& MaxIterations() { return maxIterations; }

 private:
  SDPType sdp;

  arma::mat initialX;
  arma::vec initialYsparse;
  arma::vec initialYdense;
  arma::mat initialZ;

  double tau;
  double normXzTol;
  double primalInfeasTol;
  double dualInfeasTol;

  size_t maxIterations;

};

} // namespace optimization
} // namespace mlpack

// Include implementation.
#include "primal_dual_impl.hpp"

#endif
