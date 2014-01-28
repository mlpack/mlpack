/**
 * @file lrsdp.cpp
 * @author Ryan Curtin
 *
 * An implementation of Monteiro and Burer's formulation of low-rank
 * semidefinite programs (LR-SDP).
 */
#include "lrsdp.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace std;

LRSDP::LRSDP(const size_t numConstraints,
             const arma::mat& initialPoint) :
    a(numConstraints),
    b(numConstraints),
    aModes(numConstraints),
    initialPoint(initialPoint),
    augLagInternal(*this),
    augLag(augLagInternal)
{ }

LRSDP::LRSDP(const size_t numConstraints,
             const arma::mat& initialPoint,
             AugLagrangian<LRSDP>& augLag) :
    a(numConstraints),
    b(numConstraints),
    aModes(numConstraints),
    initialPoint(initialPoint),
    augLagInternal(*this),
    augLag(augLag)
{ }

double LRSDP::Optimize(arma::mat& coordinates)
{
  augLag.Sigma() = 20;
  augLag.Optimize(coordinates, 1000);

  return Evaluate(coordinates);
}

double LRSDP::Evaluate(const arma::mat& coordinates) const
{
  return -accu(coordinates * trans(coordinates));
}

void LRSDP::Gradient(const arma::mat& /*coordinates*/,
                     arma::mat& /*gradient*/) const
{
  Log::Fatal << "LRSDP::Gradient() called!  Not implemented!  Uh-oh..." << std::endl;
}

double LRSDP::EvaluateConstraint(const size_t index,
                                 const arma::mat& coordinates) const
{
  arma::mat rrt = coordinates * trans(coordinates);
  if (aModes[index] == 0)
    return trace(a[index] * rrt) - b[index];
  else
  {
    double value = -b[index];
    for (size_t i = 0; i < a[index].n_cols; ++i)
      value += a[index](2, i) * rrt(a[index](0, i), a[index](1, i));

    return value;
  }
}

void LRSDP::GradientConstraint(const size_t /* index */,
                               const arma::mat& /* coordinates */,
                               arma::mat& /* gradient */) const
{
  Log::Fatal << "LRSDP::GradientConstraint() called!  Not implemented!  Uh-oh..." << std::endl;
}

const arma::mat& LRSDP::GetInitialPoint()
{
  return initialPoint;
}
