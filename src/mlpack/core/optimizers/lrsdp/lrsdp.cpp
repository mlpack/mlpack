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

LRSDP::LRSDP(const size_t numSparseConstraints,
             const size_t numDenseConstraints,
             const arma::mat& initialPoint) :
    function(numSparseConstraints, numDenseConstraints, initialPoint),
    augLag(function)
{ }

double LRSDP::Optimize(arma::mat& coordinates)
{
  augLag.Sigma() = 20;
  augLag.Optimize(coordinates, 1000);

  return augLag.Function().Evaluate(coordinates);
}

// Convert the object to a string.
std::string LRSDP::ToString() const
{
  std::ostringstream convert;
  convert << "LRSDP [" << this << "]" << std::endl;
  convert << "  Optimizer: " << util::Indent(augLag.ToString(), 1) << std::endl;
  return convert.str();
}
