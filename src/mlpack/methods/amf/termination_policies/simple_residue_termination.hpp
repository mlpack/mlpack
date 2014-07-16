/**
 * @file simple_residue_termination.hpp
 * @author Sumedh Ghaisas
 */
#ifndef _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
#define _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

class SimpleResidueTermination
{
 public:
  SimpleResidueTermination(const double minResidue = 1e-10,
                           const size_t maxIterations = 10000)
        : minResidue(minResidue), maxIterations(maxIterations) { }

  template<typename MatType>
  void Initialize(const MatType& V)
  {
    residue = minResidue;
    iteration = 1;
    normOld = 0;

    const size_t n = V.n_rows;
    const size_t m = V.n_cols;

    nm = n * m;
  }

  bool IsConverged(arma::mat& W, arma::mat& H)
  {
    // Calculate norm of WH after each iteration.
    arma::mat WH;

    WH = W * H;
    double norm = sqrt(accu(WH % WH) / nm);

    if (iteration != 0)
    {
      residue = fabs(normOld - norm);
      residue /= normOld;
    }

    normOld = norm;

    iteration++;
    
    if(residue < minResidue || iteration > maxIterations) return true;
    else return false;
  }

  const double& Index() { return residue; }
  const size_t& Iteration() { return iteration; }
  const size_t& MaxIterations() { return maxIterations; }

public:
  double minResidue;
  size_t maxIterations;

  double residue;
  size_t iteration;
  double normOld;

  size_t nm;
}; // class SimpleResidueTermination

}; // namespace amf
}; // namespace mlpack


#endif // _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
