/**
 * @file simple_residue_termination.hpp
 * @author Sumedh Ghaisas
 *
 * Termination policy used in AMF (Alternating Matrix Factorization).
 */
#ifndef _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
#define _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

/**
 * This class implements simple residue based termination policy. Termination 
 * decision depends on two factors, value of residue and number of iteration. 
 * If the current value of residue drops below the threshold or the number of 
 * iterations goes above the threshold, positive termination signal is passed 
 * to AMF.
 *
 * @see AMF
 */
class SimpleResidueTermination
{
 public:
  //! empty constructor
  SimpleResidueTermination(const double minResidue = 1e-10,
                           const size_t maxIterations = 10000)
        : minResidue(minResidue), maxIterations(maxIterations) { }

  /**
   * Initializes the termination policy before stating the factorization.
   *
   * @param V Input matrix being factorized.
   */
  template<typename MatType>
  void Initialize(const MatType& V)
  {
    // set resisue to minimum value
    residue = minResidue;
    // set iteration to minimum value
    iteration = 1;
    // remove history
    normOld = 0;

    // initialize required variables
    const size_t n = V.n_rows;
    const size_t m = V.n_cols;
    nm = n * m;
  }

  /**
   * Check if termination criterio is met.
   *
   * @param W Basis matrix of output.
   * @param H Encoding matrix of output.
   */
  bool IsConverged(arma::mat& W, arma::mat& H)
  {
    // Calculate norm of WH after each iteration.
    arma::mat WH;

    // calculate the norm and compute the residue 
    WH = W * H;
    double norm = sqrt(accu(WH % WH) / nm);
    residue = fabs(normOld - norm);
    residue /= normOld;

    // store the residue into history
    normOld = norm;
    
    // increment iteration count
    iteration++;
    
    // check if termination criterion is met
    if(residue < minResidue || iteration > maxIterations) return true;
    else return false;
  }

  //! Get current value of residue
  const double& Index() const { return residue; }

  //! Get current iteration count  
  const size_t& Iteration() const { return iteration; }
  
  //! Access max iteration count
  const size_t& MaxIterations() const { return maxIterations; }
  size_t& MaxIterations() { return maxIterations; }
  
  //! Access minimum residue value
  const double& MinResidue() const { return minResidue; }
  double& MinResidue() { return minResidue; }

public:
  //! residue threshold
  double minResidue;
  //! iteration threshold
  size_t maxIterations;

  //! current value of residue
  double residue;
  //! current iteration count
  size_t iteration;
  //! norm of previous iteration
  double normOld;

  size_t nm;
}; // class SimpleResidueTermination

}; // namespace amf
}; // namespace mlpack


#endif // _MLPACK_METHODS_AMF_SIMPLERESIDUETERMINATION_HPP_INCLUDED
