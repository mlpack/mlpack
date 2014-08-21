/**
 * @file complete_incremental_termination.hpp
 * @author Sumedh Ghaisas
 *
 * Termination policy used in AMF (Alternating Matrix Factorization).
 */
#ifndef _MLPACK_METHODS_AMF_COMPLETE_INCREMENTAL_TERMINATION_HPP_INCLUDED
#define _MLPACK_METHODS_AMF_COMPLETE_INCREMENTAL_TERMINATION_HPP_INCLUDED

namespace mlpack
{
namespace amf
{

/**
 * This class acts as a wrapper for basic termination policies to be used by 
 * SVDCompleteIncrementalLearning. This class calls the wrapped class functions
 * after every n calls to main class functions where n is the number of non-zero
 * entries in the matrix being factorized. 
 *
 * @see AMF, SVDCompleteIncrementalLearning
 */
template <class TerminationPolicy>
class CompleteIncrementalTermination
{
 public:
  /**
   * Empty constructor
   *
   * @param t_policy object of wrapped class.
   */
  CompleteIncrementalTermination(TerminationPolicy t_policy = TerminationPolicy())
            : t_policy(t_policy) {}

  /**
   * Initializes the termination policy before stating the factorization.
   *
   * @param V Input matrix to be factorized.
   */
  template <class MatType>
  void Initialize(const MatType& V)
  {
    t_policy.Initialize(V);

    //! get number of non-zero entries
    incrementalIndex = accu(V != 0);
    iteration = 0;
  }

  /**
   * Initializes the termination policy before stating the factorization.
   *
   * @param V Input matrix to be factorized.
   */
  void Initialize(const arma::sp_mat& V)
  {
    t_policy.Initialize(V);

    // get number of non-zero entries
    incrementalIndex = V.n_nonzero;
    iteration = 0;
  }

  /**
   * Check if termination criterio is met.
   *
   * @param W Basis matrix of output.
   * @param H Encoding matrix of output.
   */
  bool IsConverged(arma::mat& W, arma::mat& H)
  {
    // increment iteration count
    iteration++;
    
    // if iteration count is multiple of incremental index,
    // return wrapped class function
    if(iteration % incrementalIndex == 0)
      return t_policy.IsConverged(W, H);
    // else just return false
    else return false;
  }

  //! Get current value of residue
  const double& Index() const { return t_policy.Index(); }

  //! Get current iteration count  
  const size_t& Iteration() const { return iteration; }
  
  //! Access upper limit of iteration count
  const size_t& MaxIterations() const { return t_policy.MaxIterations(); }
  size_t& MaxIterations() { return t_policy.MaxIterations(); }
  
  //! Access the wrapped class object
  const TerminationPolicy& TPolicy() const { return t_policy; }
  TerminationPolicy& TPolicy() { return t_policy; }

 private:
  //! wrapped class object
  TerminationPolicy t_policy;
  
  //! number of iterations after which wrapped class object will be called
  size_t incrementalIndex;
  //! current iteration count
  size_t iteration;
}; // class CompleteIncrementalTermination

} // namespace amf
} // namespace mlpack


#endif // COMPLETE_INCREMENTAL_TERMINATION_HPP_INCLUDED

