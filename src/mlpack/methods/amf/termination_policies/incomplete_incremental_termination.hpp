/**
 * @file incomplete_incremental_termination.hpp
 * @author Sumedh Ghaisas
 */
#ifndef _INCOMPLETE_INCREMENTAL_TERMINATION_HPP_INCLUDED
#define _INCOMPLETE_INCREMENTAL_TERMINATION_HPP_INCLUDED

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

template <class TerminationPolicy>
class IncompleteIncrementalTermination
{
 public:
  IncompleteIncrementalTermination(TerminationPolicy t_policy = TerminationPolicy())
            : t_policy(t_policy) {}

  template <class MatType>
  void Initialize(const MatType& V)
  {
    t_policy.Initialize(V);

    incrementalIndex = V.n_rows;
    iteration = 0;
  }

  bool IsConverged(arma::mat& W, arma::mat& H)
  {
    return t_policy.IsConverged(W, H);
  }

  void Step(const arma::mat& W, const arma::mat& H)
  {
    if(iteration % incrementalIndex == 0) t_policy.Step(W, H);
    iteration++;
  }

  const double& Index()
  {
    return t_policy.Index();
  }
  const size_t& Iteration()
  {
    return iteration;
  }

 private:
  TerminationPolicy t_policy;

  size_t incrementalIndex;
  size_t iteration;
};

}; // namespace amf
}; // namespace mlpack

#endif

