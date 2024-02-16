#ifndef MDL_PENALTY_HPP
#define MDL_PENALTY_HPP

#include <cmath> 
#include <armadillo> 

namespace mlpack {
namespace tree {

template<typename FitnessFunction>
class MDLPenalty {
 public:
  MDLPenalty(const FitnessFunction& fitnessFunction);
  double operator()(const arma::vec& childCounts,
                    const arma::vec& childGains,
                    const double delta,
                    const size_t numClasses,
                    const double numChildren,
                    const double sumWeights,
                    const double epsilon) const;

 private:
  const FitnessFunction& fitnessFunction;
};

#include "mdl_penalty_dtimpl.hpp" // Include the implementation file at the end of the header.

} // namespace tree
} // namespace mlpack

#endif // MDL_PENALTY_HPP
