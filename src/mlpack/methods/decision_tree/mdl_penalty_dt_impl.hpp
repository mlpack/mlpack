#ifndef MDL_PENALTY_HPP
#define MDL_PENALTY_HPP

#include <cmath> // Include cmath for std::log2
#include <armadillo> // Include armadillo for arma::vec

namespace mlpack {
namespace tree {

// Define the MDLPenalty class template.
template<typename FitnessFunction>
class MDLPenalty {
 public:
  // Constructor.
  MDLPenalty(const FitnessFunction& fitnessFunction);

  // Operator overloading for calculating penalized gain.
  double operator()(const arma::vec& childCounts,
                    const arma::vec& childGains,
                    const double delta,
                    const size_t numClasses,
                    const double numChildren,
                    const double sumWeights,
                    const double epsilon) const;

 private:
  const FitnessFunction& fitnessFunction; // Reference to the fitness function.
};

// Define the constructor.
template<typename FitnessFunction>
MDLPenalty<FitnessFunction>::MDLPenalty(
    const FitnessFunction& fitnessFunction) :
    fitnessFunction(fitnessFunction)
{
  // Nothing to do here.
}

// Define the operator.
template<typename FitnessFunction>
double MDLPenalty<FitnessFunction>::operator()(
    const arma::vec& childCounts,
    const arma::vec& childGains,
    const double delta,
    const size_t numClasses,
    const double numChildren,
    const double sumWeights,
    const double epsilon) const
{
  // Calculate the original gain without penalty.
  const double gain = fitnessFunction(childCounts, childGains, delta,
      numClasses, numChildren, sumWeights, epsilon);

  // Calculate the penalty term using the MDL formula.
  const double penalty = std::log2(numChildren / sumWeights);

  // Return the penalized gain.
  return gain - penalty;
}

} // namespace tree
} // namespace mlpack

#endif // MDL_PENAL
