/**
 * @file stagnation.hpp
 * @author Marcus Edel
 *
 * Definition of the Stagnation class.
 */
#ifndef MLPACK_METHODS_NE_STAGNATION_HPP
#define MLPACK_METHODS_NE_STAGNATION_HPP

#include <mlpack/core.hpp>

#include "fitness_policies/mean_fitness.hpp"
#include "species.hpp"

namespace mlpack {
namespace ne {

template<typename FitnessPolicy = MeanFitness>
class Stagnation
{
 public:
  Stagnation(const size_t maxStagnation) : maxStagnation(maxStagnation)
  {
    // Nothing to do here.
  }

  void Update(std::vector<Species>& species)
  {
    for (std::vector<Species>::iterator it = species.begin();
        it != species.end();)
    {
      it->Fitness() = FitnessPolicy::Fitness(*it);

      if(it->Fitness() < it->PreviousFitness())
      {
        it->PreviousFitness() = it->Fitness();
        it->Stagnation() = 0;
      }
      else
      {
        it->Stagnation() += 1;
      }

      if (it->Stagnation() >= maxStagnation)
      {
        it = species.erase(it);
      }
      else
      {
        it++;
      }
    }
  }

  double AdjustedFitness(const std::vector<Species>& species)
  {
    arma::vec fitnesses;
    return AdjustedFitness(species, fitnesses);
  }

  template<typename Policy = FitnessPolicy>
  static typename std::enable_if<
      std::is_same<Policy, MeanFitness>::value, double>::type
  AdjustedFitness(const std::vector<Species>& species, arma::vec& fitnesses)
  {
    fitnesses = arma::vec(species.size());

    double adjustedFitness = 0;
    size_t i = 0;
    for (std::vector<Species>::const_iterator it = species.begin();
        it != species.end(); ++it, ++i)
    {
      fitnesses(i) = it->Fitness();
      adjustedFitness += it->Fitness();
    }

    return species.size() > 0 ? adjustedFitness / species.size() : -1;
  }

  template<typename Policy = FitnessPolicy>
  static typename std::enable_if<
      !std::is_same<Policy, MeanFitness>::value, double>::type
  AdjustedFitness(const std::vector<Species>& species, arma::vec& fitnesses)
  {
    fitnesses = arma::vec(species.size());

    double adjustedFitness = 0;
    size_t i = 0;
    for (std::vector<Species>::const_iterator it = species.begin();
        it != species.end(); ++it, ++i)
    {
      fitnesses(i) = MeanFitness::Fitness(*it);
      adjustedFitness += fitnesses(i);
    }

    return species.size() > 0 ? adjustedFitness / species.size() : -1;
  }

 private:
  size_t maxStagnation;
};

}  // namespace ne
}  // namespace mlpack

#endif  // MLPACK_METHODS_NE_STAGNATION_HPP
