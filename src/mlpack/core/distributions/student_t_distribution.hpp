/**
 * @file core/distributions/student_t_distribution.hpp
 * @author Kiner Shah
 *
 * Implementation of the Student t-distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DISTRIBUTIONS_STUDENT_T_DISTRIBUTION_HPP
#define MLPACK_CORE_DISTRIBUTIONS_STUDENT_T_DISTRIBUTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * A multivariate Student t-distribution.
 * 
 * The Student t-distribution is commonly used in t-SNE for computing
 * probabilities in the low-dimensional space. It has heavier tails than
 * the Gaussian distribution, which helps prevent crowding of points in
 * the center of the map.
 */
template<typename MatType = arma::mat>
class StudentTDistribution
{
 public:
  // Convenience typedefs for derived types of MatType.
  using VecType = typename GetColType<MatType>::type;
  using ElemType = typename MatType::elem_type;

 private:
  //! Degrees of freedom for the distribution.
  ElemType degreesOfFreedom;

  //! log(pi)
  static const constexpr ElemType logPi = 1.14472988584940017414;

 public:
  /**
   * Default constructor, which creates a Student t-distribution with 
   * 1 degree of freedom.
   */
  StudentTDistribution() : degreesOfFreedom(1.0) { /* nothing to do */ }

  /**
   * Create a Student t-distribution with the given degrees of freedom.
   *
   * @param degreesOfFreedom Degrees of freedom for the distribution.
   */
  StudentTDistribution(const ElemType degreesOfFreedom) :
      degreesOfFreedom(degreesOfFreedom)
  {
    if (degreesOfFreedom <= 0.0)
    {
      Log::Fatal << "StudentTDistribution::StudentTDistribution(): "
          << "degrees of freedom must be positive!" << std::endl;
    }
  }

  /**
   * Return the probability of the given observation under a univariate
   * Student t-distribution centered at the given location.
   * 
   * @param observation The observation.
   * @param location The center/location of the distribution.
   */
  ElemType Probability(const ElemType observation,
                       const ElemType location = 0.0) const
  {
    return std::exp(LogProbability(observation, location));
  }

  /**
   * Return the log probability of the given observation under a univariate
   * Student t-distribution centered at the given location.
   * 
   * @param observation The observation.
   * @param location The center/location of the distribution.
   */
  ElemType LogProbability(const ElemType observation,
                          const ElemType location = 0.0) const;

  /**
   * Compute the probability of pairwise distances under a multivariate
   * Student t-distribution (specifically for t-SNE).
   * 
   * This computes: (1 + ||y_i - y_j||^2 / df)^(-(df + 1)/2)
   * where df is the degrees of freedom.
   * 
   * @param squaredDistances Matrix of squared Euclidean distances.
   * @param probabilities Output matrix of probabilities.
   */
  void PairwiseProbabilities(const MatType& squaredDistances,
                             MatType& probabilities) const;

  /**
   * Compute the log probability of pairwise distances under a multivariate
   * Student t-distribution.
   * 
   * @param squaredDistances Matrix of squared Euclidean distances.
   * @param logProbabilities Output matrix of log probabilities.
   */
  void PairwiseLogProbabilities(const MatType& squaredDistances,
                                MatType& logProbabilities) const;

  /**
   * Return the degrees of freedom.
   */
  ElemType DegreesOfFreedom() const { return degreesOfFreedom; }

  /**
   * Modify the degrees of freedom.
   */
  ElemType& DegreesOfFreedom() { return degreesOfFreedom; }

  /**
   * Serialize the distribution.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(degreesOfFreedom));
  }
};

} // namespace mlpack

// Include implementation.
#include "student_t_distribution_impl.hpp"

#endif
