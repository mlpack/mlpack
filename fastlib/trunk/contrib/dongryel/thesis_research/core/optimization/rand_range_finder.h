/** @file rand_range_finder.h
 *
 *  Implements Algorithm 4.2 in "Finding Structure With Randomness:
 *  Probabilistic Algorithms for Constructing Approximate Matrix
 *  Decompositions" by Halko et al.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_OPTIMIZATION_RAND_RANGE_FINDER_H
#define CORE_OPTIMIZATION_RAND_RANGE_FINDER_H

#include <armadillo>
#include "core/table/cyclic_dense_matrix.h"

namespace core {
namespace optimization {
class RandRangeFinder {
  private:

    /** @brief Projects the data matrix onto a random set of Gaussian
     *         noise vectors.
     */
    static void ApplyGaussianNoise_(
      const arma::mat &input, int rank,
      core::table::CyclicArmaMat *output) {

      output->zeros(input.n_rows, rank);
      for(unsigned int j = 0; j < input.n_cols; j++) {
        for(unsigned int i = 0; i < input.n_rows; i++) {
          for(int k = 0; k < rank; k++) {

            // Generate a standard Gaussian random number.
            double rand_gaussian = core::math::RandGaussian(1.0);
            output->at(i, k) += rand_gaussian * input.at(i, j);
          }
        }
      }
    }

    static void ApplyGaussianNoise_(
      const arma::mat &matrix, arma::vec *output) {

      output->zeros(input.n_rows);
      for(unsigned int j = 0; j < input.n_cols; j++) {

        // Generate a standard Gaussian random number.
        double rand_gaussian = core::math::RandGaussian(1.0);

        // Add the data point weighted by the Gaussian noise.
        (*output) += rand_gaussian * matrix.col(j);
      }
    }

    static double MaxL2Norm_(const arma::mat &matrix) {
      double max_l2_norm = 0.0;
      for(unsigned int i = 0; i < matrix.n_cols; i++) {
        max_l2_norm = std::max(max_l2_norm, 2);
      }
      return max_l2_norm;
    }

  public:

    /** @brief Finds $r$ orthogonal basis set such that the input
     *         matrix can be approximated within the $\epsilon$
     *         tolerance with probability at least $1 - \min\{m, n}
     *         10^{-r}$.
     */
    static void Compute(
      const arma::mat &input,
      double epsilon, double probability,
      arma::mat *output) {

      // Compute the rank given the required probability level.
      int min_dimension = std::min(input.n_rows, input.n_cols);
      int rank =
        static_cast<int>(
          ceil(log10(min_dimension / (1.0 - probability))));
      if(isnan(rank) || isinf(rank) || rank <= 0) {
        rank = 1;
      }
      core::table::CyclicArmaMat candidate_basis_set;
      ApplyGaussianNoise_(input, rank, &candidate_basis_set);
      double max_l2_norm = MaxL2Norm_(output);
      double threshold = epsilon / (10.0 * sqrt(2.0 / arma::math::pi()));

      // Loop while the maximum norm is above the threshold.
      std::vector< arma::vec * > collected_basis_set;
      while(max_l2_norm > threshold) {

        // Get the first vector in the list.
        arma::vec first_vector;
        candidate_basis_set.col(0, & first_vector);
        ComputeResidual_(collected_basis_set, &first_vector);
        arma::vec *normalized_residual = new arma::vec();
        *normalized_residual = first_vector / arma::norm(first_vector, 2);
        collected_basis_set->push_back(normalized_residual);

      } // end of the while-loop.
    }
};
}
}

#endif
