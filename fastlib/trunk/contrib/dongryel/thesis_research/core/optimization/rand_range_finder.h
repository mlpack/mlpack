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
#include "core/math/math_lib.h"

namespace core {
namespace optimization {
class RandRangeFinder {
  private:

    static void ComputeResidual_(
      const std::vector< arma::vec *> &collected_basis_set,
      arma::vec *residual_out) {

      for(unsigned int i = 0; i < collected_basis_set.size(); i++) {

        // Project onto the basis and subtract.
        double dot_product =
          arma::dot(*(collected_basis_set[i]), (*residual_out)) ;
        (*residual_out) -= dot_product * (* (collected_basis_set[i]));
      }
    }

    /** @brief Projects the data matrix onto a random set of Gaussian
     *         noise vectors.
     */
    static void ApplyGaussianNoise_(
      const arma::mat &input, arma::vec *output) {

      output->zeros(input.n_rows);
      for(unsigned int j = 0; j < input.n_cols; j++) {
        // Generate a standard Gaussian random number.
        double rand_gaussian = core::math::RandGaussian(1.0);
        for(unsigned int i = 0; i < input.n_rows; i++) {
          (*output)[i] += rand_gaussian * input.at(i, j);
        }
      }
    }

    /** @brief Projects the data matrix onto a random set of Gaussian
     *         noise vectors.
     */
    static void ApplyGaussianNoise_(
      const arma::mat &input, int rank,
      std::vector< arma::vec * > *output) {

      output->resize(rank);
      for(int k = 0; k < rank; k++) {
        (*output)[k] = new arma::vec();
        ((*output)[k])->zeros(input.n_rows);
      }
      for(unsigned int j = 0; j < input.n_cols; j++) {
        for(int k = 0; k < rank; k++) {

          // Generate a standard Gaussian random number.
          double rand_gaussian = core::math::RandGaussian(1.0);
          for(unsigned int i = 0; i < input.n_rows; i++) {
            (*((*output)[k]))[i] += rand_gaussian * input.at(i, j);
          }
        }
      }
    }

    static double MaxL2Norm_(
      const std::vector< arma::vec * > &matrix, int start_index) {
      double max_l2_norm = 0.0;
      for(unsigned int i = start_index; i < matrix.size(); i++) {
        max_l2_norm = std::max(
                        max_l2_norm, arma::norm(*(matrix[i]), 2));
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
      std::vector< arma::vec * > candidate_basis_set;
      ApplyGaussianNoise_(input, rank, &candidate_basis_set);
      double max_l2_norm = MaxL2Norm_(candidate_basis_set, 0);
      double threshold = epsilon / (10.0 * sqrt(2.0 / arma::math::pi()));

      // The basis vectors collected so far.
      std::vector< arma::vec * > collected_basis_set;

      // Iteration number.
      int iteration_num = -1;

      // Loop while the maximum norm is above the threshold.
      do {

        // Increment the iteration number.
        iteration_num++;

        // Get the first vector in the list.
        arma::vec &overwrite_vector = *(candidate_basis_set[iteration_num]);
        ComputeResidual_(collected_basis_set, &overwrite_vector);
        arma::vec *normalized_residual = new arma::vec();
        *normalized_residual =
          overwrite_vector / arma::norm(overwrite_vector, 2);

        // Grow the basis set.
        collected_basis_set.push_back(normalized_residual);

        // Draw a new candidate vector.
        arma::vec *candidate_vector = new arma::vec();
        candidate_vector->zeros(input.n_rows);
        ApplyGaussianNoise_(input, candidate_vector);
        ComputeResidual_(collected_basis_set, candidate_vector);
        candidate_basis_set.push_back(candidate_vector);
        for(int i = iteration_num + 1; i <= iteration_num + rank - 1; i++) {
          arma::vec &column_alias =  *(candidate_basis_set[i]);
          double dot_product =
            arma::dot(*normalized_residual, column_alias);
          column_alias =
            column_alias - dot_product * (*normalized_residual);
        }

        // Update the max L2 norm.
        max_l2_norm = MaxL2Norm_(candidate_basis_set, iteration_num + 1);

      }
      while(max_l2_norm > threshold);     // end of the while-loop.

      // Copy out the basis set.
      output->zeros(input.n_rows, collected_basis_set.size());
      for(unsigned int i = 0; i < collected_basis_set.size(); i++) {
        output->col(i) = * (collected_basis_set[i]) ;
        delete collected_basis_set[i];
      }
      for(unsigned int i = 0; i < candidate_basis_set.size(); i++) {
        delete candidate_basis_set[i];
      }
    }
};
}
}

#endif
