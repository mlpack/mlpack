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
    template<typename MatrixType>
    static void ApplyGaussianNoise_(
      const arma::mat &input, int rank, MatrixType *output) {

      for(unsigned int j = 0; j < input.n_cols; j++) {
        for(int k = 0; k < rank; k++) {

          // Generate a standard Gaussian random number.
          double rand_gaussian = core::math::RandGaussian(1.0);
          for(unsigned int i = 0; i < input.n_rows; i++) {
            output->at(i, k) += rand_gaussian * input.at(i, j);
          }
        }
      }
    }

    static double MaxL2Norm_(const core::table::CyclicArmaMat &matrix) {
      double max_l2_norm = 0.0;
      for(unsigned int i = 0; i < matrix.n_cols; i++) {
        arma::vec matrix_column;
        matrix.col(i, &matrix_column);
        max_l2_norm = std::max(max_l2_norm, arma::norm(matrix_column, 2));
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
      candidate_basis_set.zeros(input.n_rows, rank);
      ApplyGaussianNoise_(input, rank, &candidate_basis_set);
      double max_l2_norm = MaxL2Norm_(candidate_basis_set);
      double threshold = epsilon / (10.0 * sqrt(2.0 / arma::math::pi()));

      // Loop while the maximum norm is above the threshold.
      std::vector< arma::vec * > collected_basis_set;
      arma::vec candidate_vector;
      while(max_l2_norm > threshold) {

        // Get the first vector in the list.
        arma::vec first_vector;
        candidate_basis_set.col(0, & first_vector);
        ComputeResidual_(collected_basis_set, &first_vector);
        arma::vec *normalized_residual = new arma::vec();
        *normalized_residual = first_vector / arma::norm(first_vector, 2);

        // Grow the basis set.
        collected_basis_set->push_back(normalized_residual);

        // Draw a new candidate vector.
        candidate_basis_set.col(0, &candidate_vector);
        ApplyGaussianNoise_(input, 1, &candidate_vector);
        ComputeResidual_(collected_basis_set, &candidate_vector);
        for(int i = 1; i < rank; i++) {
          double dot_product =
            arma::dot(
              *normalized_residual, candidate_basis_set.col(i));
          candidate_basis_set.col(i) =
            candidate_basis_set.col(i) - dot_product * (*normalized_residual);
        }

        // Shift the starting index by one.
        candidate_basis_set.ShiftStartingIndex();

        // Update the max L2 norm.
        max_l2_norm = MaxL2Norm_(candidate_basis_set);

      } // end of the while-loop.

      // Copy out the basis set.
      output->zeros(input.n_rows, collected_basis_set.size());
      for(unsigned int i = 0; i < collected_basis_set.size(); i++) {
        output->col(i) = * (collected_basis_set[i]) ;
      }
    }
};
}
}

#endif
