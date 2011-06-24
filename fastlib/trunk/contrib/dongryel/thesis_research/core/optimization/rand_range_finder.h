/** @file rand_range_finder.h
 *
 *  Implements Algorithm 4.2 and Algorithm 4.4 in "Finding Structure
 *  With Randomness: Probabilistic Algorithms for Constructing
 *  Approximate Matrix Decompositions" by Halko et al.
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

    static void ModifiedGramSchmidt_(
      const arma::mat &input, arma::mat *basis_out) {

      basis_out->zeros(input.n_rows, input.n_cols);
      for(unsigned int i = 0; i < input.n_cols; i++) {
        basis_out->col(i) = input.col(i);
        for(unsigned int j = 0; j < i; j++) {
          double dot_product =
            arma::dot(basis_out->col(i), basis_out->col(j));
          basis_out->col(i) -= dot_product * basis_out->col(j);
        }
        double vector_norm = arma::norm(basis_out->col(i), 2);
        if(vector_norm > 1.0e-6) {
          basis_out->col(i) /= vector_norm;
        }
        else {
          basis_out->col(i) = arma::zeros<arma::vec>(input.n_rows);
        }
      }
    }

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

    /** @brief Projects the data matrix onto a random set of Gaussian
     *         noise vectors.
     */
    static void ApplyGaussianNoise_(
      const arma::mat &input, int rank,
      arma::mat *output) {

      output->zeros(input.n_rows, rank);
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

    static void Compute(
      const arma::mat &input, int rank, int num_iterations_in,
      arma::mat *q_factor_out) {

      // Apply the Gaussian noise.
      arma::mat noise_applied_input;
      ApplyGaussianNoise_(
        input, rank, &noise_applied_input);

      // Initial Q and R factor.
      ModifiedGramSchmidt_(noise_applied_input, q_factor_out);

      // Apply some number of iterations.
      for(int i = 0; i < num_iterations_in; i++) {
        arma::mat tmp_product = arma::trans(input) * (*q_factor_out);
        ModifiedGramSchmidt_(tmp_product, q_factor_out);
        tmp_product = input * (* q_factor_out);
        ModifiedGramSchmidt_(tmp_product, q_factor_out);
      }
    }
};
}
}

#endif
