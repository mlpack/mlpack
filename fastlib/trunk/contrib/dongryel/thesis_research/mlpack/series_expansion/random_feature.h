/** @file random_feature.h
 *
 *  An implementation of Rahimi's random feature extraction.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_RANDOM_FEATURE_H
#define MLPACK_SERIES_EXPANSION_RANDOM_FEATURE_H

#include <vector>
#include "core/monte_carlo/mean_variance_pair_matrix.h"

namespace mlpack {
namespace series_expansion {
class RandomFeature {
  public:

    template<typename KernelType, typename TreeIteratorType>
    static void EvaluateAverageField(
      const KernelType &kernel_in,
      TreeIteratorType &rnode_it,
      TreeIteratorType &qnode_it,
      int num_random_fourier_features,
      std::vector<core::monte_carlo::MeanVariancePair> *kernel_sums) {

      // The list of random Fourier features drawn in this round.
      std::vector< arma::vec > random_variates(num_random_fourier_features);
      for(int i = 0; i < num_random_fourier_features; i++) {

        // Draw a random Fourier feature.
        kernel_in.DrawRandomVariate(
          rnode_it.table()->n_attributes(), & random_variates[i]);
      }

      // Compute the sum of Fourier component projections of reference
      // set.
      arma::vec sum_reference_projections;
      sum_reference_projections.zeros(2);

      for(int j = 0; j < num_random_fourier_features; j++) {

        // First compute the sum of the projections for the reference
        // node for the current random Fourier feature.
        rnode_it.Reset();
        sum_reference_projections.zeros();
        while(rnode_it.HasNext()) {
          arma::vec reference_point;
          rnode_it.Next(&reference_point);
          double dot_product = arma::dot(random_variates[j], reference_point);
          sum_reference_projections[0] += cos(dot_product);
          sum_reference_projections[1] += sin(dot_product);
        }
        sum_reference_projections /= static_cast<double>(rnode_it.count());

        // Compute the projection of each query point.
        qnode_it.Reset();
        while(qnode_it.HasNext()) {
          arma::vec query_point;
          int query_point_index;
          qnode_it.Next(&query_point, &query_point_index);
          double dot_product = arma::dot(random_variates[j], query_point);
          double contribution =
            cos(dot_product) * sum_reference_projections[0] +
            sin(dot_product) * sum_reference_projections[1];
          (*kernel_sums)[query_point_index].push_back(contribution);
        }

      } // end of looping over each random Fourier feature.
    }

    /** @brief Generates a random Fourier features from the given
     *         table and rotates it by a given set of eigenvectors.
     */
    template<typename TableType>
    static void AccumulateRotationTransform(
      const TableType &table_in,
      const core::table::DenseMatrix &covariance_eigenvectors,
      const std::vector< arma::vec > &random_variates,
      core::monte_carlo::MeanVariancePairMatrix *accumulants) {

      int num_random_fourier_features = random_variates.size();
      double normalization_factor = 1.0 / sqrt(num_random_fourier_features);
      arma::vec tmp_coordinate;
      for(int i = 0; i < table_in.n_entries(); i++) {
        arma::vec old_point;
        table_in.get(i, &old_point);
        tmp_coordinate.zeros(covariance_eigenvectors.n_cols());

        for(int j = 0; j < num_random_fourier_features; j++) {
          double dot_product = arma::dot(random_variates[j], old_point);
          double first_value = cos(dot_product) * normalization_factor;
          double second_value = sin(dot_product) * normalization_factor;

          // For each column of eigenvectors,
          for(int k = 0; k < covariance_eigenvectors.n_cols(); k++) {
            tmp_coordinate[k] +=
              covariance_eigenvectors.get(j, k) * first_value +
              covariance_eigenvectors.get(
                j + num_random_fourier_features, k) * second_value;
          }
        }
        for(int k = 0; k < covariance_eigenvectors.n_cols(); k++) {
          accumulants->get(k, i).push_back(tmp_coordinate[k]);
        }
      }
    }

    template<typename TableType>
    static void CovarianceTransform(
      const TableType &table_in,
      bool do_centering,
      const core::monte_carlo::MeanVariancePairMatrix &global_mean,
      const std::vector< arma::vec > &random_variates,
      core::monte_carlo::MeanVariancePairMatrix *covariance_transformation,
      core::table::DenseMatrix *table_projections) {

      // Allocate the projection matrix.
      table_projections->Init(
        2 * random_variates.size(), table_in.n_entries());

      int num_random_fourier_features = random_variates.size();
      double normalization_factor = 1.0 / sqrt(num_random_fourier_features);
      covariance_transformation->Init(
        2 * num_random_fourier_features, 2 * num_random_fourier_features);
      covariance_transformation->set_total_num_terms(table_in.n_entries());

      for(int i = 0; i < table_in.n_entries(); i++) {
        arma::vec old_point;
        table_in.get(i , &old_point);
        for(int j = 0; j < num_random_fourier_features; j++) {
          double dot_product = arma::dot(random_variates[j], old_point);
          double first_correction_factor =
            (do_centering) ? global_mean.get(0, j).sample_mean() : 0.0;
          double second_correction_factor =
            (do_centering) ?
            global_mean.get(
              0, j + num_random_fourier_features).sample_mean() : 0.0;
          table_projections->set(
            j, i, cos(dot_product) * normalization_factor -
            first_correction_factor);
          table_projections->set(
            j + num_random_fourier_features, i,
            sin(dot_product) * normalization_factor - second_correction_factor);
        }

        // Now Accumulate the covariance.
        for(int k = 0; k < table_projections->n_rows(); k++) {
          for(int j = 0; j < table_projections->n_rows(); j++) {
            covariance_transformation->get(j, k).push_back(
              table_projections->get(j, i) *
              table_projections->get(k, i));
          }
        }
      }
    }

    /** @brief Computes an expected random Fourier feature, normalized
     *         in the dot product sense.
     */
    template<typename TableType>
    static void NormalizedAverageTransform(
      const TableType &table_in,
      const std::vector< arma::vec > &random_variates,
      core::monte_carlo::MeanVariancePairMatrix *average_transformation) {

      int num_random_fourier_features = random_variates.size();
      double normalization_factor = 1.0 / sqrt(num_random_fourier_features);
      average_transformation->Init(1, 2 * num_random_fourier_features);
      average_transformation->set_total_num_terms(table_in.n_entries());

      for(int i = 0; i < table_in.n_entries(); i++) {
        arma::vec old_point;
        table_in.get(i, &old_point);
        for(int j = 0; j < num_random_fourier_features; j++) {
          double dot_product = arma::dot(random_variates[j], old_point);
          average_transformation->get(0, j).push_back(
            cos(dot_product) * normalization_factor);
          average_transformation->get(
            0, j + num_random_fourier_features).push_back(
              sin(dot_product) * normalization_factor);
        }
      }
    }

    /** @brief Computes an expected random Fourier feature, where it
     *         runs over a sample of points weighted by a set of
     *         weights.
     */
    template<typename TableType>
    static void WeightedAverageTransform(
      const TableType &table_in,
      const core::table::DenseMatrix &weights_in,
      int num_reference_samples,
      const std::vector< arma::vec > &random_variates,
      std::vector<int> *random_combination,
      core::monte_carlo::MeanVariancePairMatrix *average_transformation) {

      int num_random_fourier_features = random_variates.size();
      average_transformation->Init(
        weights_in.n_rows(), 2 * num_random_fourier_features);
      average_transformation->set_total_num_terms(table_in.n_entries());

      // Generate a random combination.
      core::math::RandomCombination(
        0, table_in.n_entries(), num_reference_samples, random_combination);

      for(unsigned int i = 0; i < random_combination->size(); i++) {
        arma::vec old_point;
        table_in.get((*random_combination)[i] , &old_point);
        for(int j = 0; j < num_random_fourier_features; j++) {
          double dot_product = arma::dot(random_variates[j], old_point);
          for(int k = 0; k < weights_in.n_rows(); k++) {
            double weight = weights_in.get(k, (*random_combination)[i]);
            average_transformation->get(k, j).push_back(
              weight * cos(dot_product));
            average_transformation->get(
              k, j + num_random_fourier_features).push_back(
                weight * sin(dot_product));
          }
        }
      }
    }

    template<typename TableType>
    static void SumTransform(
      const TableType &table_in,
      const std::vector< arma::vec > &random_variates,
      core::table::DensePoint *sum_transformations) {

      int num_random_fourier_features = random_variates.size();
      sum_transformations->Init(2 * num_random_fourier_features);
      sum_transformations->SetZero();

      for(int i = 0; i < table_in.n_entries(); i++) {
        arma::vec old_point;
        table_in.get(i, &old_point);
        for(int j = 0; j < num_random_fourier_features; j++) {
          double dot_product = arma::dot(random_variates[j], old_point);
          (*sum_transformations)[j] += cos(dot_product);
          (*sum_transformations)[j + num_random_fourier_features] +=
            sin(dot_product);
        }
      }
    }

    static void Transform(
      const arma::vec &point_in,
      const std::vector< arma::vec > &random_variates,
      arma::vec *point_out) {

      int num_random_fourier_features = random_variates.size();
      point_out->set_size(2 * num_random_fourier_features);
      for(int j = 0; j < num_random_fourier_features; j++) {
        double dot_product = arma::dot(random_variates[j], point_in);
        (*point_out)[j] = cos(dot_product);
        (*point_out)[j + num_random_fourier_features] = sin(dot_product);
      }
    }

    template<typename TableType, typename PointType>
    static void Transform(
      const TableType &table_in,
      const std::vector< PointType > &random_variates,
      bool normalize,
      TableType *table_out) {

      // The normalization factor.
      int num_random_fourier_features = random_variates.size();
      table_out->Init(2 * num_random_fourier_features, table_in.n_entries());
      double normalization_factor =
        (normalize) ? (1.0 / sqrt(num_random_fourier_features)) : 1.0;

      for(int i = 0; i < table_in.n_entries(); i++) {
        arma::vec old_point;
        table_in.get(i, &old_point);
        arma::vec new_point;
        table_out->get(i, &new_point);
        for(int j = 0; j < num_random_fourier_features; j++) {
          double dot_product = arma::dot(random_variates[j], old_point);
          new_point[j] = cos(dot_product) * normalization_factor;
          new_point[j + num_random_fourier_features] =
            sin(dot_product) * normalization_factor;
        }
      }
    }

    template<typename TableType, typename KernelType>
    static void Transform(
      const TableType &table_in,
      const KernelType &kernel_in,
      int num_random_fourier_features,
      TableType *table_out) {

      // The dimensionality of the new table is twice the requested
      // number of random fourier features (cosine and sine bases).
      std::vector< arma::vec > random_variates(num_random_fourier_features);
      for(int i = 0; i < num_random_fourier_features; i++) {

        // Draw a random Fourier feature.
        kernel_in.DrawRandomVariate(
          table_in.n_attributes(), & random_variates[i]);
      }

      // Compute the features.
      Transform(
        table_in, random_variates, true, table_out, (arma::vec *) NULL);
    }
};
}
}

#endif
