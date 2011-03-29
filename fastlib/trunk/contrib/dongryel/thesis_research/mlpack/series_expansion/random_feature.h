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

    template<typename TableType>
    static void SumTransform(
      const TableType &table_in,
      const std::vector< core::table::DensePoint > &random_variates,
      core::table::DensePoint *sum_transformations) {

      int num_random_fourier_features = random_variates.size();
      sum_transformations->Init(2 * num_random_fourier_features);
      sum_transformations->SetZero();

      // Build aliases.
      std::vector< arma::vec > random_variate_aliases;
      for(unsigned int i = 0; i < random_variates.size(); i++) {
        core::table::DensePointToArmaVec(
          random_variates[i], &(random_variate_aliases[i]));
      }

      for(int i = 0; i < table_in.n_entries(); i++) {
        arma::vec old_point;
        table_in.get(i, &old_point);
        for(int j = 0; j < num_random_fourier_features; j++) {
          double dot_product = arma::dot(random_variate_aliases[j], old_point);
          (*sum_transformations)[j] += cos(dot_product);
          (*sum_transformations)[j + num_random_fourier_features] +=
            sin(dot_product);
        }
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
