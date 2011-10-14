/** @file random_feature.h
 *
 *  An implementation of Rahimi's random feature extraction.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_RANDOM_FEATURE_H
#define MLPACK_SERIES_EXPANSION_RANDOM_FEATURE_H

#include <boost/scoped_array.hpp>
#include <omp.h>
#include <vector>
#include "core/monte_carlo/mean_variance_pair_matrix.h"

namespace mlpack {
namespace series_expansion {

template<typename TableType>
class RandomFeature {
  private:

    struct RandomFeatureArgument {
      int begin_;
      int end_;
      const TableType *table_;
      bool do_centering_;
      const core::monte_carlo::MeanVariancePairMatrix *global_mean_;
      const boost::scoped_array< arma::vec > *random_variates_;
      core::monte_carlo::MeanVariancePairMatrix *covariance_transformation_;
      core::monte_carlo::MeanVariancePairMatrix *average_transformation_;
      arma::mat *table_projections_;
      int num_random_fourier_features_;

      RandomFeatureArgument() {
        begin_ = 0;
        end_ = 0;
        table_ = NULL;
        do_centering_ = false;
        global_mean_ = NULL;
        random_variates_ = NULL;
        covariance_transformation_ = NULL;
        average_transformation_ = NULL;
        table_projections_ = NULL;
        num_random_fourier_features_ = 0;
      }

      void NormalizedAverageTransformInit_(
        int begin,
        int end,
        const TableType &table_in,
        const boost::scoped_array<arma::vec> &random_variates,
        int num_random_fourier_features,
        core::monte_carlo::MeanVariancePairMatrix *average_transformation) {

        begin_ = begin;
        end_ = end;
        table_ = &table_in;
        random_variates_ = &random_variates;
        num_random_fourier_features_ = num_random_fourier_features;
        average_transformation_ = average_transformation;
      }

      void CovarianceTransformInit_(
        int begin,
        int end,
        const TableType &table_in,
        bool do_centering,
        const core::monte_carlo::MeanVariancePairMatrix &global_mean,
        const boost::scoped_array< arma::vec > &random_variates,
        int num_random_fourier_features,
        core::monte_carlo::MeanVariancePairMatrix *covariance_transformation,
        arma::mat *table_projections) {

        begin_ = begin;
        end_ = end;
        table_ = &table_in;
        do_centering_ = do_centering;
        global_mean_ = &global_mean;
        random_variates_ = &random_variates;
        num_random_fourier_features_ = num_random_fourier_features;
        covariance_transformation_ = covariance_transformation;
        table_projections_ = table_projections;
      }
    };

  private:

    static void *CovarianceTransform_(void *args_in) {

      RandomFeatureArgument &args =
        *(static_cast<RandomFeatureArgument *>(args_in));
      double normalization_factor =
        1.0 / sqrt(args.num_random_fourier_features_);
      args.covariance_transformation_->Init(
        2 * args.num_random_fourier_features_,
        2 * args.num_random_fourier_features_);
      args.covariance_transformation_->set_total_num_terms(
        args.end_ - args.begin_);

      for(int i = args.begin_; i < args.end_; i++) {
        arma::vec old_point;
        args.table_->get(i , &old_point);
        for(int j = 0; j < args.num_random_fourier_features_; j++) {
          double dot_product =
            arma::dot((*args.random_variates_)[j], old_point);
          double first_correction_factor =
            (args.do_centering_) ?
            args.global_mean_->get(0, j).sample_mean() : 0.0;
          double second_correction_factor =
            (args.do_centering_) ?
            args.global_mean_->get(
              0, j + args.num_random_fourier_features_).sample_mean() : 0.0;
          args.table_projections_->at(j, i) =
            cos(dot_product) * normalization_factor - first_correction_factor;
          args.table_projections_->at(
            j + args.num_random_fourier_features_, i) =
              sin(dot_product) * normalization_factor - second_correction_factor;
        }

        // Now Accumulate the covariance.
        for(unsigned int k = 0; k < args.table_projections_->n_rows; k++) {
          for(unsigned int j = 0; j < args.table_projections_->n_rows; j++) {
            args.covariance_transformation_->get(j, k).push_back(
              args.table_projections_->at(j, i) *
              args.table_projections_->at(k, i));
          }
        }
      }
      return NULL;
    }

    /** @brief Computes an expected random Fourier feature, normalized
     *         in the dot product sense.
     */
    static void *NormalizedAverageTransform_(void *args_in) {

      RandomFeatureArgument &args =
        *(static_cast<RandomFeatureArgument *>(args_in));
      double normalization_factor =
        1.0 / sqrt(args.num_random_fourier_features_);
      args.average_transformation_->Init(
        1, 2 * args.num_random_fourier_features_);
      args.average_transformation_->set_total_num_terms(
        args.end_ - args.begin_);

      for(int i = args.begin_; i < args.end_; i++) {
        arma::vec old_point;
        args.table_->get(i, &old_point);
        for(int j = 0; j < args.num_random_fourier_features_; j++) {
          double dot_product =
            arma::dot((*args.random_variates_)[j], old_point);
          args.average_transformation_->get(0, j).push_back(
            cos(dot_product) * normalization_factor);
          args.average_transformation_->get(
            0, j + args.num_random_fourier_features_).push_back(
              sin(dot_product) * normalization_factor);
        }
      }
      return NULL;
    }

  public:

    /** @brief Computes an expected random Fourier feature, normalized
     *         in the dot product sense.
     */
    static void ThreadedNormalizedAverageTransform(
      int num_threads,
      const TableType &table_in,
      const boost::scoped_array< arma::vec > &random_variates,
      int num_random_fourier_features,
      core::monte_carlo::MeanVariancePairMatrix *average_transformation_in) {

      // Allocate the projection matrix.
      average_transformation_in->Init(
        1, 2 * num_random_fourier_features);

      // Basically, store sub-results and combine them later after all
      // threads are joined.
      std::vector < RandomFeatureArgument > tmp_arguments(num_threads);
      boost::scoped_array<core::monte_carlo::MeanVariancePairMatrix>
      sub_average_transformations(
        new core::monte_carlo::MeanVariancePairMatrix[num_threads]);

      // The grain size per thread.
      int grain_size = table_in.n_entries() / num_threads;

      // OpenMP parallel region.
      #pragma omp parallel
      {
        int i = omp_get_thread_num();
        int begin = i * grain_size;
        int end = (i < num_threads - 1) ?
                  (i + 1) * grain_size : table_in.n_entries();
        tmp_arguments[i].NormalizedAverageTransformInit_(
          begin, end, table_in, random_variates,
          num_random_fourier_features, &(sub_average_transformations[i]));
        mlpack::series_expansion::RandomFeature <
        TableType >::NormalizedAverageTransform_(&tmp_arguments[i]);
      }

      // By here, all threads have exited.
      for(int i = 0; i < num_threads; i++) {
        average_transformation_in->CombineWith(
          sub_average_transformations[i]);
      }
    }

    /** @brief A private function for launching threads.
     */
    static void ThreadedCovarianceTransform(
      int num_threads,
      const TableType &table_in,
      bool do_centering,
      const core::monte_carlo::MeanVariancePairMatrix &global_mean,
      const boost::scoped_array< arma::vec > &random_variates,
      int num_random_fourier_features,
      core::monte_carlo::MeanVariancePairMatrix *covariance_transformation,
      arma::mat *table_projections) {

      // Allocate the projection matrix.
      table_projections->set_size(
        2 * num_random_fourier_features, table_in.n_entries());

      // Basically, store sub-results and combine them later after all
      // threads are joined.
      std::vector <RandomFeatureArgument > tmp_arguments(num_threads);
      boost::scoped_array<core::monte_carlo::MeanVariancePairMatrix>
      sub_covariance_transformations(
        new core::monte_carlo::MeanVariancePairMatrix[num_threads]);

      // The block size.
      int grain_size = table_in.n_entries() / num_threads;

      #pragma omp parallel
      {
        int i = omp_get_thread_num();
        int begin = i * grain_size;
        int end = (i < num_threads - 1) ?
                  (i + 1) * grain_size : table_in.n_entries();
        tmp_arguments[i].CovarianceTransformInit_(
          begin, end, table_in, do_centering,
          global_mean, random_variates, num_random_fourier_features,
          &(sub_covariance_transformations[i]),
          table_projections);
        mlpack::series_expansion::RandomFeature <
        TableType >::CovarianceTransform_(&tmp_arguments[i]);
      }

      // By here, all threads have exited.
      covariance_transformation->Init(
        sub_covariance_transformations[0].n_rows(),
        sub_covariance_transformations[0].n_cols());
      for(int i = 0; i < num_threads; i++) {
        covariance_transformation->CombineWith(
          sub_covariance_transformations[i]);
      }
    }

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
    static void AccumulateRotationTransform(
      const TableType &table_in,
      const arma::mat &covariance_eigenvectors,
      const boost::scoped_array< arma::vec > &random_variates,
      int num_random_fourier_features,
      core::monte_carlo::MeanVariancePairMatrix *accumulants) {

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
          for(unsigned int k = 0; k < covariance_eigenvectors.n_cols; k++) {
            tmp_coordinate[k] +=
              covariance_eigenvectors.at(j, k) * first_value +
              covariance_eigenvectors.at(
                j + num_random_fourier_features, k) * second_value;
          }
        }
        for(unsigned int k = 0; k < covariance_eigenvectors.n_cols; k++) {
          accumulants->get(k, i).push_back(tmp_coordinate[k]);
        }
      }
    }

    /** @brief Computes an expected random Fourier feature, where it
     *         runs over a sample of points weighted by a set of
     *         weights.
     */
    static void WeightedAverageTransform(
      const TableType &table_in,
      const arma::mat &weights_in,
      int num_reference_samples,
      const boost::scoped_array< arma::vec > &random_variates,
      int num_random_fourier_features,
      const core::monte_carlo::MeanVariancePairMatrix &global_mean,
      std::vector<int> *random_combination,
      core::monte_carlo::MeanVariancePairMatrix *average_transformation) {

      average_transformation->Init(
        weights_in.n_rows, 2 * num_random_fourier_features);
      average_transformation->set_total_num_terms(table_in.n_entries());

      // Generate a random combination.
      core::math::RandomCombination(
        0, table_in.n_entries(), num_reference_samples, random_combination);

      for(unsigned int i = 0; i < random_combination->size(); i++) {
        arma::vec old_point;
        table_in.get((*random_combination)[i] , &old_point);
        for(int j = 0; j < num_random_fourier_features; j++) {
          double dot_product = arma::dot(random_variates[j], old_point);
          for(unsigned int k = 0; k < weights_in.n_rows; k++) {
            double weight = weights_in.at(k, (*random_combination)[i]);
            average_transformation->get(k, j).push_back(
              weight * (
                cos(dot_product) - global_mean.get(0, j).sample_mean()));
            average_transformation->get(
              k, j + num_random_fourier_features).push_back(
                weight *
                (sin(dot_product) -
                 global_mean.get(
                   0, j + num_random_fourier_features).sample_mean())) ;
          }
        }
      }
    }

    static void SumTransform(
      const TableType &table_in,
      const boost::scoped_array< arma::vec > &random_variates,
      int num_random_fourier_features,
      arma::vec *sum_transformations) {

      sum_transformations->zeros(2 * num_random_fourier_features);

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
      const boost::scoped_array< arma::vec > &random_variates,
      int num_random_fourier_features,
      arma::vec *point_out) {

      point_out->set_size(2 * num_random_fourier_features);
      for(int j = 0; j < num_random_fourier_features; j++) {
        double dot_product = arma::dot(random_variates[j], point_in);
        (*point_out)[j] = cos(dot_product);
        (*point_out)[j + num_random_fourier_features] = sin(dot_product);
      }
    }

    template<typename PointType>
    static void Transform(
      const TableType &table_in,
      const boost::scoped_array< PointType > &random_variates,
      int num_random_fourier_features,
      bool normalize,
      TableType *table_out) {

      // The normalization factor.
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

    template<typename KernelType>
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
