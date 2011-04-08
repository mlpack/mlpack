/** @file threaded_convergence.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KPCA_THREADED_CONVERGENCE_H
#define MLPACK_DISTRIBUTED_KPCA_THREADED_CONVERGENCE_H

#include <boost/scoped_array.hpp>
#include <queue>
#include "mlpack/series_expansion/random_feature.h"

namespace mlpack {
namespace distributed_kpca {

template<typename DistributedTableType>
class ThreadedConvergence {

  public:
    typedef typename DistributedTableType::TableType TableType;

  private:

    int begin_;
    int end_;
    double relative_error_;
    double absolute_error_;
    double num_standard_deviations_;
    int num_reference_samples_;
    int num_random_fourier_features_;
    DistributedTableType *reference_table_;
    DistributedTableType *query_table_;
    const core::table::DenseMatrix *weights_;
    core::monte_carlo::MeanVariancePairMatrix *kernel_sums_;
    std::deque<bool> *converged_;
    boost::scoped_array< arma::vec > *random_variate_aliases_;
    core::monte_carlo::MeanVariancePairVector *l1_norm_history_;
    int max_num_iterations_;
    int num_iterations_;
    bool all_local_query_converged_;
    core::monte_carlo::MeanVariancePairMatrix *global_reference_average_;

  private:

    void Check_() {

      all_local_query_converged_ = true;
      for(int i = begin_; i < end_; i++) {

        if((*converged_)[i]) {

          // If already converged, skip.
          continue;
        }

        arma::vec query_point;
        query_table_->local_table()->get(i, &query_point);
        arma::vec query_point_projected;
        mlpack::series_expansion::RandomFeature<TableType>::Transform(
          query_point, *random_variate_aliases_,
          num_random_fourier_features_, &query_point_projected);

        double l1_norm = 0.0;
        for(int k = 0; k < weights_->n_rows(); k++) {
          for(int j = 0; j < num_random_fourier_features_; j++) {

            // You need to multiply by the factor of two since Fourier
            // features come in pairs of cosine and sines.
            kernel_sums_->get(k, i).ScaledCombineWith(
              2.0 * query_point_projected[j],
              global_reference_average_->get(k, j));
            kernel_sums_->get(k, i).ScaledCombineWith(
              2.0 * query_point_projected[j + num_random_fourier_features_],
              global_reference_average_->get(
                k, j + num_random_fourier_features_));
          }

          // Add up the frobenius norm contribution.
          l1_norm += fabs(kernel_sums_->get(k, i).sample_mean());

        } // end of checking the given KPCA component.

        // Add to the history.
        (*l1_norm_history_)[i].push_back(l1_norm);

        // Start checking the convergence after 10 iterations.
        if(num_iterations_ > 10) {
          double left_hand_side =
            num_standard_deviations_ *
            sqrt(
              (*l1_norm_history_)[i].sample_mean_variance());
          double right_hand_side =
            relative_error_ * (*l1_norm_history_)[i].sample_mean() +
            absolute_error_;
          (*converged_)[i] = (left_hand_side <= right_hand_side);
        }
        all_local_query_converged_ =
          (all_local_query_converged_ && (*converged_)[i]) ||
          (num_iterations_ > max_num_iterations_);

      } // end of looping over each local query.
    }

    void Init_(
      int begin,
      int end,
      double relative_error_in,
      double absolute_error_in,
      double num_standard_deviations,
      int num_reference_samples,
      DistributedTableType *reference_table_in,
      DistributedTableType *query_table_in,
      const core::table::DenseMatrix &weights,
      core::monte_carlo::MeanVariancePairMatrix *kernel_sums,
      std::deque<bool> &converged,
      boost::scoped_array< arma::vec > &random_variate_aliases,
      int num_random_fourier_features,
      core::monte_carlo::MeanVariancePairVector &l1_norm_history,
      int num_iterations,
      int max_num_iterations,
      core::monte_carlo::MeanVariancePairMatrix *global_reference_average) {

      begin_ = begin;
      end_ = end;
      relative_error_ = relative_error_in;
      absolute_error_ = absolute_error_in;
      num_standard_deviations_ = num_standard_deviations;
      num_reference_samples_ = num_reference_samples;
      num_random_fourier_features_ = num_random_fourier_features;
      reference_table_ = reference_table_in;
      query_table_ = query_table_in;
      weights_ = &weights;
      kernel_sums_ = kernel_sums;
      converged_ = &converged;
      random_variate_aliases_ = &random_variate_aliases;
      l1_norm_history_ = &l1_norm_history;
      num_iterations_ = num_iterations;
      max_num_iterations_ = max_num_iterations;
      global_reference_average_ = global_reference_average;
    }

  public:

    ThreadedConvergence() {
      begin_ = 0;
      end_ = 0;
      relative_error_ = 0.0;
      absolute_error_ = 0.0;
      num_standard_deviations_ = 0.0;
      num_reference_samples_ = 0;
      num_random_fourier_features_ = 0;
      reference_table_ = NULL;
      query_table_ = NULL;
      weights_ = NULL;
      kernel_sums_ = NULL;
      converged_ = NULL;
      random_variate_aliases_ = NULL;
      l1_norm_history_ = NULL;
      num_iterations_ = 0;
      max_num_iterations_ = 0;
      global_reference_average_ = NULL;
    }

    bool all_local_query_converged() const {
      return all_local_query_converged_;
    }

    static bool ThreadedCheck(
      int num_threads,
      double relative_error_in,
      double absolute_error_in,
      double num_standard_deviations,
      int num_reference_samples,
      DistributedTableType *reference_table_in,
      DistributedTableType *query_table_in,
      const core::table::DenseMatrix &weights,
      core::monte_carlo::MeanVariancePairMatrix *kernel_sums,
      std::deque<bool> &converged,
      boost::scoped_array< arma::vec > &random_variate_aliases,
      int num_random_fourier_features,
      core::monte_carlo::MeanVariancePairVector &l1_norm_history,
      int num_iterations, int max_num_iterations,
      core::monte_carlo::MeanVariancePairMatrix *global_reference_average) {

      // Basically, store sub-results and combine them later after all
      // threads are joined.
      boost::thread_group thread_group;
      std::vector <
      mlpack::distributed_kpca::ThreadedConvergence<DistributedTableType> >
      tmp_objects(num_threads);

      // The block size.
      int grain_size = query_table_in->n_entries() / num_threads;
      for(int i = 0; i < num_threads; i++) {
        int begin = i * grain_size;
        int end = (i < num_threads - 1) ?
                  (i + 1) * grain_size : query_table_in->n_entries();
        tmp_objects[i].Init_(
          begin, end, relative_error_in, absolute_error_in,
          num_standard_deviations, num_reference_samples,
          reference_table_in, query_table_in, weights, kernel_sums, converged,
          random_variate_aliases, num_random_fourier_features,
          l1_norm_history, num_iterations,
          max_num_iterations, global_reference_average);

        if(i == 0) {
          tmp_objects[0].ThreadedConvergence<DistributedTableType>::Check_();
        }
        else {
          thread_group.add_thread(
            new boost::thread(
              &mlpack::distributed_kpca::ThreadedConvergence <
              DistributedTableType >::Check_,
              &tmp_objects[i]));
        }
      }
      thread_group.join_all();

      bool converged_result = true;
      for(int i = 0; converged_result && i < num_threads; i++) {
        converged_result =
          converged_result && tmp_objects[i].all_local_query_converged();
      }
      return converged_result;
    }
};
}
}

#endif
