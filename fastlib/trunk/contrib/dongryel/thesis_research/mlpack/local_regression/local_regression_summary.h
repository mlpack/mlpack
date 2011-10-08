/** @file local_regression_summary.h
 *
 *  The summary results that must be tested for pruning in local
 *  regression dual-tree algorithm. These are also stored in the query
 *  nodes.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_SUMMARY_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_SUMMARY_H

#include "mlpack/local_regression/local_regression_sampling.h"

namespace mlpack {
namespace local_regression {

template<typename KernelType>
class CanProbabilisticSummarizeTrait {
  public:

    template<typename MetricType, typename GlobalType, typename TreeType>
    static bool QuickTest(
      const MetricType &metric_in,
      const GlobalType &global_in,
      const core::math::Range &squared_distance_range,
      TreeType *rnode) {
      return false;
    }
};

template<>
class CanProbabilisticSummarizeTrait<core::metric_kernels::GaussianKernel> {
  public:
    template<typename MetricType, typename GlobalType, typename TreeType>
    static bool QuickTest(
      const MetricType &metric_in,
      const GlobalType &global_in,
      const core::math::Range &squared_distance_range,
      TreeType *rnode) {

      double mid_distance =
        0.5 * (
          sqrt(squared_distance_range.lo) + sqrt(squared_distance_range.hi));

      return fabs(mid_distance - sqrt(global_in.kernel().bandwidth_sq())) <=
             0.5 * mid_distance &&
             rnode->count() >= GlobalType::min_sampling_threshold;
    }
};

/** @brief The summary statistics for the local regression object.
 */
class LocalRegressionSummary {

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

  public:

    arma::mat left_hand_side_l_;

    arma::mat left_hand_side_u_;

    arma::vec right_hand_side_l_;

    arma::vec right_hand_side_u_;

    double pruned_l_;

    double left_hand_side_used_error_u_;

    double right_hand_side_used_error_u_;

    void Seed(double initial_pruned_in) {
      pruned_l_ += initial_pruned_in;
    }

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & left_hand_side_l_;
      ar & left_hand_side_u_;
      ar & right_hand_side_l_;
      ar & right_hand_side_u_;
      ar & pruned_l_;
      ar & left_hand_side_used_error_u_;
      ar & right_hand_side_used_error_u_;
    }

    void Copy(const LocalRegressionSummary &summary_in) {
      left_hand_side_l_ = summary_in.left_hand_side_l_;
      left_hand_side_u_ = summary_in.left_hand_side_u_;
      right_hand_side_l_ = summary_in.right_hand_side_l_;
      right_hand_side_u_ = summary_in.right_hand_side_u_;
      pruned_l_ = summary_in.pruned_l_;
      left_hand_side_used_error_u_ = summary_in.left_hand_side_used_error_u_;
      right_hand_side_used_error_u_ = summary_in.right_hand_side_used_error_u_;
    }

    LocalRegressionSummary() {
      this->SetZero();
    }

    LocalRegressionSummary(const LocalRegressionSummary &summary_in) {
      this->Copy(summary_in);
    }

    template < typename MetricType, typename GlobalType,
             typename PostponedType, typename DeltaType,
             typename TreeType, typename ResultType >
    bool CanProbabilisticSummarize(
      const MetricType &metric,
      GlobalType &global,
      const PostponedType &postponed, DeltaType &delta,
      const core::math::Range &squared_distance_range,
      TreeType *qnode, int qnode_rank, TreeType *rnode, int rnode_rank,
      bool qnode_and_rnode_are_equal,
      double failure_probability, ResultType *query_results) const {

      // If there is a sufficient overlap for the bandwidth and max
      // node distance, do Monte Carlo.
      if(! CanProbabilisticSummarizeTrait <
          typename GlobalType::KernelType >::QuickTest(
            metric, global, squared_distance_range, rnode)) {
        return false;
      }

      // The number of standard deviations corresponding to the
      // failure probability.
      double num_standard_deviations = global.compute_quantile(
                                         failure_probability);

      // The iterators for the query node and the reference node.
      typename GlobalType::TableType::TreeIterator qnode_it =
        global.query_table()->get_node_iterator(qnode);
      typename GlobalType::TableType::TreeIterator rnode_it =
        global.reference_table()->get_node_iterator(rnode);

      // Declare the sampling object.
      LocalRegressionSampling <
      DeltaType, mlpack::local_regression::LocalRegressionSummary > sampling;
      sampling.Init(global, delta, qnode_it);

      // More temporary variables.
      arma::vec random_variate;

      // Repeat until all queries are converged.
      do {

        for(int f = 0; f < global.num_random_features(); f++) {

          // Reset the accumulants for every random feature.
          sampling.Reset(delta);

          // Generate a random feature.
          global.kernel().DrawRandomVariate(
            rnode_it.table()->n_attributes(), & random_variate);

          // Now loop over each query point and accumulate the averages.
          sampling.AccumulateContributions(
            global, random_variate, qnode_it, rnode_it,
            num_standard_deviations);

        } // end of looping over each random Fourier feature.

        // If converged, break.
        if(sampling.Converged(
              global, postponed, delta, squared_distance_range,
              qnode, qnode_rank, rnode, rnode_rank,
              qnode_and_rnode_are_equal, query_results, qnode_it,
              num_standard_deviations)) {
          break;
        }
      }
      while(true);

      // Set the delta to point to the query deltas.
      delta.query_deltas_ = &(global.query_deltas());
      qnode_it.Reset();
      do {

        // Set the correct number of terms for each query.
        int qpoint_id;
        qnode_it.Next(&qpoint_id);
        (* delta.query_deltas_)[qpoint_id].set_total_num_terms(
          static_cast<int>(delta.pruned_));
      }
      while(qnode_it.HasNext());

      return true;
    }

    template < typename GlobalType, typename DeltaType, typename TreeType,
             typename ResultType >
    bool CanSummarize(
      const GlobalType &global, DeltaType &delta,
      const core::math::Range &squared_distance_range,
      TreeType *qnode, int qnode_rank, TreeType *rnode, int rnode_rank,
      bool qnode_and_rnode_are_equal,
      ResultType *query_results) const {

      double left_hand_side_for_left = delta.left_hand_side_used_error_;
      double left_hand_side_for_right = delta.right_hand_side_used_error_;
      double lower_bound_l1_norm_for_left = 0.0;
      double lower_bound_l1_norm_for_right = 0.0;
      for(unsigned int j = 0; j < left_hand_side_l_.n_cols; j++) {
        lower_bound_l1_norm_for_right += right_hand_side_l_[j];
        for(unsigned int i = 0; i < left_hand_side_l_.n_rows; i++) {
          lower_bound_l1_norm_for_left += left_hand_side_l_.at(i, j);
        }
      }

      double right_hand_side_for_left =
        delta.pruned_ * (
          global.adjusted_relative_error() * lower_bound_l1_norm_for_left +
          global.effective_num_reference_points() * global.absolute_error() -
          left_hand_side_used_error_u_) /
        static_cast<double>(
          global.effective_num_reference_points() - pruned_l_);
      double right_hand_side_for_right =
        delta.pruned_ * (
          global.adjusted_relative_error() * lower_bound_l1_norm_for_right +
          global.effective_num_reference_points() * global.absolute_error() -
          right_hand_side_used_error_u_) /
        static_cast<double>(
          global.effective_num_reference_points() - pruned_l_);

      // Prunable by finite-difference.
      return left_hand_side_for_left <= right_hand_side_for_left &&
             left_hand_side_for_right <= right_hand_side_for_right;
    }

    /** @brief Initializes the postponed quantities with the given
     *         dimension.
     */
    template<typename GlobalType>
    void Init(const GlobalType &global_in) {
      left_hand_side_l_.zeros(
        global_in.problem_dimension(),
        global_in.problem_dimension());
      left_hand_side_u_.zeros(
        global_in.problem_dimension(),
        global_in.problem_dimension());
      right_hand_side_l_.zeros(global_in.problem_dimension());
      right_hand_side_u_.zeros(global_in.problem_dimension());
      SetZero();
    }

    void SetZero() {
      left_hand_side_l_.zeros();
      left_hand_side_u_.zeros();
      right_hand_side_l_.zeros();
      right_hand_side_u_.zeros();
      pruned_l_ = 0.0;
      left_hand_side_used_error_u_ = 0.0;
      right_hand_side_used_error_u_ = 0.0;
    }

    void Init() {
      SetZero();
    }

    /** @brief Resets the summary statistics so that it can be
     *         re-accumulated.
     */
    void StartReaccumulate() {
      left_hand_side_l_.fill(std::numeric_limits<double>::max());
      left_hand_side_u_.fill(- std::numeric_limits<double>::max());
      right_hand_side_l_.fill(std::numeric_limits<double>::max());
      right_hand_side_u_.fill(- std::numeric_limits<double>::max());
      pruned_l_ = std::numeric_limits<double>::max();
      left_hand_side_used_error_u_ = 0.0;
      right_hand_side_used_error_u_ = 0.0;
    }

    /** @brief Accumulates the given query result into the summary
     *         statistics.
     */
    template<typename GlobalType, typename ResultType>
    void Accumulate(
      const GlobalType &global, const ResultType &results, int q_index) {

      for(unsigned int j = 0; j < left_hand_side_l_.n_cols; j++) {
        right_hand_side_l_[j] =
          std::min(
            right_hand_side_l_[j],
            results.right_hand_side_l_[q_index][j].sample_mean() *
            results.pruned_[q_index]);
        right_hand_side_u_[j] =
          std::max(
            right_hand_side_u_[j],
            results.right_hand_side_u_[q_index][j].sample_mean() *
            results.pruned_[q_index]);
        for(unsigned int i = 0; i < left_hand_side_l_.n_rows; i++) {
          left_hand_side_l_.at(i, j) =
            std::min(
              left_hand_side_l_.at(i, j),
              results.left_hand_side_l_[q_index].get(i, j).sample_mean() *
              results.pruned_[q_index]);
          left_hand_side_u_.at(i, j) =
            std::max(
              left_hand_side_u_.at(i, j),
              results.left_hand_side_u_[q_index].get(i, j).sample_mean() *
              results.pruned_[q_index]);
        }
      }
      pruned_l_ = std::min(pruned_l_, results.pruned_[q_index]);
      left_hand_side_used_error_u_ =
        std::max(
          left_hand_side_used_error_u_,
          results.left_hand_side_used_error_[q_index]);
      right_hand_side_used_error_u_ =
        std::max(
          right_hand_side_used_error_u_,
          results.right_hand_side_used_error_[q_index]);
    }

    template<typename GlobalType, typename LocalRegressionPostponedType>
    void Accumulate(
      const GlobalType &global, const LocalRegressionSummary &summary_in,
      const LocalRegressionPostponedType &postponed_in) {

      for(unsigned int j = 0; j < left_hand_side_l_.n_cols; j++) {
        right_hand_side_l_[j] =
          std::min(
            right_hand_side_l_[j],
            summary_in.right_hand_side_l_[j] +
            postponed_in.right_hand_side_l_[j].sample_mean() *
            postponed_in.pruned_);
        right_hand_side_u_[j] =
          std::max(
            right_hand_side_u_[j],
            summary_in.right_hand_side_u_[j] +
            postponed_in.right_hand_side_u_[j].sample_mean() *
            postponed_in.pruned_);
        for(unsigned int i = 0; i < left_hand_side_l_.n_rows; i++) {
          left_hand_side_l_.at(i, j) =
            std::min(
              left_hand_side_l_.at(i, j),
              summary_in.left_hand_side_l_.at(i, j) +
              postponed_in.left_hand_side_l_.get(i, j).sample_mean() *
              postponed_in.pruned_);
          left_hand_side_u_.at(i, j) =
            std::max(
              left_hand_side_u_.at(i, j),
              summary_in.left_hand_side_u_.at(i, j) +
              postponed_in.left_hand_side_u_.get(i, j).sample_mean() *
              postponed_in.pruned_);
        }
      }
      pruned_l_ = std::min(
                    pruned_l_, summary_in.pruned_l_ + postponed_in.pruned_);
      left_hand_side_used_error_u_ =
        std::max(
          left_hand_side_used_error_u_,
          summary_in.left_hand_side_used_error_u_ +
          postponed_in.left_hand_side_used_error_);
      right_hand_side_used_error_u_ =
        std::max(
          right_hand_side_used_error_u_,
          summary_in.right_hand_side_used_error_u_ +
          postponed_in.right_hand_side_used_error_);
    }

    /** @brief Applies the delta change computed deterministically.
     */
    void ApplyDelta(const LocalRegressionDelta &delta_in) {
      for(unsigned int j = 0; j < left_hand_side_l_.n_cols; j++) {
        right_hand_side_l_[j] +=
          delta_in.right_hand_side_l_[j].sample_mean() * delta_in.pruned_;
        right_hand_side_u_[j] +=
          delta_in.right_hand_side_u_[j].sample_mean() * delta_in.pruned_;
        for(unsigned int i = 0; i < left_hand_side_l_.n_rows; i++) {
          left_hand_side_l_.at(i, j) +=
            delta_in.left_hand_side_l_.get(i, j).sample_mean() *
            delta_in.pruned_;
          left_hand_side_u_.at(i, j) +=
            delta_in.left_hand_side_u_.get(i, j).sample_mean() *
            delta_in.pruned_;
        }
      }
    }

    /** @brief Applies a delta that represents the result of a Monte
     *         Carlo sampling result.
     */
    void ApplyProbabilisticDelta(
      const LocalRegressionDelta &delta_in, double num_standard_deviations) {
      for(unsigned int j = 0; j < left_hand_side_l_.n_cols; j++) {
        core::math::Range delta_right_hand_side;
        delta_in.right_hand_side_e_[j].scaled_interval(
          delta_in.pruned_, num_standard_deviations, &delta_right_hand_side);

        // For the lower bound, apply something larger.
        right_hand_side_l_[j] += delta_in.right_hand_side_e_[j].sample_mean() *
                                 delta_in.pruned_;
        right_hand_side_u_[j] += delta_right_hand_side.hi;
        for(unsigned int i = 0; i < left_hand_side_l_.n_rows; i++) {
          core::math::Range delta_left_hand_side;
          delta_in.left_hand_side_e_.get(i, j).scaled_interval(
            delta_in.pruned_, num_standard_deviations, &delta_left_hand_side);
          delta_left_hand_side.lo = std::max(delta_left_hand_side.lo, 0.0);

          // Same here.
          left_hand_side_l_.at(i, j) +=
            delta_in.left_hand_side_e_.get(i, j).sample_mean() *
            delta_in.pruned_;
          left_hand_side_u_.at(i, j) += delta_left_hand_side.hi;
        }
      }
    }

    template<typename LocalRegressionPostponedType>
    void ApplyPostponed(const LocalRegressionPostponedType &postponed_in) {
      for(unsigned int j = 0; j < left_hand_side_l_.n_cols; j++) {
        right_hand_side_l_[j] +=
          postponed_in.right_hand_side_l_[j].sample_mean() *
          postponed_in.pruned_;
        right_hand_side_u_[j] +=
          postponed_in.right_hand_side_u_[j].sample_mean() *
          postponed_in.pruned_;
        for(unsigned int i = 0; i < left_hand_side_l_.n_rows; i++) {
          left_hand_side_l_.at(i, j) +=
            postponed_in.left_hand_side_l_.get(i, j).sample_mean() *
            postponed_in.pruned_;
          left_hand_side_u_.at(i, j) +=
            postponed_in.left_hand_side_u_.get(i, j).sample_mean() *
            postponed_in.pruned_;
        }
      }
      pruned_l_ = pruned_l_ + postponed_in.pruned_;
      left_hand_side_used_error_u_ =
        left_hand_side_used_error_u_ + postponed_in.left_hand_side_used_error_;
      right_hand_side_used_error_u_ =
        right_hand_side_used_error_u_ +
        postponed_in.right_hand_side_used_error_;
    }
};
}
}

#endif
