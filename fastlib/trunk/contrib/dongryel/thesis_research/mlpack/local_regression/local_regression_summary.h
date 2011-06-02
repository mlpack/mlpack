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

namespace mlpack {
namespace local_regression {

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
      this->SetZero();
      pruned_l_ = initial_pruned_in;
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
      TreeType *qnode, TreeType *rnode,
      double failure_probability, ResultType *query_results) const {

      return false;
    }

    template < typename GlobalType, typename DeltaType, typename TreeType,
             typename ResultType >
    bool CanSummarize(
      const GlobalType &global, DeltaType &delta,
      const core::math::Range &squared_distance_range,
      TreeType *qnode, TreeType *rnode, ResultType *query_results) const {

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
        rnode->count() * (
          global.adjusted_relative_error() * lower_bound_l1_norm_for_left +
          global.effective_num_reference_points() * global.absolute_error() -
          left_hand_side_used_error_u_) /
        static_cast<double>(
          global.effective_num_reference_points() - pruned_l_);
      double right_hand_side_for_right =
        rnode->count() * (
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
