/** @file local_regression_delta.h
 *
 *  The delta result that must be tested for pruning in local
 *  regression dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_DELTA_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_DELTA_H

namespace mlpack {
namespace local_regression {

class LocalRegressionDelta {

  public:

    core::monte_carlo::MeanVariancePairMatrix left_hand_side_l_;

    core::monte_carlo::MeanVariancePairMatrix left_hand_side_e_;

    core::monte_carlo::MeanVariancePairMatrix left_hand_side_u_;

    core::monte_carlo::MeanVariancePairVector right_hand_side_l_;

    core::monte_carlo::MeanVariancePairVector right_hand_side_e_;

    core::monte_carlo::MeanVariancePairVector right_hand_side_u_;

    double pruned_;

    double left_hand_side_used_error_;

    double right_hand_side_used_error_;

    /** @brief The pointer to the scratch space for accumulating each
     *         query's Monte Carlo result.
     */
    boost::scoped_array< LocalRegressionDelta > *query_deltas_;

    LocalRegressionDelta() {
      SetZero();
    }

    void SetZero() {
      left_hand_side_l_.SetZero();
      left_hand_side_e_.SetZero();
      left_hand_side_u_.SetZero();
      right_hand_side_l_.SetZero();
      right_hand_side_e_.SetZero();
      right_hand_side_u_.SetZero();
      pruned_ = left_hand_side_used_error_ = right_hand_side_used_error_ = 0.0;
      query_deltas_ = NULL;
    }

    template<typename GlobalType>
    void push_back(
      const GlobalType &global,
      double scaled_cosine_value,
      double scaled_sine_value,
      const std::pair < core::monte_carlo::MeanVariancePairMatrix,
      core::monte_carlo::MeanVariancePairMatrix >
      &avg_left_hand_side_for_reference,
      const std::pair < core::monte_carlo::MeanVariancePairVector,
      core::monte_carlo::MeanVariancePairVector >
      &avg_right_hand_side_for_reference,
      double num_standard_deviations) {

      for(int j = 0; j < global.problem_dimension(); j++) {
        right_hand_side_e_[j].ScaledCombineWith(
          scaled_cosine_value,
          avg_right_hand_side_for_reference.first[j]);
        right_hand_side_e_[j].ScaledCombineWith(
          scaled_sine_value,
          avg_right_hand_side_for_reference.second[j]);

        for(int i = 0; i < global.problem_dimension(); i++) {
          left_hand_side_e_.get(i, j).ScaledCombineWith(
            scaled_cosine_value,
            avg_left_hand_side_for_reference.first.get(i, j));
          left_hand_side_e_.get(i, j).ScaledCombineWith(
            scaled_sine_value,
            avg_left_hand_side_for_reference.second.get(i, j));
        }
      }

      // Set the error as well.
      left_hand_side_used_error_ =
        left_hand_side_e_.max_scaled_deviation(
          pruned_, num_standard_deviations);
      right_hand_side_used_error_ =
        right_hand_side_e_.max_scaled_deviation(
          pruned_, num_standard_deviations);
    }

    template<typename MetricType, typename GlobalType, typename TreeType>
    void DeterministicCompute(
      const MetricType &metric,
      const GlobalType &global, TreeType *qnode, TreeType *rnode,
      const core::math::Range &squared_distance_range) {

      // The maximum deviation between the lower and the upper
      // estimated quantities for the left hand side and the right
      // hand side.
      double left_hand_side_max_deviation = 0.0;
      double right_hand_side_max_deviation = 0.0;

      // Lower and upper bound on the kernels.
      double lower_kernel_value =
        global.kernel().EvalUnnormOnSq(squared_distance_range.hi);
      double upper_kernel_value =
        global.kernel().EvalUnnormOnSq(squared_distance_range.lo);

      // Initialize the left hand sides and the right hand sides.
      int total_num_terms = (global.is_monochromatic() && qnode == rnode) ?
                            rnode->count() - 1 : rnode->count();
      left_hand_side_l_.Init(
        global.problem_dimension() ,
        global.problem_dimension());
      left_hand_side_l_.set_total_num_terms(total_num_terms);
      left_hand_side_e_.Init(
        global.problem_dimension(),
        global.problem_dimension());
      left_hand_side_e_.set_total_num_terms(total_num_terms);
      left_hand_side_u_.Init(
        global.problem_dimension(),
        global.problem_dimension());
      left_hand_side_u_.set_total_num_terms(total_num_terms);
      right_hand_side_l_.Init(global.problem_dimension());
      right_hand_side_l_.set_total_num_terms(total_num_terms);
      right_hand_side_e_.Init(global.problem_dimension());
      right_hand_side_e_.set_total_num_terms(total_num_terms);
      right_hand_side_u_.Init(global.problem_dimension());
      right_hand_side_u_.set_total_num_terms(total_num_terms);

      double kernel_average = 0.5 * (lower_kernel_value + upper_kernel_value);
      for(int j = 0; j < global.problem_dimension(); j++) {
        right_hand_side_l_[j].push_back(
          lower_kernel_value *
          rnode->stat().weighted_average_info_[j].sample_mean());
        right_hand_side_e_[j].push_back(
          kernel_average *
          rnode->stat().weighted_average_info_[j].sample_mean());
        right_hand_side_u_[j].push_back(
          upper_kernel_value *
          rnode->stat().weighted_average_info_[j].sample_mean());
        for(int i = 0; i < global.problem_dimension() ; i++) {
          left_hand_side_l_.get(i, j).push_back(
            lower_kernel_value *
            rnode->stat().average_info_.get(i, j).sample_mean());
          left_hand_side_e_.get(i, j).push_back(
            kernel_average *
            rnode->stat().average_info_.get(i, j).sample_mean());
          left_hand_side_u_.get(i, j).push_back(
            upper_kernel_value *
            rnode->stat().average_info_.get(i, j).sample_mean());
        }
      }

      // Compute the maximum deviation.
      for(int j = 0; j < left_hand_side_l_.n_cols(); j++) {
        right_hand_side_max_deviation =
          std::max(
            right_hand_side_max_deviation,
            upper_kernel_value *
            rnode->stat().max_weighted_average_info_[j] -
            lower_kernel_value *
            rnode->stat().min_weighted_average_info_[j]);
        for(int i = 0; i < left_hand_side_l_.n_rows(); i++) {
          left_hand_side_max_deviation =
            std::max(
              left_hand_side_max_deviation,
              left_hand_side_u_.get(i, j).sample_mean() -
              left_hand_side_l_.get(i, j).sample_mean());
        }
      }

      pruned_ = static_cast<double>(total_num_terms);
      left_hand_side_used_error_ =
        0.5 * left_hand_side_max_deviation * total_num_terms;
      right_hand_side_used_error_ =
        right_hand_side_max_deviation * total_num_terms;
    }
};
}
}

#endif
