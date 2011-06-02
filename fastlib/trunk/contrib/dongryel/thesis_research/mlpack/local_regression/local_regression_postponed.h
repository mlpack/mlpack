/** @file local_regression_postponed.h
 *
 *  The postponed quantities in local regression in a dual-tree
 *  algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_POSTPONED_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_POSTPONED_H

namespace mlpack {
namespace local_regression {

/** @brief The postponed quantities for local regression.
 */
class LocalRegressionPostponed {

  private:

    // For boost serialization.
    friend class boost::serialization::access;

  public:

    /** @brief The lower bound on the postponed quantities for the
     *         left hand side.
     */
    core::monte_carlo::MeanVariancePairMatrix left_hand_side_l_;

    /** @brief The finite-difference postponed quantities for the left
     *         hand side.
     */
    core::monte_carlo::MeanVariancePairMatrix left_hand_side_e_;

    /** @brief The upper bound on the postponed quantities for the
     *         left hand side.
     */
    core::monte_carlo::MeanVariancePairMatrix left_hand_side_u_;

    /** @brief The lower bound on the postponed quantities for the
     *         right hand side.
     */
    core::monte_carlo::MeanVariancePairVector right_hand_side_l_;

    /** @brief The finite-difference postponed quantities for the left
     *         right side.
     */
    core::monte_carlo::MeanVariancePairVector right_hand_side_e_;

    /** @brief The upper bound on the postponed quantities for the
     *         right hand side.
     */
    core::monte_carlo::MeanVariancePairVector right_hand_side_u_;

    /** @brief The amount of pruned quantities.
     */
    double pruned_;

    /** @brief The upper bound on the used error on the left hand
     *         side.
     */
    double left_hand_side_used_error_;

    /** @brief The upper bound on the used error on the right hand
     *         side.
     */
    double right_hand_side_used_error_;

    /** @brief Serialize the postponed quantities.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & left_hand_side_l_;
      ar & left_hand_side_e_;
      ar & left_hand_side_u_;
      ar & right_hand_side_l_;
      ar & right_hand_side_e_;
      ar & right_hand_side_u_;
      ar & pruned_;
      ar & left_hand_side_used_error_;
      ar & right_hand_side_used_error_;
    }

    /** @brief The default constructor.
     */
    LocalRegressionPostponed() {
      SetZero();
    }

    /** @brief Copies another postponed object.
     */
    void Copy(const LocalRegressionPostponed &postponed_in) {
      left_hand_side_l_.CopyValues(postponed_in.left_hand_side_l_);
      left_hand_side_e_.CopyValues(postponed_in.left_hand_side_e_);
      left_hand_side_u_.CopyValues(postponed_in.left_hand_side_u_);
      right_hand_side_l_.CopyValues(postponed_in.right_hand_side_l_);
      right_hand_side_e_.CopyValues(postponed_in.right_hand_side_e_);
      right_hand_side_u_.CopyValues(postponed_in.right_hand_side_u_);
      pruned_ = postponed_in.pruned_;
      left_hand_side_used_error_ = postponed_in.left_hand_side_used_error_;
      right_hand_side_used_error_ = postponed_in.right_hand_side_used_error_;
    }

    /** @brief Initializes the postponed quantities with the given
     *         dimension.
     */
    template<typename GlobalType>
    void Init(const GlobalType &global_in) {
      left_hand_side_l_.Init(
        global_in.problem_dimension() ,
        global_in.problem_dimension());
      left_hand_side_e_.Init(
        global_in.problem_dimension() ,
        global_in.problem_dimension());
      left_hand_side_u_.Init(
        global_in.problem_dimension() ,
        global_in.problem_dimension());
      right_hand_side_l_.Init(global_in.problem_dimension());
      right_hand_side_e_.Init(global_in.problem_dimension());
      right_hand_side_u_.Init(global_in.problem_dimension());
      SetZero();
    }

    /** @brief Initializes the postponed quantities given a global
     *         object and a query reference pair.
     */
    template<typename GlobalType, typename TreeType>
    void Init(const GlobalType &global_in, TreeType *qnode, TreeType *rnode) {
      left_hand_side_l_.SetZero();
      left_hand_side_e_.SetZero();
      left_hand_side_u_.SetZero();
      right_hand_side_l_.SetZero();
      right_hand_side_e_.SetZero();
      right_hand_side_u_.SetZero();

      // Set the total number of terms.
      int total_num_terms = (global_in.is_monochromatic() && qnode == rnode) ?
                            rnode->count() - 1 : rnode->count() ;
      left_hand_side_l_.set_total_num_terms(total_num_terms);
      left_hand_side_e_.set_total_num_terms(total_num_terms);
      left_hand_side_u_.set_total_num_terms(total_num_terms);
      right_hand_side_l_.set_total_num_terms(total_num_terms);
      right_hand_side_e_.set_total_num_terms(total_num_terms);
      right_hand_side_u_.set_total_num_terms(total_num_terms);
      pruned_ = static_cast<double>(total_num_terms);

      // Used error is zero.
      left_hand_side_used_error_ = 0.0;
      right_hand_side_used_error_ = 0.0;
    }

    /** @brief Applies the incoming delta contribution to the
     *         postponed quantities, optionally to the query results
     *         as well.
     */
    template < typename TreeType, typename GlobalType,
             typename LocalRegressionDelta, typename ResultType >
    void ApplyDelta(
      TreeType *qnode, TreeType *rnode,
      const GlobalType &global, const LocalRegressionDelta &delta_in,
      ResultType *query_results) {

      // Combine the delta.
      left_hand_side_l_.CombineWith(delta_in.left_hand_side_l_);
      left_hand_side_e_.CombineWith(delta_in.left_hand_side_e_);
      left_hand_side_u_.CombineWith(delta_in.left_hand_side_u_);
      right_hand_side_l_.CombineWith(delta_in.right_hand_side_l_);
      right_hand_side_e_.CombineWith(delta_in.right_hand_side_e_);
      right_hand_side_u_.CombineWith(delta_in.right_hand_side_u_);

      // Add the pruned and used error quantities.
      pruned_ = pruned_ + delta_in.pruned_;
      left_hand_side_used_error_ = left_hand_side_used_error_ +
                                   delta_in.left_hand_side_used_error_;
      right_hand_side_used_error_ = right_hand_side_used_error_ +
                                    delta_in.right_hand_side_used_error_;
    }

    /** @brief Applies the incoming postponed contribution.
     */
    void ApplyPostponed(const LocalRegressionPostponed &other_postponed) {

      // Combine the postponed quantities.
      left_hand_side_l_.CombineWith(other_postponed.left_hand_side_l_);
      left_hand_side_e_.CombineWith(other_postponed.left_hand_side_e_);
      left_hand_side_u_.CombineWith(other_postponed.left_hand_side_u_);
      right_hand_side_l_.CombineWith(other_postponed.right_hand_side_l_);
      right_hand_side_e_.CombineWith(other_postponed.right_hand_side_e_);
      right_hand_side_u_.CombineWith(other_postponed.right_hand_side_u_);

      // Add the pruned and used error quantities.
      pruned_ = pruned_ + other_postponed.pruned_;
      left_hand_side_used_error_ = left_hand_side_used_error_ +
                                   other_postponed.left_hand_side_used_error_;
      right_hand_side_used_error_ = right_hand_side_used_error_ +
                                    other_postponed.right_hand_side_used_error_;
    }

    /** @brief Applies the incoming postponed contribution during the
     *         postprocessing stage.
     */
    template<typename GlobalType>
    void FinalApplyPostponed(
      const GlobalType &global, LocalRegressionPostponed &other_postponed) {

      ApplyPostponed(other_postponed);
    }

    /** @brief Called from an exact pairwise evaluation method
     *         (i.e. the base case) which incurs no error.
     */
    template<typename GlobalType, typename MetricType>
    void ApplyContribution(
      const GlobalType &global,
      const MetricType &metric,
      const arma::vec &query_point,
      double query_weight,
      const arma::vec &reference_point,
      double reference_weight) {

      // If monochromatic, return.
      if(global.is_monochromatic() &&
          query_point.memptr() == reference_point.memptr()) {
        return;
      }

      double distsq = metric.DistanceSq(query_point, reference_point);
      double kernel_value = global.kernel().EvalUnnormOnSq(distsq);
      left_hand_side_l_.get(0, 0).push_back(kernel_value);
      left_hand_side_e_.get(0, 0).push_back(kernel_value);
      left_hand_side_u_.get(0, 0).push_back(kernel_value);
      right_hand_side_l_[0].push_back(kernel_value * reference_weight);
      right_hand_side_e_[0].push_back(kernel_value * reference_weight);
      right_hand_side_u_[0].push_back(kernel_value * reference_weight);
      for(int j = 1; j < left_hand_side_l_.n_cols(); j++) {

        // The row update for the left hand side.
        double left_hand_side_increment = kernel_value * reference_point[j - 1];
        left_hand_side_l_.get(0, j).push_back(left_hand_side_increment);
        left_hand_side_e_.get(0, j).push_back(left_hand_side_increment);
        left_hand_side_u_.get(0, j).push_back(left_hand_side_increment);

        // The column update for the left hand side.
        left_hand_side_l_.get(j, 0).push_back(left_hand_side_increment);
        left_hand_side_e_.get(j, 0).push_back(left_hand_side_increment);
        left_hand_side_u_.get(j, 0).push_back(left_hand_side_increment);

        // The right hand side.
        double right_hand_side_increment =
          kernel_value * reference_weight * reference_point[j - 1];
        right_hand_side_l_[j].push_back(right_hand_side_increment);
        right_hand_side_e_[j].push_back(right_hand_side_increment);
        right_hand_side_u_[j].push_back(right_hand_side_increment);

        for(int i = 1; i < left_hand_side_l_.n_rows(); i++) {

          double inner_increment = kernel_value * reference_point[i - 1] *
                                   reference_point[j - 1];
          left_hand_side_l_.get(i, j).push_back(inner_increment);
          left_hand_side_e_.get(i, j).push_back(inner_increment);
          left_hand_side_u_.get(i, j).push_back(inner_increment);
        }
      }
    }

    /** @brief Sets everything to zero.
     */
    void SetZero() {
      left_hand_side_l_.SetZero();
      left_hand_side_e_.SetZero();
      left_hand_side_u_.SetZero();
      right_hand_side_l_.SetZero();
      right_hand_side_e_.SetZero();
      right_hand_side_u_.SetZero();
      pruned_ = 0.0;
      left_hand_side_used_error_ = 0.0;
      right_hand_side_used_error_ = 0.0;
    }

    /** @brief Sets everything to zero in the post-processing step.
     */
    void FinalSetZero() {
      this->SetZero();
    }
};
}
}

#endif
