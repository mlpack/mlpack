/** @file kde_summary.h
 *
 *  The summary quantities for computing the kernel density estimate
 *  using a dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_SUMMARY_H
#define MLPACK_KDE_KDE_SUMMARY_H

namespace mlpack {
namespace kde {

class KdeSummary {

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

    /** @brief Determines whether the contribution of the reference
     *         node can be approximated using series expansion.
     */
    template <
    typename GlobalType, typename DeltaType, typename TreeType,
             typename ResultType >
    bool CanSummarizeSeriesExpansion_(
      const GlobalType &global, DeltaType &delta,
      const core::math::Range &squared_distance_range,
      TreeType *qnode, TreeType *rnode, double right_hand_side,
      ResultType *query_results) const {

      // The far-field expansion of the reference node.
      const typename GlobalType::KernelAuxType::FarFieldType &
      farfield_expansion = rnode->stat().farfield_expansion_;

      // Actual amount of error incurred per each query/ref pair.
      double actual_err_farfield_to_local = 0;
      double actual_err_farfield = 0;
      double actual_err_local = 0;

      // The allowed error per each query/ref pair.
      double allowed_err =
        right_hand_side / static_cast<double>(rnode->count());

      // Estimated computational cost.
      int cost_farfield_to_local = std::numeric_limits<int>::max();
      int cost_farfield = std::numeric_limits<int>::max();
      int cost_local = std::numeric_limits<int>::max();
      int cost_exhaustive = (qnode->count()) * (rnode->count()) *
                            (qnode->bound().dim());
      int min_cost = 0;

      // Get the order of approximations.
      delta.order_farfield_to_local_ =
        global.kernel_aux().OrderForConvertingFromFarFieldToLocal(
          rnode->bound(), qnode->bound(),
          squared_distance_range.lo, squared_distance_range.hi,
          allowed_err, &actual_err_farfield_to_local);

      // Farfield evaluations are possible when the query poins are
      // available.
      delta.order_farfield_ =
        global.kernel_aux().OrderForEvaluatingFarField(
          rnode->bound(), qnode->bound(),
          squared_distance_range.lo, squared_distance_range.hi,
          allowed_err, &actual_err_farfield);

      // Direct local accumulations are possible when the reference
      // points are available.
      delta.order_local_ =
        global.kernel_aux().OrderForEvaluatingLocal(
          rnode->bound(), qnode->bound(),
          squared_distance_range.lo, squared_distance_range.hi,
          allowed_err, &actual_err_local);

      // Update computational cost and compute the minimum.
      if(delta.order_farfield_to_local_ >= 0) {
        cost_farfield_to_local =
          static_cast<int>(
            global.kernel_aux().global().FarFieldToLocalTranslationCost(
              delta.order_farfield_to_local_));
      }
      if(delta.order_farfield_ >= 0) {
        cost_farfield =
          static_cast<int>(
            global.kernel_aux().global().FarFieldEvaluationCost(
              delta.order_farfield_) * (qnode->count()));
      }
      if(delta.order_local_ >= 0) {
        cost_local =
          static_cast<int>(
            global.kernel_aux().global().DirectLocalAccumulationCost(
              delta.order_local_) * (rnode->count()));
      }

      min_cost =
        std::min(
          cost_farfield_to_local,
          std::min(
            cost_farfield, std::min(
              cost_local, cost_exhaustive)));

      if(cost_farfield_to_local == min_cost) {
        delta.used_error_ = farfield_expansion.get_weight_sum() *
                            actual_err_farfield_to_local;
        delta.order_farfield_ = -1;
        delta.order_local_ = -1;
        return true;
      }

      if(cost_farfield == min_cost) {
        delta.used_error_ = farfield_expansion.get_weight_sum() *
                            actual_err_farfield;
        delta.order_farfield_to_local_ = -1;
        delta.order_local_ = -1;
        return true;
      }

      if(cost_local == min_cost) {
        delta.used_error_ = farfield_expansion.get_weight_sum() *
                            actual_err_local;
        delta.order_farfield_to_local_ = -1;
        delta.order_farfield_ = -1;
        return true;
      }

      delta.order_farfield_to_local_ = -1;
      delta.order_farfield_ = -1;
      delta.order_local_ = -1;
      return false;
    }

  public:

    double densities_l_;

    double densities_u_;

    double pruned_l_;

    double used_error_u_;

    void Seed(double initial_pruned_in) {
      this->SetZero();
      pruned_l_ = initial_pruned_in;
    }

    void Print() const {
      printf("Lower bound/upper bound on the densities: [ %g %g ], "
             "Lower bound on the pruned components: %g, "
             "Upper bound on the used error: %g\n",
             densities_l_, densities_u_, pruned_l_, used_error_u_);
    }

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & densities_l_;
      ar & densities_u_;
      ar & pruned_l_;
      ar & used_error_u_;
    }

    void Copy(const KdeSummary &summary_in) {
      densities_l_ = summary_in.densities_l_;
      densities_u_ = summary_in.densities_u_;
      pruned_l_ = summary_in.pruned_l_;
      used_error_u_ = summary_in.used_error_u_;
    }

    KdeSummary() {
      SetZero();
    }

    KdeSummary(const KdeSummary &summary_in) {
      densities_l_ = summary_in.densities_l_;
      densities_u_ = summary_in.densities_u_;
      pruned_l_ = summary_in.pruned_l_;
      used_error_u_ = summary_in.used_error_u_;
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

      // The number of samples.
      const int num_samples = 25;
      if(rnode->count() < 50 ||
          2.0 * global.kernel_aux().kernel().bandwidth_sq() <
          squared_distance_range.hi) {
        return false;
      }

      // Get the iterator for the query node.
      typename GlobalType::TableType::TreeIterator qnode_it =
        global.query_table()->get_node_iterator(qnode);
      arma::vec qpoint;
      int qpoint_index;

      // Get the iterator for the reference node.
      typename GlobalType::TableType::TreeIterator rnode_it =
        global.reference_table()->get_node_iterator(rnode);
      arma::vec rpoint;
      int rpoint_index;

      // Interval for the pivot query point.
      double num_standard_deviations = global.compute_quantile(
                                         failure_probability);
      delta.mean_variance_pair_ = ((GlobalType &) global).mean_variance_pair();

      // The flag saying whether the pruning is a success.
      bool prunable = true;
      do {

        // Get each query point.
        qnode_it.Next(&qpoint, &qpoint_index);

        // Clear the sample mean variance pair for the current query.
        (*delta.mean_variance_pair_)[qpoint_index].SetZero();

        for(int i = 0; i < num_samples; i++) {

          // Pick a random reference point and compute the kernel
          // difference.
          rnode_it.RandomPick(&rpoint, &rpoint_index);
          double squared_dist = metric.DistanceSq(qpoint, rpoint);
          double kernel_value =
            global.kernel().EvalUnnormOnSq(squared_dist);

          // Accumulate the sample.
          (*delta.mean_variance_pair_)[qpoint_index].push_back(kernel_value);
        }

        // Add the correction.
        core::math::Range correction;
        (*delta.mean_variance_pair_)[qpoint_index].scaled_interval(
          delta.pruned_, num_standard_deviations, &correction);
        correction.lo = std::max(correction.lo, 0.0);
        correction.hi = std::min(correction.hi, delta.pruned_);

        // Technically, though not correct, just use the mid point of
        // the Monte Carlo contribution.
        double modified_densities_l =
          query_results->densities_l_[qpoint_index] + correction.mid() +
          postponed.densities_l_;
        double left_hand_side = correction.width() * 0.5;
        double right_hand_side =
          rnode->count() * (
            global.relative_error() * modified_densities_l +
            global.effective_num_reference_points() * global.absolute_error() -
            used_error_u_) /
          static_cast<double>(
            global.effective_num_reference_points() -
            query_results->pruned_[qpoint_index]);

        // Prunable if the left hand side is less than right hand side.
        prunable = (left_hand_side <= right_hand_side);
      }
      while(qnode_it.HasNext() && prunable);
      return prunable;
    }

    template < typename GlobalType, typename DeltaType, typename TreeType,
             typename ResultType >
    bool CanSummarize(
      const GlobalType &global, DeltaType &delta,
      const core::math::Range &squared_distance_range,
      TreeType *qnode, int qnode_rank, TreeType *rnode, int rnode_rank,
      bool qnode_and_rnode_are_equal,
      ResultType *query_results) const {

      double left_hand_side = delta.used_error_;
      double right_hand_side =
        rnode->count() * (
          global.relative_error() * densities_l_ +
          global.effective_num_reference_points() * global.absolute_error() -
          used_error_u_) /
        static_cast<double>(
          global.effective_num_reference_points() - pruned_l_);

      // Prunable by finite-difference.
      if(left_hand_side <= right_hand_side) {
        return true;
      }

      // Otherwise, try series expansion.
      else if(((! global.is_monochromatic()) || (! qnode_and_rnode_are_equal)) &&
              global.kernel_aux().global().get_max_order() > 0) {
        return CanSummarizeSeriesExpansion_(
                 global, delta, squared_distance_range,
                 qnode, rnode, right_hand_side, query_results);
      }
      return false;
    }

    void SetZero() {
      densities_l_ = 0;
      densities_u_ = 0;
      pruned_l_ = 0;
      used_error_u_ = 0;
    }

    void StartReaccumulate() {
      densities_l_ = std::numeric_limits<double>::max();
      densities_u_ = 0;
      pruned_l_ = densities_l_;
      used_error_u_ = 0;
    }

    template<typename GlobalType, typename ResultType>
    void Accumulate(
      const GlobalType &global, const ResultType &results, int q_index) {
      densities_l_ = std::min(densities_l_, results.densities_l_[q_index]);
      densities_u_ = std::max(densities_u_, results.densities_u_[q_index]);
      pruned_l_ = std::min(pruned_l_, results.pruned_[q_index]);
      used_error_u_ = std::max(used_error_u_, results.used_error_[q_index]);
    }

    template<typename GlobalType, typename KdePostponedType>
    void Accumulate(
      const GlobalType &global, const KdeSummary &summary_in,
      const KdePostponedType &postponed_in) {
      densities_l_ = std::min(
                       densities_l_,
                       summary_in.densities_l_ + postponed_in.densities_l_);
      densities_u_ = std::max(
                       densities_u_,
                       summary_in.densities_u_ + postponed_in.densities_u_);
      pruned_l_ = std::min(
                    pruned_l_, summary_in.pruned_l_ + postponed_in.pruned_);
      used_error_u_ = std::max(
                        used_error_u_,
                        summary_in.used_error_u_ + postponed_in.used_error_);
    }

    void ApplyDelta(const KdeDelta &delta_in) {
      densities_l_ = densities_l_ + delta_in.densities_l_;
      densities_u_ = densities_u_ + delta_in.densities_u_;
    }

    template<typename KdePostponedType>
    void ApplyPostponed(const KdePostponedType &postponed_in) {
      densities_l_ = densities_l_ + postponed_in.densities_l_;
      densities_u_ = densities_u_ + postponed_in.densities_u_;
      pruned_l_ = pruned_l_ + postponed_in.pruned_;
      used_error_u_ = used_error_u_ + postponed_in.used_error_;
    }
};
}
}

#endif
