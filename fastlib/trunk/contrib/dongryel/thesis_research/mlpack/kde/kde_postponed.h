/** @file kde_postponed.h
 *
 *  The postponed quantities in kde dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_POSTPONED_H
#define MLPACK_KDE_KDE_POSTPONED_H

namespace mlpack {
namespace kde {

/** @brief The postponed quantities for KDE.
 */
template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
class KdePostponed {

  private:

    // For boost serialization.
    friend class boost::serialization::access;

  public:

    /** @brief The local expansion representing the postponed
     *         quantities.
     */
    mlpack::series_expansion::CartesianLocal <
    ExpansionType > local_expansion_;

    /** @brief The lower bound on the postponed quantities.
     */
    double densities_l_;

    /** @brief The finite-difference postponed quantities.
     */
    double densities_e_;

    /** @brief The upper bound on the postponed quantities.
     */
    double densities_u_;

    /** @brief The amount of pruned quantities.
     */
    double pruned_;

    /** @brief The upper bound on the used error.
     */
    double used_error_;

    /** @brief Serialize the postponed quantities.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & densities_l_;
      ar & densities_e_;
      ar & densities_u_;
      ar & pruned_;
      ar & used_error_;
    }

    /** @brief The default constructor.
     */
    KdePostponed() {
      SetZero();
    }

    /** @brief Copies another postponed object.
     */
    void Copy(const KdePostponed &postponed_in) {
      densities_l_ = postponed_in.densities_l_;
      densities_e_ = postponed_in.densities_e_;
      densities_u_ = postponed_in.densities_u_;
      pruned_ = postponed_in.pruned_;
      used_error_ = postponed_in.used_error_;
    }

    /** @brief Initializes the postponed quantities.
     */
    template<typename GlobalType>
    void Init(const GlobalType &global_in) {
      SetZero();
    }

    /** @brief Initializes the postponed quantities given a global
     *         object and a query reference pair.
     */
    template<typename GlobalType, typename TreeType>
    void Init(
      const GlobalType &global_in, TreeType *qnode, TreeType *rnode,
      bool qnode_and_rnode_are_equal) {
      int rnode_count =
        (global_in.is_monochromatic() && qnode_and_rnode_are_equal) ?
        rnode->count() - 1 : rnode->count();
      densities_l_ = densities_u_ = 0.0;
      densities_e_ = 0.0;
      pruned_ = static_cast<double>(rnode_count);
      used_error_ = 0.0;
    }

    /** @brief Applies the incoming delta contribution to the
     *         postponed quantities, optionally to the query results
     *         as well.
     */
    template < typename TreeType, typename GlobalType,
             typename KdeDelta, typename ResultType >
    void ApplyDelta(
      TreeType *qnode, TreeType *rnode,
      const GlobalType &global, const KdeDelta &delta_in,
      ResultType *query_results) {

      if(delta_in.order_farfield_to_local_ >= 0) {

        // Far-to-local translation.
        query_results->num_farfield_to_local_prunes_++;
        rnode->stat().farfield_expansion_.TranslateToLocal(
          global.kernel_aux(),
          delta_in.order_farfield_to_local_,
          & (qnode->stat().postponed_.local_expansion_));
      }
      else if(delta_in.order_farfield_ >= 0) {

        // Far-field evaluation.
        query_results->num_farfield_prunes_++;
        typename GlobalType::TableType::TreeIterator qnode_it =
          const_cast <
          typename GlobalType::TableType * >(
            global.query_table())->get_node_iterator(qnode);
        while(qnode_it.HasNext()) {
          arma::vec qpoint;
          int qpoint_id;
          qnode_it.Next(&qpoint, &qpoint_id);
          query_results->densities_[qpoint_id] +=
            rnode->stat().farfield_expansion_.EvaluateField(
              global.kernel_aux(), qpoint, delta_in.order_farfield_);
        }
      }
      else if(delta_in.order_local_ >= 0) {

        // Direct local accumulation.
        typename GlobalType::TableType::TreeIterator rnode_it =
          const_cast<GlobalType &>(global).
          reference_table()->get_node_iterator(rnode);
        query_results->num_local_prunes_++;
        qnode->stat().postponed_.local_expansion_.AccumulateCoeffs(
          global.kernel_aux(), rnode_it, delta_in.order_local_);
      }
      else {

        // Finite-difference.
        densities_e_ += 0.5 * (delta_in.densities_l_ + delta_in.densities_u_);
      }

      densities_l_ = densities_l_ + delta_in.densities_l_;
      densities_u_ = densities_u_ + delta_in.densities_u_;
      pruned_ = pruned_ + delta_in.pruned_;
      used_error_ = used_error_ + delta_in.used_error_;
    }

    /** @brief Applies the incoming postponed contribution.
     */
    void ApplyPostponed(const KdePostponed &other_postponed) {
      densities_l_ = densities_l_ + other_postponed.densities_l_;
      densities_e_ = densities_e_ + other_postponed.densities_e_;
      densities_u_ = densities_u_ + other_postponed.densities_u_;
      pruned_ = pruned_ + other_postponed.pruned_;
      used_error_ = used_error_ + other_postponed.used_error_;
    }

    /** @brief Applies the incoming postponed contribution.
     */
    template<typename GlobalType>
    void FinalApplyPostponed(
      const GlobalType &global, KdePostponed &other_postponed) {

      // Translate the local expansion.
      other_postponed.local_expansion_.TranslateToLocal(
        global.kernel_aux(), &local_expansion_);
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
      int query_point_rank,
      int query_point_dfs_index,
      double query_weight,
      const arma::vec &reference_point,
      int reference_point_rank,
      int reference_point_dfs_index,
      double reference_weight) {

      if(query_point_rank == reference_point_rank &&
          query_point_dfs_index == reference_point_dfs_index) {
        return;
      }

      double distsq = metric.DistanceSq(query_point, reference_point);
      double density_incoming = global.kernel().EvalUnnormOnSq(distsq);
      densities_l_ = densities_l_ + density_incoming;
      densities_e_ = densities_e_ + density_incoming;
      densities_u_ = densities_u_ + density_incoming;
    }

    /** @brief Sets everything to zero except for the local expansion.
     */
    void SetZero() {
      densities_l_ = 0.0;
      densities_e_ = 0.0;
      densities_u_ = 0.0;
      pruned_ = 0.0;
      used_error_ = 0.0;
    }

    /** @brief Sets everything to zero.
     */
    void FinalSetZero() {
      local_expansion_.SetZero();
      this->SetZero();
    }
};
}
}

#endif
