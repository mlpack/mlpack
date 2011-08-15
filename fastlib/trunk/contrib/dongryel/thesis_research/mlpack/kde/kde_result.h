/** @file kde_result.h
 *
 *  The computed results for kde dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_RESULT_H
#define MLPACK_KDE_KDE_RESULT_H

#include "core/parallel/map_vector.h"

namespace mlpack {
namespace kde {

/** @brief Represents the storage of KDE computation results.
 */
class KdeResult {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

    typedef core::parallel::MapVector<double> ContainerType;

  public:

    /** @brief The lower bound on the density sum.
     */
    ContainerType densities_l_;

    /** @brief The approximate density sum per query.
     */
    ContainerType densities_;

    /** @brief The upper bound on the density sum.
     */
    ContainerType densities_u_;

    /** @brief The number of points pruned per each query.
     */
    ContainerType pruned_;

    /** @brief The amount of maximum error incurred per each query.
     */
    ContainerType used_error_;

    /** @brief The number of far-to-local translations.
     */
    int num_farfield_to_local_prunes_;

    /** @brief The number of far-field evaluations.
     */
    int num_farfield_prunes_;

    /** @brief The number of direct local accumulations.
     */
    int num_local_prunes_;

    /** @brief Serialize the KDE result object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & densities_l_;
      ar & densities_;
      ar & densities_u_;
      ar & pruned_;
      ar & used_error_;
    }

    void Seed(int qpoint_index, double initial_pruned_in) {
      pruned_[qpoint_index] = initial_pruned_in;
    }

    /** @brief The default constructor.
     */
    KdeResult() {
      SetZero();
    }

    /** @brief Normalizes the density of each query.
     */
    template<typename GlobalType>
    void Normalize(const GlobalType &global) {
      for(unsigned int q_index = 0; q_index < densities_l_.size(); q_index++) {
        densities_l_[q_index] *= global.get_mult_const();
        densities_[q_index] *= global.get_mult_const();
        densities_u_[q_index] *= global.get_mult_const();
      }
    }

    template<typename MetricType, typename GlobalType>
    void PostProcess(
      const MetricType &metric,
      const core::table::DensePoint &qpoint,
      int q_index,
      double q_weight,
      const GlobalType &global,
      const bool is_monochromatic) {
      if(global.normalize_densities()) {
        densities_l_[q_index] *= global.get_mult_const();
        densities_[q_index] *= global.get_mult_const();
        densities_u_[q_index] *= global.get_mult_const();
      }
    }

    void Print(const std::string &file_name) const {
      FILE *file_output = fopen(file_name.c_str(), "w+");
      for(unsigned int i = 0; i < densities_.size(); i++) {
        fprintf(file_output, "%g %g %g %g\n", densities_l_[i],
                densities_[i], densities_u_[i], pruned_[i]);
      }
      fclose(file_output);
    }

    template<typename GlobalType, typename TreeType, typename DeltaType>
    void ApplyProbabilisticDelta(
      GlobalType &global, TreeType *qnode, double failure_probability,
      const DeltaType &delta_in) {

      // Get the iterator for the query node.
      typename GlobalType::TableType::TreeIterator qnode_it =
        global.query_table()->get_node_iterator(qnode);
      int qpoint_index;

      // Look up the number of standard deviations.
      double num_standard_deviations = global.compute_quantile(
                                         failure_probability);

      do {
        // Get each query point.
        qnode_it.Next(&qpoint_index);
        core::math::Range contribution;
        (*delta_in.mean_variance_pair_)[qpoint_index].scaled_interval(
          delta_in.pruned_, num_standard_deviations, &contribution);
        contribution.lo = std::max(contribution.lo, 0.0);
        contribution.hi = std::min(contribution.hi, delta_in.pruned_);
        densities_l_[qpoint_index] += contribution.lo;
        densities_[qpoint_index] += contribution.mid();
        densities_u_[qpoint_index] += contribution.hi;
        pruned_[qpoint_index] += delta_in.pruned_;
      }
      while(qnode_it.HasNext());
    }

    template<typename GlobalType>
    void Init(const GlobalType &global_in, int num_points) {
      this->Init(num_points);
    }

    void Init(int num_points) {
      densities_l_.Init(num_points);
      densities_.Init(num_points);
      densities_u_.Init(num_points);
      pruned_.Init(num_points);
      used_error_.Init(num_points);
      SetZero();
    }

    void SetZero() {
      for(int i = 0; i < static_cast<int>(densities_l_.size()); i++) {
        densities_l_[i] = 0;
        densities_[i] = 0;
        densities_u_[i] = 0;
        pruned_[i] = 0;
        used_error_[i] = 0;
      }
      num_farfield_to_local_prunes_ = 0;
      num_farfield_prunes_ = 0;
      num_local_prunes_ = 0;
    }

    template<typename KdePostponedType>
    void ApplyPostponed(
      int q_index, const KdePostponedType &postponed_in) {
      densities_l_[q_index] = densities_l_[q_index] + postponed_in.densities_l_;
      densities_[q_index] = densities_[q_index] + postponed_in.densities_e_;
      densities_u_[q_index] = densities_u_[q_index] + postponed_in.densities_u_;
      pruned_[q_index] = pruned_[q_index] + postponed_in.pruned_;
      used_error_[q_index] = used_error_[q_index] + postponed_in.used_error_;
    }

    template<typename GlobalType, typename KdePostponedType>
    void FinalApplyPostponed(
      const GlobalType &global,
      const core::table::DensePoint &qpoint,
      int q_index,
      const KdePostponedType &postponed_in) {

      // Evaluate the local expansion.
      densities_[q_index] +=
        postponed_in.local_expansion_.EvaluateField(
          global.kernel_aux(), qpoint);

      // Apply postponed.
      ApplyPostponed(q_index, postponed_in);
    }
};
}
}

#endif
