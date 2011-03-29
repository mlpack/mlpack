/** @file kpca_result.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KPCA_KPCA_RESULT_H
#define MLPACK_DISTRIBUTED_KPCA_KPCA_RESULT_H

#include <boost/math/distributions/normal.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include "core/metric_kernels/kernel.h"
#include "core/tree/statistic.h"
#include "core/table/table.h"

namespace mlpack {
namespace distributed_kpca {

/** @brief Represents the storage of KPCA computation results.
 */
class KpcaResult {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

  public:

    /** @brief Serialize the KDE result object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
    }

    /** @brief The default constructor.
     */
    KpcaResult() {
      SetZero();
    }

    void Print(const std::string &file_name) {
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
      core::table::DensePoint qpoint;
      int qpoint_index;

      // Look up the number of standard deviations.
      double num_standard_deviations = global.compute_quantile(
                                         failure_probability);

      do {
        // Get each query point.
        qnode_it.Next(&qpoint, &qpoint_index);
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

    void Init(int num_points) {
      densities_l_.resize(num_points);
      densities_.resize(num_points);
      densities_u_.resize(num_points);
      pruned_.resize(num_points);
      used_error_.resize(num_points);

      SetZero();
    }

    void SetZero() {
      self_contribution_subtracted_ = false;
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
};

}
}

#endif
