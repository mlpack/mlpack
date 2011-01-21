/** @file distributed_kde_arguments.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KDE_DISTRIBUTED_KDE_ARGUMENTS_H
#define MLPACK_DISTRIBUTED_KDE_DISTRIBUTED_KDE_ARGUMENTS_H

#include <boost/interprocess/offset_ptr.hpp>
#include "core/table/table.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace mlpack {
namespace distributed_kde {
template<typename DistributedTableType>
class DistributedKdeArguments {
  public:

    /** @brief The name of the output file that will hold the density
     *         estimates.
     */
    std::string densities_out_;

    /** @brief The size of each leaf node.
     */
    int leaf_size_;

    /** @brief The pointer to the distributed reference table.
     */
    DistributedTableType *reference_table_;

    /** @brief The pointer to the distributed query table.
     */
    DistributedTableType *query_table_;

    /** @brief The bandwidth value being used.
     */
    double bandwidth_;

    /** @brief The absolute error.
     */
    double absolute_error_;

    /** @brief The relative error.
     */
    double relative_error_;

    /** @brief The probability level.
     */
    double probability_;

    /** @brief The probability at which each data point is sampled for
     *         building the top sample tree.
     */
    double top_tree_sample_probability_;

    /** @brief The name of the kernel.
     */
    std::string kernel_;

    /** @brief The metric that is being used.
     */
    core::metric_kernels::LMetric<2> *metric_;

    int max_num_levels_to_serialize_;

    int max_num_work_to_dequeue_per_stage_;

  public:

    /** @brief The default constructor.
     */
    DistributedKdeArguments() {
      leaf_size_ = 0;
      reference_table_ = NULL;
      query_table_ = NULL;
      bandwidth_ = 0.0;
      absolute_error_ = 0.0;
      relative_error_ = 0.0;
      top_tree_sample_probability_ = 0.0;
      probability_ = 0.0;
      kernel_ = "";
      metric_ = NULL;
      max_num_levels_to_serialize_ = 0;
      max_num_work_to_dequeue_per_stage_ = 0;
    }

    /** @brief The destructor.
     */
    ~DistributedKdeArguments() {
      if(reference_table_ == query_table_) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(reference_table_);
        }
        else {
          delete reference_table_;
        }
      }
      else {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(reference_table_);
          if(query_table_ != NULL) {
            core::table::global_m_file_->DestroyPtr(query_table_);
          }
        }
        else {
          delete reference_table_;
          if(query_table_ != NULL) {
            delete query_table_;
          }
        }
      }
      reference_table_ = NULL;
      query_table_ = NULL;

      if(metric_ != NULL) {
        delete metric_;
        metric_ = NULL;
      }

      // Assumes that distributed KDE argument is the last argument
      // that is being destroyed.
      if(core::table::global_m_file_ != NULL) {
        if(core::table::global_m_file_->AllMemoryDeallocated()) {
          std::cerr << "All memory have been deallocated.\n";
        }
        else {
          std::cerr << "There are memory leaks.\n";
        }
        delete core::table::global_m_file_;
        core::table::global_m_file_ = NULL;
      }
    }
};
}
}

#endif
