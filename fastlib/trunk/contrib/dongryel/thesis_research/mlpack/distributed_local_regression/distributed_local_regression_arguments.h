/** @file distributed_local_regression_arguments.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_LOCAL_REGRESSION_DISTRIBUTED_LOCAL_REGRESSION_ARGUMENTS_H
#define MLPACK_DISTRIBUTED_LOCAL_REGRESSION_DISTRIBUTED_LOCAL_REGRESSION_ARGUMENTS_H

#include <boost/interprocess/offset_ptr.hpp>
#include "core/table/table.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace mlpack {
namespace distributed_local_regression {
template<typename DistributedTableType, typename MetricType>
class DistributedLocalRegressionArguments {
  public:

    /** @brief The name of the output file that will hold the
     *         regression estimates.
     */
    std::string predictions_out_;

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
    MetricType metric_;

    /** @brief The maximum size of the subtree to serialize.
     */
    int max_subtree_size_;

    /** @brief The maximum number of work to grab in total per stage.
     */
    int max_num_work_to_dequeue_per_stage_;

    /** @brief The number of threads to use.
     */
    int num_threads_;

    /** @brief The order of local polynomial regression.
     */
    int order_;

    /** @brief The effective number of reference points.
     */
    int effective_num_reference_points_;

    /** @brief Whether the computation is monochromatic or not.
     */
    bool is_monochromatic_;

    /** @brief The dimension of the linear system. For NWR, it is 1 by
     *         1 system, For local linear, it is (D + 1) by (D + 1).
     */
    int problem_dimension_;

  public:

    /** @brief The default constructor.
     */
    DistributedLocalRegressionArguments() {
      leaf_size_ = 0;
      reference_table_ = NULL;
      query_table_ = NULL;
      bandwidth_ = 0.0;
      absolute_error_ = 0.0;
      relative_error_ = 0.0;
      top_tree_sample_probability_ = 0.0;
      probability_ = 0.0;
      kernel_ = "";
      max_subtree_size_ = 0;
      max_num_work_to_dequeue_per_stage_ = 0;
      num_threads_ = 1;
      effective_num_reference_points_ = 0;
      is_monochromatic_ = true;
      order_ = 0;
      problem_dimension_ = 0;
    }

    /** @brief The destructor.
     */
    ~DistributedLocalRegressionArguments() {
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

      // Assumes that distributed local regression argument is the
      // last argument that is being destroyed.
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
