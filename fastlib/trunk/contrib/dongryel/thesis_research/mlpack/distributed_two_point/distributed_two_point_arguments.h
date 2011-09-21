/*
 *  distributed_two_point_arguments.h
 *  
 *
 *  Created by William March on 9/20/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MLPACK_DISTRIBUTED_TWO_POINT_ARGUMENTS_H
#define MLPACK_DISTRIBUTED_TWO_POINT_ARGUMENTS_H

#include <boost/interprocess/offset_ptr.hpp>
#include "core/table/table.h"

namespace core {
  namespace table {
    extern core::table::MemoryMappedFile *global_m_file_;
  }
}


namespace mlpack {
  namespace distributed_two_point {
    template<typename DistributedTableType>
    class DistributedTwoPointArguments {
    public:
      
      /** @brief Whether to do load balancing.
       */
      bool do_load_balancing_;
      
      /** @brief The size of each leaf node.
       */
      int leaf_size_;
      
      /** @brief The pointer to the distributed reference table.
       */
      DistributedTableType *points_table_1_;
      
      /** @brief The pointer to the distributed query table.
       */
      DistributedTableType *points_table_2_;
      
      /** @brief The probability at which each data point is sampled for
       *         building the top sample tree.
       */
      double top_tree_sample_probability_;
      
      /** @brief The name of the kernel.
       */
      std::string kernel_;
      
      /** @brief The type of series expansion used.
       */
      std::string series_expansion_type_;
      
      /** @brief The metric that is being used.
       */
      core::metric_kernels::LMetric<2> *metric_;
      
      /** @brief The maximum size of the subtree to serialize.
       */
      int max_subtree_size_;
      
      /** @brief The maximum number of work to grab in total per stage.
       */
      int max_num_work_to_dequeue_per_stage_;
      
      /** @brief The number of threads to use.
       */
      int num_threads_;
      
      /** @brief The scale of the matcher. (r)
       */
      double matcher_distance_;
      
      /** @brief The width of the matcher (\Delta r).
       */
      double matcher_thickness_;
      
      /** @brief The file to save the counts to.
       */
      std::string counts_out_;
      
      
    public:
      
      /** @brief The default constructor.
       */
      DistributedTwoPointArguments() {
        do_load_balancing_ = false;
        leaf_size_ = 0;
        points_table_1_ = NULL;
        points_table_2_ = NULL;
        top_tree_sample_probability_ = 0.0;
        metric_ = NULL;
        max_subtree_size_ = 0;
        max_num_work_to_dequeue_per_stage_ = 0;
        num_threads_ = 1;
        matcher_distance_ = 0.0;
        matcher_thickness_ = 0.0;
      }
      
      /** @brief The destructor.
       */
      ~DistributedTwoPointArguments() {
        if(points_table_1_ == points_table_2_) {
          if(core::table::global_m_file_) {
            core::table::global_m_file_->DestroyPtr(points_table_1_);
          }
          else {
            delete points_table_1_;
          }
        }
        else {
          if(core::table::global_m_file_) {
            core::table::global_m_file_->DestroyPtr(points_table_1_);
            if(points_table_2_ != NULL) {
              core::table::global_m_file_->DestroyPtr(points_table_2_);
            }
          }
          else {
            delete points_table_1_;
            if(points_table_2_ != NULL) {
              delete points_table_2_;
            }
          }
        }
        points_table_1_ = NULL;
        points_table_2_ = NULL;
        
        if(metric_ != NULL) {
          delete metric_;
          metric_ = NULL;
        }
        
        // Assumes that distributed TwoPoint argument is the last argument
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