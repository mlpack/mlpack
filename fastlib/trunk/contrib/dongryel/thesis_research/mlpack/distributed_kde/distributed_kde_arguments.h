/** @file distributed_kde_arguments.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KDE_DISTRIBUTED_KDE_ARGUMENTS_H
#define MLPACK_DISTRIBUTED_KDE_DISTRIBUTED_KDE_ARGUMENTS_H

#include <boost/interprocess/offset_ptr.hpp>
#include "core/table/table.h"
#include "core/metric_kernels/abstract_metric.h"

namespace core {
namespace table {
extern core::table::MemoryMappedFile *global_m_file_;
};
};

namespace mlpack {
namespace distributed_kde {
template<typename DistributedTableType>
class DistributedKdeArguments {
  public:

    std::string densities_out_;

    int leaf_size_;

    boost::interprocess::offset_ptr<DistributedTableType> *reference_table_;

    boost::interprocess::offset_ptr<DistributedTableType> *query_table_;

    double bandwidth_;

    double relative_error_;

    double probability_;

    double top_tree_sample_probability_;

    std::string kernel_;

    core::metric_kernels::AbstractMetric *metric_;

  public:
    DistributedKdeArguments() {
      leaf_size_ = 0;
      reference_table_ = NULL;
      query_table_ = NULL;
      bandwidth_ = 0.0;
      relative_error_ = 0.0;
      top_tree_sample_probability_ = 0.0;
      probability_ = 0.0;
      kernel_ = "";
      metric_ = NULL;
    }

    ~DistributedKdeArguments() {
      if(reference_table_ == query_table_) {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(reference_table_.get());
        }
        else {
          delete reference_table_.get();
        }
      }
      else {
        if(core::table::global_m_file_) {
          core::table::global_m_file_->DestroyPtr(reference_table_.get());
          core::table::global_m_file_->DestroyPtr(query_table_.get());
        }
        else {
          delete reference_table_;
          delete query_table_;
        }
      }
      reference_table_ = NULL;
      query_table_ = NULL;

      if(metric_ != NULL) {
        delete metric_;
        metric_ = NULL;
      }
    }
};
};
};

#endif
