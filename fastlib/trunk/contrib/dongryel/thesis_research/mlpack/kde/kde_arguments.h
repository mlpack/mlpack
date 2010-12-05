/** @file kde_arguments.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_ARGUMENTS_H
#define MLPACK_KDE_KDE_ARGUMENTS_H

#include "core/table/table.h"
#include "core/metric_kernels/abstract_metric.h"

namespace mlpack {
namespace kde {
template<typename TableType>
class KdeArguments {
  public:

    std::string densities_out_;

    int leaf_size_;

    TableType *reference_table_;

    TableType *query_table_;

    double bandwidth_;

    double relative_error_;

    double probability_;

    std::string kernel_;

    core::metric_kernels::AbstractMetric *metric_;

  public:
    KdeArguments() {
      leaf_size_ = 0;
      reference_table_ = NULL;
      query_table_ = NULL;
      bandwidth_ = 0.0;
      relative_error_ = 0.0;
      probability_ = 0.0;
      kernel_ = "";
      metric_ = NULL;
    }

    ~KdeArguments() {
      if(reference_table_ == query_table_) {
        if(reference_table_ != NULL) {
          delete reference_table_;
        }
      }
      else {
        if(reference_table_ != NULL) {
          delete reference_table_;
        }
        if(query_table_ != NULL) {
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
