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

    bool tables_are_aliased_;

  public:

    template<typename GlobalType>
    void Init(
      TableType *reference_table_in, TableType *query_table_in,
      GlobalType &global_in) {
      reference_table_ = reference_table_in;
      query_table_ = query_table_in;
      bandwidth_ = global_in.bandwidth();
      relative_error_ = global_in.relative_error();
      probability_ = global_in.probability();
      kernel_ = global_in.kernel().name();
      tables_are_aliased_ = true;
    }

    template<typename GlobalType>
    void Init(GlobalType &global_in) {
      reference_table_ = global_in.reference_table()->local_table();
      if(reference_table_ != query_table_) {
        query_table_ = global_in.query_table()->local_table();
      }
      bandwidth_ = global_in.bandwidth();
      relative_error_ = global_in.relative_error();
      probability_ = global_in.probability();
      kernel_ = global_in.kernel().name();
      tables_are_aliased_ = true;
    }

    KdeArguments() {
      leaf_size_ = 0;
      reference_table_ = NULL;
      query_table_ = NULL;
      bandwidth_ = 0.0;
      relative_error_ = 0.0;
      probability_ = 0.0;
      kernel_ = "";
      metric_ = NULL;
      tables_are_aliased_ = false;
    }

    ~KdeArguments() {
      if(tables_are_aliased_ == false) {
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
