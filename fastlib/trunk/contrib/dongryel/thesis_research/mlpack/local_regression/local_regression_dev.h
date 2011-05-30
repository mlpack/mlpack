/** @file local_regression_dev.h
 *
 *  The implementation of mixed logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_DEV_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_DEV_H

#include "core/gnp/dualtree_dfs_dev.h"
#include "core/metric_kernels/lmetric.h"
#include "mlpack/local_regression/local_regression.h"

namespace mlpack {
namespace local_regression {

template<typename TableType, typename KernelType, typename MetricType>
TableType *LocalRegression <
TableType, KernelType, MetricType >::query_table() {
  return query_table_;
}

template<typename TableType, typename KernelType, typename MetricType>
TableType *LocalRegression <
TableType, KernelType, MetricType >::reference_table() {
  return reference_table_;
}

template<typename TableType, typename KernelType, typename MetricType>
typename LocalRegression <
TableType, KernelType, MetricType >::GlobalType &
LocalRegression <
TableType, KernelType, MetricType >::global() {
  return global_;
}

template<typename TableType, typename KernelType, typename MetricType>
bool LocalRegression <
TableType, KernelType, MetricType >::is_monochromatic() const {
  return is_monochromatic_;
}

template<typename TableType, typename KernelType, typename MetricType>
template<typename IncomingGlobalType>
void LocalRegression<TableType, KernelType, MetricType>::Init(
  ArgumentType &arguments_in,
  IncomingGlobalType *global_in) {

  reference_table_ = arguments_in.reference_table_;
  if(arguments_in.query_table_ == arguments_in.reference_table_) {
    is_monochromatic_ = true;
    query_table_ = reference_table_;
  }
  else {
    is_monochromatic_ = false;
    query_table_ = arguments_in.query_table_;
  }

  // Declare the global constants.
  global_.Init(arguments_in);
}

template<typename TableType, typename KernelType, typename MetricType>
void LocalRegression<TableType, KernelType, MetricType>::Compute(
  const ArgumentType &arguments_in,
  mlpack::local_regression::LocalRegressionResult *result_out) {

  // Instantiate a dual-tree algorithm of the local regression.
  typedef LocalRegression<TableType, KernelType, MetricType> ProblemType;
  core::gnp::DualtreeDfs< ProblemType > dualtree_dfs;
  dualtree_dfs.Init(*this);

  // Compute the result.
  if(arguments_in.num_iterations_in_ <= 0) {
    dualtree_dfs.Compute(arguments_in.metric_, result_out);
    printf("Number of prunes: %d\n", dualtree_dfs.num_deterministic_prunes());
    printf("Number of probabilistic prunes: %d\n",
           dualtree_dfs.num_probabilistic_prunes());
  }
  else {
    typename core::gnp::DualtreeDfs <
    ProblemType >::template iterator <
    core::metric_kernels::LMetric<2> > local_regression_it =
      dualtree_dfs.get_iterator(arguments_in.metric_, result_out);
    for(int i = 0; i < arguments_in.num_iterations_in_; i++) {
      ++local_regression_it;
    }

    // Tell the iterator that we are done using it so that the
    // result can be finalized.
    local_regression_it.Finalize();
  }
}

}
}

#endif
