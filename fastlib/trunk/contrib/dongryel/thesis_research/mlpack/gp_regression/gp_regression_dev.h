/** @file gp_regression_dev.h
 *
 *  The implementation of Gaussian process regression.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_GP_REGRESSION_GP_REGRESSION_DEV_H
#define MLPACK_GP_REGRESSION_GP_REGRESSION_DEV_H

#include "core/gnp/dualtree_dfs_dev.h"
#include "core/metric_kernels/lmetric.h"
#include "core/metric_kernels/weighted_lmetric.h"
#include "mlpack/gp_regression/gp_regression.h"

namespace mlpack {
namespace gp_regression {

template<typename TableType, typename KernelType, typename MetricType>
TableType *GpRegression <
TableType, KernelType, MetricType >::query_table() {
  return query_table_;
}

template<typename TableType, typename KernelType, typename MetricType>
TableType *GpRegression <
TableType, KernelType, MetricType >::reference_table() {
  return reference_table_;
}

template<typename TableType, typename KernelType, typename MetricType>
typename GpRegression <
TableType, KernelType, MetricType >::GlobalType &
GpRegression <
TableType, KernelType, MetricType >::global() {
  return global_;
}

template<typename TableType, typename KernelType, typename MetricType>
bool GpRegression <
TableType, KernelType, MetricType >::is_monochromatic() const {
  return is_monochromatic_;
}

template<typename TableType, typename KernelType, typename MetricType>
template<typename IncomingGlobalType>
void GpRegression<TableType, KernelType, MetricType>::Init(
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

  if(global_in != NULL) {
    arguments_in.is_monochromatic_ = global_in->is_monochromatic();
    arguments_in.do_postprocess_ = global_in->do_postprocess();
  }

  // Declare the global constants.
  global_.Init(arguments_in);
}

template<typename TableType, typename KernelType, typename MetricType>
void GpRegression<TableType, KernelType, MetricType>::Compute(
  const ArgumentType &arguments_in,
  mlpack::gp_regression::GpRegressionResult *result_out) {

  // Instantiate a dual-tree algorithm of the GP regression.
  typedef GpRegression<TableType, KernelType, MetricType> ProblemType;
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
    MetricType > gp_regression_it =
      dualtree_dfs.get_iterator(arguments_in.metric_, result_out);
    for(int i = 0; i < arguments_in.num_iterations_in_; i++) {
      ++gp_regression_it;
    }

    // Tell the iterator that we are done using it so that the
    // result can be finalized.
    gp_regression_it.Finalize();
  }
}

}
}

#endif
