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
void GpRegression<TableType, KernelType, MetricType>::Solve_(
  const std::vector< TreeType *> &reference_frontier,
  arma::mat &variable_states,
  int reference_frontier_id) {

  TreeType *reference_node = reference_frontier[reference_frontier_id];
  int rank = std::min(reference_node->count(), 10);

}

template<typename TableType, typename KernelType, typename MetricType>
void GpRegression<TableType, KernelType, MetricType>::Compute(
  const ArgumentType &arguments_in,
  mlpack::gp_regression::GpRegressionResult *result_out) {

  // Convergence flag.
  bool converged = false;

  // Get a partition using the reference tree.
  std::vector< TreeType *> reference_frontier;
  reference_table_->get_tree()->get_frontier_nodes(
    arguments_in.admm_max_subproblem_size_, &reference_frontier);

  // The variable state for each subproblem (each column).
  arma::mat variable_states(
    reference_table_->n_entries(), reference_frontier.size());
  variable_states.randn();

  do {

    // Solve the least squares problem, while fixing the other
    // variables in each strata.
    for(unsigned int i = 0; i < reference_frontier.size(); i++) {
      Solve_(reference_frontier, variable_states, i);
    }

    // Do the correction at the interfaces.
    for(unsigned int i = 1; i < reference_frontier.size(); i++) {

    }

  }
  while(converged);
}
}
}

#endif
