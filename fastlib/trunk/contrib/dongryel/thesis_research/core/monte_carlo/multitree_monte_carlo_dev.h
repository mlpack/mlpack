/** @file multitree_monte_carlo_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_MONTE_CARLO_MULTITREE_MONTE_CARLO_DEV_H
#define CORE_MONTE_CARLO_MULTITREE_MONTE_CARLO_DEV_H

#include "fastlib/monte_carlo/multitree_monte_carlo.h"
#include "fastlib/monte_carlo/strata.h"
#include "mlpack/kde/mean_variance_pair.h"

namespace core {
namespace monte_carlo {

template<typename TableType>
std::vector< MultitreeMonteCarlo<TableType> *> &
MultitreeMonteCarlo<TableType>::subproblems() {

  return subproblems_;
}

template<typename TableType>
void MultitreeMonteCarlo<TableType>::AddSubProblem(
  MultitreeMonteCarlo<TableType> &subproblem_in) {

  subproblems_.push_back(& subproblem_in);

  // The variable arguments of the self become the constant arguments
  // for the subproblems.
  for(int i = 0; i < tables_for_variable_arguments_.size(); i++) {
    subproblems_.add_constant_argument(
      tables_for_variable_arguments_[i].first);
  }
}

template<typename TableType>
void MultitreeMonteCarlo<TableType>::clear_constant_arguments() {
  constant_arguments_.resize(0);
}

template<typename TableType>
void MultitreeMonteCarlo<TableType>::set_constant_argument(
  int constant_argument_index, int point_index) {

  constant_arguments_[constant_argument_index].second = point_index;
}

template<typename TableType>
void MultitreeMonteCarlo<TableType>::add_constant_argument(
  TableType *constant_argument_in) {

  constant_arguments_.push_back(std::pair<TableType *, int>(
                                  constant_argument_in, 0));
}

template<typename TableType>
void MultitreeMonteCarlo<TableType>::add_variable_argument(
  TableType &variable_argument,
  const std::vector<int> &leave_one_out_dataset_indices) {

  tables_for_variable_arguments_.push_back(std::pair<TableType *, std::vector<int> >(
        &variable_argument, leave_one_out_dataset_indices));
}

template<typename TableType>
void MultitreeMonteCarlo<TableType>::set_error(double relative_error_in,
    double probability_in) {

  relative_error_ = relative_error_in;
  probability_ = probability_in;

  // Compute the appropriate z-score.
  double cumulative_probability_from_left_end = 0.5 + 0.5 * probability_in;
  if(cumulative_probability_from_left_end > 0.999) {
    num_standard_deviations_ = 3.0;
  }
  else {
    num_standard_deviations_ = boost::math::quantile(
                                 normal_dist_, cumulative_probability_from_left_end);
  }
}

template<typename TableType>
bool MultitreeMonteCarlo<TableType>::Converged_(
  const std::pair<double, double> &summary_mean_variance_pair) const {

  double error =
    sqrt(summary_mean_variance_pair.second) * num_standard_deviations_;

  return error <= relative_error_ * fabs(summary_mean_variance_pair.first);
}

template<typename TableType>
const std::vector< std::pair<TableType *, int> > &MultitreeMonteCarlo<TableType>::constant_arguments() const {

  return constant_arguments_;
}

template<typename TableType>
std::vector< std::pair<TableType *, int> > &MultitreeMonteCarlo<TableType>::constant_arguments() {
  return constant_arguments_;
}

template<typename TableType>
const std::vector< std::pair<TableType *, std::vector<int> > >
&MultitreeMonteCarlo<TableType>::tables_for_variable_arguments() const {

  return tables_for_variable_arguments_;
}

template<typename TableType>
std::vector< std::pair<TableType *, std::vector<int> > >
&MultitreeMonteCarlo<TableType>::tables_for_variable_arguments() {

  return tables_for_variable_arguments_;
}

template<typename TableType>
template<typename PointType>
void MultitreeMonteCarlo<TableType>::get(
  int variable_argument_index, int point_index, PointType *point_out) {

  tables_for_variable_arguments_[variable_argument_index].first->get(
    point_index, point_out);
}

template<typename TableType>
template<typename FunctionType>
void MultitreeMonteCarlo<TableType>::Compute(
  const FunctionType &function_in,
  std::vector< std::pair<double, double> > *mean_variance_pair_out) {

  // The number of samples to allocate in total in each round.
  const int total_num_samples_in_each_round = 1000;

  // For each variable argument, we maintain a strata. Initialize the
  // strata with the one with the root nodes for each variable
  // argument.
  fl::ml::Strata<TableType> strata;
  strata.Init(function_in, *this, total_num_samples_in_each_round);

  for(int trial_num = 1; ; trial_num++) {

    // Sample in this round.
    std::vector< std::pair<double, double> > summary_mean_variance_pair;
    strata.AccumulateSamples(function_in, &summary_mean_variance_pair);

    // Check convergence. If not converged, then increase the number
    // of strata by one.
    if(Converged_(summary_mean_variance_pair[0]) == false) {
      strata.OptimalAllocation();
      if(strata.size() < 8) {
        strata.Stratify(function_in);
      }
    }
    else {
      mean_variance_pair_out->resize(summary_mean_variance_pair.size());
      for(int i = 0; i < summary_mean_variance_pair.size(); i++) {
        (*mean_variance_pair_out)[i].first =
          summary_mean_variance_pair[i].first;
        (*mean_variance_pair_out)[i].second =
          summary_mean_variance_pair[i].second;
      }
      break;
    }
  }
}
}
}

#endif
