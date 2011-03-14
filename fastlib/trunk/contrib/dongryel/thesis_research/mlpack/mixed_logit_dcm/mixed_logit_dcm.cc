/** @file mixed_logit_dcm.cc
 *
 *  The main driver for the mixed logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <iostream>
#include <string>
#include "core/tree/gen_metric_tree.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_argument_parser.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_dev.h"

template<typename DistributionType>
void BranchOnDistribution(
  mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
  TableType > &mixed_logit_dcm_arguments) {

  // Instantiate a mixed logit discrete choice model object.
  mlpack::mixed_logit_dcm::MixedLogitDCM <
  TableType, DistributionType > mixed_logit_dcm_instance;
  mixed_logit_dcm_instance.Init(mixed_logit_dcm_arguments);

  // Compute the result.
  mlpack::mixed_logit_dcm::MixedLogitDCMResult mixed_logit_dcm_result;
  mixed_logit_dcm_instance.Compute(
    mixed_logit_dcm_arguments, &mixed_logit_dcm_result);

  // Output the mixed logit discrete choice model result to the file.
  std::cerr << "Writing the discrete choice predictions to the file: " <<
            mixed_logit_dcm_arguments.predictions_out_ << "\n";
  mixed_logit_dcm_result.PrintDebug(
    mixed_logit_dcm_arguments.predictions_out_);
}

int main(int argc, char *argv[]) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree<core::tree::AbstractStatistic> > TableType;

  // Parse arguments for mixed logit discrete choice model.
  mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
  TableType > mixed_logit_dcm_arguments;
  mlpack::mixed_logit_dcm::MixedLogitDCMArgumentParser::ParseArguments(
    argc, argv, &mixed_logit_dcm_arguments);

  if(mixed_logit_dcm_arguments.distribution_ == "constant") {
    BranchOnDistribution<mlpack::mixed_logit_dcm::ConstantDistribution>(
      mixed_logit_dcm_arguments);
  }
  else if(mixed_logit_dcm_arguments.distribution_ == "gaussian") {
    BranchOnDistribution<mlpack::mixed_logit_dcm::GaussianDistribution>(
      mixed_logit_dcm_arguments);
  }

  return 0;
}
