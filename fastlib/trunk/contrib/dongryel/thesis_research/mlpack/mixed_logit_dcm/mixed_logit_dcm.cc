/** @file mixed_logit_dcm.cc
 *
 *  The main driver for the mixed logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include <iostream>
#include <string>
#include "mixed_logit_dcm_dev.h"
#include "core/tree/gen_metric_tree.h"

int main(int argc, char *argv[]) {

  // Tree type: hard-coded for a metric tree.
  typedef core::table::Table <
  core::tree::GenMetricTree<core::tree::AbstractStatistic> > TableType;

  // Parse arguments for mixed logit discrete choice model.
  mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
  TableType > mixed_logit_dcm_arguments;
  mlpack::mixed_logit_dcm::MixedLogitDCM <
  TableType >::ParseArguments(argc, argv, &mixed_logit_dcm_arguments);

  // Instantiate a mixed logit discrete choice model object.
  mlpack::mixed_logit_dcm::MixedLogitDCM<TableType> mixed_logit_dcm_instance;
  mixed_logit_dcm_instance.Init(mixed_logit_dcm_arguments);

  // Compute the result.
  mlpack::mixed_logit_dcm::MixedLogitDCMResult mixed_logit_dcm_result;
  mixed_logit_dcm_instance.Compute(
    mixed_logit_dcm_arguments, &mixed_logit_dcm_result);

  // Output the mixed logit discrete choice model result to the file.
  std::cerr << "Writing the discrete choice predictions to the file: " <<
            mixed_logit_dcm_arguments.predictions_out_ << "\n";
  mixed_logit_dcm_result.PrintDebug(mixed_logit_dcm_arguments.predictions_out_);

  return 0;
}
