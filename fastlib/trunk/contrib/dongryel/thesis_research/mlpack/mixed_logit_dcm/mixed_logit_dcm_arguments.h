/** @file mixed_logit_dcm_arguments.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_ARGUMENTS_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_ARGUMENTS_H

#include "core/table/table.h"

namespace mlpack {
namespace mixed_logit_dcm {
template<typename TableType>
class MixedLogitDCMArguments {
  public:
    TableType *attribute_table_;

    TableType *num_discrete_choices_per_person_;

    int num_parameters_;

    double initial_dataset_sample_rate_;

    double initial_integration_sample_rate_;

    std::string predictions_out_;

    std::string model_out_;

  public:

    MixedLogitDCMArguments() {
      attribute_table_ = NULL;
      num_discrete_choices_per_person_ = NULL;
      num_parameters_ = 0;
      initial_dataset_sample_rate_ = 0;
      initial_integration_sample_rate_ = 0;
    }

    ~MixedLogitDCMArguments() {
      delete attribute_table_;
      attribute_table_ = NULL;
      delete num_discrete_choices_per_person_;
      num_discrete_choices_per_person_ = NULL;
    }
};
};
};

#endif
