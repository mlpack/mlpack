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
    TableType *reference_table_;

    TableType *query_table_;

    double initial_dataset_sample_rate_;

    double initial_integration_sample_rate_;

    std::string predictions_out_;

    std::string model_out_;

  public:

    MixedLogitDCMArguments() {
      reference_table_ = NULL;
      query_table_ = NULL;
      initial_dataset_sample_rate_ = 0;
      initial_integration_sample_rate_ = 0;
    }

    ~MixedLogitDCMArguments() {
      if(reference_table_ != query_table_) {
        delete reference_table_;
        delete query_table_;
      }
      else {
        delete reference_table_;
      }
      reference_table_ = query_table_ = NULL;
    }
};
};
};

#endif
