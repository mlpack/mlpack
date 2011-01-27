/** @file mixed_logit_dcm_arguments.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_ARGUMENTS_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_ARGUMENTS_H

#include "core/table/table.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_distribution.h"

namespace mlpack {
namespace mixed_logit_dcm {

template<typename TableType>
class DCMTable;

template<typename TableType>
class MixedLogitDCMArguments {
  public:
    typedef mlpack::mixed_logit_dcm::DCMTable<TableType> DCMTableType;

  public:
    TableType *attribute_table_;

    TableType *num_discrete_choices_per_person_;

    mlpack::mixed_logit_dcm::MixedLogitDCMDistribution <
    DCMTableType > *distribution_;

    double initial_dataset_sample_rate_;

    double initial_integration_sample_rate_;

    std::string predictions_out_;

    std::string model_out_;

    std::string trust_region_search_method_;

    /** @brief Used for determining the stopping condition based on
     *         the gradient norm.
     */
    double c_factor_;

    /** @brief The maximum average integration sample size per person.
     */
    int max_num_integration_samples_per_person_;

  public:

    MixedLogitDCMArguments() {
      attribute_table_ = NULL;
      num_discrete_choices_per_person_ = NULL;
      distribution_ = NULL;
      initial_dataset_sample_rate_ = 0;
      initial_integration_sample_rate_ = 0;
      c_factor_ = 0;
      max_num_integration_samples_per_person_ = 0;
    }

    ~MixedLogitDCMArguments() {
      delete attribute_table_;
      attribute_table_ = NULL;
      delete num_discrete_choices_per_person_;
      num_discrete_choices_per_person_ = NULL;
      delete distribution_;
      distribution_ = NULL;
    }
};
};
};

#endif
