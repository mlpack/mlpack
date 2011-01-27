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

    /** @brief The type of the discrete choice model table being used.
     */
    typedef mlpack::mixed_logit_dcm::DCMTable<TableType> DCMTableType;

  public:

    /** @brief Stores the attribute vectors for each person.
     */
    TableType *attribute_table_;

    /** @brief Stores the number of discrete choices per person.
     */
    TableType *num_discrete_choices_per_person_;

    /** @brief The pointer to the distribution that generates each
     *         $\beta$ attribute vector. This could be a Gaussian
     *         distribution for instance.
     */
    mlpack::mixed_logit_dcm::MixedLogitDCMDistribution <
    DCMTableType > *distribution_;

    /** @brief The initial dataset sample rate (for the outer term in
     *         the sum).
     */
    double initial_dataset_sample_rate_;

    /** @brief The proportion of the total allowable integration
     *         samples to take per person in the beginning.
     */
    double initial_integration_sample_rate_;

    /** @brief The file name to output the predictions to.
     */
    std::string predictions_out_;

    /** @brief The trust region search method to be used.
     */
    std::string trust_region_search_method_;

    /** @brief Used for determining the stopping condition based on
     *         the gradient norm.
     */
    double gradient_norm_threshold_;

    /** @brief The maximum average integration sample size per person.
     */
    int max_num_integration_samples_per_person_;

    /** @brief The threshold on the integration sample error to be
     *         considered small.
     */
    double integration_sample_error_threshold_;

  public:

    MixedLogitDCMArguments() {
      attribute_table_ = NULL;
      num_discrete_choices_per_person_ = NULL;
      distribution_ = NULL;
      initial_dataset_sample_rate_ = 0;
      initial_integration_sample_rate_ = 0;
      gradient_norm_threshold_ = 0;
      max_num_integration_samples_per_person_ = 0;
      integration_sample_error_threshold_ = 0;
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
