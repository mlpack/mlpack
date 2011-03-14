/** @file mixed_logit_dcm_arguments.h
 *
 *  The arguments used for the mixed logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_ARGUMENTS_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_ARGUMENTS_H

#include "core/table/table.h"
#include "core/optimization/trust_region.h"
#include "mlpack/mixed_logit_dcm/distribution.h"

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The argument list for the mixed logit discrete choice
 *         model.
 */
template<typename TableType>
class MixedLogitDCMArguments {
  public:

    /** @brief Stores the attribute vectors for each person.
     */
    TableType *attribute_table_;

    /** @brief Stores the number of discrete choices per person.
     */
    TableType *discrete_choice_set_info_;

    /** @brief The pointer to the distribution that generates each
     *         $\beta$ attribute vector. This could be a Gaussian
     *         distribution for instance.
     */
    std::string distribution_;

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
    enum core::optimization::TrustRegionSearchMethod::SearchType
    trust_region_search_method_;

    /** @brief Used for determining the stopping condition based on
     *         the gradient norm.
     */
    double gradient_norm_threshold_;

    /** @brief The maximum number of iterations to try after all
     *         terms have been added to the object function.
     */
    int max_num_iterations_;

    /** @brief The maximum average integration sample size per person.
     */
    int max_num_integration_samples_per_person_;

    /** @brief The threshold on the integration sample error to be
     *         considered small.
     */
    double integration_sample_error_threshold_;

    /** @brief The maximum trust region radius.
     */
    double max_trust_region_radius_;

    /** @brief The method for updating the Hessian.
     */
    std::string hessian_update_method_;

  public:

    /** @brief The default constructor.
     */
    MixedLogitDCMArguments() {
      attribute_table_ = NULL;
      discrete_choice_set_info_ = NULL;
      initial_dataset_sample_rate_ = 0;
      initial_integration_sample_rate_ = 0;
      gradient_norm_threshold_ = 0;
      max_num_iterations_ = 0;
      max_num_integration_samples_per_person_ = 0;
      integration_sample_error_threshold_ = 0;
      max_trust_region_radius_ = 0;
    }

    /** @brief The destructor.
     */
    ~MixedLogitDCMArguments() {
      delete attribute_table_;
      attribute_table_ = NULL;
      delete discrete_choice_set_info_;
      discrete_choice_set_info_ = NULL;
    }
};
}
}

#endif
