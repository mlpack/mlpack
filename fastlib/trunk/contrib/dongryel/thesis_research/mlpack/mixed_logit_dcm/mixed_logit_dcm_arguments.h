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

    /** @brief The dimensions of each component of an attribute.
     */
    std::vector<int> attribute_dimensions_;

    /** @brief The decision per each person.
     */
    TableType *decisions_table_;

    /** @brief Stores the number of discrete choices per person.
     */
    TableType *num_alternatives_table_;

    int random_seed_;

    /** @brief Stores the attribute vectors for each person (the test
     *         set).
     */
    TableType *test_attribute_table_;

    /** @brief The decision per each person (the test set).
     */
    TableType *test_decisions_table_;

    /** @brief Stores the number of discrete choices per person (the
     *         test set).
     */
    TableType *test_num_alternatives_table_;

    /** @brief Stores the initial parameter values to start with.
     */
    TableType *initial_parameters_table_;

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

    /** @brief The method used to compute the error.
     */
    std::string error_compute_method_;

    /** @brief Stores the true parameter values.
     */
    TableType *true_parameters_table_;

  public:

    /** @brief The default constructor.
     */
    MixedLogitDCMArguments() {
      attribute_table_ = NULL;
      decisions_table_ = NULL;
      num_alternatives_table_ = NULL;
      test_attribute_table_ = NULL;
      test_decisions_table_ = NULL;
      test_num_alternatives_table_ = NULL;
      initial_parameters_table_ = NULL;
      initial_dataset_sample_rate_ = 0;
      initial_integration_sample_rate_ = 0;
      gradient_norm_threshold_ = 0;
      max_num_iterations_ = 0;
      max_num_integration_samples_per_person_ = 0;
      integration_sample_error_threshold_ = 0;
      max_trust_region_radius_ = 0;
      random_seed_ = 0;
      true_parameters_table_ = NULL;
    }

    /** @brief The destructor.
     */
    ~MixedLogitDCMArguments() {
      delete attribute_table_;
      attribute_table_ = NULL;
      delete decisions_table_;
      decisions_table_ = NULL;
      delete num_alternatives_table_;
      num_alternatives_table_ = NULL;
      if(true_parameters_table_ != NULL) {
        delete true_parameters_table_;
        true_parameters_table_ = NULL;
      }

      if(test_attribute_table_ != NULL) {
        delete test_attribute_table_;
        test_attribute_table_ = NULL;
        delete test_decisions_table_;
        test_decisions_table_ = NULL;
        delete test_num_alternatives_table_;
        test_num_alternatives_table_ = NULL;
      }

      delete initial_parameters_table_;
      initial_parameters_table_ = NULL;
    }
};
}
}

#endif
