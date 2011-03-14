/** @file mixed_logit_dcm.h
 *
 *  The simulation-based mixed logit discrete choice model class.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_MIXED_LOGIT_DCM_H
#define MLPACK_MIXED_LOGIT_MIXED_LOGIT_DCM_H

#include <vector>
#include <boost/program_options.hpp>
#include "core/monte_carlo/mean_variance_pair.h"
#include "core/table/table.h"
#include "mlpack/mixed_logit_dcm/dcm_table.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_arguments.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_result.h"

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The definition of mixed logit discrete choice model using
 *         simulation-based approach.
 */
template<typename IncomingTableType>
class MixedLogitDCM {
  public:

    /** @brief The table type being used in the algorithm.
     */
    typedef IncomingTableType TableType;

    /** @brief The discrete choice model table type.
     */
    typedef mlpack::mixed_logit_dcm::DCMTable<TableType> DCMTableType;

    /** @brief The sample type.
     */
    typedef
    mlpack::mixed_logit_dcm::MixedLogitDCMSampling<DCMTableType> SamplingType;

    /** @brief The argument type.
     */
    typedef mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
    TableType > ArgumentType;

  private:

    /** @brief Update the sample allocation using optimum allocation
     *         of integration sample strategy. Updates the first
     *         sample (the current iterate) based on the second sample
     *         (the trust region stepped new iterate) and its variance
     *         difference.
     */
    void UpdateSampleAllocation_(
      const ArgumentType &arguments_in,
      double integration_sample_error,
      const SamplingType &second_sample,
      SamplingType *first_sample) const;

    /** @brief Implements the stopping condition.
     */
    bool TerminationConditionReached_(
      const ArgumentType &arguments_in,
      double model_reduction_ratio,
      double data_sample_error,
      double integration_sample_error,
      const SamplingType &first_sample,
      const arma::vec &gradient,
      int *num_iterations) const;

    /** @brief Computes the sample data error (Section 3.1).
     */
    double DataSampleError_(
      const SamplingType &first_sample,
      const SamplingType &second_sample) const;

    /** @brief Computes the integration sample error contributed by
     *         the given person.
     */
    void IntegrationSampleErrorPerPerson_(
      int person_index,
      const SamplingType &first_sample,
      const SamplingType &second_sample,
      core::monte_carlo::MeanVariancePair *integration_sample_error) const;

    /** @brief Computes the simulation error (Section 3.2).
     */
    double IntegrationSampleError_(
      const SamplingType &first_sample,
      const SamplingType &second_sample) const;

    /** @brief Computes the gradient error (Section 3.3).
     */
    double GradientError_(const SamplingType &sample) const;

    /** @brief Computes the first part of the gradient error (Section
     *         3.3).
     */
    double GradientErrorFirstPart_(const SamplingType &sample) const;

    /** @brief Computes the second part of the gradient error (Section
     *         3.3).
     */
    double GradientErrorSecondPart_(const SamplingType &sample) const;

  public:

    /** @brief Returns the table holding the discrete choice model
     *         information.
     */
    TableType *attribute_table();

    /** @brief Initializes the mixed logit discrete choice model
     *         object with a set of arguments.
     */
    void Init(ArgumentType &arguments_in);

    /** @brief Computes the result.
     */
    void Compute(
      const ArgumentType &arguments_in,
      mlpack::mixed_logit_dcm::MixedLogitDCMResult *result_out);

  private:

    /** @brief The table that holds the discrete choice model
     *         information.
     */
    DCMTableType table_;
};
}
}

#endif
