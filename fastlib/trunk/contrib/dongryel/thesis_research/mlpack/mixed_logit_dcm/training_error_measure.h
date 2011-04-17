/** @file training_error_measure.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_TRAINING_ERROR_MEASURE_H
#define MLPACK_MIXED_LOGIT_DCM_TRAINING_ERROR_MEASURE_H

namespace mlpack {
namespace mixed_logit_dcm {


class TrainingErrorMeasure {
  public:

    /** @brief Computes the overall zero one measure, the overall
     *         expected error, and the percentage alternative ratios
     *         for each alternative.
     */
    template<typename DCMTableType>
    static void Compute(
      const DCMTableType &discrete_choice_model_table,
      const arma::vec &parameters,
      double *zero_one_error,
      double *expected_error,
      arma::vec *percentage_alternative_ratios) {

      // For now, fix the number of samples.
      const int total_num_samples = 100;

      double zero_one_measure = 0.0;
      int total_num_people = discrete_choice_model_table.num_people();

      // Given the parameters, recompute the simulated choice
      // probability for the given person.
      std::vector< std::vector< core::monte_carlo::MeanVariancePair > >
      simulated_choice_probabilities;

      for(int j = 0; j < total_num_samples; j++) {

        // Draw the beta from the parameter theta.
        arma::vec random_beta;
        discrete_choice_model_table.distribution().DrawBeta(
          parameters, &random_beta);

        for(int i = 0; i < total_num_people; i++) {


          zero_one_measure += ;
        }
      }
      *zero_one_error /= static_cast<double>(total_num_people);
      *expected_error *= (2.0 / static_cast<double>(total_num_people));
    }
};
}
}

#endif
