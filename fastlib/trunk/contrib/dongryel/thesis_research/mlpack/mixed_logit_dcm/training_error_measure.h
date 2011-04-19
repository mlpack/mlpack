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
      const DCMTableType &dcm_table,
      const arma::vec &parameters,
      std::vector<double> *zero_one_error_per_discrete_choice,
      double *zero_one_error,
      double *expected_error,
      arma::vec *percentage_alternative_ratios) {

      // For now, fix the number of samples.
      const int total_num_samples = 100;

      // The total number of people in the dataset.
      int total_num_people = dcm_table.num_people();

      // The total number of discrete choices.
      int total_num_discrete_choices = dcm_table.total_num_discrete_choices();
      zero_one_error_per_discrete_choice->resize(total_num_discrete_choices);
      std::fill(
        zero_one_error_per_discrete_choice->begin(),
        zero_one_error_per_discrete_choice->end(), 0.0);

      // Given the parameters, recompute the simulated choice
      // probability for the given person.
      std::vector< std::vector< core::monte_carlo::MeanVariancePair > >
      simulated_choice_probabilities(total_num_people);
      for(int j = 0; j < total_num_people; j++) {
        simulated_choice_probabilities[j].resize(
          dcm_table.num_discrete_choices(j));
      }

      // First accumulate the simulated choice probabilities.
      arma::vec choice_probabilities;
      for(int j = 0; j < total_num_samples; j++) {

        // Draw the beta from the parameter theta.
        arma::vec beta_vector;
        dcm_table.distribution().DrawBeta(parameters, &beta_vector);

        for(int i = 0; i < total_num_people; i++) {

          // Compute the choice probabilities given the beta.
          dcm_table.choice_probabilities(
            i, beta_vector, &choice_probabilities);
          for(int k = 0; k < choice_probabilities.n_elem; k++) {
            simulated_choice_probabilities[i][k].push_back(
              choice_probabilities[k]);
          }
        }
      }

      // Given the simulated choice probabilities, compute the error.
      *zero_one_error = 0.0;
      *expected_error = 0.0;
      for(int i = 0; i < total_num_people; i++) {

        // Among the simulated choice proabilities for the given
        // person, find out the maximum index and see whether it
        // matches the person's actual discrete choice index.
        std::pair<int, double> max_simulated_choice_probability(-1, 0.0);
        for(unsigned int k = 0;
            k < simulated_choice_probabilities[i].size(); k++) {
          if(simulated_choice_probabilities[i][k].sample_mean() >
              max_simulated_choice_probability.second) {
            max_simulated_choice_probability.first = k;
            max_simulated_choice_probability.second =
              simulated_choice_probabilities[i][k].sample_mean();
          }
        }

        // Get the person's discrete choice index.
        int discrete_choice_index = dcm_table.get_discrete_choice_index(i);

        if(discrete_choice_index == max_simulated_choice_probability.first) {
          (*zero_one_error) += 1.0;
          (*zero_one_error_per_discrete_choice)[discrete_choice_index] += 1.0;
        }
        (*expected_error) +=
          (1.0 - simulated_choice_probabilities[i][
             max_simulated_choice_probability.first].sample_mean());
      }

      (*zero_one_error) /= static_cast<double>(total_num_people);
      for(int i = 0; i < total_num_discrete_choices; i++) {
        (*zero_one_error_per_discrete_choice)[i] /=
          static_cast<double>(dcm_table.num_people_per_discrete_choice(i));
      }
      (*expected_error) *= (2.0 / static_cast<double>(total_num_people));
    }
};
}
}

#endif
