/** @file mixed_logit_dcm_result.h
 *
 *  The comptued results for the mixed logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_RESULT_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_RESULT_H

#include <armadillo>
#include <boost/scoped_array.hpp>

namespace mlpack {
namespace mixed_logit_dcm {
class MixedLogitDCMResult {

  private:

    /** @brief The number of test people.
     */
    int num_test_people_;

    /** @brief Predictions outputted for the test set using the
     *         trained parameters.
     */
    arma::ivec predicted_discrete_choices_;

    /** @brief The simulated choice probabilities for each person
     *         using the trained parameters.
     */
    boost::scoped_array< arma::vec > predicted_simulated_choice_probabilities_;

    /** @brief The trained parameters using the train set.
     */
    arma::vec trained_parameters_;

  public:

    MixedLogitDCMResult() {
      num_test_people_ = 0;
    }

    void set_test_prediction(
      int person_index,
      const core::monte_carlo::MeanVariancePairVector
      &simulated_choice_probabilities) {

      int max_index = -1;
      double max_simulated_choice_probability = -1.0;
      int num_discrete_choices = simulated_choice_probabilities.length();
      predicted_simulated_choice_probabilities_[person_index].zeros(
        num_discrete_choices);
      for(int i = 0; i < num_discrete_choices; i++) {
        predicted_simulated_choice_probabilities_[ person_index ][i] =
          simulated_choice_probabilities[i].sample_mean();
        if(max_simulated_choice_probability <
            simulated_choice_probabilities[i].sample_mean()) {
          max_index = i;
          max_simulated_choice_probability =
            simulated_choice_probabilities[i].sample_mean();
        }
      }
      predicted_discrete_choices_[ person_index ] = max_index;
    }

    /** @brief Returns the trained parameters.
     */
    const arma::vec &trained_parameters() const {
      return trained_parameters_;
    }

    /** @brief Initializes the prediction vector for the given number
     *         of people.
     */
    void Init(int num_test_people) {
      num_test_people_ = num_test_people;
      predicted_discrete_choices_.zeros(num_test_people_);
      boost::scoped_array< arma::vec > tmp_array(new arma::vec[num_test_people_]);
      predicted_simulated_choice_probabilities_.swap(tmp_array);
    }

    /** @brief Sets the trained parameters.
     */
    void set_trained_parameters(const arma::vec &trained_parameters_in) {
      trained_parameters_ = trained_parameters_in;
    }

    /** @brief Exports the mixed logit discrete choice outputs to the
     *         file.
     */
    void Print(const std::string &file_out) {
      FILE *output = fopen(file_out.c_str(), "w+");
      for(int i = 0; i < num_test_people_; i++) {
        for(unsigned int j = 0;
            j < predicted_simulated_choice_probabilities_[i].n_elem; j++) {
          fprintf(output, "%g,", predicted_simulated_choice_probabilities_[i][j]);
        }
        fprintf(
          output, "%d\n", predicted_discrete_choices_[i]);
      }
      fclose(output);
    }
};
};
};

#endif
