/** @file mixed_logit_dcm_result.h
 *
 *  The comptued results for the mixed logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_RESULT_H
#define MLPACK_MIXED_LOGIT_DCM_MIXED_LOGIT_DCM_RESULT_H

#include <armadillo>

namespace mlpack {
namespace mixed_logit_dcm {
class MixedLogitDCMResult {

  private:

    /** @brief The trained parameters using the train set.
     */
    arma::vec trained_parameters_;

    /** @brief Predictions outputted for the test set using the
     *         trained parameters.
     */
    arma::ivec predicted_discrete_choices_;

    /** @brief The maximum simulated choice probability achieved for
     *         each test person using the trained parameters.
     */
    arma::vec predicted_simulated_choice_probabilities_;

  public:

    void set_test_prediction(
      int person_index, int discrete_choice_index, double probability) {
      predicted_discrete_choices_[ person_index ] = discrete_choice_index;
      predicted_simulated_choice_probabilities_[ person_index ] = probability;
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
      predicted_discrete_choices_.zeros(num_test_people);
      predicted_simulated_choice_probabilities_.zeros(num_test_people);
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
      for(unsigned int i = 0; i < predicted_discrete_choices_.n_elem; i++) {
        fprintf(
          output, "%d,%g\n", predicted_discrete_choices_[i],
          predicted_simulated_choice_probabilities_[i]);
      }
      fclose(output);
    }
};
};
};

#endif
