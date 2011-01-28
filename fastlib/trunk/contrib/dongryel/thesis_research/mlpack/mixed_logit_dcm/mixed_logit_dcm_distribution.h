/** @file mixed_logit_dcm_distribution.h
 *
 *  A virtual class that defines the specification for which the mixed
 *  logit discrete choice model distribution should satisfy.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_DCM_DISTRIBUTION_H
#define MLPACK_MIXED_LOGIT_DCM_DCM_DISTRIBUTION_H

#include <armadillo>
#include "core/table/dense_point.h"
#include "core/table/dense_matrix.h"

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The base abstract class for the distribution that generates
 *         each $\beta$ in mixed logit models. This distribution is
 *         parametrized by $\theta$.
 */
template<typename DCMTableType>
class MixedLogitDCMDistribution {
  public:

    /** @brief Returns the (row, col)-th entry of
     *         $\frac{\partial}{\partial \theta} \beta^{\nu}(\theta)$
     */
    virtual double AttributeGradientWithRespectToParameter(
      const arma::vec &parameters, int row_index, int col_index) const = 0;

    /** @brief Returns $\frac{\partial}{\partial \theta} \beta^{\nu}(\theta)$.
     */
    void AttributeGradientWithRespectToParameter(
      const arma::vec &parameters,
      int num_attributes, arma::mat *gradient_out) const {

      gradient_out->set_size(this->num_parameters(), num_attributes);

      for(int j = 0; j < num_attributes; j++) {
        for(int i = 0; i < this->num_parameters(); i++) {
          gradient_out->at(
            i, j) =
              this->AttributeGradientWithRespectToParameter(parameters, i, j);
        }
      }
    }

    virtual void DrawBeta(
      const arma::vec &parameters, arma::vec *beta_out) const = 0;

    virtual int num_parameters() const = 0;

    virtual void Init(const std::string &file_name) const = 0;

    void ChoiceProbabilityWeightedAttributeVector(
      const DCMTableType &dcm_table_in, int person_index,
      const arma::vec &choice_probabilities,
      arma::vec *choice_prob_weighted_attribute_vector) const {

      // Get the number of discrete choices for the given person.
      int num_discrete_choices = dcm_table_in.num_discrete_choices(
                                   person_index);
      choice_prob_weighted_attribute_vector->set_size(
        dcm_table_in.num_attributes());
      choice_prob_weighted_attribute_vector->zeros();
      for(int i = 0; i < choice_probabilities.n_elem; i++) {
        arma::vec attribute_vector;
        dcm_table_in.get_attribute_vector(person_index, i, &attribute_vector);
        (*choice_prob_weighted_attribute_vector) +=
          choice_probabilities[i] * attribute_vector;
      }
    }

    /** @brief Returns the (row, col)-th entry of
     *         $\frac{\partial^2}{\partial \beta^2} P_{i j_i^*} (
     *         \beta^{\nu}(\theta))$ (Equation 8.5)
     */
    double ChoiceProbabilityHessianWithRespectToAttribute(
      const DCMTableType &dcm_table_in, int person_index,
      int row_index, int col_index,
      const arma::vec &choice_probabilities,
      const arma::vec &choice_prob_weighted_attribute_vector) const {

      // Get the discrete choice index.
      int discrete_choice_index =
        dcm_table_in.get_discrete_choice_index(person_index);

      // Get the choice probability for the chosen discrete choice.
      double choice_probability = choice_probabilities[discrete_choice_index];
      arma::vec discrete_choice_attribute_vector;
      dcm_table_in.get_attribute_vector(
        person_index, discrete_choice_index, &discrete_choice_attribute_vector);

      // The (row, col)-th entry of $\bar{X}_i res_{i,j_i^*}(\beta) (
      // \bar_{X}_i res_{i, j_i^*} (\beta) )'$
      double first_part = (
                            discrete_choice_attribute_vector[row_index] -
                            choice_prob_weighted_attribute_vector[row_index]) *
                          (
                            discrete_choice_attribute_vector[col_index] -
                            choice_prob_weighted_attribute_vector[col_index]);

      // The (row, col)-th entry of $\bar{X}_i \bar{P}_i(\beta)
      // ( \bar{X}_i \bar{P}_i (\beta) )$.
      double second_part = choice_prob_weighted_attribute_vector[row_index] *
                           choice_prob_weighted_attribute_vector[col_index];

      // The (row, col)-th entry of $\bar{X}_i \tilde{P}_i(\beta)
      // \bar{X}_i'$
      double third_part = 0;
      int num_discrete_choices = dcm_table_in.num_discrete_choices(
                                   person_index);
      for(int i = 0; i < num_discrete_choices; i++) {
        arma::vec attribute_vector;
        dcm_table_in.get_attribute_vector(person_index, i, &attribute_vector);
        third_part += attribute_vector[row_index] * choice_probabilities[i] *
                      attribute_vector[col_index];
      }

      // Scale the entry by the choice probability.
      return choice_probability * (first_part + second_part - third_part);
    }

    void ChoiceProbabilityHessianWithRespectToAttribute(
      const DCMTableType &dcm_table_in, int person_index,
      const arma::vec &choice_probabilities,
      const arma::vec &choice_prob_weighted_attribute_vector,
      arma::mat *hessian_out) const {

      hessian_out->set_size(
        dcm_table_in.num_attributes(), dcm_table_in.num_attributes());
      for(int j = 0; j < dcm_table_in.num_attributes(); j++) {
        for(int i = 0; i < dcm_table_in.num_attributes(); i++) {
          hessian_out->at(
            i, j) =
              this->ChoiceProbabilityHessianWithRespectToAttribute(
                dcm_table_in, person_index, i, j,
                choice_probabilities, choice_prob_weighted_attribute_vector);
        }
      }
    }

    /** @brief Computes the required quantities in Equation 8.14 (see
     *         dcm_table.h) for a realization of $\beta$ for a given
     *         person.
     */
    void HessianProducts(
      const arma::vec &parameters_in,
      const DCMTableType &dcm_table_in, int person_index,
      const arma::vec &choice_probabilities,
      const arma::vec &choice_prob_weighted_attribute_vector,
      arma::mat *hessian_first_part,
      arma::vec *hessian_second_part) const {

      // Get the discrete choice index.
      int discrete_choice_index =
        dcm_table_in.get_discrete_choice_index(person_index);

      // Compute $\frac{\partial}{\partial \theta} \beta^{\nu}(\theta)
      // \frac{\partial^2}{\partial \beta^2} P_{i j_i^*} (
      // \beta^{\nu}(\theta)) ( \frac{\partial}{\partial \theta}
      // \beta^{\nu}(\theta) )'$.
      arma::mat first;
      arma::mat second;
      this->AttributeGradientWithRespectToParameter(
        parameters_in, dcm_table_in.num_attributes(), &first);
      this->ChoiceProbabilityHessianWithRespectToAttribute(
        dcm_table_in, person_index, choice_probabilities,
        choice_prob_weighted_attribute_vector, &second);
      (*hessian_first_part) = first * second * arma::trans(first);

      // Compute $\frac{\partial}{\partial \theta}
      // \beta^{\nu}(\theta)\frac{\partial}{\partial \beta} P_{i j_i^*} (
      // \beta^{\nu}(\theta))$.
      arma::vec third;
      this->ChoiceProbabilityGradientWithRespectToAttribute(
        dcm_table_in, person_index, choice_probabilities,
        choice_prob_weighted_attribute_vector, &third);
      (*hessian_second_part) = first * third;
    }

    /** @brief Computes $\frac{\partial}{\partial \beta}
     *         P_{i,j}(\beta)$ (Equation 8.2).
     */
    void ChoiceProbabilityGradientWithRespectToAttribute(
      const DCMTableType &dcm_table_in, int person_index,
      const arma::vec &choice_probabilities,
      const arma::vec &choice_prob_weighted_attribute_vector,
      arma::vec *gradient_out) const {

      // Get the discrete choice inde.
      int discrete_choice_index =
        dcm_table_in.get_discrete_choice_index(person_index);

      // Get the discrete choice attribute vector.
      arma::vec discrete_choice_attribute;
      dcm_table_in.get_attribute_vector(
        person_index, discrete_choice_index, &discrete_choice_attribute);
      (*gradient_out) = discrete_choice_attribute -
                        choice_prob_weighted_attribute_vector;

      // Scale the gradient by the choice probability of the discrete
      // choice.
      (*gradient_out) =
        choice_probabilities[discrete_choice_index] * (*gradient_out);
    }

    /** @brief Computes $\frac{\partial}{\partial \theta}
     *         \beta^{\nu}(\theta) \bar{X}_i res_{i,j_i^*} (
     *         \beta^{\nu}(\theta))$ for a realization of $\beta$ for
     *         a given person.
     */
    void ChoiceProbabilityGradientWithRespectToParameter(
      const arma::vec &parameters_in,
      const DCMTableType &dcm_table_in,
      int person_index, const arma::vec &choice_probabilities,
      const arma::vec &choice_prob_weighted_attribute_vector,
      arma::vec *product_out) const {

      // Find the person's discrete choice index.
      int discrete_choice_index =
        dcm_table_in.get_discrete_choice_index(person_index);

      // Initialize the product.
      product_out->set_size(this->num_parameters());

      // Compute the gradient matrix times vector for the person's
      // discrete choice.
      arma::vec attribute_vector;
      dcm_table_in.get_attribute_vector(
        person_index, discrete_choice_index, &attribute_vector);

      // Compute $\bar{X}_i res_{i,j_i^*}(\beta)$.
      arma::vec attribute_vec_sub_choice_prob_weighted_vec =
        attribute_vector - choice_prob_weighted_attribute_vector;

      // For each row index of the gradient,
      for(int k = 0; k < this->num_parameters(); k++) {

        // For each column index of the gradient,
        double dot_product = 0;
        for(unsigned int j = 0; j < attribute_vector.n_elem; j++) {
          dot_product +=
            attribute_vec_sub_choice_prob_weighted_vec[j] *
            this->AttributeGradientWithRespectToParameter(
              parameters_in, k, j);
        }
        (*product_out)[k] = dot_product;
      }
    }
};
}
}

#endif
