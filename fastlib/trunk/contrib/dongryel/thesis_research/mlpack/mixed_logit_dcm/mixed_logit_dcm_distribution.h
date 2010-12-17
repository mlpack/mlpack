/** @file mixed_logit_dcm_distribution.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_DCM_DISTRIBUTION_H
#define MLPACK_MIXED_LOGIT_DCM_DCM_DISTRIBUTION_H

#include "core/table/dense_point.h"
#include "core/table/dense_matrix.h"

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The base abstract class for the distribution that
 *         generates each $\beta$ parameter in mixed logit
 *         models. This distribution is parametrized by $\theta$.
 */
template<typename DCMTableType>
class MixedLogitDCMDistribution {
  private:
    core::table::DensePoint parameters_;

  public:

    /** @brief Returns the (row, col)-th entry of
     *         $\frac{\partial}{\partial \theta} \beta^{\nu}(\theta)$
     */
    virtual double AttributeGradientWithRespectToParameter(
      int row_index, int col_index) const = 0;

    virtual int num_parameters() const = 0;

    virtual void Init(const std::string &file_name) const = 0;

    void ChoiceProbabilityWeightedAttributeVector(
      DCMTableType *dcm_table_in, int person_index,
      const core::table::DensePoint &choice_probabilities,
      core::table::DensePoint *choice_prob_weighted_attribute_vector) const {

      // Get the number of discrete choices for the given person.
      int num_discrete_choices = dcm_table_in->num_discrete_choices(
                                   person_index);
      choice_prob_weighted_attribute_vector->Init(
        dcm_table_in->num_attributes());
      choice_prob_weighted_attribute_vector->SetZero();
      for(int i = 0; i < choice_probabilities.length(); i++) {
        core::table::DensePoint attribute_vector;
        dcm_table_in->get_attribute_vector(person_index, i);
        core::math::AddExpert(
          choice_probabilities[i], attribute_vector,
          choice_prob_weighted_attribute_vector);
      }
    }

    /** @brief Returns the (row, col)-th entry of
     *         $\frac{\partial}{\partial \beta^2} P_{i j_i^*} (
     *         \beta^{\nu}(\theta))$ (Equation 8.5)
     */
    double HessianChoiceProbability(
      int row_index, int col_index,
      const core::table::DensePoint &choice_probabilities,
      int discrete_choice_index) const {

      double choice_probability = choice_probabilities[discrete_choice_index];
      double unnormalized_entry = 0;

      // Scale the entry by the choice probability.
      return choice_probability * unnormalized_entry;
    }

    /** @brief Computes the required quantities in Equation 8.14 (see
     *         dcm_table.h) for a realization of $\beta$ for a given
     *         person.
     */
    void HessianProducts(
      DCMTableType *dcm_table_in,
      int person_index, int discrete_choice_index,
      const core::table::DensePoint &parameter_vector,
      const core::table::DensePoint &choice_probabilities,
      core::table::DenseMatrix *hessian_first_part,
      core::table::DensePoint *hessian_second_part) const {

      // Intialize the output matrices.
      hessian_first_part->Init(
        this->num_parameters(), this->num_parameters());
      hessian_second_part->Init(this->num_parameters());
      hessian_first_part->SetZero();
      hessian_second_part->SetZero();


    }

    /** @brief Computes $\frac{\partial}{\partial \theta}
     *         \beta^{\nu}(\theta) \bar{X}_i res_{i,j_i^*} (
     *         \beta^{\nu}(\theta))$ for a realization of $\beta$ for
     *         a given person.
     */
    void ProductAttributeGradientWithRespectToParameter(
      DCMTableType *dcm_table_in,
      int person_index, int discrete_choice_index,
      const core::table::DensePoint &parameter_vector,
      const core::table::DensePoint &choice_probabilities,
      core::table::DensePoint *product_out) const {

      // Initialize the product.
      product_out->Init(this->num_parameters());

      // Compute the gradient matrix times vector for the person's
      // discrete choice.
      core::table::DensePoint attribute_vector;
      dcm_table_in->get_attribute_vector(
        person_index, discrete_choice_index, &attribute_vector);

      // Compute the choice probability weighted attribute
      // vector. This is $\bar{X}_i \bar{P}_i(\beta)$.
      core::table::DensePoint choice_prob_weighted_attribute_vector;
      ChoiceProbabilityWeightedAttributeVector(
        dcm_table_in, person_index, choice_probabilities,
        &choice_prob_weighted_attribute_vector);

      core::table::DensePoint attribute_vec_sub_choice_prob_weighted_vec;
      core::math::SubInit(
        choice_prob_weighted_attribute_vector, attribute_vector,
        &attribute_vec_sub_choice_prob_weighted_vec);

      // For each row index of the gradient,
      for(int k = 0; k < this->num_parameters(); k++) {

        // For each column index of the gradient,
        double dot_product = 0;
        for(int j = 0; j < attribute_vector.length(); j++) {
          dot_product += attribute_vec_sub_choice_prob_weighted_vec[j] *
                         this->AttributeGradientWithRespectToParameter(k, j);
        }
        (*product_out)[k] = dot_product;
      }
    }
};
};
};

#endif
