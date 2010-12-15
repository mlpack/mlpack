/** @file mixed_logit_dcm_distribution.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_DCM_DCM_DISTRIBUTION_H
#define MLPACK_MIXED_LOGIT_DCM_DCM_DISTRIBUTION_H

namespace mlpack {
namespace mixed_logit_dcm {

/** @brief The base abstract class for the distribution that
 *         generates each $\beta$ parameter in mixed logit
 *         models. This distribution is parametrized by $\theta$.
 */
template<typename DCMTableType>
class MixedLogitDCMDistribution {
  public:
    virtual double MixedLogitParameterGradient(
      int row_index, int col_index) const = 0;

    virtual int num_parameters() const = 0;

    void MixedLogitParameterGradientProducts(
      DCMTableType *dcm_table_in,
      int person_index,
      const core::table::DensePoint &parameter_vector,
      const core::table::DensePoint &choice_probabilities,
      core::table::DenseMatrix *product_out) {

      // Get the number of discrete choices available for the given
      // person.
      int num_discrete_choices =
        dcm_table_in->num_discrete_choices(person_index);
      product_out->Init(this->num_parameters(), num_discrete_choices);

      // Compute the gradient matrix times vector for each discrete
      // choice.
      for(int i = 0; i < num_discrete_choices; i++) {
        core::table::DensePoint attribute_vector;
        dcm_table_in->get_attribute_vector(person_index, i, &attribute_vector);

        core::table::DensePoint attribute_vector_sub_choice_probability_vector;
        core::math::SubInit(
          choice_probabilities, attribute_vector,
          &attribute_vector_sub_choice_probability_vector);

        // For each row index of the gradient,
        for(int k = 0; k < this->num_parameters(); k++) {

          // For each column index of the gradient,
          double dot_product = 0;
          for(int j = 0; j < attribute_vector.length(); j++) {
            dot_product += attribute_vector_sub_choice_probability_vector[j] *
                           this->MixedLogitParameterGradient(k, j);
          }
          product_out->set(k, i, dot_product);
        }
      }
    }
};
};
};

#endif
