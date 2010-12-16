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
    virtual double MixedLogitParameterGradient(
      int row_index, int col_index) const = 0;

    virtual int num_parameters() const = 0;

    virtual void Init(const std::string &file_name) const = 0;

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


    }

    /** @brief Computes $\frac{\partial}{\partial \theta}
     *         \beta^{\nu}(\theta) \bar{X}_i res_{i,j_i^*} (
     *         \beta^{\nu}(\theta))$ for a realization of $\beta$ for
     *         a given person.
     */
    void MixedLogitParameterGradientProducts(
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
        (*product_out)[k] = dot_product;
      }
    }
};
};
};

#endif
