/** @file inverse_pow_dist_kernel_aux.h
 *
 *  Defines the kernel of inverse distance power of the form $1 /
 *  r^{\alpha}$.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_INVERSE_POW_DIST_KERNEL_AUX_H
#define MLPACK_SERIES_EXPANSION_INVERSE_POW_DIST_KERNEL_AUX_H

#include <vector>
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"
#include "mlpack/series_expansion/cartesian_expansion_type.h"
#include "mlpack/series_expansion/inverse_pow_dist_kernel.h"

namespace mlpack {
namespace series_expansion {

/** @brief The auxilary class for $1 / r^{\lambda}$ kernels using
 *         $O(D^p)$ expansion.
 */
class InversePowDistKernelAux {

  private:
    void SubFrom_(
      int dimension, int decrement,
      const std::vector<short int> &subtract_from,
      std::vector<short int> &result) const {

      for(int d = 0; d < static_cast<int>(subtract_from.size()); d++) {
        if(d == dimension) {
          result[d] = subtract_from[d] - decrement;
        }
        else {
          result[d] = subtract_from[d];
        }
      }
    }

  public:

    static const
    enum mlpack::series_expansion::CartesianExpansionType ExpansionType =
      mlpack::series_expansion::MULTIVARIATE;

    typedef mlpack::series_expansion::InversePowDistKernel KernelType;

    typedef mlpack::series_expansion::CartesianExpansionGlobal<ExpansionType>
    ExpansionGlobalType;

    typedef mlpack::series_expansion::CartesianFarField<ExpansionType>
    FarFieldType;

    typedef mlpack::series_expansion::CartesianLocal<ExpansionType> LocalType;

  private:

    /** @brief The actual kernel object.
     */
    KernelType kernel_;

    /** @brief The series expansion object.
     */
    ExpansionGlobalType global_;

  public:

    const KernelType &kernel() const {
      return kernel_;
    }

    const ExpansionGlobalType &global() const {
      return global_;
    }

    void Init(double power, int max_order, int dim) {
      kernel_.Init(power, dim);
      global_.Init(max_order, dim);
    }

    void AllocateDerivativeMap(
      int dim, int order, core::table::DenseMatrix *derivative_map) const {
      derivative_map->Init(global_.get_total_num_coeffs(order), 1);
    }

    void ComputeDirectionalDerivatives(
      const core::table::DensePoint &x,
      core::table::DenseMatrix *derivative_map, int order) const {

      derivative_map->SetZero();

      // Squared L2 norm of the vector.
      arma::vec x_alias;
      core::table::DensePointToArmaVec(x, &x_alias);
      double squared_l2_norm = arma::dot(x_alias, x_alias);

      // Temporary variable to look for arithmetic operations on
      // multiindex.
      std::vector<short int> tmp_multiindex(global_.get_dimension());

      // Get the inverse multiindex factorial factors.
      const core::table::DensePoint &inv_multiindex_factorials =
        global_.get_inv_multiindex_factorials();

      for(int i = 0; i < derivative_map->n_rows(); i++) {

        // Contribution to the current multiindex position.
        double contribution = 0;

        // Retrieve the multiindex mapping.
        const std::vector<short int> &multiindex = global_.get_multiindex(i);

        // $D_{x}^{0} \phi_{\nu, d}(x)$ should be computed normally.
        if(i == 0) {
          derivative_map->set(0, 0, kernel_.EvalUnnorm(x));
          continue;
        }

        // The sum of the indices.
        int sum_of_indices = 0;
        for(int d = 0; d < x.length(); d++) {
          sum_of_indices += multiindex[d];
        }

        // The first factor multiplied.
        double first_factor = 2 * sum_of_indices + kernel_.lambda() - 2;

        // The second factor multilied.
        double second_factor = sum_of_indices + kernel_.lambda() - 2;

        // Compute the contribution of $D_{x}^{n - e_d} \phi_{\nu,
        // d}(x)$ component for each $d$.
        for(int d = 0; d < x.length(); d++) {

          // Subtract 1 from the given dimension.
          SubFrom_(d, 1, multiindex, tmp_multiindex);
          int n_minus_e_d_position =
            global_.ComputeMultiindexPosition(tmp_multiindex);
          if(n_minus_e_d_position >= 0) {
            contribution += first_factor * x[d] *
                            derivative_map->get(n_minus_e_d_position, 0) *
                            inv_multiindex_factorials[n_minus_e_d_position];
          }

          // Subtract 2 from the given dimension.
          SubFrom_(d, 2, multiindex, tmp_multiindex);
          int n_minus_two_e_d_position =
            global_.ComputeMultiindexPosition(tmp_multiindex);
          if(n_minus_two_e_d_position >= 0) {
            contribution += second_factor *
                            derivative_map->get(n_minus_two_e_d_position, 0) *
                            inv_multiindex_factorials[n_minus_two_e_d_position];
          }

        } // end of iterating over each dimension.

        // Set the final contribution for this multiindex.
        if(squared_l2_norm == 0) {
          derivative_map->set(i, 0, 0);
        }
        else {
          derivative_map->set(i, 0, -contribution / squared_l2_norm /
                              sum_of_indices / inv_multiindex_factorials[i]);
        }

      } // end of iterating over all required multiindex positions...

      // Iterate again, and invert the sum if the sum of the indices of
      // the current mapping is odd.
      for(int i = 1; i < derivative_map->n_rows(); i++) {

        // Retrieve the multiindex mapping.
        const std::vector<short int> &multiindex = global_.get_multiindex(i);

        // The sum of the indices.
        int sum_of_indices = 0;
        for(int d = 0; d < x.length(); d++) {
          sum_of_indices += multiindex[d];
        }

        if(sum_of_indices % 2 == 1) {
          derivative_map->set(i, 0, -derivative_map->get(i, 0));
        }
      }
    }

    double ComputePartialDerivative(
      const core::table::DenseMatrix &derivative_map,
      const std::vector<short int> &mapping) const {

      return derivative_map.get(global_.ComputeMultiindexPosition(mapping), 0);
    }
};
}
}

#endif
