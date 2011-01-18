/** @file kernel_aux.h
 *
 *  The header file for the class for computing auxiliary stuffs for
 *  the kernel functions (derivative, truncation error bound)
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_KERNEL_AUX_H
#define MLPACK_SERIES_EXPANSION_KERNEL_AUX_H

#include <vector>
#include "core/metric_kernels/kernel.h"
#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"
#include "mlpack/series_expansion/bounds_aux.h"
#include "mlpack/series_expansion/cartesian_expansion_global_dev.h"
#include "mlpack/series_expansion/cartesian_expansion_type.h"
#include "mlpack/series_expansion/hypercube_farfield_dev.h"
#include "mlpack/series_expansion/hypercube_local_dev.h"
#include "mlpack/series_expansion/multivariate_farfield_dev.h"
#include "mlpack/series_expansion/multivariate_local_dev.h"

namespace mlpack {
namespace series_expansion {

/** @brief Auxiliary class for multiplicative p^D expansion for
 *         Gaussian kernel.
 */
class GaussianKernelMultAux {

  public:

    static const
    enum mlpack::series_expansion::CartesianExpansionType ExpansionType =
      mlpack::series_expansion::HYPERCUBE;

    typedef core::metric_kernels::GaussianKernel KernelType;

    typedef mlpack::series_expansion::CartesianExpansionGlobal<ExpansionType>
    ExpansionGlobalType;

    typedef mlpack::series_expansion::CartesianFarField <
    ExpansionType > FarFieldType;

    typedef mlpack::series_expansion::CartesianLocal<ExpansionType> LocalType;

  private:

    /** @brief The Gaussian kernel object.
     */
    KernelType kernel_;

    /** @ The series expansion global object.
     */
    ExpansionGlobalType global_;

  public:

    const KernelType &kernel() const {
      return kernel_;
    }

    const ExpansionGlobalType &global() const {
      return global_;
    }

    void Init(double bandwidth, int max_order, int dim) {
      kernel_.Init(bandwidth);
      global_.Init(max_order, dim);
    }

    double BandwidthFactor(double bandwidth_sq) const {
      return sqrt(2 * bandwidth_sq);
    }

    void AllocateDerivativeMap(
      int dim, int order, core::table::DenseMatrix *derivative_map) const {
      derivative_map->Init(dim, order + 1);
    }

    void ComputeDirectionalDerivatives(
      const core::table::DensePoint &x,
      core::table::DenseMatrix *derivative_map, int order) const {

      int dim = x.length();

      // Precompute necessary Hermite polynomials based on coordinate
      // difference.
      for(int d = 0; d < dim; d++) {

        double coord_div_band = x[d];
        double d2 = 2 * coord_div_band;
        double facj = exp(-coord_div_band * coord_div_band);

        derivative_map->set(d, 0, facj);

        if(order > 0) {

          derivative_map->set(d, 1, d2 * facj);

          if(order > 1) {
            for(int k = 1; k < order; k++) {
              int k2 = k * 2;
              derivative_map->set(d, k + 1, d2 * derivative_map->get(d, k) -
                                  k2 * derivative_map->get(d, k - 1));
            }
          }
        }
      } // end of looping over each dimension
    }

    double ComputePartialDerivative(
      const core::table::DenseMatrix &derivative_map,
      const std::vector<short int> &mapping) const {

      double partial_derivative = 1.0;
      for(unsigned int d = 0; d < mapping.size(); d++) {
        partial_derivative *= derivative_map.get(d, mapping[d]);
      }
      return partial_derivative;
    }

    template<typename BoundType>
    int OrderForEvaluatingFarField(
      const BoundType &far_field_region, const BoundType &local_field_region,
      double min_dist_sqd_regions, double max_dist_sqd_regions,
      double max_error, double *actual_error) const {

      // Find out the maximum side length of the bounding box of the
      // far-field region.
      double max_far_field_length =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          far_field_region);

      double two_times_bandwidth = sqrt(kernel_.bandwidth_sq()) * 2;
      double r = max_far_field_length / two_times_bandwidth;

      int dim = global_.get_dimension();
      double r_raised_to_p_alpha = 1.0;
      double ret, ret2;
      int p_alpha = 0;
      double factorialvalue = 1.0;
      double first_factor, second_factor;
      double one_minus_r;

      // In this case, it is "impossible" to prune for the Gaussian kernel.
      if(r >= 1.0) {
        return -1;
      }
      one_minus_r = 1.0 - r;
      ret = 1.0 / pow(one_minus_r, dim);

      do {
        factorialvalue *= (p_alpha + 1);

        if(factorialvalue < 0.0 || p_alpha > global_.get_max_order()) {
          return -1;
        }

        r_raised_to_p_alpha *= r;
        first_factor = 1.0 - r_raised_to_p_alpha;
        second_factor = r_raised_to_p_alpha / sqrt(factorialvalue);

        ret2 = ret * (pow((first_factor + second_factor), dim) -
                      pow(first_factor, dim));

        if(ret2 <= max_error) {
          break;
        }

        p_alpha++;

      }
      while(1);

      *actual_error = ret2;
      return p_alpha;
    }

    template<typename BoundType>
    int OrderForConvertingFromFarFieldToLocal(
      const BoundType &far_field_region,
      const BoundType &local_field_region, double min_dist_sqd_regions,
      double max_dist_sqd_regions, double max_error,
      double *actual_error) const {

      // Find out the maximum side length of the bounding box of the
      // far-field region and the local-field regions.
      double max_far_field_length =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          far_field_region);
      double max_local_field_length =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          local_field_region);

      double two_times_bandwidth = sqrt(kernel_.bandwidth_sq()) * 2;
      double r = max_far_field_length / two_times_bandwidth;
      double r2 = max_local_field_length / two_times_bandwidth;

      int dim = global_.get_dimension();
      double r_raised_to_p_alpha = 1.0;
      double ret, ret2;
      int p_alpha = 0;
      double factorialvalue = 1.0;
      double first_factor, second_factor;
      double one_minus_two_r, two_r;

      // In this case, it is "impossible" to prune for the Gaussian kernel.
      if(r >= 0.5 || r2 >= 0.5) {
        return -1;
      }

      r = max(r, r2);
      two_r = 2.0 * r;
      one_minus_two_r = 1.0 - two_r;
      ret = 1.0 / pow(one_minus_two_r * one_minus_two_r, dim);

      do {
        factorialvalue *= (p_alpha + 1);

        if(factorialvalue < 0.0 || p_alpha > global_.get_max_order()) {
          return -1;
        }

        r_raised_to_p_alpha *= two_r;
        first_factor = 1.0 - r_raised_to_p_alpha;
        first_factor *= first_factor;
        second_factor = r_raised_to_p_alpha * (2.0 - r_raised_to_p_alpha)
                        / sqrt(factorialvalue);

        ret2 = ret * (pow((first_factor + second_factor), dim) -
                      pow(first_factor, dim));

        if(ret2 <= max_error) {
          break;
        }

        p_alpha++;

      }
      while(1);

      *actual_error = ret2;
      return p_alpha;
    }

    template<typename BoundType>
    int OrderForEvaluatingLocal(
      const BoundType &far_field_region,
      const BoundType &local_field_region,
      double min_dist_sqd_regions,
      double max_dist_sqd_regions, double max_error,
      double *actual_error) const {

      // Find out the maximum side length of the local field region.
      double max_local_field_length =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          local_field_region);

      double two_times_bandwidth = sqrt(kernel_.bandwidth_sq()) * 2;
      double r = max_local_field_length / two_times_bandwidth;

      int dim = global_.get_dimension();
      double r_raised_to_p_alpha = 1.0;
      double ret, ret2;
      int p_alpha = 0;
      double factorialvalue = 1.0;
      double first_factor, second_factor;
      double one_minus_r;

      // In this case, it is "impossible" to prune for the Gaussian kernel.
      if(r >= 1.0) {
        return -1;
      }
      one_minus_r = 1.0 - r;
      ret = 1.0 / pow(one_minus_r, dim);

      do {
        factorialvalue *= (p_alpha + 1);

        if(factorialvalue < 0.0 || p_alpha > global_.get_max_order()) {
          return -1;
        }

        r_raised_to_p_alpha *= r;
        first_factor = 1.0 - r_raised_to_p_alpha;
        second_factor = r_raised_to_p_alpha / sqrt(factorialvalue);

        ret2 = ret * (pow((first_factor + second_factor), dim) -
                      pow(first_factor, dim));

        if(ret2 <= max_error) {
          break;
        }

        p_alpha++;

      }
      while(1);

      *actual_error = ret2;
      return p_alpha;
    }
};

/** @brief Auxiliary class for Gaussian kernel for performing an
 *         $O(D^p)$ expansion.
 */
class GaussianKernelAux {

  public:

    static const
    enum mlpack::series_expansion::CartesianExpansionType ExpansionType =
      mlpack::series_expansion::MULTIVARIATE;

    typedef core::metric_kernels::GaussianKernel KernelType;

    typedef mlpack::series_expansion::CartesianExpansionGlobal<ExpansionType>
    ExpansionGlobalType;

    typedef mlpack::series_expansion::CartesianFarField<ExpansionType>
    FarFieldType;

    typedef mlpack::series_expansion::CartesianLocal<ExpansionType> LocalType;

  private:

    /** pointer to the Gaussian kernel */
    KernelType kernel_;

    /** pointer to the series expansion auxiliary object */
    ExpansionGlobalType global_;

  public:

    const KernelType &kernel() const {
      return kernel_;
    }

    const ExpansionGlobalType &global() const {
      return global_;
    }

    void Init(double bandwidth, int max_order, int dim) {
      kernel_.Init(bandwidth);
      global_.Init(max_order, dim);
    }

    double BandwidthFactor(double bandwidth_sq) const {
      return sqrt(2 * bandwidth_sq);
    }

    void AllocateDerivativeMap(int dim, int order,
                               core::table::DenseMatrix *derivative_map) const {
      derivative_map->Init(dim, order + 1);
    }

    void ComputeDirectionalDerivatives(
      const core::table::DensePoint &x,
      core::table::DenseMatrix *derivative_map, int order) const {

      int dim = x.length();

      // Precompute necessary Hermite polynomials based on coordinate
      // difference.
      for(int d = 0; d < dim; d++) {

        double coord_div_band = x[d];
        double d2 = 2 * coord_div_band;
        double facj = exp(-coord_div_band * coord_div_band);

        derivative_map->set(d, 0, facj);

        if(order > 0) {

          derivative_map->set(d, 1, d2 * facj);

          if(order > 1) {
            for(int k = 1; k < order; k++) {
              int k2 = k * 2;
              derivative_map->set(
                d, k + 1, d2 * derivative_map->get(d, k) -
                k2 * derivative_map->get(d, k - 1));
            }
          }
        }
      } // end of looping over each dimension
    }

    double ComputePartialDerivative(
      const core::table::DenseMatrix &derivative_map,
      const std::vector<short int> &mapping) const {

      double partial_derivative = 1.0;

      for(int d = 0; d < mapping.size(); d++) {
        partial_derivative *= derivative_map.get(d, mapping[d]);
      }
      return partial_derivative;
    }

    template<typename BoundType>
    int OrderForConvolvingFarField(
      const BoundType &far_field_region,
      const core::table::DensePoint &far_field_region_centroid,
      const BoundType &local_field_region,
      const core::table::DensePoint &local_field_region_centroid,
      double min_dist_sqd_regions,
      double max_dist_sqd_regions,
      double max_error,
      double *actual_error) const {

      core::metric_kernels::LMetric<2> l2_metric;
      double squared_distance_between_two_centroids =
        l2_metric.DistanceSq(
          far_field_region_centroid, local_field_region_centroid);
      double frontfactor =
        exp(-squared_distance_between_two_centroids /
            (4 * kernel_.bandwidth_sq()));
      int max_order = global_.get_max_order();

      // Find out the widest dimension of the far-field region and its
      // length.
      double far_field_widest_width =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          far_field_region);
      double local_field_widest_width =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          local_field_region);

      double two_bandwidth = 2 * sqrt(kernel_.bandwidth_sq());
      double r = (far_field_widest_width + local_field_widest_width) /
                 two_bandwidth;

      double r_raised_to_p_alpha = 1.0;
      double ret;
      int p_alpha = 0;

      do {

        if(p_alpha > max_order - 1) {
          return -1;
        }

        r_raised_to_p_alpha *= r;
        frontfactor /= sqrt(p_alpha + 1);

        ret = frontfactor * r_raised_to_p_alpha;

        if(ret > max_error) {
          p_alpha++;
        }
        else {
          break;
        }
      }
      while(1);

      *actual_error = ret;
      return p_alpha;
    }

    template<typename BoundType>
    int OrderForEvaluatingFarField(
      const BoundType &far_field_region,
      const BoundType &local_field_region,
      double min_dist_sqd_regions,
      double max_dist_sqd_regions,
      double max_error,
      double *actual_error) const {

      double frontfactor =
        exp(-min_dist_sqd_regions / (4 * kernel_.bandwidth_sq()));
      int max_order = global_.get_max_order();

      // Find out the widest dimension of the far-field region and its
      // length.
      double widest_width =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          far_field_region);

      double two_bandwidth = 2 * sqrt(kernel_.bandwidth_sq());
      double r = widest_width / two_bandwidth;

      double r_raised_to_p_alpha = 1.0;
      double ret;
      int p_alpha = 0;

      do {

        if(p_alpha > max_order - 1) {
          return -1;
        }

        r_raised_to_p_alpha *= r;
        frontfactor /= sqrt(p_alpha + 1);

        ret = frontfactor * r_raised_to_p_alpha;

        if(ret > max_error) {
          p_alpha++;
        }
        else {
          break;
        }
      }
      while(1);

      *actual_error = ret;
      return p_alpha;
    }

    template<typename BoundType>
    int OrderForConvertingFromFarFieldToLocal(
      const BoundType &far_field_region, const BoundType &local_field_region,
      double min_dist_sqd_regions, double max_dist_sqd_regions,
      double max_error, double *actual_error) const {

      // Find out the maximum side length of the reference and the query
      // regions.
      double max_ref_length =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          far_field_region);
      double max_query_length =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          local_field_region);

      double two_times_bandwidth = sqrt(kernel_.bandwidth_sq()) * 2;
      double r_R = max_ref_length / two_times_bandwidth;
      double r_Q = max_query_length / two_times_bandwidth;

      int p_alpha = -1;
      double r_Q_raised_to_p = 1.0;
      double r_R_raised_to_p = 1.0;
      double ret2;
      double frontfactor =
        exp(-min_dist_sqd_regions / (4.0 * kernel_.bandwidth_sq()));
      double first_factorial = 1.0;
      double second_factorial = 1.0;
      double r_Q_raised_to_p_cumulative = 1;

      do {
        p_alpha++;

        if(p_alpha > global_.get_max_order() - 1) {
          return -1;
        }

        first_factorial *= (p_alpha + 1);
        if(p_alpha > 0) {
          second_factorial *= sqrt(2 * p_alpha * (2 * p_alpha + 1));
        }
        r_Q_raised_to_p *= r_Q;
        r_R_raised_to_p *= r_R;

        ret2 = frontfactor *
               (1.0 / first_factorial * r_R_raised_to_p * second_factorial *
                r_Q_raised_to_p_cumulative +
                1.0 / sqrt(first_factorial) * r_Q_raised_to_p);

        r_Q_raised_to_p_cumulative +=
          r_Q_raised_to_p /
          ((p_alpha > 0) ? (first_factorial / (p_alpha + 1)) : first_factorial);

      }
      while(ret2 >= max_error);

      *actual_error = ret2;
      return p_alpha;
    }

    template<typename BoundType>
    int OrderForEvaluatingLocal(
      const BoundType &far_field_region, const BoundType &local_field_region,
      double min_dist_sqd_regions, double max_dist_sqd_regions,
      double max_error, double *actual_error) const {

      double frontfactor =
        exp(-min_dist_sqd_regions / (4 * kernel_.bandwidth_sq()));
      int max_order = global_.get_max_order();

      // Find out the widest dimension of the local field region and its
      // length.
      double widest_width =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          local_field_region);

      double two_bandwidth = 2 * sqrt(kernel_.bandwidth_sq());
      double r = widest_width / two_bandwidth;

      double r_raised_to_p_alpha = 1.0;
      double ret;
      int p_alpha = 0;

      do {

        if(p_alpha > max_order - 1) {
          return -1;
        }

        r_raised_to_p_alpha *= r;
        frontfactor /= sqrt(p_alpha + 1);

        ret = frontfactor * r_raised_to_p_alpha;

        if(ret > max_error) {
          p_alpha++;
        }
        else {
          break;
        }
      }
      while(1);

      *actual_error = ret;
      return p_alpha;
    }
};

/** @brief Auxilairy computer class for Epanechnikov kernel.
 */
class EpanKernelAux {

  public:

    static const
    enum mlpack::series_expansion::CartesianExpansionType ExpansionType =
      mlpack::series_expansion::MULTIVARIATE;

    typedef core::metric_kernels::EpanKernel KernelType;

    typedef mlpack::series_expansion::CartesianExpansionGlobal<ExpansionType>
    ExpansionGlobalType;

    typedef mlpack::series_expansion::CartesianFarField<ExpansionType>
    FarFieldType;

    typedef mlpack::series_expansion::CartesianLocal<ExpansionType> LocalType;

  private:

    KernelType kernel_;

    ExpansionGlobalType global_;

    InversePowDistKernelAux squared_component_;

  public:

    const KernelType &kernel() const {
      return kernel_;
    }

    const ExpansionGlobalType &global() const {
      return global_;
    }

    void Init(double bandwidth, int max_order, int dim) {
      kernel_.Init(bandwidth);
      global_.Init(max_order, dim);

      // This is for doing an expansion on $||x||^2$ part.
      squared_component_.Init(-2, max_order, dim);
    }

    double BandwidthFactor(double bandwidth_sq) const {
      return sqrt(bandwidth_sq);
    }

    void AllocateDerivativeMap(
      int dim, int order, core::table::DenseMatrix *derivative_map) const {
      derivative_map->Init(global_.get_total_num_coeffs(order), 1);
    }

    void ComputeDirectionalDerivatives(
      const core::table::DensePoint &x,
      core::table::DenseMatrix *derivative_map, int order) const {

      // Compute the derivatives for $||x||^2$ and negate it. Then, add
      // $(1, 0, 0, ... 0)$ to it.
      squared_component_.ComputeDirectionalDerivatives(
        x, derivative_map, order);

      la::Scale(derivative_map->n_rows(), -1, derivative_map->GetColumnPtr(0));

      (derivative_map->GetColumnPtr(0))[0] += 1.0;
    }

    double ComputePartialDerivative(
      const core::table::DenseMatrix &derivative_map,
      const std::vector<short int> &mapping) const {

      return derivative_map.get(global_.ComputeMultiindexPosition(mapping), 0);
    }

    template<typename BoundType>
    int OrderForEvaluatingFarField(
      const BoundType &far_field_region, const BoundType &local_field_region,
      double min_dist_sqd_regions, double max_dist_sqd_regions,
      double max_error, double *actual_error) const {

      // First check that the maximum distances between the two regions are
      // within the bandwidth, otherwise the expansion is not valid.
      if(max_dist_sqd_regions > kernel_.bandwidth_sq()) {
        return -1;
      }

      // Find out the widest dimension and its length of the far-field
      // region.
      double widest_width =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          far_field_region);

      // Find out the max distance between query and reference region in L1
      // sense.
      int dim;
      double farthest_distance_manhattan =
        mlpack::series_expansion::BoundsAux::MaxL1Distance(
          far_field_region, local_field_region, &dim);

      // Divide by the two times the bandwidth to find out how wide it is
      // in terms of the bandwidth.
      double two_bandwidth = 2 * sqrt(kernel_.bandwidth_sq());
      double r = widest_width / two_bandwidth;
      farthest_distance_manhattan /= sqrt(kernel_.bandwidth_sq());

      // Try the 0-th order approximation first.
      double error = 2 * dim * farthest_distance_manhattan * r;
      if(error < max_error) {
        *actual_error = error;
        return 0;
      }

      // Try the 1st order approximation later.
      error = dim * r * r;
      if(error < max_error) {
        *actual_error = error;
        return 1;
      }

      // Failing all above, take up to 2nd terms.
      *actual_error = 0;
      return 2;
    }

    template<typename BoundType>
    int OrderForConvertingFromFarFieldToLocal(
      const BoundType &far_field_region, const BoundType &local_field_region,
      double min_dist_sqd_regions, double max_dist_sqd_regions,
      double max_error, double *actual_error) const {

      // First check that the maximum distances between the two regions are
      // within the bandwidth, otherwise the expansion is not valid.
      if(max_dist_sqd_regions > kernel_.bandwidth_sq() ||
          min_dist_sqd_regions == 0) {
        return -1;
      }
      else {
        *actual_error = 0;
        return 2;
      }
    }

    template<typename BoundType>
    int OrderForEvaluatingLocal(
      const BoundType &far_field_region,
      const BoundType &local_field_region,
      double min_dist_sqd_regions, double max_dist_sqd_regions,
      double max_error, double *actual_error) const {

      // First check that the maximum distances between the two regions are
      // within the bandwidth, otherwise the expansion is not valid.
      if(max_dist_sqd_regions > kernel_.bandwidth_sq()) {
        return -1;
      }

      // Find out the widest dimension and its length of the local
      // region.
      double widest_width =
        mlpack::series_expansion::BoundsAux::MaxSideLengthOfBoundingBox(
          local_field_region);

      // Find out the max distance between query and reference region in L1
      // sense
      int dim;
      double farthest_distance_manhattan =
        mlpack::series_expansion::BoundsAux::MaxL1Distance(
          far_field_region, local_field_region, &dim);

      // divide by the two times the bandwidth to find out how wide it is
      // in terms of the bandwidth
      double two_bandwidth = 2 * sqrt(kernel_.bandwidth_sq());
      double r = widest_width / two_bandwidth;
      farthest_distance_manhattan /= sqrt(kernel_.bandwidth_sq());

      // try the 0-th order approximation first
      double error = 2 * dim * farthest_distance_manhattan * r;
      if(error < max_error) {
        *actual_error = error;
        return 0;
      }

      // try the 1st order approximation later
      error = dim * math::Sqr(farthest_distance_manhattan);
      if(error < max_error) {
        *actual_error = error;
        return 1;
      }

      // failing all above, take up to 2nd terms
      *actual_error = 0;
      return 2;
    }
};
}
}

#endif
