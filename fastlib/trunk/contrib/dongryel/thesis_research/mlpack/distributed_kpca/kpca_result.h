/** @file kpca_result.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KPCA_KPCA_RESULT_H
#define MLPACK_DISTRIBUTED_KPCA_KPCA_RESULT_H

#include <boost/math/distributions/normal.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include "core/metric_kernels/kernel.h"
#include "core/monte_carlo/mean_variance_pair_matrix.h"
#include "core/tree/statistic.h"
#include "core/table/table.h"

namespace mlpack {
namespace distributed_kpca {

/** @brief Represents the storage of KPCA computation results.
 */
class KpcaResult {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

    /** @brief The lower bound on the projected KPCA projections.
     */
    core::table::DenseMatrix kpca_projections_l_;

    /** @brief The projected KPCA projections.
     */
    core::table::DenseMatrix kpca_projections_;

    /** @brief The upper bound on the projected KPCA projections.
     */
    core::table::DenseMatrix kpca_projections_u_;

  public:

    /** @brief Serialize the KPCA result object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & kpca_projections_l_;
      ar & kpca_projections_;
      ar & kpca_projections_u_;
    }

    /** @brief The default constructor.
     */
    KpcaResult() {
      SetZero();
    }

    void Export(
      double num_standard_deviations,
      const core::monte_carlo::MeanVariancePairVector &kernel_sum) {
      for(int i = 0; i < kpca_projections_.n_cols(); i++) {
        double deviation = num_standard_deviations *
                           sqrt(kernel_sum[i].sample_mean_variance());
        kpca_projections_.set(0, i, kernel_sum[i].sample_mean() - deviation);
        kpca_projections_.set(1, i, kernel_sum[i].sample_mean());
        kpca_projections_.set(2, i, kernel_sum[i].sample_mean() + deviation);
      }
    }

    void Print(const std::string &file_name) const {
      FILE *file_output = fopen(file_name.c_str(), "w+");
      for(int i = 0; i < kpca_projections_.n_cols(); i++) {
        for(int j = 0; j < kpca_projections_.n_rows(); j++) {
          fprintf(
            file_output, "(%g %g %g) ",
            kpca_projections_l_.get(j, i),
            kpca_projections_.get(j, i),
            kpca_projections_u_.get(j, i));
        }
        fprintf(file_output, "\n");
      }
      fclose(file_output);
    }

    void Init(int num_components, int query_points) {
      kpca_projections_l_.Init(num_components, query_points);
      kpca_projections_.Init(num_components, query_points);
      kpca_projections_u_.Init(num_components, query_points);
      SetZero();
    }

    void SetZero() {
      kpca_projections_l_.SetZero();
      kpca_projections_.SetZero();
      kpca_projections_u_.SetZero();
    }
};
}
}

#endif
