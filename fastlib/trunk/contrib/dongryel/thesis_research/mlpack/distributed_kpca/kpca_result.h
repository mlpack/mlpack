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

    /** The projected KPCA projections.
     */
    core::table::DenseMatrix kpca_projections_;

  public:

    /** @brief Serialize the KPCA result object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
    }

    /** @brief The default constructor.
     */
    KpcaResult() {
      SetZero();
    }

    void Init(int num_components, int query_points) {
      kpca_projections_.Init(num_components, query_points);
      SetZero();
    }

    void SetZero() {
      kpca_projections_.SetZero();
    }
};
}
}

#endif
