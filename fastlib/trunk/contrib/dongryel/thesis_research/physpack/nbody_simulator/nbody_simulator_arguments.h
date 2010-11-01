/** @file nbody_simulator_arguments.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef PHYSPACK_NBODY_SIMULATOR_NBODY_SIMULATOR_ARGUMENTS_H
#define PHYSPACK_NBODY_SIMULATOR_NBODY_SIMULATOR_ARGUMENTS_H

#include "core/table/table.h"
#include "core/metric_kernels/abstract_metric.h"

namespace physpack {
namespace nbody_simulator {
template<typename TableType>
class NbodySimulatorArguments {
  public:

    std::string potentials_out_;

    int leaf_size_;

    TableType *table_;

    double relative_error_;

    double probability_;

    core::metric_kernels::AbstractMetric *metric_;

    double summary_compute_quantile_;

  public:
    NbodySimulatorArguments() {
      leaf_size_ = 0;
      table_ = NULL;
      relative_error_ = 0.0;
      probability_ = 0.0;
      metric_ = NULL;
      summary_compute_quantile_ = 0.0;
    }

    ~NbodySimulatorArguments() {
      if(table_ != NULL) {
        delete table_;
        table_ = NULL;
      }
      if(metric_ != NULL) {
        delete metric_;
        metric_ = NULL;
      }
    }
};
};
};

#endif
