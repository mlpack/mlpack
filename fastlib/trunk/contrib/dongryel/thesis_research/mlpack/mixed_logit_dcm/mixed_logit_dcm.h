/** @file mixed_logit_dcm.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_MIXED_LOGIT_DCM_H
#define MLPACK_MIXED_LOGIT_MIXED_LOGIT_DCM_H

#include <vector>
#include "boost/program_options.hpp"
#include "core/table/table.h"
#include "mlpack/mixed_logit_dcm/dcm_table.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_arguments.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_result.h"

namespace mlpack {
namespace mixed_logit_dcm {
template<typename IncomingTableType>
class MixedLogitDCM {
  public:
    typedef IncomingTableType TableType;

    typedef mlpack::mixed_logit_dcm::DCMTable<TableType> DCMTableType;

    typedef mlpack::mixed_logit_dcm::MixedLogitDCMSampling<DCMTableType> SamplingType;

  private:

    /** @brief Computes the sample data error (Section 3.1)
     */
    double SampleDataError_(
      const SamplingType &first_sample,
      const SamplingType &second_sample) const;

  public:

    TableType *attribute_table();

    void Init(
      mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
      TableType > &arguments_in);

    void Compute(
      const mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
      TableType > &arguments_in,
      mlpack::mixed_logit_dcm::MixedLogitDCMResult *result_out);

    static void ParseArguments(
      const std::vector<std::string> &args,
      mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
      TableType > *arguments_out);

    static void ParseArguments(
      int argc,
      char *argv[],
      mlpack::mixed_logit_dcm::MixedLogitDCMArguments <
      TableType > *arguments_out);

  private:

    DCMTableType table_;

  private:

    static bool ConstructBoostVariableMap_(
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm);
};
};
};

#endif
