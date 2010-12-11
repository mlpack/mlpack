/** @file mixed_logit_dcm.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MIXED_LOGIT_MIXED_LOGIT_DCM_H
#define MLPACK_MIXED_LOGIT_MIXED_LOGIT_DCM_H

#include "boost/program_options.hpp"
#include "core/table/table.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_arguments.h"
#include "mlpack/mixed_logit_dcm/mixed_logit_dcm_result.h"

namespace mlpack {
namespace mixed_logit_dcm {
template<typename IncomingTableType>
class MixedLogitDCM {
  public:
    typedef IncomingTableType TableType;

  public:

    /**
     * @brief returns a pointer to the query table
     */
    TableType *query_table();

    /**
     * @brief returns a pointer to the reference table
     */
    TableType *reference_table();

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

    TableType *query_table_;
    TableType *reference_table_;
    bool is_monochromatic_;

  private:

    static bool ConstructBoostVariableMap_(
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm);
};
};
};

#endif
