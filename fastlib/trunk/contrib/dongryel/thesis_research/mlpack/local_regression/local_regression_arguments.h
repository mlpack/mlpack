/** @file local_regression_arguments.h
 *
 *  The arguments used for the local regression.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_ARGUMENTS_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_ARGUMENTS_H

#include "core/table/table.h"

namespace mlpack {
namespace local_regression {

/** @brief The argument list for the local regression.
 */
template<typename TableType>
class LocalRegressionArguments {
  public:

    /** @brief The bandwidth.
     */
    double bandwidth_;

    /** @brief The kernel.
     */
    std::string kernel_;

    /** @brief The leaf size.
     */
    int leaf_size_;

    /** @brief The file name to output the predictions to.
     */
    std::string predictions_out_;

    /** @brief The prescale option.
     */
    std::string prescale_;

    /** @brief Stores the query table.
     */
    TableType *query_table_;

    /** @brief Stores the reference table.
     */
    TableType *reference_table_;

    /** @brief The relative error.
     */
    double relative_error_;

  public:

    /** @brief The default constructor.
     */
    LocalRegressionArguments() {
      bandwidth_ = 0.0;
      leaf_size_ = 0;
      query_table_ = NULL;
      reference_table_ = NULL;
      relative_error_ = 0.0;
    }

    /** @brief The destructor.
     */
    ~LocalRegressionArguments() {
      if(query_table_ != reference_table_) {
        delete query_table_;
        delete reference_table_;
      }
      else {
        delete reference_table_;
      }
      query_table_ = NULL;
      reference_table_ = NULL;
    }
};
}
}

#endif
