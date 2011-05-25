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
template<typename TableType, typename MetricType>
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

    /** @brief The absolute error.
     */
    double absolute_error_;

    /** @brief The relative error.
     */
    double relative_error_;

    /** @brief The probability level.
     */
    double probability_;

    /** @brief The metric used for computing the distances.
     */
    MetricType metric_;

    /** @brief The number of iterations to run (on iterative mode).
     */
    int num_iterations_in_;

    /** @brief The effective number of reference points (discounting
     *         the self-contribution when monochromatic).
     */
    int effective_num_reference_points_;

  public:

    /** @brief The default constructor.
     */
    LocalRegressionArguments() {
      bandwidth_ = 0.0;
      leaf_size_ = 0;
      query_table_ = NULL;
      reference_table_ = NULL;
      absolute_error_ = 0.0;
      relative_error_ = 0.0;
      probability_ = 1.0;
      num_iterations_in_ = -1;
      effective_num_reference_points_ = 0;
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
