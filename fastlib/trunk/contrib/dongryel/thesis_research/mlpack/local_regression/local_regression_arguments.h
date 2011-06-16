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

    /** @brief The local polynomial order.
     */
    int order_;

    /** @brief The file name to output the predictions to.
     */
    std::string predictions_out_;

    /** @brief The prescale option.
     */
    std::string prescale_;

    /** @brief The problem dimension.
     */
    int problem_dimension_;

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
    double effective_num_reference_points_;

    /** @brief The flag denoting whether the computation is
     *         monochromatic or not.
     */
    bool is_monochromatic_;

    /** @brief Whether the tables are aliased or should be freed when
     *         the argument gets destructed.
     */
    bool tables_are_aliased_;

    /** @brief Whether the linear system is solved at the end.
     */
    bool do_postprocess_;

  public:

    template<typename GlobalType>
    void Init(
      TableType *reference_table_in, TableType *query_table_in,
      GlobalType &global_in) {
      reference_table_ = reference_table_in;
      query_table_ = query_table_in;
      effective_num_reference_points_ =
        global_in.effective_num_reference_points();
      bandwidth_ = global_in.bandwidth();
      absolute_error_ = global_in.absolute_error();
      relative_error_ = global_in.relative_error();
      probability_ = global_in.probability();
      kernel_ = global_in.kernel().name();
      tables_are_aliased_ = true;
      do_postprocess_ = global_in.do_postprocess();
    }

    template<typename GlobalType>
    void Init(GlobalType &global_in) {
      reference_table_ = global_in.reference_table()->local_table();
      if(reference_table_ != query_table_) {
        query_table_ = global_in.query_table()->local_table();
      }
      effective_num_reference_points_ =
        global_in.effective_num_reference_points();
      bandwidth_ = global_in.bandwidth();
      absolute_error_ = global_in.absolute_error();
      relative_error_ = global_in.relative_error();
      probability_ = global_in.probability();
      kernel_ = global_in.kernel().name();
      tables_are_aliased_ = true;
      do_postprocess_ = global_in.do_postprocess();
    }

    /** @brief The default constructor.
     */
    LocalRegressionArguments() {
      bandwidth_ = 0.0;
      leaf_size_ = 0;
      order_ = 0;
      problem_dimension_ = 1;
      query_table_ = NULL;
      reference_table_ = NULL;
      absolute_error_ = 0.0;
      relative_error_ = 0.0;
      probability_ = 1.0;
      num_iterations_in_ = -1;
      effective_num_reference_points_ = 0;
      is_monochromatic_ = false;
      tables_are_aliased_ = false;
      do_postprocess_ = true;
    }

    /** @brief The destructor.
     */
    ~LocalRegressionArguments() {
      if(! tables_are_aliased_) {
        if(query_table_ != reference_table_) {
          delete query_table_;
          delete reference_table_;
        }
        else {
          delete reference_table_;
        }
      }
      query_table_ = NULL;
      reference_table_ = NULL;
    }
};
}
}

#endif
