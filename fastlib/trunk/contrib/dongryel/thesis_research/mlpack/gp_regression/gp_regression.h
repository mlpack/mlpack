/** @file gp_regression.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_GP_REGRESSION_GP_REGRESSION_H
#define MLPACK_GP_REGRESSION_GP_REGRESSION_H

#include <boost/program_options.hpp>
#include <vector>
#include "core/table/table.h"
#include "mlpack/gp_regression/gp_regression_arguments.h"

namespace mlpack {
namespace gp_regression {

/** @brief The class computing Gaussian process regression.
 */
template <
typename IncomingTableType, typename IncomingKernelType,
         typename IncomingMetricType >
class GpRegression {
  public:

    /** @brief The table type being used in the algorithm.
     */
    typedef IncomingTableType TableType;

    /** @brief The tree type.
     */
    typedef typename TableType::TreeType TreeType;

    /** @brief The kernel type.
     */
    typedef IncomingKernelType KernelType;

    /** @brief The metric type.
     */
    typedef IncomingMetricType MetricType;

    /** @brief The argument type.
     */
    typedef mlpack::gp_regression::GpRegressionArguments <
    TableType, MetricType > ArgumentType;

    typedef mlpack::gp_regression::GpRegressionPostponed PostponedType;

    typedef mlpack::gp_regression::GpRegressionGlobal <
    TableType, KernelType > GlobalType;

    typedef mlpack::gp_regression::GpRegressionResult ResultType;

    typedef mlpack::gp_regression::GpRegressionDelta DeltaType;

    typedef mlpack::gp_regression::GpRegressionSummary SummaryType;

    typedef mlpack::gp_regression::GpRegressionStatistic StatisticType;

  public:

    /** @brief returns a pointer to the query table.
     */
    TableType *query_table();

    /** @brief returns a pointer to the reference table.
     */
    TableType *reference_table();

    /** @brief returns a GlobalType structure that has the
     *         normalization statistics.
     */
    GlobalType &global();

    /** @brief When the reference table and the query table are the
     *         same then the Kde is called monochromatic.
     */
    bool is_monochromatic() const;

    /** @brief Initializes the local regression object with a set of
     *         arguments.
     */
    template<typename IncomingGlobalType>
    void Init(
      ArgumentType &arguments_in, IncomingGlobalType *global_in);

    /** @brief Computes the result.
     */
    void Compute(
      const ArgumentType &arguments_in,
      mlpack::gp_regression::GpRegressionResult *result_out);

  private:

    void Solve_(
      const std::vector< TreeType *> &reference_frontier,
      arma::mat &variable_states,
      int reference_frontier_id);

  private:

    /** @brief The query table.
     */
    TableType *query_table_;

    /** @brief The reference table.
     */
    TableType *reference_table_;

    /** @brief The globa constants.
     */
    GlobalType global_;

    /** @brief The flag that tells whether the computation is
     *         monochromatic or not.
     */
    bool is_monochromatic_;
};
}
}

#endif
