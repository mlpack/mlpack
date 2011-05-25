/** @file local_regression.h
 *
 *  The fixed bandwidth local regression.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_H

#include <boost/program_options.hpp>
#include <vector>
#include "core/table/table.h"
#include "mlpack/local_regression/local_regression_arguments.h"
#include "mlpack/local_regression/local_regression_dualtree.h"

namespace mlpack {
namespace local_regression {

/** @brief The definition of the fixed bandwidth local regression.
 */
template <
typename IncomingTableType, typename IncomingKernelType,
         typename IncomingMetricType >
class LocalRegression {
  public:

    /** @brief The table type being used in the algorithm.
     */
    typedef IncomingTableType TableType;

    /** @brief The kernel type.
     */
    typedef IncomingKernelType KernelType;

    /** @brief The metric type.
     */
    typedef IncomingMetricType MetricType;

    /** @brief The argument type.
     */
    typedef mlpack::local_regression::LocalRegressionArguments <
    TableType, MetricType > ArgumentType;

    typedef mlpack::local_regression::LocalRegressionPostponed PostponedType;

    typedef mlpack::local_regression::LocalRegressionGlobal <
    TableType, KernelType > GlobalType;

    typedef mlpack::local_regression::LocalRegressionResult ResultType;

    typedef mlpack::local_regression::LocalRegressionDelta DeltaType;

    typedef mlpack::local_regression::LocalRegressionSummary SummaryType;

    typedef mlpack::local_regression::LocalRegressionStatistic StatisticType;

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
      mlpack::local_regression::LocalRegressionResult *result_out);

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
