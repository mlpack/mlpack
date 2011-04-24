/** @file local_regression.h
 *
 *  The fixed bandwidth local regression.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_H

#include <vector>
#include <boost/program_options.hpp>
#include "core/table/table.h"
#include "mlpack/local_regression/local_regression_arguments.h"
#include "mlpack/local_regression/local_regression_result.h"

namespace mlpack {
namespace local_regression {

/** @brief The definition of the fixed bandwidth local regression.
 */
template<typename IncomingTableType, typename IncomingKernelType>
class LocalRegression {
  public:

    /** @brief The table type being used in the algorithm.
     */
    typedef IncomingTableType TableType;

    typedef IncomingKernelType KernelType;

    /** @brief The argument type.
     */
    typedef mlpack::local_regression::LocalRegressionArguments <
    TableType > ArgumentType;

  public:

    /** @brief Initializes the local regression object with a set of
     *         arguments.
     */
    void Init(ArgumentType &arguments_in);

    /** @brief Computes the result.
     */
    void Compute(
      const ArgumentType &arguments_in,
      mlpack::local_regression::LocalRegressionResult *result_out);
};
}
}

#endif
