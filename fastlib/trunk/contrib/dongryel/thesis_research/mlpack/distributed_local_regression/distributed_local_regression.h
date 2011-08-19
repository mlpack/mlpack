/** @file distributed_local_regression.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_LOCAL_REGRESSION_DISTRIBUTED_LOCAL_REGRESSION_H
#define MLPACK_DISTRIBUTED_LOCAL_REGRESSION_DISTRIBUTED_LOCAL_REGRESSION_H

#include <boost/program_options.hpp>
#include <boost/mpi/communicator.hpp>
#include "core/table/distributed_table.h"
#include "mlpack/local_regression/local_regression_dev.h"
#include "mlpack/local_regression/local_regression_dualtree.h"
#include "mlpack/local_regression/local_regression_arguments.h"
#include "mlpack/distributed_local_regression/distributed_local_regression_arguments.h"

namespace mlpack {
namespace distributed_local_regression {

/** @brief The argument parsing class for distributed local regression
 *         computation.
 */
class DistributedLocalRegressionArgumentParser {
  public:
    template<typename DistributedTableType, typename MetricType>
    static bool ParseArguments(
      boost::mpi::communicator &world,
      boost::program_options::variables_map &vm,
      mlpack::distributed_local_regression::
      DistributedLocalRegressionArguments <
      DistributedTableType, MetricType > *arguments_out);

    static bool ConstructBoostVariableMap(
      boost::mpi::communicator &world,
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm);

    static bool ConstructBoostVariableMap(
      boost::mpi::communicator &world,
      int argc,
      char *argv[],
      boost::program_options::variables_map *vm);

    template<typename TableType>
    static void RandomGenerate(
      boost::mpi::communicator &world,
      const std::string &file_name,
      const std::string *weight_file_name,
      int num_dimensions,
      int num_points,
      const std::string &prescale_option);
};

template <
typename IncomingDistributedTableType,
         typename IncomingKernelType,
         typename IncomingMetricType >
class DistributedLocalRegression {
  public:

    typedef IncomingDistributedTableType DistributedTableType;

    typedef IncomingKernelType KernelType;

    typedef IncomingMetricType MetricType;

    typedef typename DistributedTableType::TableType TableType;

    typedef mlpack::local_regression::LocalRegressionPostponed PostponedType;

    typedef mlpack::local_regression::LocalRegressionGlobal <
    DistributedTableType, KernelType > GlobalType;

    typedef mlpack::local_regression::LocalRegressionResult ResultType;

    typedef mlpack::local_regression::LocalRegressionDelta DeltaType;

    typedef mlpack::local_regression::LocalRegressionSummary SummaryType;

    typedef mlpack::local_regression::LocalRegressionStatistic StatisticType;

    typedef mlpack::local_regression::LocalRegressionArguments <
    TableType, MetricType > ArgumentType;

    typedef mlpack::local_regression::LocalRegression <
    TableType, KernelType, MetricType > ProblemType;

  public:

    /** @brief The default constructor.
     */
    DistributedLocalRegression() {
      world_ = NULL;
    }

    /** @brief sets the bandwidth
     */
    void set_bandwidth(double bandwidth_in);

    /** @brief returns a pointer to the query table
     */
    DistributedTableType *query_table();

    /** @brief returns a pointer to the reference table
     */
    DistributedTableType *reference_table();

    /** @brief returns a GlobalType structure that has the
     *         normalization statistics
     */
    GlobalType &global();

    /** @brief When the reference table and the query table are the
     *         same then the LocalRegression is called monochromatic
     */
    bool is_monochromatic() const;

    /** @brief Initialize a local regresion engine with the arguments.
     */
    void Init(
      boost::mpi::communicator &world_in,
      mlpack::distributed_local_regression::
      DistributedLocalRegressionArguments <
      DistributedTableType, MetricType > &arguments_in);

    void Compute(
      const mlpack::distributed_local_regression::
      DistributedLocalRegressionArguments <
      DistributedTableType, MetricType > &arguments_in,
      ResultType *result_out);

  private:

    boost::mpi::communicator *world_;

    /** @brief The distributed query table.
     */
    DistributedTableType *query_table_;

    /** @brief The distributed reference table.
     */
    DistributedTableType *reference_table_;

    /** @brief The relevant global variables for the distributed local
     *         regression computation.
     */
    GlobalType global_;

    /** @brief The flag that tells whether the computation is
     *         monochromatic.
     */
    bool is_monochromatic_;
};
}
}

#endif
