/** @file distributed_allknn.h
 *
 *  The header declaration of distributed all knn algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_ALLKNN_DISTRIBUTED_ALLKNN_H
#define MLPACK_DISTRIBUTED_ALLKNN_DISTRIBUTED_ALLKNN_H

#include <boost/program_options.hpp>
#include <boost/mpi/communicator.hpp>
#include "core/table/distributed_table.h"
#include "mlpack/allknn/allknn_dev.h"
#include "mlpack/allknn/allknn_dualtree.h"
#include "mlpack/allknn/allknn_arguments.h"
#include "mlpack/distributed_allknn/distributed_allknn_arguments.h"

namespace mlpack {
namespace distributed_allknn {

/** @brief The class for storing necessary arguments to initiate the
 *         distributed allknn computation.
 */
template<typename IncomingDistributedTableType>
class DistributedAllknn {
  public:

    typedef IncomingDistributedTableType DistributedTableType;

    typedef typename DistributedTableType::TableType TableType;

    typedef mlpack::allknn::KdePostponed PostponedType;

    typedef mlpack::allknn::KdeGlobal<DistributedTableType> GlobalType;

    typedef mlpack::allknn::KdeResult< std::vector<double> > ResultType;

    typedef mlpack::allknn::KdeDelta DeltaType;

    typedef mlpack::allknn::KdeSummary SummaryType;

    typedef mlpack::allknn::KdeStatistic StatisticType;

    typedef mlpack::allknn::KdeArguments<TableType> ArgumentType;

    typedef mlpack::allknn::Allknn< TableType > ProblemType;

  public:

    /** @brief The default constructor.
     */
    DistributedAllknn() {
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
     *         same then the allknn is called monochromatic
     */
    bool is_monochromatic() const;

    /** @brief Initialize a allknn engine with the arguments.
     */
    void Init(
      boost::mpi::communicator &world_in,
      mlpack::distributed_allknn::DistributedAllknnArguments <
      DistributedTableType > &arguments_in);

    void Compute(
      const mlpack::distributed_allknn::DistributedAllknnArguments <
      DistributedTableType > &arguments_in,
      ResultType *result_out);

    static void RandomGenerate(
      boost::mpi::communicator &world, const std::string &file_name,
      int num_dimensions, int num_points);

    static void ParseArguments(
      boost::mpi::communicator &world,
      const std::vector<std::string> &args,
      mlpack::distributed_allknn::DistributedAllknnArguments <
      DistributedTableType > *arguments_out);

    static void ParseArguments(
      int argc,
      char *argv[],
      boost::mpi::communicator &world,
      mlpack::distributed_allknn::DistributedAllknnArguments <
      DistributedTableType > *arguments_out);

  private:

    boost::mpi::communicator *world_;

    /** @brief The distributed query table.
     */
    DistributedTableType *query_table_;

    /** @brief The distributed reference table.
     */
    DistributedTableType *reference_table_;

    /** @brief The relevant global variables for the distributed
     *         allknn computation.
     */
    GlobalType global_;

    /** @brief The flag that tells whether the computation is
     *         monochromatic.
     */
    bool is_monochromatic_;

  private:

    static bool ConstructBoostVariableMap_(
      boost::mpi::communicator &world,
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm);
};
}
}

#endif
