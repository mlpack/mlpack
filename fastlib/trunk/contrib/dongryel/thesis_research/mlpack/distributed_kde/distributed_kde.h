/** @file distributed_kde.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KDE_DISTRIBUTED_KDE_H
#define MLPACK_DISTRIBUTED_KDE_DISTRIBUTED_KDE_H

#include <boost/program_options.hpp>
#include <boost/mpi/communicator.hpp>
#include "core/table/distributed_table.h"
#include "mlpack/kde/kde_dualtree.h"
#include "mlpack/distributed_kde/distributed_kde_arguments.h"

namespace mlpack {
namespace distributed_kde {
template<typename TreeSpecType>
class DistributedKde {
  public:
    typedef core::table::Table<TreeSpecType> TableType;

    typedef core::table::DistributedTable<TreeSpecType> DistributedTableType;

    typedef mlpack::kde::KdePostponed PostponedType;

    typedef mlpack::kde::KdeGlobal<TableType> GlobalType;

    typedef mlpack::kde::KdeResult< std::vector<double> > ResultType;

    typedef mlpack::kde::KdeDelta DeltaType;

    typedef mlpack::kde::KdeSummary SummaryType;

    typedef mlpack::kde::KdeStatistic StatisticType;

  public:

    /**
     * @brief sets the bandwidth
     */
    void set_bandwidth(double bandwidth_in);

    /**
     * @brief returns a pointer to the query table
     */
    core::table::DistributedTable<TreeSpecType> *query_table();

    /**
     * @brief returns a pointer to the reference table
     */
    core::table::DistributedTable<TreeSpecType> *reference_table();

    /**
     * @brief returns a GlobalType structure that has the
     *        normalization statistics
     */
    GlobalType &global();

    /**
     * @brief When the reference table and the query table are the same then
     *        the Kde is called monochromatic
     */
    bool is_monochromatic() const;

    /**
     * @brief Initialize a Kde engine with the arguments.
     */
    void Init(
      boost::mpi::communicator &world_in,
      mlpack::distributed_kde::DistributedKdeArguments <
      DistributedTableType > &arguments_in);

    void Compute(
      const mlpack::distributed_kde::DistributedKdeArguments <
      DistributedTableType > &arguments_in,
      ResultType *result_out);

    static void RandomGenerate(
      boost::mpi::communicator &world, const std::string &file_name,
      int num_dimensions, int num_points);

    static void ParseArguments(
      boost::mpi::communicator &world,
      const std::vector<std::string> &args,
      mlpack::distributed_kde::DistributedKdeArguments <
      DistributedTableType > *arguments_out);

    static void ParseArguments(
      int argc,
      char *argv[],
      mlpack::distributed_kde::DistributedKdeArguments <
      DistributedTableType > *arguments_out);

  private:

    boost::mpi::communicator &world_;

    /** @brief The distributed query table.
     */
    DistributedTableType *query_table_;

    /** @brief The distributed reference table.
     */
    DistributedTableType *reference_table_;

    /** @brief The relevant global variables for the distributed KDE
     *         computation.
     */
    GlobalType global_;

    /** @brief The flag that tells whether the computation is
     *         monochromatic.
     */
    bool is_monochromatic_;

  private:

    static bool ConstructBoostVariableMap_(
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm);
};
};
};

#endif
