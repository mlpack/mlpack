/** @file nbody_simulator.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef PHYSPACK_NBODY_SIMULATOR_NBODY_SIMULATOR_H
#define PHYSPACK_NBODY_SIMULATOR_NBODY_SIMULATOR_H

#include <armadillo>
#include "boost/program_options.hpp"

#include "core/table/table.h"
#include "nbody_simulator_tripletree.h"
#include "nbody_simulator_arguments.h"

namespace physpack {
namespace nbody_simulator {
template<typename IncomingTableType>
class NbodySimulator {
  public:

    typedef IncomingTableType TableType;

    typedef physpack::nbody_simulator::NbodySimulatorPostponed PostponedType;

    typedef physpack::nbody_simulator::NbodySimulatorGlobal<TableType> GlobalType;

    typedef physpack::nbody_simulator::NbodySimulatorResult ResultType;

    typedef physpack::nbody_simulator::NbodySimulatorDelta DeltaType;

    typedef physpack::nbody_simulator::NbodySimulatorSummary SummaryType;

    typedef physpack::nbody_simulator::NbodySimulatorStatistic StatisticType;

  public:

    /**
     * @brief returns a pointer to the table
     */
    TableType *table();

    /**
     * @brief returns a GlobalType structure that has the
     *        normalization statistics
     */
    GlobalType &global();

    /**
     * @brief Initialize a nbody simulator engine with the arguments.
     */
    void Init(
      physpack::nbody_simulator::NbodySimulatorArguments<TableType> &arguments_in);

    void NaiveCompute(
      const physpack::nbody_simulator::NbodySimulatorArguments<TableType> &arguments_in,
      const physpack::nbody_simulator::NbodySimulatorResult &approx_result_in,
      physpack::nbody_simulator::NbodySimulatorResult *naive_result_out);

    void Compute(
      const physpack::nbody_simulator::NbodySimulatorArguments<TableType> &arguments_in,
      ResultType *result_out);

    static void ParseArguments(
      const std::vector<std::string> &args,
      physpack::nbody_simulator::NbodySimulatorArguments<TableType> *arguments_out);

    static void ParseArguments(
      int argc,
      char *argv[],
      physpack::nbody_simulator::NbodySimulatorArguments<TableType> *arguments_out);

  private:

    TableType *table_;
    GlobalType global_;

  private:

    static bool ConstructBoostVariableMap_(
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm);
};
};
};

#endif
