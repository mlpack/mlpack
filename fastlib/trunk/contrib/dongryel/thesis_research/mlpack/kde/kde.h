/** @file kde.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_H
#define MLPACK_KDE_KDE_H

#include <armadillo>
#include "boost/program_options.hpp"

#include "core/table/table.h"
#include "kde_dualtree.h"
#include "kde_arguments.h"

namespace mlpack {
namespace kde {
template<typename IncomingTableType>
class Kde {
  public:
    typedef IncomingTableType TableType;

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
    TableType *query_table();

    /**
     * @brief returns a pointer to the reference table
     */
    TableType *reference_table();

    /**
     * @brief returns a GlobalType structure that has the normalization statistics
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
    void Init(mlpack::kde::KdeArguments<TableType> &arguments_in);

    void Compute(
      const mlpack::kde::KdeArguments<TableType> &arguments_in,
      ResultType *result_out);

    static void ParseArguments(
      const std::vector<std::string> &args,
      mlpack::kde::KdeArguments<TableType> *arguments_out);

    static void ParseArguments(
      int argc,
      char *argv[],
      mlpack::kde::KdeArguments<TableType> *arguments_out);

  private:

    TableType *query_table_;
    TableType *reference_table_;
    GlobalType global_;
    bool is_monochromatic_;

  private:

    static bool ConstructBoostVariableMap_(
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm);
};
};
};

#endif
