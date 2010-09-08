/** @file kde.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_H
#define MLPACK_KDE_KDE_H

#include <armadillo>
#include "boost/program_options.hpp"
#include "boost/mpl/map.hpp"
#include "boost/mpl/at.hpp"
#include "boost/mpl/if.hpp"
#include "boost/mpl/has_key.hpp"
#include "boost/mpl/int.hpp"
#include "boost/mpl/insert.hpp"
#include "boost/mpl/assert.hpp"
#include "boost/mpl/vector.hpp"

#include "core/table/table.h"

#include "../gnp/dualtree_trace.h"
#include "kde_dualtree.h"
#include "kde_arguments.h"

namespace ml {
class Kde {
  public:
    typedef ml::KdePostponed PostponedType;

    typedef ml::KdeGlobal GlobalType;

    typedef ml::KdeResult< std::vector<double> > ResultType;

    typedef ml::KdeDelta DeltaType;

    typedef ml::KdeSummary SummaryType;

    typedef ml::KdeStatistic StatisticType;

    typedef core::table::Table TableType;

  public:

    /**
     * @brief sets the bandwidth
     */
    void set_bandwidth(double bandwidth_in);

    /**
     * @brief returns a pointer to the query table
     */
    core::table::Table *query_table();

    /**
     * @brief returns a pointer to the reference table
     */
    core::table::Table *reference_table();

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
    void Init(ml::KdeArguments &arguments_in);

    void Compute(ResultType *result_out);

    static void ParseArguments(
      int argc,
      char *argv[],
      ml::KdeArguments *arguments_out);

  private:

    core::table::Table *query_table_;
    core::table::Table *reference_table_;
    GlobalType global_;
    bool is_monochromatic_;

  private:

    static bool ConstructBoostVariableMap_(
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm);
};
};

#endif
