/** @file kde.h
 *
 *  The header file for the kernel density estimation.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_H
#define MLPACK_KDE_KDE_H

#include <boost/program_options.hpp>
#include "core/table/table.h"
#include "mlpack/kde/kde_arguments.h"
#include "mlpack/kde/kde_dualtree.h"

namespace mlpack {
namespace kde {

/** @brief The argument parsing class for KDE computation.
 */
class KdeArgumentParser {
  private:
    template<typename TableType>
    static void PrescaleTable_(
      const std::string &prescale_option, TableType *table);

  public:
    template<typename TableType>
    static bool ParseArguments(
      boost::program_options::variables_map &vm,
      mlpack::kde::KdeArguments<TableType> *arguments_out);

    static bool ConstructBoostVariableMap(
      const std::vector<std::string> &args,
      boost::program_options::variables_map *vm);

    static bool ConstructBoostVariableMap(
      int argc,
      char *argv[],
      boost::program_options::variables_map *vm);
};

/** @brief The main entry point of the KDE that manages the
 *         computation.
 */
template<typename IncomingTableType, typename IncomingKernelAuxType>
class Kde {
  public:

    typedef IncomingTableType TableType;

    typedef IncomingKernelAuxType KernelAuxType;

    static const
    enum mlpack::series_expansion::CartesianExpansionType ExpansionType =
      KernelAuxType::ExpansionType;

    typedef mlpack::kde::KdePostponed<ExpansionType> PostponedType;

    typedef mlpack::kde::KdeGlobal<TableType, KernelAuxType> GlobalType;

    typedef mlpack::kde::KdeResult< std::vector<double> > ResultType;

    typedef mlpack::kde::KdeDelta DeltaType;

    typedef mlpack::kde::KdeSummary SummaryType;

    typedef mlpack::kde::KdeStatistic<ExpansionType> StatisticType;

  public:

    /** @brief Sets the bandwidth.
     */
    void set_bandwidth(double bandwidth_in);

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

    /** @brief Initialize a Kde engine with the arguments.
     */
    void Init(mlpack::kde::KdeArguments<TableType> &arguments_in);

    void Compute(
      const mlpack::kde::KdeArguments<TableType> &arguments_in,
      ResultType *result_out);

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
