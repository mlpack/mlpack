/** @file distributed_kpca.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KPCA_DISTRIBUTED_KPCA_H
#define MLPACK_DISTRIBUTED_KPCA_DISTRIBUTED_KPCA_H

#include <boost/math/distributions/normal.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/program_options.hpp>
#include "core/table/distributed_table.h"
#include "mlpack/distributed_kpca/distributed_kpca_arguments.h"
#include "mlpack/distributed_kpca/kpca_result.h"

namespace mlpack {
namespace distributed_kpca {

/** @brief The argument parsing class for distributed KPCA computation.
 */
class DistributedKpcaArgumentParser {
  public:
    template<typename DistributedTableType>
    static bool ParseArguments(
      boost::mpi::communicator &world,
      boost::program_options::variables_map &vm,
      mlpack::distributed_kpca::DistributedKpcaArguments <
      DistributedTableType > *arguments_out);

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
      boost::mpi::communicator &world, const std::string &file_name,
      int num_dimensions, int num_points, const std::string &prescale_option);
};

template<typename IncomingDistributedTableType, typename IncomingKernelType>
class DistributedKpca {
  public:

    typedef IncomingDistributedTableType DistributedTableType;

    typedef IncomingKernelType KernelType;

    typedef typename DistributedTableType::TableType TableType;

    typedef mlpack::distributed_kpca::KpcaResult ResultType;

    typedef mlpack::distributed_kpca::DistributedKpcaArguments <
    TableType > ArgumentType;

  private:

    void FinalizeKernelEigenvectors_(
      double num_standard_deviations_in,
      int num_reference_samples_in,
      const mlpack::distributed_kpca::DistributedKpcaArguments <
      DistributedTableType > &arguments_in,
      ResultType *result_out);

    void ComputeEigenDecomposition_(
      double num_standard_deviations_in,
      int num_reference_samples_in,
      const mlpack::distributed_kpca::DistributedKpcaArguments <
      DistributedTableType > &arguments_in,
      ResultType *result_out);

    void GenerateRandomFourierFeatures_(
      int num_random_fourier_features,
      core::table::DenseMatrix *random_variates,
      std::vector< arma::vec > *random_variate_aliases);

    void NaiveKernelEigenvectors_(
      int num_kpca_components_in_,
      bool do_centering,
      DistributedTableType *reference_table_in,
      const core::table::DenseMatrix &kpca_components);

    void NaiveWeightedKernelAverage_(
      double relative_error_in,
      double absolute_error_in,
      DistributedTableType *reference_table_in,
      DistributedTableType *query_table_in,
      const core::table::DenseMatrix &weights,
      const core::monte_carlo::MeanVariancePairMatrix &kernel_sums);

    void ComputeWeightedKernelAverage_(
      int num_threads_in,
      double relative_error_in,
      double absolute_error_in,
      double num_standard_deviations,
      int num_reference_samples,
      int num_random_fourier_features,
      DistributedTableType *reference_table_in,
      DistributedTableType *query_table_in,
      const core::table::DenseMatrix &weights,
      core::monte_carlo::MeanVariancePairMatrix *kernel_sums);

    void PostProcessKpcaProjections_(
      const core::monte_carlo::MeanVariancePairMatrix &reference_kernel_sum,
      const core::monte_carlo::MeanVariancePairMatrix &query_kernel_sums,
      const core::monte_carlo::MeanVariancePair &average_reference_kernel_sum,
      const core::table::DenseMatrix &kpca_components,
      core::monte_carlo::MeanVariancePairMatrix *query_kpca_projections);

  public:

    DistributedKpca();

    /** @brief Initialize a Kpca engine with the arguments.
     */
    void Init(
      boost::mpi::communicator &world_in,
      mlpack::distributed_kpca::DistributedKpcaArguments <
      DistributedTableType > &arguments_in);

    void Compute(
      const mlpack::distributed_kpca::DistributedKpcaArguments <
      DistributedTableType > &arguments_in,
      ResultType *result_out);

  private:

    /** @brief The communicator.
     */
    boost::mpi::communicator *world_;

    /** @brief The normal distribution object.
     */
    boost::math::normal normal_dist_;

    double mult_const_;

    int effective_num_reference_points_;

    double correction_term_;

    int max_num_iterations_;

    int num_random_fourier_features_eigen_;

    /** @brief The dimensionality of the problem.
     */
    int num_dimensions_;

    /** @brief The default L2 metric.
     */
    core::metric_kernels::LMetric<2> metric_;

    /** @brief The kernel.
     */
    KernelType kernel_;
};
}
}

#endif
