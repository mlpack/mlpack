/** @file local_regression_global.h
 *
 *  The global quantities in local regression in a dual-tree
 *  algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_GLOBAL_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_GLOBAL_H

#include "mlpack/local_regression/local_regression_delta.h"

namespace mlpack {
namespace local_regression {

template<typename KernelType>
class ConsiderExtrinsicPruneTrait {
  public:
    static bool Compute(
      const KernelType &kernel_aux_in,
      const core::math::Range &squared_distance_range_in) {
      return false;
    }
};

template<>
class ConsiderExtrinsicPruneTrait <
    core::metric_kernels::EpanKernel > {
  public:
    static bool Compute(
      const core::metric_kernels::EpanKernel &kernel_in,
      const core::math::Range &squared_distance_range_in) {

      return
        kernel_in.bandwidth_sq() <= squared_distance_range_in.lo;
    }
};

/** @brief The global constant struct passed around for local
 *         regression computation.
 */
template<typename IncomingTableType, typename IncomingKernelType>
class LocalRegressionGlobal {

  public:
    typedef IncomingTableType TableType;

    typedef IncomingKernelType KernelType;

    static const int min_sampling_threshold = 2000;

    static const int sampling_batch_size = 20;

  private:

    /** @brief The absolute error approximation level.
     */
    double absolute_error_;

    /** @brief The relative error approximation level.
     */
    double relative_error_;

    /** @brief The adjusted relative error factor to guarantee overall
     *         relative error.
     */
    double adjusted_relative_error_;

    /** @brief For the probabilistic approximation.
     */
    double probability_;

    /** @brief The kernel type.
     */
    KernelType kernel_;

    /** @brief The effective number of reference points used for
     *         normalization.
     */
    double effective_num_reference_points_;

    /** @brief The dimensionality of the problem.
     */
    int problem_dimension_;

    /** @brief The number of random features to use.
     */
    int num_random_features_;

    /** @brief The query table.
     */
    TableType *query_table_;

    /** @brief The reference table.
     */
    TableType *reference_table_;

    /** @brief Whether the computation is monochromatic or not.
     */
    bool is_monochromatic_;

    /** @brief The temporary space for marking convergence of each
     *         query.
     */
    std::deque<bool> converged_flags_;

    /** @brief The temporary space for accumulating Monte Carlo
     *         samples for the left hand side for each query.
     */
    boost::scoped_array <
    mlpack::local_regression::LocalRegressionDelta > query_deltas_;

    /** @brief The normal distribution object.
     */
    boost::math::normal normal_dist_;

  public:

    /** @brief Returns the standard score corresponding to the
     *         cumulative distribution of the unit variance normal
     *         distribution with the given tail mass.
     */
    double compute_quantile(double tail_mass) const {
      double mass = 1 - 0.5 * tail_mass;
      if(mass > 0.999) {
        return 3;
      }
      else {
        return boost::math::quantile(normal_dist_, mass);
      }
    }

    boost::scoped_array< mlpack::local_regression::LocalRegressionDelta > &
    query_deltas() {
      return query_deltas_;
    }

    /** @brief Tells whether the given squared distance range is
     *         sufficient for pruning for any pair of query/reference
     *         pair that satisfies the range.
     */
    bool ConsiderExtrinsicPrune(
      const core::math::Range &squared_distance_range) const {

      return
        ConsiderExtrinsicPruneTrait<KernelType>::Compute(
          kernel_, squared_distance_range);
    }

    int problem_dimension() const {
      return problem_dimension_;
    }

    std::deque<bool> &converged_flags() {
      return converged_flags_;
    }

    /** @brief Returns whether the computation is monochromatic or
     *         not.
     */
    bool is_monochromatic() const {
      return is_monochromatic_;
    }

    /** @brief Returns the effective number of reference points.
     */
    double effective_num_reference_points() const {
      return effective_num_reference_points_;
    }

    /** @brief Sets the effective number of reference points given a
     *         pair of distributed table of points.
     */
    template<typename DistributedTableType>
    void set_effective_num_reference_points(
      boost::mpi::communicator &comm,
      DistributedTableType *reference_table_in,
      DistributedTableType *query_table_in) {

      double total_sum = 0;
      for(int i = 0; i < comm.size(); i++) {
        total_sum += reference_table_in->local_n_entries(i);
      }
      effective_num_reference_points_ =
        (reference_table_in == query_table_in) ?
        (total_sum - 1.0) : total_sum;
    }

    /** @brief The constructor.
     */
    LocalRegressionGlobal() {
      absolute_error_ = 0.0;
      relative_error_ = 0.0;
      adjusted_relative_error_ = 0.0;
      probability_ = 1.0;
      problem_dimension_ = 1;
      num_random_features_ = 0;
      effective_num_reference_points_ = 0.0;
      query_table_ = NULL;
      reference_table_ = NULL;
      is_monochromatic_ = true;
    }

    /** @brief Returns the query table.
     */
    TableType *query_table() {
      return query_table_;
    }

    /** @brief Returns the query table.
     */
    const TableType *query_table() const {
      return query_table_;
    }

    /** @brief Returns the reference table.
     */
    TableType *reference_table() {
      return reference_table_;
    }

    /** @brief Returns the reference table.
     */
    const TableType *reference_table() const {
      return reference_table_;
    }

    /** @brief Returns the absolute error.
     */
    double absolute_error() const {
      return absolute_error_;
    }

    /** @brief Returns the relative error.
     */
    double relative_error() const {
      return relative_error_;
    }

    /** @brief Returns the adjust relative error.
     */
    double adjusted_relative_error() const {
      return adjusted_relative_error_;
    }

    /** @brief Returns the probability.
     */
    double probability() const {
      return probability_;
    }

    /** @brief Returns the bandwidth value being used.
     */
    double bandwidth() const {
      return sqrt(kernel_.bandwidth_sq());
    }

    /** @brief Sets the bandwidth.
     */
    void set_bandwidth(double bandwidth_in) {
      kernel_.Init(bandwidth_in);
    }

    /** @brief Returns the kernel.
     */
    const KernelType &kernel() const {
      return kernel_;
    }

    /** @brief Returns the number of random features to sample.
     */
    int num_random_features() const {
      return num_random_features_;
    }

    /** @brief Initializes the local regression global object.
     */
    template<typename ArgumentType>
    void Init(ArgumentType &arguments_in) {

      effective_num_reference_points_ =
        arguments_in.effective_num_reference_points_;

      // Initialize the kernel.
      kernel_.Init(arguments_in.bandwidth_);

      relative_error_ = arguments_in.relative_error_;
      adjusted_relative_error_ =
        relative_error_ / (relative_error_ + 2.0);
      absolute_error_ = arguments_in.absolute_error_;
      probability_ = arguments_in.probability_;
      query_table_ = arguments_in.query_table_;
      reference_table_ = arguments_in.reference_table_;

      // Set the monochromatic flag.
      is_monochromatic_ = arguments_in.is_monochromatic_;

      // Set the problem dimension.
      problem_dimension_ = arguments_in.problem_dimension_;

      // Set the number of random features to use.
      num_random_features_ = std::min(
                               reference_table_->n_attributes() + 5, 20);

      // Allocate the space for storing the convergence flags.
      converged_flags_.resize(query_table_->n_entries());
      for(unsigned int i = 0; i < converged_flags_.size(); i++) {
        converged_flags_[i] = false;
      }

      // Allocate temporary spaces for Monte Carlo accumulations.
      boost::scoped_array< mlpack::local_regression::LocalRegressionDelta >
      scratch_deltas(
        new mlpack::local_regression::LocalRegressionDelta[
          query_table_->n_entries()]);
      query_deltas_.swap(scratch_deltas);
      for(int i = 0; i < query_table_->n_entries(); i++) {
        query_deltas_[i].Init(*this);
      }
    }
};
}
}

#endif
