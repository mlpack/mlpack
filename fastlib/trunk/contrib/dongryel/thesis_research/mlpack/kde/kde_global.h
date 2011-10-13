/** @file kde_global.h
 *
 *  The global quantities for kde dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_GLOBAL_H
#define MLPACK_KDE_KDE_GLOBAL_H

namespace mlpack {
namespace kde {

template<typename KernelAuxType>
class ConsiderExtrinsicPruneTrait {
  public:
    static bool Compute(
      const KernelAuxType &kernel_aux_in,
      const core::math::Range &squared_distance_range_in) {
      return false;
    }
};

template<>
class ConsiderExtrinsicPruneTrait <
    mlpack::series_expansion::EpanKernelMultivariateAux > {
  public:
    static bool Compute(
      const mlpack::series_expansion::EpanKernelMultivariateAux &kernel_aux_in,
      const core::math::Range &squared_distance_range_in) {

      return
        kernel_aux_in.kernel().bandwidth_sq() <= squared_distance_range_in.lo;
    }
};

/** @brief The global constant struct passed around for KDE
 *         computation.
 */
template<typename IncomingTableType, typename IncomingKernelAuxType>
class KdeGlobal {

  public:
    typedef IncomingTableType TableType;

    typedef IncomingKernelAuxType KernelAuxType;

    typedef std::vector <
    core::monte_carlo::MeanVariancePair > MeanVariancePairListType;

  private:

    /** @brief Whether to normalize the kernel sums at the end or not.
     */
    bool normalize_densities_;

    /** @brief The absolute error approximation level.
     */
    double absolute_error_;

    /** @brief The relative error approximation level.
     */
    double relative_error_;

    /** @brief For the probabilistic approximation.
     */
    double probability_;

    /** @brief The kernel type.
     */
    KernelAuxType *kernel_aux_;

    /** @brief Tells whether the kernel aux object is an alias or not.
     */
    bool kernel_aux_is_alias_;

    /** @brief The effective number of reference points used for
     *         normalization.
     */
    unsigned long int effective_num_reference_points_;

    /** @brief The normalization constant.
     */
    double mult_const_;

    /** @brief The query table.
     */
    TableType *query_table_;

    /** @brief The reference table.
     */
    TableType *reference_table_;

    /** @brief Whether the computation is monochromatic or not.
     */
    bool is_monochromatic_;

    /** @brief The normal distribution object.
     */
    boost::math::normal normal_dist_;

    /** @brief The scratch space for doing a Monte Carlo sum.
     */
    MeanVariancePairListType *mean_variance_pair_;

  public:

    /** @brief Tells whether the given squared distance range is
     *         sufficient for pruning for any pair of query/reference
     *         pair that satisfies the range.
     */
    bool ConsiderExtrinsicPrune(
      const core::math::Range &squared_distance_range) const {

      return
        ConsiderExtrinsicPruneTrait<KernelAuxType>::Compute(
          *kernel_aux_, squared_distance_range);
    }

    /** @brief Returns whether the computation is monochromatic or
     *         not.
     */
    bool is_monochromatic() const {
      return is_monochromatic_;
    }

    /** @brief Returns whether we should normalize the sum at the end.
     */
    bool normalize_densities() const {
      return normalize_densities_;
    }

    /** @brief Returns the effective number of reference points.
     */
    unsigned long int effective_num_reference_points() const {
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

      unsigned long int total_sum = 0;
      for(int i = 0; i < comm.size(); i++) {
        total_sum += reference_table_in->local_n_entries(i);
      }
      effective_num_reference_points_ =
        (query_table_in == reference_table_in) ? total_sum - 1 : total_sum ;
      mult_const_ = 1.0 /
                    (kernel_aux_->kernel().CalcNormConstant(
                       reference_table_in->n_attributes()) *
                     ((double) effective_num_reference_points_));
    }

    /** @brief The constructor.
     */
    KdeGlobal() {
      normalize_densities_ = true;
      absolute_error_ = 0.0;
      relative_error_ = 0.0;
      probability_ = 1.0;
      kernel_aux_ = NULL;
      kernel_aux_is_alias_ = false;
      effective_num_reference_points_ = 0.0;
      mult_const_ = 0.0;
      query_table_ = NULL;
      reference_table_ = NULL;
      is_monochromatic_ = true;
      mean_variance_pair_ = NULL;
    }

    /** @brief The destructor.
     */
    ~KdeGlobal() {
      if(! kernel_aux_is_alias_) {
        delete kernel_aux_;
        delete mean_variance_pair_;
      }
      kernel_aux_ = NULL;
      mean_variance_pair_ = NULL;
    }

    /** @brief Returns the mean variance pair object.
     */
    MeanVariancePairListType *mean_variance_pair() {
      return mean_variance_pair_;
    }

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

    /** @brief Returns the probability.
     */
    double probability() const {
      return probability_;
    }

    /** @brief Returns the bandwidth value being used.
     */
    double bandwidth() const {
      return sqrt(kernel_aux_->kernel().bandwidth_sq());
    }

    /** @brief Sets the bandwidth.
     */
    void set_bandwidth(double bandwidth_in) {
      kernel_aux_->kernel().Init(bandwidth_in);
    }

    /** @brief Returns the kernel auxilary object.
     */
    const KernelAuxType &kernel_aux() const {
      return *kernel_aux_;
    }

    /** @brief Returns the kernel.
     */
    const typename KernelAuxType::KernelType &kernel() const {
      return kernel_aux_->kernel();
    }

    /** @brief Returns the series expansion type.
     */
    const std::string series_expansion_type() const {
      return kernel_aux_->series_expansion_type();
    }

    /** @brief Initializes the KDE global object.
     */
    void Init(
      TableType *reference_table_in,
      TableType *query_table_in,
      double effective_num_reference_points_in, KernelAuxType *kernel_aux_in,
      double bandwidth_in, MeanVariancePairListType *mean_variance_pair_in,
      const bool is_monochromatic,
      double relative_error_in, double absolute_error_in, double probability_in,
      bool normalize_densities_in = true) {

      effective_num_reference_points_ = effective_num_reference_points_in;

      // Initialize the kernel.
      if(kernel_aux_in) {
        kernel_aux_ = kernel_aux_in;
        kernel_aux_is_alias_ = true;
        mean_variance_pair_ = mean_variance_pair_in;
      }
      else {
        kernel_aux_ = new KernelAuxType();
        kernel_aux_is_alias_ = false;
        kernel_aux_->kernel().Init(bandwidth_in);
        mean_variance_pair_ = new MeanVariancePairListType();
      }
      mult_const_ = 1.0 /
                    (kernel_aux_->kernel().CalcNormConstant(
                       reference_table_in->n_attributes()) *
                     ((double) effective_num_reference_points_));

      relative_error_ = relative_error_in;
      absolute_error_ = absolute_error_in;
      probability_ = probability_in;
      query_table_ = query_table_in;
      reference_table_ = reference_table_in;

      // Initialize the temporary vector for storing the Monte Carlo
      // results.
      if(! kernel_aux_is_alias_) {
        mean_variance_pair_->resize(query_table_->n_entries());
      }

      // Set the normalize flag.
      normalize_densities_ = normalize_densities_in;

      // Set the monochromatic flag.
      is_monochromatic_ = is_monochromatic;

      // Initialize the kernel series expansion object.
      if(! kernel_aux_is_alias_) {
        if(kernel_aux_->series_expansion_type() == "multivariate") {
          if(reference_table_->n_attributes() <= 2) {
            kernel_aux_->Init(
              bandwidth_in, 7, reference_table_->n_attributes());
          }
          else if(reference_table_->n_attributes() <= 3) {
            kernel_aux_->Init(
              bandwidth_in, 5, reference_table_->n_attributes());
          }
          else if(reference_table_->n_attributes() <= 5) {
            kernel_aux_->Init(
              bandwidth_in, 3, reference_table_->n_attributes());
          }
          else if(reference_table_->n_attributes() <= 6) {
            kernel_aux_->Init(
              bandwidth_in, 1, reference_table_->n_attributes());
          }
          else {
            kernel_aux_->Init(
              bandwidth_in, 0, reference_table_->n_attributes());
          }
        }
        else {
          if(reference_table_->n_attributes() <= 2) {
            kernel_aux_->Init(
              bandwidth_in, 5, reference_table_->n_attributes());
          }
          else if(reference_table_->n_attributes() <= 3) {
            kernel_aux_->Init(
              bandwidth_in, 3, reference_table_->n_attributes());
          }
          else if(reference_table_->n_attributes() <= 5) {
            kernel_aux_->Init(
              bandwidth_in, 1, reference_table_->n_attributes());
          }
          else if(reference_table_->n_attributes() <= 6) {
            kernel_aux_->Init(
              bandwidth_in, 0, reference_table_->n_attributes());
          }
          else {
            kernel_aux_->Init(
              bandwidth_in, 0, reference_table_->n_attributes());
          }
        }
      }
    }

    /** @brief Gets the multiplicative normalization constant.
     */
    double get_mult_const() const {
      return mult_const_;
    }
};
}
}

#endif
