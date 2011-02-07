/** @file kde_dualtree.h
 *
 *  The template stub filled out for computing the kernel density
 *  estimate using a dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_DUALTREE_H
#define MLPACK_KDE_KDE_DUALTREE_H

#include <boost/math/distributions/normal.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include "core/monte_carlo/mean_variance_pair.h"
#include "core/metric_kernels/kernel.h"
#include "core/tree/statistic.h"
#include "core/table/table.h"
#include "mlpack/series_expansion/multivariate_farfield_dev.h"
#include "mlpack/series_expansion/multivariate_local_dev.h"

namespace mlpack {
namespace kde {

/** @brief The postponed quantities for KDE.
 */
class KdePostponed {

  private:

    // For boost serialization.
    friend class boost::serialization::access;

  public:

    /** @brief The lower bound on the postponed quantities.
     */
    double densities_l_;

    /** @brief The upper bound on the postponed quantities.
     */
    double densities_u_;

    /** @brief The amount of pruned quantities.
     */
    double pruned_;

    /** @brief The upper bound on the used error.
     */
    double used_error_;

    /** @brief Serialize the postponed quantities.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & densities_l_;
      ar & densities_u_;
      ar & pruned_;
      ar & used_error_;
    }

    /** @brief The default constructor.
     */
    KdePostponed() {
      SetZero();
    }

    /** @brief Copies another postponed object.
     */
    void Copy(const KdePostponed &postponed_in) {
      densities_l_ = postponed_in.densities_l_;
      densities_u_ = postponed_in.densities_u_;
      pruned_ = postponed_in.pruned_;
      used_error_ = postponed_in.used_error_;
    }

    /** @brief Initializes the postponed quantities.
     */
    void Init() {
      SetZero();
    }

    /** @brief Initializes the postponed quantities given a global
     *         object and a query reference pair.
     */
    template<typename GlobalType, typename TreeType>
    void Init(const GlobalType &global_in, TreeType *qnode, TreeType *rnode) {
      densities_l_ = densities_u_ = 0;
      pruned_ = (qnode == rnode && global_in.is_monochromatic()) ?
                static_cast<double>(rnode->count() - 1) :
                static_cast<double>(rnode->count());
      used_error_ = 0;
    }

    /** @brief Applies the incoming delta contribution to the
     *         postponed quantities, optionally to the query results
     *         as well.
     */
    template<typename KdeDelta, typename ResultType>
    void ApplyDelta(
      const KdeDelta &delta_in, ResultType *query_results) {
      densities_l_ = densities_l_ + delta_in.densities_l_;
      densities_u_ = densities_u_ + delta_in.densities_u_;
      pruned_ = pruned_ + delta_in.pruned_;
      used_error_ = used_error_ + delta_in.used_error_;
    }

    /** @brief Applies the incoming postponed contribution.
     */
    void ApplyPostponed(const KdePostponed &other_postponed) {
      densities_l_ = densities_l_ + other_postponed.densities_l_;
      densities_u_ = densities_u_ + other_postponed.densities_u_;
      pruned_ = pruned_ + other_postponed.pruned_;
      used_error_ = used_error_ + other_postponed.used_error_;
    }

    /** @brief Called from an exact pairwise evaluation method
     *         (i.e. the base case) which incurs no error.
     */
    template<typename GlobalType, typename MetricType, typename PointType>
    void ApplyContribution(
      const GlobalType &global,
      const MetricType &metric,
      const PointType &query_point, const PointType &reference_point) {

      double distsq = metric.DistanceSq(query_point, reference_point);
      double density_incoming = global.kernel().EvalUnnormOnSq(distsq);
      densities_l_ = densities_l_ + density_incoming;
      densities_u_ = densities_u_ + density_incoming;
    }

    /** @brief Sets everything to zero.
     */
    void SetZero() {
      densities_l_ = 0;
      densities_u_ = 0;
      pruned_ = 0;
      used_error_ = 0;
    }
};

/** @brief The global constant struct passed around for KDE
 *         computation.
 */
template<typename IncomingTableType>
class KdeGlobal {

  public:
    typedef IncomingTableType TableType;

    typedef core::metric_kernels::AbstractKernel KernelType;

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
    KernelType *kernel_;

    /** @brief The effective number of reference points used for
     *         normalization.
     */
    double effective_num_reference_points_;

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
    std::vector< core::monte_carlo::MeanVariancePair > mean_variance_pair_;

  public:

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
      mult_const_ = 1.0 /
                    (kernel_->CalcNormConstant(
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
      kernel_ = NULL;
      effective_num_reference_points_ = 0.0;
      mult_const_ = 0.0;
      query_table_ = NULL;
      reference_table_ = NULL;
      is_monochromatic_ = true;
    }

    /** @brief The destructor.
     */
    ~KdeGlobal() {
      delete kernel_;
      kernel_ = NULL;
    }

    /** @brief Returns the mean variance pair object.
     */
    std::vector< core::monte_carlo::MeanVariancePair > *mean_variance_pair() {
      return &mean_variance_pair_;
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
      return sqrt(kernel_->bandwidth_sq());
    }

    /** @brief Sets the bandwidth.
     */
    void set_bandwidth(double bandwidth_in) {
      kernel_->Init(bandwidth_in);
    }

    /** @brief Returns the kernel.
     */
    const KernelType &kernel() const {
      return *kernel_;
    }

    /** @brief Initializes the KDE global object.
     */
    void Init(
      TableType *reference_table_in,
      TableType *query_table_in,
      double effective_num_reference_points_in,
      double bandwidth_in, const bool is_monochromatic,
      double relative_error_in, double absolute_error_in, double probability_in,
      const std::string &kernel_type_in,
      bool normalize_densities_in = true) {

      effective_num_reference_points_ = effective_num_reference_points_in;

      if(kernel_type_in == "gaussian") {
        kernel_ = new core::metric_kernels::GaussianKernel();
      }
      else if(kernel_type_in == "epan") {
        kernel_ = new core::metric_kernels::EpanKernel();
      }

      kernel_->Init(bandwidth_in);
      mult_const_ = 1.0 /
                    (kernel_->CalcNormConstant(
                       reference_table_in->n_attributes()) *
                     ((double) effective_num_reference_points_));

      relative_error_ = relative_error_in;
      absolute_error_ = absolute_error_in;
      probability_ = probability_in;
      query_table_ = query_table_in;
      reference_table_ = reference_table_in;

      // Initialize the temporary vector for storing the Monte Carlo
      // results.
      mean_variance_pair_.resize(query_table_->n_entries());

      // Set the normalize flag.
      normalize_densities_ = normalize_densities_in;

      // Set the monochromatic flag.
      is_monochromatic_ = is_monochromatic;
    }

    /** @brief Gets the multiplicative normalization constant.
     */
    double get_mult_const() const {
      return mult_const_;
    }
};

/** @brief Represents the storage of KDE computation results.
 */
template<typename ContainerType>
class KdeResult {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

  public:

    /** @brief The lower bound on the density sum.
     */
    ContainerType densities_l_;

    /** @brief The approximate density sum per query.
     */
    ContainerType densities_;

    /** @brief The upper bound on the density sum.
     */
    ContainerType densities_u_;

    /** @brief The number of points pruned per each query.
     */
    ContainerType pruned_;

    /** @brief The amount of maximum error incurred per each query.
     */
    ContainerType used_error_;

    /** @brief Serialize the KDE result object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & densities_l_;
      ar & densities_;
      ar & densities_u_;
      ar & pruned_;
      ar & used_error_;
    }

    /** @brief The default constructor.
     */
    KdeResult() {
      SetZero();
    }

    /** @brief Normalizes the density of each query.
     */
    template<typename GlobalType>
    void Normalize(const GlobalType &global) {
      for(unsigned int q_index = 0; q_index < densities_l_.size(); q_index++) {
        densities_l_[q_index] *= global.get_mult_const();
        densities_[q_index] *= global.get_mult_const();
        densities_u_[q_index] *= global.get_mult_const();
      }
    }

    template<typename MetricType, typename GlobalType>
    void PostProcess(
      const MetricType &metric,
      int q_index, const GlobalType &global,
      const bool is_monochromatic) {

      densities_[q_index] = 0.5 * (
                              densities_l_[q_index] + densities_u_[q_index]);
      if(global.normalize_densities()) {
        densities_l_[q_index] *= global.get_mult_const();
        densities_[q_index] *= global.get_mult_const();
        densities_u_[q_index] *= global.get_mult_const();
      }
    }

    void PrintDebug(const std::string &file_name) {
      FILE *file_output = fopen(file_name.c_str(), "w+");
      for(unsigned int i = 0; i < densities_.size(); i++) {
        fprintf(file_output, "%g %g %g %g\n", densities_l_[i],
                densities_[i], densities_u_[i], pruned_[i]);
      }
      fclose(file_output);
    }

    template<typename GlobalType, typename TreeType, typename DeltaType>
    void ApplyProbabilisticDelta(
      GlobalType &global, TreeType *qnode, double failure_probability,
      const DeltaType &delta_in) {

      // Get the iterator for the query node.
      typename GlobalType::TableType::TreeIterator qnode_it =
        global.query_table()->get_node_iterator(qnode);
      core::table::DensePoint qpoint;
      int qpoint_index;

      // Look up the number of standard deviations.
      double num_standard_deviations = global.compute_quantile(
                                         failure_probability);

      do {
        // Get each query point.
        qnode_it.Next(&qpoint, &qpoint_index);
        core::math::Range contribution;
        (*delta_in.mean_variance_pair_)[qpoint_index].scaled_interval(
          delta_in.pruned_, num_standard_deviations, &contribution);
        densities_l_[qpoint_index] += contribution.lo;
        densities_u_[qpoint_index] += contribution.hi;
        pruned_[qpoint_index] += delta_in.pruned_;
      }
      while(qnode_it.HasNext());
    }

    void Init(int num_points) {
      densities_l_.resize(num_points);
      densities_.resize(num_points);
      densities_u_.resize(num_points);
      pruned_.resize(num_points);
      used_error_.resize(num_points);

      SetZero();
    }

    void SetZero() {
      for(int i = 0; i < static_cast<int>(densities_l_.size()); i++) {
        densities_l_[i] = 0;
        densities_[i] = 0;
        densities_u_[i] = 0;
        pruned_[i] = 0;
        used_error_[i] = 0;
      }
    }

    void ApplyPostponed(
      int q_index,
      const KdePostponed &postponed_in) {

      densities_l_[q_index] = densities_l_[q_index] + postponed_in.densities_l_;
      densities_u_[q_index] = densities_u_[q_index] + postponed_in.densities_u_;
      pruned_[q_index] = pruned_[q_index] + postponed_in.pruned_;
      used_error_[q_index] = used_error_[q_index] + postponed_in.used_error_;
    }
};

class KdeDelta {

  public:

    double densities_l_;

    double densities_u_;

    double pruned_;

    double used_error_;

    std::vector< core::monte_carlo::MeanVariancePair > *mean_variance_pair_;

    KdeDelta() {
      SetZero();
    }

    ~KdeDelta() {
    }

    void SetZero() {
      densities_l_ = densities_u_ = pruned_ = used_error_ = 0;
      mean_variance_pair_ = NULL;
    }

    template<typename MetricType, typename GlobalType, typename TreeType>
    void DeterministicCompute(
      const MetricType &metric,
      const GlobalType &global, TreeType *qnode, TreeType *rnode,
      const core::math::Range &squared_distance_range) {

      int rnode_count = (qnode == rnode) ?
                        (rnode->count() - 1) : rnode->count();
      densities_l_ = rnode_count *
                     global.kernel().EvalUnnormOnSq(squared_distance_range.hi);
      densities_u_ = rnode_count *
                     global.kernel().EvalUnnormOnSq(squared_distance_range.lo);
      pruned_ = static_cast<double>(rnode_count);
      used_error_ = 0.5 * (densities_u_ - densities_l_);
    }
};

class KdeSummary {

  private:

    friend class boost::serialization::access;

  public:

    double densities_l_;

    double densities_u_;

    double pruned_l_;

    double used_error_u_;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & densities_l_;
      ar & densities_u_;
      ar & pruned_l_;
      ar & used_error_u_;
    }

    void Copy(const KdeSummary &summary_in) {
      densities_l_ = summary_in.densities_l_;
      densities_u_ = summary_in.densities_u_;
      pruned_l_ = summary_in.pruned_l_;
      used_error_u_ = summary_in.used_error_u_;
    }

    KdeSummary() {
      SetZero();
    }

    KdeSummary(const KdeSummary &summary_in) {
      densities_l_ = summary_in.densities_l_;
      densities_u_ = summary_in.densities_u_;
      pruned_l_ = summary_in.pruned_l_;
      used_error_u_ = summary_in.used_error_u_;
    }

    template < typename MetricType, typename GlobalType, typename DeltaType,
             typename TreeType, typename ResultType >
    bool CanProbabilisticSummarize(
      const MetricType &metric,
      GlobalType &global, DeltaType &delta, TreeType *qnode, TreeType *rnode,
      double failure_probability, ResultType *query_results) const {

      const int speedup_factor = 10;
      int num_samples = rnode->count() / speedup_factor;

      if(num_samples > rnode->count()) {
        return false;
      }

      // Get the iterator for the query node.
      typename GlobalType::TableType::TreeIterator qnode_it =
        global.query_table()->get_node_iterator(qnode);
      core::table::DensePoint qpoint;
      int qpoint_index;

      // Get the iterator for the reference node.
      typename GlobalType::TableType::TreeIterator rnode_it =
        global.reference_table()->get_node_iterator(rnode);
      core::table::DensePoint rpoint;
      int rpoint_index;

      // Interval for the pivot query point.
      double num_standard_deviations = global.compute_quantile(
                                         failure_probability);
      delta.mean_variance_pair_ = ((GlobalType &) global).mean_variance_pair();

      // The flag saying whether the pruning is a success.
      bool prunable = true;

      // The min kernel value determined by the bounding box.
      double min_kernel_value = delta.densities_l_ /
                                static_cast<double>(rnode->count());

      int prev_qpoint_index = -1;
      double bandwidth = sqrt(global.kernel().bandwidth_sq());
      double movement_threshold = 0.05 * bandwidth;
      int movement_count = 0;
      do {

        // Get each query point.
        qnode_it.Next(&qpoint, &qpoint_index);
        bool skip = false;
        if(prev_qpoint_index >= 0) {
          core::table::DensePoint prev_qpoint;
          global.query_table()->get(prev_qpoint_index, &prev_qpoint);
          double dist = sqrt(metric.DistanceSq(qpoint, prev_qpoint));
          if(dist <= movement_threshold && movement_count < 5) {
            (*delta.mean_variance_pair_)[qpoint_index].CopyValues(
              (*delta.mean_variance_pair_)[prev_qpoint_index]);
            skip = true;
            movement_count++;
          }
          else {
            movement_count = 0;
          }
        }

        // Clear the sample mean variance pair for the current query.
        if(skip == false) {
          (*delta.mean_variance_pair_)[qpoint_index].SetZero();

          for(int i = 0; i < num_samples; i++) {

            // Pick a random reference point and compute the kernel
            // difference.
            rnode_it.RandomPick(&rpoint, &rpoint_index);
            double squared_dist = metric.DistanceSq(qpoint, rpoint);
            double kernel_value =
              global.kernel().EvalUnnormOnSq(squared_dist);
            double new_sample = kernel_value - min_kernel_value;

            // Accumulate the sample.
            (*delta.mean_variance_pair_)[qpoint_index].push_back(new_sample);
          }
        }

        // Add the correction.
        core::math::Range correction;
        (*delta.mean_variance_pair_)[qpoint_index].scaled_interval(
          delta.pruned_, num_standard_deviations, &correction);
        correction.lo = std::max(correction.lo, 0.0);
        correction += delta.densities_l_;

        // Take the middle estimate, though technically it is not correct.
        double modified_densities_l =
          query_results->densities_l_[qpoint_index] + correction.lo;
        double left_hand_side = correction.width() * 0.5;
        double right_hand_side =
          rnode->count() *
          (global.relative_error() * modified_densities_l /
           static_cast<double>(global.reference_table()->n_entries()) +
           global.absolute_error());

        prunable = (left_hand_side <= right_hand_side);

        prev_qpoint_index = qpoint_index;
      }
      while(qnode_it.HasNext() && prunable);
      return prunable;
    }

    template < typename GlobalType, typename DeltaType, typename TreeType,
             typename ResultType >
    bool CanSummarize(
      const GlobalType &global, const DeltaType &delta,
      TreeType *qnode, TreeType *rnode, ResultType *query_results) const {

      double left_hand_side = delta.used_error_;
      double right_hand_side =
        rnode->count() * (
          (global.relative_error() * densities_l_ +
           global.effective_num_reference_points() * global.absolute_error() -
           used_error_u_) /
          static_cast<double>(
            global.effective_num_reference_points() - pruned_l_));
      return left_hand_side <= right_hand_side;
    }

    void SetZero() {
      densities_l_ = 0;
      densities_u_ = 0;
      pruned_l_ = 0;
      used_error_u_ = 0;
    }

    void Init() {
      SetZero();
    }

    void StartReaccumulate() {
      densities_l_ = std::numeric_limits<double>::max();
      densities_u_ = 0;
      pruned_l_ = densities_l_;
      used_error_u_ = 0;
    }

    template<typename GlobalType, typename ResultType>
    void Accumulate(
      const GlobalType &global, const ResultType &results, int q_index) {
      if(results.pruned_[q_index] < global.effective_num_reference_points()) {
        densities_l_ = std::min(densities_l_, results.densities_l_[q_index]);
        densities_u_ = std::max(densities_u_, results.densities_u_[q_index]);
        pruned_l_ = std::min(pruned_l_, results.pruned_[q_index]);
        used_error_u_ = std::max(used_error_u_, results.used_error_[q_index]);
      }
    }

    template<typename GlobalType>
    void Accumulate(
      const GlobalType &global, const KdeSummary &summary_in,
      const KdePostponed &postponed_in) {

      if(
        summary_in.pruned_l_ + postponed_in.pruned_ <
        global.effective_num_reference_points()) {
        densities_l_ = std::min(
                         densities_l_,
                         summary_in.densities_l_ + postponed_in.densities_l_);
        densities_u_ = std::max(
                         densities_u_,
                         summary_in.densities_u_ + postponed_in.densities_u_);
        pruned_l_ = std::min(
                      pruned_l_, summary_in.pruned_l_ + postponed_in.pruned_);
        used_error_u_ = std::max(
                          used_error_u_,
                          summary_in.used_error_u_ + postponed_in.used_error_);
      }
    }

    void ApplyDelta(const KdeDelta &delta_in) {
      densities_l_ = densities_l_ + delta_in.densities_l_;
      densities_u_ = densities_u_ + delta_in.densities_u_;
    }

    void ApplyPostponed(const KdePostponed &postponed_in) {
      densities_l_ = densities_l_ + postponed_in.densities_l_;
      densities_u_ = densities_u_ + postponed_in.densities_u_;
      pruned_l_ = pruned_l_ + postponed_in.pruned_;
      used_error_u_ = used_error_u_ + postponed_in.used_error_;
    }
};

class KdeStatistic {

  private:

    friend class boost::serialization::access;

    KdeStatistic(const KdeStatistic &stat_in) {
    }

  public:

    mlpack::kde::KdePostponed postponed_;

    mlpack::kde::KdeSummary summary_;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & postponed_;
      ar & summary_;
    }

    /** @brief Copies another abstract statistics (does not do anything).
     */
    void Copy(const KdeStatistic &stat_in) {
      postponed_.Copy(stat_in.postponed_);
      summary_.Copy(stat_in.summary_);
    }

    KdeStatistic() {
      SetZero();
    }

    void SetZero() {
      postponed_.SetZero();
      summary_.SetZero();
    }

    /**
     * Initializes by taking statistics on raw data.
     */
    template<typename TreeIteratorType>
    void Init(TreeIteratorType &iterator) {
      SetZero();
    }

    /** @brief Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    template<typename TreeIteratorType>
    void Init(
      TreeIteratorType &iterator,
      const KdeStatistic &left_stat,
      const KdeStatistic &right_stat) {
      SetZero();
    }
};
}
}

#endif
