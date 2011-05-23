/** @file local_regression_dualtree.h
 *
 *  The template stub filled out for computing the local regression
 *  estimate using a dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_DUALTREE_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_DUALTREE_H

#include <boost/math/distributions/normal.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include <deque>
#include "core/monte_carlo/mean_variance_pair.h"
#include "core/metric_kernels/kernel.h"
#include "core/tree/statistic.h"
#include "core/table/table.h"

namespace mlpack {
namespace local_regression {

/** @brief The postponed quantities for local regression.
 */
class LocalRegressionPostponed {

  private:

    // For boost serialization.
    friend class boost::serialization::access;

  public:

    /** @brief The lower bound on the postponed quantities for the
     *         left hand side.
     */
    arma::mat left_hand_side_l_;

    /** @brief The finite-difference postponed quantities for the left
     *         hand side.
     */
    arma::mat left_hand_side_e_;

    /** @brief The upper bound on the postponed quantities for the
     *         left hand side.
     */
    arma::mat left_hand_side_u_;

    /** @brief The lower bound on the postponed quantities for the
     *         right hand side.
     */
    arma::vec right_hand_side_l_;

    /** @brief The finite-difference postponed quantities for the left
     *         right side.
     */
    arma::vec right_hand_side_e_;

    /** @brief The upper bound on the postponed quantities for the
     *         right hand side.
     */
    arma::vec right_hand_side_u_;

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
      ar & left_hand_side_l_;
      ar & left_hand_side_e_;
      ar & left_hand_side_u_;
      ar & right_hand_side_l_;
      ar & right_hand_side_e_;
      ar & right_hand_side_u_;
      ar & pruned_;
      ar & used_error_;
    }

    /** @brief The default constructor.
     */
    LocalRegressionPostponed() {
      SetZero();
    }

    /** @brief Copies another postponed object.
     */
    void Copy(const LocalRegressionPostponed &postponed_in) {
      left_hand_side_l_ = postponed_in.left_hand_side_l_;
      left_hand_side_e_ = postponed_in.left_hand_side_e_;
      left_hand_side_u_ = postponed_in.left_hand_side_u_;
      right_hand_side_l_ = postponed_in.right_hand_side_l_;
      right_hand_side_e_ = postponed_in.right_hand_side_e_;
      right_hand_side_u_ = postponed_in.right_hand_side_u_;
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
      left_hand_side_l_.zeros();
      left_hand_side_e_.zeros();
      left_hand_side_u_.zeros();
      right_hand_side_l_.zeros();
      right_hand_side_e_.zeros();
      right_hand_side_u_.zeros();
      pruned_ = static_cast<double>(rnode->count());
      used_error_ = 0;
    }

    /** @brief Applies the incoming delta contribution to the
     *         postponed quantities, optionally to the query results
     *         as well.
     */
    template < typename TreeType, typename GlobalType,
             typename LocalRegressionDelta, typename ResultType >
    void ApplyDelta(
      TreeType *qnode, TreeType *rnode,
      const GlobalType &global, const LocalRegressionDelta &delta_in,
      ResultType *query_results) {

      // Finite-difference.
      left_hand_side_e_ +=
        0.5 * (
          delta_in.left_hand_side_l_ + delta_in.left_hand_side_u_);
      left_hand_side_l_ = left_hand_side_l_ + delta_in.left_hand_side_l_;
      left_hand_side_u_ = left_hand_side_u_ + delta_in.left_hand_side_u_;
      right_hand_side_e_ +=
        0.5 * (
          delta_in.right_hand_side_l_ + delta_in.right_hand_side_u_);
      right_hand_side_l_ = right_hand_side_l_ + delta_in.right_hand_side_l_;
      right_hand_side_u_ = right_hand_side_u_ + delta_in.right_hand_side_u_;
      pruned_ = pruned_ + delta_in.pruned_;
      used_error_ = used_error_ + delta_in.used_error_;
    }

    /** @brief Applies the incoming postponed contribution.
     */
    void ApplyPostponed(const LocalRegressionPostponed &other_postponed) {
      left_hand_side_l_ = left_hand_side_l_ + other_postponed.left_hand_side_l_;
      left_hand_side_e_ = left_hand_side_e_ + other_postponed.left_hand_side_e_;
      left_hand_side_u_ = left_hand_side_u_ + other_postponed.left_hand_side_u_;
      right_hand_side_l_ = right_hand_side_l_ +
                           other_postponed.right_hand_side_l_;
      right_hand_side_e_ = right_hand_side_e_ +
                           other_postponed.right_hand_side_e_;
      right_hand_side_u_ = right_hand_side_u_ +
                           other_postponed.right_hand_side_u_;
      pruned_ = pruned_ + other_postponed.pruned_;
      used_error_ = used_error_ + other_postponed.used_error_;
    }

    /** @brief Applies the incoming postponed contribution during the
     *         postprocessing stage.
     */
    template<typename GlobalType>
    void FinalApplyPostponed(
      const GlobalType &global, LocalRegressionPostponed &other_postponed) {

      ApplyPostponed(other_postponed);
    }

    /** @brief Called from an exact pairwise evaluation method
     *         (i.e. the base case) which incurs no error.
     */
    template<typename GlobalType, typename MetricType>
    void ApplyContribution(
      const GlobalType &global,
      const MetricType &metric,
      const arma::vec &query_point,
      double query_weight,
      const arma::vec &reference_point,
      double reference_weight) {

      double distsq = metric.DistanceSq(query_point, reference_point);
      double kernel_value = global.kernel().EvalUnnormOnSq(distsq);
      left_hand_side_l_.at(0, 0) += kernel_value;
      left_hand_side_e_.at(0, 0) += kernel_value;
      left_hand_side_u_.at(0, 0) += kernel_value;
      right_hand_side_l_.at(0, 0) += kernel_value * reference_weight;
      right_hand_side_e_.at(0, 0) += kernel_value * reference_weight;
      right_hand_side_u_.at(0, 0) += kernel_value * reference_weight;
      for(unsigned int j = 1; j <= reference_point.n_elem; j++) {

        // The row update for the left hand side.
        double left_hand_side_increment = kernel_value * reference_point[j - 1];
        left_hand_side_l_.at(0, j - 1) += left_hand_side_increment;
        left_hand_side_e_.at(0, j - 1) += left_hand_side_increment;
        left_hand_side_u_.at(0, j - 1) += left_hand_side_increment;

        // The column update for the left hand side.
        left_hand_side_l_.at(j - 1, 0) += left_hand_side_increment;
        left_hand_side_e_.at(j - 1, 0) += left_hand_side_increment;
        left_hand_side_u_.at(j - 1, 0) += left_hand_side_increment;

        // The right hand side.
        double right_hand_side_increment =
          kernel_value * reference_weight * reference_point[j - 1];
        right_hand_side_l_[j - 1] += right_hand_side_increment;
        right_hand_side_e_[j - 1] += right_hand_side_increment;
        right_hand_side_u_[j - 1] += right_hand_side_increment;

        for(unsigned int i = 1; i <= reference_point.n_elem; i++) {

          double inner_increment = kernel_value * reference_point[i - 1] *
                                   reference_point[j - 1];
          left_hand_side_l_.at(i, j) += inner_increment;
          left_hand_side_e_.at(i, j) += inner_increment;
          left_hand_side_u_.at(i, j) += inner_increment;
        }
      }
    }

    /** @brief Sets everything to zero.
     */
    void SetZero() {
      left_hand_side_l_.zeros();
      left_hand_side_e_.zeros();
      left_hand_side_u_.zeros();
      right_hand_side_l_.zeros();
      right_hand_side_e_.zeros();
      right_hand_side_u_.zeros();
      pruned_ = 0;
      used_error_ = 0;
    }

    /** @brief Sets everything to zero in the post-processing step.
     */
    void FinalSetZero() {
      this->SetZero();
    }
};

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
    mlpack::series_expansion::EpanKernel > {
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
    KernelType kernel_;

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
    MeanVariancePairListType *mean_variance_pair_;

  public:

    /** @brief Tells whether the given squared distance range is
     *         sufficient for pruning for any pair of query/reference
     *         pair that satisfies the range.
     */
    bool ConsiderExtrinsicPrune(
      const core::math::Range &squared_distance_range) const {

      return
        ConsiderExtrinsicPruneTrait<KernelType>::Compute(
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
                    (kernel_aux_->kernel().CalcNormConstant(
                       reference_table_in->n_attributes()) *
                     ((double) effective_num_reference_points_));
    }

    /** @brief The constructor.
     */
    LocalRegressionGlobal() {
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
    ~LocalRegressionGlobal() {
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

    /** @brief Returns the kernel.
     */
    const KernelType &kernel() const {
      return kernel_;
    }

    /** @brief Initializes the KDE global object.
     */
    void Init(
      TableType *reference_table_in,
      TableType *query_table_in,
      double effective_num_reference_points_in, KernelType *kernel_in,
      double bandwidth_in, MeanVariancePairListType *mean_variance_pair_in,
      const bool is_monochromatic,
      double relative_error_in, double absolute_error_in, double probability_in,
      bool normalize_densities_in = true) {

      effective_num_reference_points_ = effective_num_reference_points_in;

      // Initialize the kernel.
      kernel_.Init(bandwidth_in);
      mean_variance_pair_ = new MeanVariancePairListType();
      mult_const_ = 1.0 /
                    (kernel_.CalcNormConstant(
                       reference_table_in->n_attributes()) *
                     ((double) effective_num_reference_points_));

      relative_error_ = relative_error_in;
      absolute_error_ = absolute_error_in;
      probability_ = probability_in;
      query_table_ = query_table_in;
      reference_table_ = reference_table_in;

      // Initialize the temporary vector for storing the Monte Carlo
      // results.
      mean_variance_pair_->resize(query_table_->n_entries());

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
class LocalRegressionResult {
  private:

    // For BOOST serialization.
    friend class boost::serialization::access;

  public:

    /** @brief The flag that tells whether the self contribution has
     *         been subtracted or not.
     */
    std::deque<bool> self_contribution_subtracted_;

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

    /** @brief The number of far-to-local translations.
     */
    int num_farfield_to_local_prunes_;

    /** @brief The number of far-field evaluations.
     */
    int num_farfield_prunes_;

    /** @brief The number of direct local accumulations.
     */
    int num_local_prunes_;

    /** @brief Serialize the KDE result object.
     */
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & self_contribution_subtracted_;
      ar & densities_l_;
      ar & densities_;
      ar & densities_u_;
      ar & pruned_;
      ar & used_error_;
    }

    void Seed(int qpoint_index, double initial_pruned_in) {
      pruned_[qpoint_index] = initial_pruned_in;
    }

    /** @brief The default constructor.
     */
    LocalRegressionResult() {
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

      // If monochromatic, then do post correction by subtracing the
      // self-contribution.
      if((! self_contribution_subtracted_[q_index]) && is_monochromatic) {
        double self_contribution =
          global.kernel_aux().kernel().EvalUnnormOnSq(0.0);
        densities_l_[q_index] -= self_contribution;
        densities_[q_index] -= self_contribution;
        densities_u_[q_index] -= self_contribution;
        self_contribution_subtracted_[q_index] = true;
      }

      if(global.normalize_densities()) {
        densities_l_[q_index] *= global.get_mult_const();
        densities_[q_index] *= global.get_mult_const();
        densities_u_[q_index] *= global.get_mult_const();
      }
    }

    void Print(const std::string &file_name) const {
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
        contribution.lo = std::max(contribution.lo, 0.0);
        contribution.hi = std::min(contribution.hi, delta_in.pruned_);
        densities_l_[qpoint_index] += contribution.lo;
        densities_[qpoint_index] += contribution.mid();
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
      self_contribution_subtracted_.resize(num_points);
      SetZero();
    }

    void SetZero() {
      for(int i = 0; i < static_cast<int>(densities_l_.size()); i++) {
        densities_l_[i] = 0;
        densities_[i] = 0;
        densities_u_[i] = 0;
        pruned_[i] = 0;
        used_error_[i] = 0;
        self_contribution_subtracted_[i] = false;
      }
      num_farfield_to_local_prunes_ = 0;
      num_farfield_prunes_ = 0;
      num_local_prunes_ = 0;
    }

    template<typename LocalRegressionPostponedType>
    void ApplyPostponed(
      int q_index, const LocalRegressionPostponedType &postponed_in) {
      densities_l_[q_index] = densities_l_[q_index] + postponed_in.densities_l_;
      densities_[q_index] = densities_[q_index] + postponed_in.densities_e_;
      densities_u_[q_index] = densities_u_[q_index] + postponed_in.densities_u_;
      pruned_[q_index] = pruned_[q_index] + postponed_in.pruned_;
      used_error_[q_index] = used_error_[q_index] + postponed_in.used_error_;
    }

    template<typename GlobalType, typename LocalRegressionPostponedType>
    void FinalApplyPostponed(
      const GlobalType &global,
      const core::table::DensePoint &qpoint,
      int q_index,
      const LocalRegressionPostponedType &postponed_in) {

      // Evaluate the local expansion.
      densities_[q_index] +=
        postponed_in.local_expansion_.EvaluateField(
          global.kernel_aux(), qpoint);

      // Apply postponed.
      ApplyPostponed(q_index, postponed_in);
    }
};

class LocalRegressionDelta {

  public:

    double densities_l_;

    double densities_u_;

    double pruned_;

    double used_error_;

    int order_farfield_to_local_;

    int order_farfield_;

    int order_local_;

    std::vector< core::monte_carlo::MeanVariancePair > *mean_variance_pair_;

    LocalRegressionDelta() {
      SetZero();
    }

    void SetZero() {
      densities_l_ = densities_u_ = pruned_ = used_error_ = 0;
      order_farfield_to_local_ = -1;
      order_farfield_ = -1;
      order_local_ = -1;
      mean_variance_pair_ = NULL;
    }

    template<typename MetricType, typename GlobalType, typename TreeType>
    void DeterministicCompute(
      const MetricType &metric,
      const GlobalType &global, TreeType *qnode, TreeType *rnode,
      const core::math::Range &squared_distance_range) {

      int rnode_count = rnode->count();
      densities_l_ = rnode_count *
                     global.kernel().EvalUnnormOnSq(squared_distance_range.hi);
      densities_u_ = rnode_count *
                     global.kernel().EvalUnnormOnSq(squared_distance_range.lo);
      pruned_ = static_cast<double>(rnode_count);
      used_error_ = 0.5 * (densities_u_ - densities_l_);
    }
};

class LocalRegressionSummary {

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

  public:

    double densities_l_;

    double densities_u_;

    double pruned_l_;

    double used_error_u_;

    void Seed(double initial_pruned_in) {
      this->SetZero();
      pruned_l_ = initial_pruned_in;
    }

    void Print() const {
      printf("Lower bound/upper bound on the densities: [ %g %g ], "
             "Lower bound on the pruned components: %g, "
             "Upper bound on the used error: %g\n",
             densities_l_, densities_u_, pruned_l_, used_error_u_);
    }

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & densities_l_;
      ar & densities_u_;
      ar & pruned_l_;
      ar & used_error_u_;
    }

    void Copy(const LocalRegressionSummary &summary_in) {
      densities_l_ = summary_in.densities_l_;
      densities_u_ = summary_in.densities_u_;
      pruned_l_ = summary_in.pruned_l_;
      used_error_u_ = summary_in.used_error_u_;
    }

    LocalRegressionSummary() {
      SetZero();
    }

    LocalRegressionSummary(const LocalRegressionSummary &summary_in) {
      densities_l_ = summary_in.densities_l_;
      densities_u_ = summary_in.densities_u_;
      pruned_l_ = summary_in.pruned_l_;
      used_error_u_ = summary_in.used_error_u_;
    }

    template < typename MetricType, typename GlobalType,
             typename PostponedType, typename DeltaType,
             typename TreeType, typename ResultType >
    bool CanProbabilisticSummarize(
      const MetricType &metric,
      GlobalType &global,
      const PostponedType &postponed, DeltaType &delta,
      const core::math::Range &squared_distance_range,
      TreeType *qnode, TreeType *rnode,
      double failure_probability, ResultType *query_results) const {

      // The number of samples.
      const int num_samples = 25;
      if(rnode->count() < 50 ||
          2.0 * global.kernel_aux().kernel().bandwidth_sq() <
          squared_distance_range.hi) {
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
      do {

        // Get each query point.
        qnode_it.Next(&qpoint, &qpoint_index);

        // Clear the sample mean variance pair for the current query.
        (*delta.mean_variance_pair_)[qpoint_index].SetZero();

        for(int i = 0; i < num_samples; i++) {

          // Pick a random reference point and compute the kernel
          // difference.
          rnode_it.RandomPick(&rpoint, &rpoint_index);
          double squared_dist = metric.DistanceSq(qpoint, rpoint);
          double kernel_value =
            global.kernel().EvalUnnormOnSq(squared_dist);

          // Accumulate the sample.
          (*delta.mean_variance_pair_)[qpoint_index].push_back(kernel_value);
        }

        // Add the correction.
        core::math::Range correction;
        (*delta.mean_variance_pair_)[qpoint_index].scaled_interval(
          delta.pruned_, num_standard_deviations, &correction);
        correction.lo = std::max(correction.lo, 0.0);
        correction.hi = std::min(correction.hi, delta.pruned_);

        // Technically, though not correct, just use the mid point of
        // the Monte Carlo contribution.
        double modified_densities_l =
          query_results->densities_l_[qpoint_index] + correction.mid() +
          postponed.densities_l_;
        double left_hand_side = correction.width() * 0.5;
        double right_hand_side =
          rnode->count() * (
            global.relative_error() * modified_densities_l +
            global.effective_num_reference_points() * global.absolute_error() -
            used_error_u_) /
          static_cast<double>(
            global.effective_num_reference_points() -
            query_results->pruned_[qpoint_index]);

        // Prunable if the left hand side is less than right hand side.
        prunable = (left_hand_side <= right_hand_side);
      }
      while(qnode_it.HasNext() && prunable);
      return prunable;
    }

    template < typename GlobalType, typename DeltaType, typename TreeType,
             typename ResultType >
    bool CanSummarize(
      const GlobalType &global, DeltaType &delta,
      const core::math::Range &squared_distance_range,
      TreeType *qnode, TreeType *rnode, ResultType *query_results) const {

      double left_hand_side = delta.used_error_;
      double right_hand_side =
        rnode->count() * (
          global.relative_error() * densities_l_ +
          global.effective_num_reference_points() * global.absolute_error() -
          used_error_u_) /
        static_cast<double>(
          global.effective_num_reference_points() - pruned_l_);

      // Prunable by finite-difference.
      if(left_hand_side <= right_hand_side) {
        return true;
      }

      // Otherwise, try series expansion.
      else if(global.kernel_aux().global().get_max_order() > 0) {
        return CanSummarizeSeriesExpansion_(
                 global, delta, squared_distance_range,
                 qnode, rnode, right_hand_side, query_results);
      }
      return false;
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
      densities_l_ = std::min(densities_l_, results.densities_l_[q_index]);
      densities_u_ = std::max(densities_u_, results.densities_u_[q_index]);
      pruned_l_ = std::min(pruned_l_, results.pruned_[q_index]);
      used_error_u_ = std::max(used_error_u_, results.used_error_[q_index]);
    }

    template<typename GlobalType, typename LocalRegressionPostponedType>
    void Accumulate(
      const GlobalType &global, const LocalRegressionSummary &summary_in,
      const LocalRegressionPostponedType &postponed_in) {
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

    void ApplyDelta(const LocalRegressionDelta &delta_in) {
      densities_l_ = densities_l_ + delta_in.densities_l_;
      densities_u_ = densities_u_ + delta_in.densities_u_;
    }

    template<typename LocalRegressionPostponedType>
    void ApplyPostponed(const LocalRegressionPostponedType &postponed_in) {
      densities_l_ = densities_l_ + postponed_in.densities_l_;
      densities_u_ = densities_u_ + postponed_in.densities_u_;
      pruned_l_ = pruned_l_ + postponed_in.pruned_;
      used_error_u_ = used_error_u_ + postponed_in.used_error_;
    }
};

class LocalRegressionStatistic {

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

  public:

    mlpack::kde::LocalRegressionPostponed<ExpansionType> postponed_;

    mlpack::kde::LocalRegressionSummary summary_;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & postponed_;
      ar & summary_;
    }

    /** @brief Copies another KDE statistic.
     */
    void Copy(const LocalRegressionStatistic &stat_in) {
      postponed_.Copy(stat_in.postponed_);
      summary_.Copy(stat_in.summary_);
    }

    /** @brief The default constructor.
     */
    LocalRegressionStatistic() {
      SetZero();
    }

    /** @brief Sets the postponed and the summary statistics to zero.
     */
    void SetZero() {
      postponed_.SetZero();
      summary_.SetZero();
    }

    void Seed(double initial_pruned_in) {
      postponed_.SetZero();
      summary_.Seed(initial_pruned_in);
    }

    /** @brief Initializes by taking statistics on raw data.
     */
    template<typename GlobalType, typename TreeType>
    void Init(const GlobalType &global, TreeType *node) {

      // Sets the postponed quantities and summary statistics to zero.
      SetZero();
    }

    /** @brief Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    template<typename GlobalType, typename TreeType>
    void Init(
      const GlobalType &global,
      TreeType *node,
      const LocalRegressionStatistic &left_stat,
      const LocalRegressionStatistic &right_stat) {

      // Sets the postponed quantities and summary statistics to zero.
      SetZero();
    }
};
}
}

#endif
