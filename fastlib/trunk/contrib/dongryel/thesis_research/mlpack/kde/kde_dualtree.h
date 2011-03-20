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
#include "mlpack/series_expansion/hypercube_farfield_dev.h"
#include "mlpack/series_expansion/hypercube_local_dev.h"
#include "mlpack/series_expansion/multivariate_farfield_dev.h"
#include "mlpack/series_expansion/multivariate_local_dev.h"

namespace mlpack {
namespace kde {

/** @brief The postponed quantities for KDE.
 */
template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
class KdePostponed {

  private:

    // For boost serialization.
    friend class boost::serialization::access;

  public:

    /** @brief The local expansion representing the postponed
     *         quantities.
     */
    mlpack::series_expansion::CartesianLocal <
    ExpansionType > local_expansion_;

    /** @brief The lower bound on the postponed quantities.
     */
    double densities_l_;

    /** @brief The finite-difference postponed quantities.
     */
    double densities_e_;

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
      ar & densities_e_;
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
      densities_e_ = postponed_in.densities_e_;
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
      densities_e_ = 0;
      pruned_ = (qnode == rnode && global_in.is_monochromatic()) ?
                static_cast<double>(rnode->count() - 1) :
                static_cast<double>(rnode->count());
      used_error_ = 0;
    }

    /** @brief Applies the incoming delta contribution to the
     *         postponed quantities, optionally to the query results
     *         as well.
     */
    template < typename TreeType, typename GlobalType,
             typename KdeDelta, typename ResultType >
    void ApplyDelta(
      TreeType *qnode, TreeType *rnode,
      const GlobalType &global, const KdeDelta &delta_in,
      ResultType *query_results) {

      if(delta_in.order_farfield_to_local_ >= 0) {

        // Far-to-local translation.
        query_results->num_farfield_to_local_prunes_++;
        rnode->stat().farfield_expansion_.TranslateToLocal(
          global.kernel_aux(),
          delta_in.order_farfield_to_local_,
          & (qnode->stat().postponed_.local_expansion_));
      }
      else if(delta_in.order_farfield_ >= 0) {

        // Far-field evaluation.
        query_results->num_farfield_prunes_++;
        typename GlobalType::TableType::TreeIterator qnode_it =
          const_cast <
          typename GlobalType::TableType * >(
            global.query_table())->get_node_iterator(qnode);
        while(qnode_it.HasNext()) {
          core::table::DensePoint qpoint;
          int qpoint_id;
          qnode_it.Next(&qpoint, &qpoint_id);
          query_results->densities_[qpoint_id] +=
            rnode->stat().farfield_expansion_.EvaluateField(
              global.kernel_aux(), qpoint, delta_in.order_farfield_);
        }
      }
      else if(delta_in.order_local_ >= 0) {

        // Direct local accumulation.
        typename GlobalType::TableType::TreeIterator rnode_it =
          const_cast<GlobalType &>(global).
          reference_table()->get_node_iterator(rnode);
        query_results->num_local_prunes_++;
        qnode->stat().postponed_.local_expansion_.AccumulateCoeffs(
          global.kernel_aux(), rnode_it, delta_in.order_local_);
      }
      else {

        // Finite-difference.
        densities_e_ += 0.5 * (delta_in.densities_l_ + delta_in.densities_u_);
      }

      densities_l_ = densities_l_ + delta_in.densities_l_;
      densities_u_ = densities_u_ + delta_in.densities_u_;
      pruned_ = pruned_ + delta_in.pruned_;
      used_error_ = used_error_ + delta_in.used_error_;
    }

    /** @brief Applies the incoming postponed contribution.
     */
    void ApplyPostponed(const KdePostponed &other_postponed) {
      densities_l_ = densities_l_ + other_postponed.densities_l_;
      densities_e_ = densities_e_ + other_postponed.densities_e_;
      densities_u_ = densities_u_ + other_postponed.densities_u_;
      pruned_ = pruned_ + other_postponed.pruned_;
      used_error_ = used_error_ + other_postponed.used_error_;
    }

    /** @brief Applies the incoming postponed contribution.
     */
    template<typename GlobalType>
    void FinalApplyPostponed(
      const GlobalType &global, KdePostponed &other_postponed) {

      // Translate the local expansion.
      other_postponed.local_expansion_.TranslateToLocal(
        global.kernel_aux(), &local_expansion_);
      ApplyPostponed(other_postponed);
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
      densities_e_ = densities_e_ + density_incoming;
      densities_u_ = densities_u_ + density_incoming;
    }

    /** @brief Sets everything to zero except for the local expansion.
     */
    void SetZero() {
      densities_l_ = 0;
      densities_e_ = 0;
      densities_u_ = 0;
      pruned_ = 0;
      used_error_ = 0;
    }

    /** @brief Sets everything to zero.
     */
    void FinalSetZero() {
      local_expansion_.SetZero();
      this->SetZero();
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

      return kernel_aux_->kernel().bandwidth_sq() <= squared_distance_range.lo;
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
      if(global.normalize_densities()) {
        densities_l_[q_index] *= global.get_mult_const();
        densities_[q_index] *= global.get_mult_const();
        densities_u_[q_index] *= global.get_mult_const();
      }
    }

    void Print(const std::string &file_name) {
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
      num_farfield_to_local_prunes_ = 0;
      num_farfield_prunes_ = 0;
      num_local_prunes_ = 0;
    }

    template<typename KdePostponedType>
    void ApplyPostponed(
      int q_index, const KdePostponedType &postponed_in) {
      densities_l_[q_index] = densities_l_[q_index] + postponed_in.densities_l_;
      densities_[q_index] = densities_[q_index] + postponed_in.densities_e_;
      densities_u_[q_index] = densities_u_[q_index] + postponed_in.densities_u_;
      pruned_[q_index] = pruned_[q_index] + postponed_in.pruned_;
      used_error_[q_index] = used_error_[q_index] + postponed_in.used_error_;
    }

    template<typename GlobalType, typename KdePostponedType>
    void FinalApplyPostponed(
      const GlobalType &global,
      const core::table::DensePoint &qpoint,
      int q_index,
      const KdePostponedType &postponed_in) {

      // Evaluate the local expansion.
      densities_[q_index] +=
        postponed_in.local_expansion_.EvaluateField(
          global.kernel_aux(), qpoint);

      // Apply postponed.
      ApplyPostponed(q_index, postponed_in);
    }
};

class KdeDelta {

  public:

    double densities_l_;

    double densities_u_;

    double pruned_;

    double used_error_;

    int order_farfield_to_local_;

    int order_farfield_;

    int order_local_;

    std::vector< core::monte_carlo::MeanVariancePair > *mean_variance_pair_;

    KdeDelta() {
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

    // For Boost serialization.
    friend class boost::serialization::access;

    /** @brief Determines whether the contribution of the reference
     *         node can be approximated using series expansion.
     */
    template <
    typename GlobalType, typename DeltaType, typename TreeType,
             typename ResultType >
    bool CanSummarizeSeriesExpansion_(
      const GlobalType &global, DeltaType &delta,
      const core::math::Range &squared_distance_range,
      TreeType *qnode, TreeType *rnode, double right_hand_side,
      ResultType *query_results) const {

      // The far-field expansion of the reference node.
      const typename GlobalType::KernelAuxType::FarFieldType &
      farfield_expansion = rnode->stat().farfield_expansion_;

      // Actual amount of error incurred per each query/ref pair.
      double actual_err_farfield_to_local = 0;
      double actual_err_farfield = 0;
      double actual_err_local = 0;

      // The allowed error per each query/ref pair.
      double allowed_err =
        right_hand_side / static_cast<double>(rnode->count());

      // Estimated computational cost.
      int cost_farfield_to_local = std::numeric_limits<int>::max();
      int cost_farfield = std::numeric_limits<int>::max();
      int cost_local = std::numeric_limits<int>::max();
      int cost_exhaustive = (qnode->count()) * (rnode->count()) *
                            (qnode->bound().dim());
      int min_cost = 0;

      // Get the order of approximations.
      delta.order_farfield_to_local_ =
        global.kernel_aux().OrderForConvertingFromFarFieldToLocal(
          rnode->bound(), qnode->bound(),
          squared_distance_range.lo, squared_distance_range.hi,
          allowed_err, &actual_err_farfield_to_local);

      if(global.query_table()->points_available_underneath(qnode)) {

        // Farfield evaluations are possible when the query poins are
        // available.
        delta.order_farfield_ =
          global.kernel_aux().OrderForEvaluatingFarField(
            rnode->bound(), qnode->bound(),
            squared_distance_range.lo, squared_distance_range.hi,
            allowed_err, &actual_err_farfield);
      }

      if(global.reference_table()->points_available_underneath(rnode)) {

        // Direct local accumulations are possible when the reference
        // points are available.
        delta.order_local_ =
          global.kernel_aux().OrderForEvaluatingLocal(
            rnode->bound(), qnode->bound(),
            squared_distance_range.lo, squared_distance_range.hi,
            allowed_err, &actual_err_local);
      }

      // Update computational cost and compute the minimum.
      if(delta.order_farfield_to_local_ >= 0) {
        cost_farfield_to_local =
          static_cast<int>(
            global.kernel_aux().global().FarFieldToLocalTranslationCost(
              delta.order_farfield_to_local_));
      }
      if(delta.order_farfield_ >= 0) {
        cost_farfield =
          static_cast<int>(
            global.kernel_aux().global().FarFieldEvaluationCost(
              delta.order_farfield_) * (qnode->count()));
      }
      if(delta.order_local_ >= 0) {
        cost_local =
          static_cast<int>(
            global.kernel_aux().global().DirectLocalAccumulationCost(
              delta.order_local_) * (rnode->count()));
      }

      min_cost =
        std::min(
          cost_farfield_to_local,
          std::min(
            cost_farfield, std::min(
              cost_local, cost_exhaustive)));

      if(cost_farfield_to_local == min_cost) {
        delta.used_error_ = farfield_expansion.get_weight_sum() *
                            actual_err_farfield_to_local;
        delta.order_farfield_ = -1;
        delta.order_local_ = -1;
        return true;
      }

      if(cost_farfield == min_cost) {
        delta.used_error_ = farfield_expansion.get_weight_sum() *
                            actual_err_farfield;
        delta.order_farfield_to_local_ = -1;
        delta.order_local_ = -1;
        return true;
      }

      if(cost_local == min_cost) {
        delta.used_error_ = farfield_expansion.get_weight_sum() *
                            actual_err_local;
        delta.order_farfield_to_local_ = -1;
        delta.order_farfield_ = -1;
        return true;
      }

      delta.order_farfield_to_local_ = -1;
      delta.order_farfield_ = -1;
      delta.order_local_ = -1;
      return false;
    }

  public:

    double densities_l_;

    double densities_u_;

    double pruned_l_;

    double used_error_u_;

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

      if((! global.query_table()->points_available_underneath(qnode)) ||
          (! global.reference_table()->points_available_underneath(rnode))) {
        return false;
      }

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
      else {
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

    template<typename GlobalType, typename KdePostponedType>
    void Accumulate(
      const GlobalType &global, const KdeSummary &summary_in,
      const KdePostponedType &postponed_in) {
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

    void ApplyDelta(const KdeDelta &delta_in) {
      densities_l_ = densities_l_ + delta_in.densities_l_;
      densities_u_ = densities_u_ + delta_in.densities_u_;
    }

    template<typename KdePostponedType>
    void ApplyPostponed(const KdePostponedType &postponed_in) {
      densities_l_ = densities_l_ + postponed_in.densities_l_;
      densities_u_ = densities_u_ + postponed_in.densities_u_;
      pruned_l_ = pruned_l_ + postponed_in.pruned_;
      used_error_u_ = used_error_u_ + postponed_in.used_error_;
    }
};

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
class KdeStatistic {

  private:

    // For Boost serialization.
    friend class boost::serialization::access;

  public:

    mlpack::series_expansion::CartesianFarField <
    ExpansionType > farfield_expansion_;

    mlpack::kde::KdePostponed<ExpansionType> postponed_;

    mlpack::kde::KdeSummary summary_;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
      ar & farfield_expansion_;
      ar & postponed_;
      ar & summary_;
    }

    /** @brief Copies another KDE statistic.
     */
    void Copy(const KdeStatistic &stat_in) {
      farfield_expansion_.Copy(stat_in.farfield_expansion_);
      postponed_.Copy(stat_in.postponed_);
      summary_.Copy(stat_in.summary_);
    }

    /** @brief The default constructor.
     */
    KdeStatistic() {
      SetZero();
    }

    /** @brief Sets the postponed and the summary statistics to zero.
     */
    void SetZero() {
      postponed_.SetZero();
      summary_.SetZero();
    }

    /** @brief Initializes by taking statistics on raw data.
     */
    template<typename GlobalType, typename TreeType>
    void Init(const GlobalType &global, TreeType *node) {

      // The node iterator.
      typename GlobalType::TableType::TreeIterator node_it =
        const_cast<GlobalType &>(global).
        reference_table()->get_node_iterator(node);

      // Sets the postponed quantities and summary statistics to zero.
      SetZero();

      // Form the far-field moments.
      core::table::DensePoint node_center;
      node->bound().center(&node_center);
      farfield_expansion_.Init(global.kernel_aux(), node_center);
      farfield_expansion_.AccumulateCoeffs(
        global.kernel_aux(), node_it,
        global.kernel_aux().global().get_max_order());

      // Initialize the local expansion.
      postponed_.local_expansion_.Init(global.kernel_aux(), node_center);
    }

    /** @brief Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    template<typename GlobalType, typename TreeType>
    void Init(
      const GlobalType &global,
      TreeType *node,
      const KdeStatistic &left_stat,
      const KdeStatistic &right_stat) {

      // Sets the postponed quantities and summary statistics to zero.
      SetZero();

      // Form the far-field moments.
      core::table::DensePoint node_center;
      node->bound().center(&node_center);
      farfield_expansion_.Init(global.kernel_aux(), node_center);
      farfield_expansion_.TranslateFromFarField(
        global.kernel_aux(), left_stat.farfield_expansion_);
      farfield_expansion_.TranslateFromFarField(
        global.kernel_aux(), right_stat.farfield_expansion_);

      // Initialize the local expansion.
      postponed_.local_expansion_.Init(global.kernel_aux(), node_center);
    }
};
}
}

#endif
