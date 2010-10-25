/** @file nbody_tripletree.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef PHYSPACK_NBODY_SIMULATOR_NBODY_SIMULATOR_TRIPLETREE_H
#define PHYSPACK_NBODY_SIMULATOR_NBODY_SIMULATOR_TRIPLETREE_H

#include <deque>
#include <vector>
#include <armadillo>

#include "core/gnp/triple_range_distance_sq.h"
#include "axilrod_teller.h"
#include <boost/math/distributions/normal.hpp>
#include <boost/utility.hpp>
#include "core/monte_carlo/mean_variance_pair.h"

namespace physpack {
namespace nbody_simulator {
class NbodySimulatorPostponed {

  public:

    core::math::Range negative_potential_;

    core::math::Range positive_potential_;

    double pruned_;

    double used_error_;

  public:
    NbodySimulatorPostponed() {
      Init();
    }

    NbodySimulatorPostponed(double num_tuples) {
      negative_potential_.Init(0, 0);
      positive_potential_.Init(0, 0);
      pruned_ = num_tuples;
      used_error_ = 0;
    }

    void Init() {
      SetZero();
    }

    void Init(double num_tuples) {
      negative_potential_.Init(0, 0);
      positive_potential_.Init(0, 0);
      pruned_ = num_tuples;
      used_error_ = 0;
    }

    template<typename NbodyDelta, typename ResultType>
    void ApplyDelta(
      const NbodyDelta &delta_in, int node_index, ResultType *query_results) {
      negative_potential_ = negative_potential_ +
                            delta_in.negative_potential_[node_index];
      positive_potential_ = positive_potential_ +
                            delta_in.positive_potential_[node_index];
      pruned_ = pruned_ + delta_in.pruned_[node_index];
      used_error_ = used_error_ + delta_in.used_error_[node_index];
    }

    void ApplyPostponed(const NbodySimulatorPostponed &other_postponed) {
      negative_potential_ = negative_potential_ +
                            other_postponed.negative_potential_;
      positive_potential_ = positive_potential_ +
                            other_postponed.positive_potential_;
      pruned_ = pruned_ + other_postponed.pruned_;
      used_error_ = used_error_ + other_postponed.used_error_;
    }

    void SetZero() {
      negative_potential_.Init(0, 0);
      positive_potential_.Init(0, 0);
      pruned_ = 0;
      used_error_ = 0;
    }
};

class NbodySimulatorDelta {
  public:

    std::vector< core::math::Range > negative_potential_;

    std::vector< core::math::Range > positive_potential_;

    std::vector<double> pruned_;

    std::vector<double> used_error_;

    std::vector< core::monte_carlo::MeanVariancePair > *mean_variance_pair_;

    std::vector< core::monte_carlo::MeanVariancePair > *mean_variance_pair() {
      return mean_variance_pair_;
    }

    template<typename GlobalType, typename TreeType>
    void ResetMeanVariancePairs(
      GlobalType &global,
      const std::vector<TreeType *> &nodes) {

      for(int i = 0; i < 3; i++) {
        TreeType *node = nodes[i];
        if(i == 0 || node != nodes[i - 1]) {

          // Get the iterator for the node.
          core::table::Table::TreeIterator node_it =
            global.table()->get_node_iterator(node);
          int qpoint_index;
          for(int j = 0; j < node_it.count(); j++) {
            node_it.get_id(j, &qpoint_index);
            (*mean_variance_pair_)[qpoint_index].SetZero();
          }
        }
      }
    }

    NbodySimulatorDelta() {
      negative_potential_.resize(3);
      positive_potential_.resize(3);
      pruned_.resize(3);
      used_error_.resize(3);
      SetZero();
    }

    void SetZero() {
      for(unsigned int i = 0; i < negative_potential_.size(); i++) {
        negative_potential_[i].Init(0, 0);
        positive_potential_[i].Init(0, 0);
        pruned_[i] = 0;
        used_error_[i] = 0;
      }
      mean_variance_pair_ = NULL;
    }

    template<typename GlobalType>
    void DeterministicCompute(
      const core::metric_kernels::AbstractMetric &metric,
      const GlobalType &global,
      const core::gnp::TripleRangeDistanceSq &triple_range_distance_sq) {

      // Set the mean variance pair pointer.
      mean_variance_pair_ =
        const_cast< GlobalType & >(global).mean_variance_pair();

      // Compute the potential range.
      core::math::Range potential_range = global.potential().RangeUnnormOnSq(
                                            triple_range_distance_sq);
      for(unsigned int i = 0;
          i < pruned_.size();
          i++) {
        pruned_[i] = triple_range_distance_sq.num_tuples(i);
        used_error_[i] = pruned_[i] * 0.5 * potential_range.width();
        if(potential_range.lo < 0) {
          negative_potential_[i].lo = pruned_[i] * potential_range.lo;
          positive_potential_[i].lo = 0.0;
        }
        else {
          negative_potential_[i].lo = 0.0;
          positive_potential_[i].lo = pruned_[i] * potential_range.lo;
        }
        if(potential_range.hi > 0) {
          negative_potential_[i].hi = 0.0;
          positive_potential_[i].hi = pruned_[i] * potential_range.hi;
        }
        else {
          negative_potential_[i].hi = pruned_[i] * potential_range.hi;
          positive_potential_[i].hi = 0.0;
        }
      }
    }
};

class NbodySimulatorResult {
  public:
    std::vector< core::math::Range > negative_potential_;
    std::vector< core::math::Range > positive_potential_;
    std::vector<double> potential_e_;
    std::vector<double> pruned_;
    std::vector<double> used_error_;

    template<typename GlobalType>
    void PostProcess(
      const core::metric_kernels::AbstractMetric &metric,
      int q_index, const GlobalType &global) {

      potential_e_[q_index] = negative_potential_[q_index].mid() +
                              positive_potential_[q_index].mid();
    }

    void PrintDebug(const std::string &file_name) {
      FILE *file_output = fopen(file_name.c_str(), "w+");
      for(unsigned int i = 0; i < potential_e_.size(); i++) {
        fprintf(file_output, "%g %g\n", potential_e_[i], pruned_[i]);
      }
      fclose(file_output);
    }

    void Init(int num_points) {
      negative_potential_.resize(num_points);
      positive_potential_.resize(num_points);
      potential_e_.resize(num_points);
      pruned_.resize(num_points);
      used_error_.resize(num_points);

      SetZero();
    }

    void SetZero() {
      for(int i = 0; i < static_cast<int>(negative_potential_.size()); i++) {
        negative_potential_[i].Init(0.0, 0.0);
        positive_potential_[i].Init(0.0, 0.0);
        potential_e_[i] = 0.0;
        pruned_[i] = 0.0;
        used_error_[i] = 0.0;
      }
    }

    template<typename GlobalType>
    void ApplyProbabilisticDelta(
      GlobalType &global,
      const core::gnp::TripleRangeDistanceSq &triple_range_distance_sq_in,
      const std::vector<double> &failure_probabilities,
      const NbodySimulatorDelta &delta_in) {

      for(int node_index = 0; node_index < 3; node_index++) {

        core::table::Table::TreeType *node =
          triple_range_distance_sq_in.node(node_index);
        if(node_index == 0 || node !=
            triple_range_distance_sq_in.node(node_index - 1)) {

          // Get the iterator for the node.
          core::table::Table::TreeIterator node_it =
            global.table()->get_node_iterator(node);
          core::table::DenseConstPoint qpoint;
          int qpoint_index;

          // Look up the number of standard deviations.
          double num_standard_deviations =
            global.compute_quantile(failure_probabilities[node_index]);

          do {
            // Get each point and apply contribution.
            node_it.Next(&qpoint, &qpoint_index);
            core::math::Range contribution;
            (*delta_in.mean_variance_pair_)[qpoint_index].scaled_interval(
              delta_in.pruned_[node_index], num_standard_deviations,
              &contribution);
            if(contribution.lo < 0) {
              negative_potential_[qpoint_index].lo += contribution.lo;
            }
            else {
              positive_potential_[qpoint_index].lo += contribution.lo;
            }
            if(contribution.hi < 0) {
              negative_potential_[qpoint_index].hi += contribution.hi;
            }
            else {
              positive_potential_[qpoint_index].hi += contribution.hi;
            }
            pruned_[qpoint_index] += delta_in.pruned_[node_index];
            used_error_[qpoint_index] += delta_in.used_error_[node_index];
          }
          while(node_it.HasNext());
        }
      } // end of looping over each node.
    }

    void ApplyPostponed(
      int q_index,
      const NbodySimulatorPostponed &postponed_in) {

      negative_potential_[q_index] += postponed_in.negative_potential_;
      positive_potential_[q_index] += postponed_in.positive_potential_;
      pruned_[q_index] = pruned_[q_index] + postponed_in.pruned_;
      used_error_[q_index] = used_error_[q_index] + postponed_in.used_error_;
    }
};

class NbodySimulatorGlobal {

  private:

    double relative_error_;

    double probability_;

    core::table::Table *table_;

    physpack::nbody_simulator::AxilrodTeller potential_;

    double total_num_tuples_;

    boost::math::normal normal_dist_;

    std::vector< core::monte_carlo::MeanVariancePair > mean_variance_pair_;

  public:

    std::vector< core::monte_carlo::MeanVariancePair > *mean_variance_pair() {
      return &mean_variance_pair_;
    }

    double compute_quantile(double tail_mass) const {
      double mass = 1 - 0.5 * tail_mass;
      if(mass > 0.999) {
        return 3;
      }
      else {
        return boost::math::quantile(normal_dist_, mass);
      }
    }

    const physpack::nbody_simulator::AxilrodTeller &potential() const {
      return potential_;
    }

    void ApplyContribution(
      const core::gnp::TripleDistanceSq &range_in,
      std::vector< NbodySimulatorPostponed > *postponeds) const {

      double potential_value = potential_.EvalUnnormOnSq(range_in);

      if(potential_value < 0.0) {
        for(unsigned int i = 0; i < postponeds->size(); i++) {
          (*postponeds)[i].negative_potential_.Init(
            potential_value, potential_value);
          (*postponeds)[i].positive_potential_.Init(0.0, 0.0);
        }
      }
      else {
        for(unsigned int i = 0; i < postponeds->size(); i++) {
          (*postponeds)[i].negative_potential_.Init(0.0, 0.0);
          (*postponeds)[i].positive_potential_.Init(
            potential_value, potential_value);
        }
      }
      for(unsigned int i = 0; i < postponeds->size(); i++) {
        (*postponeds)[i].pruned_ = (*postponeds)[i].used_error_ = 0.0;
      }
    }

    core::table::Table *table() {
      return table_;
    }

    const core::table::Table *table() const {
      return table_;
    }

    double relative_error() const {
      return relative_error_;
    }

    double probability() const {
      return probability_;
    }

    double total_num_tuples() const {
      return total_num_tuples_;
    }

    void Init(
      core::table::Table *table_in,
      double relative_error_in, double probability_in) {

      relative_error_ = relative_error_in;
      probability_ = probability_in;
      table_ = table_in;
      total_num_tuples_ = core::math::BinomialCoefficient<double>(
                            table_in->n_entries() - 1, 2);

      // Initialize the temporary vector for storing the Monte Carlo
      // results.
      mean_variance_pair_.resize(table_->n_entries());
    }
};

class NbodySimulatorSummary {

  private:

    void ReplacePoints_(
      const core::table::Table &table,
      const core::metric_kernels::AbstractMetric &metric_in,
      const std::vector<int> &random_combination,
      core::gnp::TripleDistanceSq *distance_sq_out) const {

      core::table::DenseConstPoint point;
      for(int i = 1; i < 3; i++) {
        table.get(random_combination[i], &point);
        distance_sq_out->ReplaceOnePoint(
          metric_in, point, random_combination[i], i);
      }
    }

    void RandomCombination_(
      const core::gnp::TripleRangeDistanceSq &range_sq_in,
      int node_index_fix,
      std::vector<int> *random_combination_out) const {

      // Detect the number of consecutive nodes that are equal.
      int iterating_node_index = node_index_fix;
      do {
        int reference_node_index = iterating_node_index;
        int count = 0;
        do {
          iterating_node_index = (iterating_node_index + 1) % 3;
          count++;
        }
        while(
          range_sq_in.node(iterating_node_index) ==
          range_sq_in.node(reference_node_index) &&
          iterating_node_index != node_index_fix);
        std::vector<int> subcombination_out;

        if(random_combination_out->size() == 1) {
          count--;
          subcombination_out.push_back((*random_combination_out)[0]);
        }
        core::math::RandomCombination(
          range_sq_in.node(reference_node_index)->begin(),
          range_sq_in.node(reference_node_index)->end(), count,
          &subcombination_out);
        random_combination_out->insert(
          random_combination_out->end(), subcombination_out.begin(),
          subcombination_out.end());
      }
      while(iterating_node_index != node_index_fix);
    }

  public:

    core::math::Range negative_potential_;

    core::math::Range positive_potential_;

    double pruned_;

    double used_error_;

    NbodySimulatorSummary() {
      SetZero();
    }

    NbodySimulatorSummary(const NbodySimulatorSummary &summary_in) {
      negative_potential_ = summary_in.negative_potential_;
      positive_potential_ = summary_in.positive_potential_;
      pruned_ = summary_in.pruned_;
      used_error_ = summary_in.used_error_;
    }

    template < typename GlobalType, typename DeltaType, typename ResultType >
    bool CanProbabilisticSummarize(
      const core::metric_kernels::AbstractMetric &metric,
      GlobalType &global, DeltaType &delta,
      const core::gnp::TripleRangeDistanceSq &range_sq_in,
      const std::vector<double> &failure_probabilities,
      int node_index,
      ResultType *query_results) {

      const int num_samples = 25;

      if(std::min(std::min(
                    range_sq_in.num_tuples(0),
                    range_sq_in.num_tuples(1)),
                  range_sq_in.num_tuples(2)) > 10000.0) {
        return false;
      }

      // Look up the number of standard deviations.
      double num_standard_deviations =
        global.compute_quantile(failure_probabilities[node_index]);

      // The prunable flag.
      bool prunable = true;

      // Get an iterator for the current node.
      typename core::table::Table::TreeType * node =
        range_sq_in.node(node_index);
      typename core::table::Table::TreeIterator node_it =
        global.table()->get_node_iterator(node);

      std::vector<int> random_combination(1, 0);
      core::table::DenseConstPoint query_point;
      int query_point_index;

      // Triple range distance square object to keep track.
      core::gnp::TripleDistanceSq triple_distance_sq;
      do {

        // The current query point.
        node_it.Next(&query_point, &query_point_index);
        triple_distance_sq.ReplaceOnePoint(
          metric, query_point, query_point_index, 0);

        // The first in the list is the query point index.
        random_combination[0] = query_point_index;
        printf("Query: %d\n", query_point_index);

        // Generate a random combination that contains the current
        // query point.
        core::monte_carlo::MeanVariancePair &mean_variance_pair =
          (* delta.mean_variance_pair())[query_point_index];
        while(mean_variance_pair.num_samples() < num_samples) {
          random_combination.resize(1);
          RandomCombination_(range_sq_in, node_index, &random_combination);

          printf("Random combination: %d %d %d\n", random_combination[0],
                 random_combination[1], random_combination[2]);
          ReplacePoints_(
            *(global.table()), metric, random_combination, &triple_distance_sq);

          // Evaluate the potential and add it to the result of each
          // point involved.
          double potential = global.potential().EvalUnnormOnSq(
                               triple_distance_sq);

          mean_variance_pair.push_back(potential);
        }
        exit(0);

        // Check whether the current query point can be pruned.
        core::math::Range delta_contribution;
        mean_variance_pair.scaled_interval(
          range_sq_in.num_tuples(node_index), num_standard_deviations,
          &delta_contribution);
        if(delta_contribution.hi < 0.0) {
          negative_potential_.hi += delta_contribution.hi;
        }
        if(delta_contribution.lo > 0.0) {
          positive_potential_.lo += delta_contribution.lo;
        }

        double left_hand_side =
          0.5 * (delta_contribution.hi - delta_contribution.lo);
        double right_hand_side =
          delta.pruned_[node_index] *
          (global.relative_error() * std::max(
             - negative_potential_.hi, positive_potential_.lo) - used_error_) /
          static_cast<double>(global.total_num_tuples() - pruned_);

        prunable = (left_hand_side <= right_hand_side);
      }
      while(node_it.HasNext() && prunable);

      return prunable;
    }

    template < typename GlobalType, typename DeltaType, typename ResultType >
    bool CanSummarize(
      const GlobalType &global, const DeltaType &delta,
      const core::gnp::TripleRangeDistanceSq &triple_range_distance_sq_in,
      int node_index, ResultType *query_results) const {

      double left_hand_side = delta.used_error_[node_index];
      double right_hand_side =
        delta.pruned_[node_index] *
        (global.relative_error() * std::max(
           - negative_potential_.hi, positive_potential_.lo) - used_error_) /
        static_cast<double>(global.total_num_tuples() - pruned_);

      return left_hand_side <= right_hand_side;
    }

    void SetZero() {
      negative_potential_.Init(0.0, 0.0);
      positive_potential_.Init(0.0, 0.0);
      pruned_ = 0.0;
      used_error_ = 0.0;
    }

    void Init() {
      SetZero();
    }

    void StartReaccumulate() {
      negative_potential_.Init(
        std::numeric_limits<double>::max(),
        - std::numeric_limits<double>::max());
      positive_potential_.Init(
        std::numeric_limits<double>::max(),
        - std::numeric_limits<double>::max());
      pruned_ = std::numeric_limits<double>::max();
      used_error_ = 0;
    }

    template<typename ResultType>
    void Accumulate(const ResultType &results, int q_index) {
      negative_potential_ |= results.negative_potential_[q_index];
      positive_potential_ |= results.positive_potential_[q_index];
      pruned_ = std::min(pruned_, results.pruned_[q_index]);
      used_error_ = std::max(used_error_, results.used_error_[q_index]);
    }

    void Accumulate(const NbodySimulatorSummary &summary_in) {
      negative_potential_ |= summary_in.negative_potential_;
      positive_potential_ |= summary_in.positive_potential_;
      pruned_ = std::min(pruned_, summary_in.pruned_);
      used_error_ = std::max(used_error_, summary_in.used_error_);
    }

    void Accumulate(
      const NbodySimulatorSummary &summary_in,
      const NbodySimulatorPostponed &postponed_in) {

      negative_potential_ |=
        (summary_in.negative_potential_ + postponed_in.negative_potential_);
      positive_potential_ =
        (summary_in.positive_potential_ + postponed_in.positive_potential_);
      pruned_ = std::min(
                  pruned_, summary_in.pruned_ + postponed_in.pruned_);
      used_error_ = std::max(
                      used_error_,
                      summary_in.used_error_ + postponed_in.used_error_);
    }

    void ApplyDelta(const NbodySimulatorDelta &delta_in, int node_index) {
      negative_potential_ += delta_in.negative_potential_[node_index];
      positive_potential_ += delta_in.positive_potential_[node_index];
    }

    void ApplyPostponed(const NbodySimulatorPostponed &postponed_in) {
      negative_potential_ += postponed_in.negative_potential_;
      positive_potential_ += postponed_in.positive_potential_;
      pruned_ = pruned_ + postponed_in.pruned_;
      used_error_ = used_error_ + postponed_in.used_error_;
    }
};

class NbodySimulatorStatistic: public core::tree::AbstractStatistic,
  public boost::noncopyable {

  public:

    physpack::nbody_simulator::NbodySimulatorPostponed postponed_;

    physpack::nbody_simulator::NbodySimulatorSummary summary_;

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

    /**
     * Initializes by combining statistics of two partitions.
     *
     * This lets you build fast bottom-up statistics when building trees.
     */
    template<typename TreeIteratorType>
    void Init(
      TreeIteratorType &iterator,
      const NbodySimulatorStatistic *left_stat,
      const NbodySimulatorStatistic *right_stat) {
      SetZero();
    }
};
};
};

#endif
