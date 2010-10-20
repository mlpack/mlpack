#ifndef FL_LITE_MLPACK_KCDE_STRATUM_H
#define FL_LITE_MLPACK_KCDE_STRATUM_H

#include <vector>
#include "multitree_monte_carlo.h"
#include "mlpack/kde/mean_variance_pair.h"

namespace fl {
namespace ml {

template<typename TableType>
class Stratum {

  public:
    typedef typename TableType::Tree_t Tree_t;

  private:

    int num_samples_;

    std::vector<Tree_t *> nodes_;

    fl::ml::MultitreeMonteCarlo<TableType> *monte_carlo_engine_;

    std::vector< std::pair<TableType *, int> > *constant_arguments_;

    std::vector< std::pair<TableType *, std::vector<int> > >
    *tables_for_variable_arguments_;

    std::vector< fl::ml::MeanVariancePair > mean_variance_pairs_;

  private:

    void ChooseVariableArguments_(
      std::vector<int> *chosen_variable_arguments) {

      for(int i = 0; i < tables_for_variable_arguments_->size(); i++) {
        TableType *table = (*tables_for_variable_arguments_)[i].first;
        Tree_t *node = nodes_[i];
        const std::vector<int> &violation_indices =
          (*tables_for_variable_arguments_)[i].second;
        bool valid_index = true;
        int random_index;
        do {
          valid_index = true;
          random_index = fl::math::Random(table->get_node_begin(node),
                                          table->get_node_end(node));

          // Examine the LOO candidate lists, and see if it is one of
          // them, in which case we repeat.
          for(int j = 0; valid_index == true &&
              j < violation_indices.size(); j++) {
            valid_index = (random_index != violation_indices[j]);
          }
        }
        while(valid_index == false);
        (*chosen_variable_arguments)[i] = random_index;
      }
    }

  public:

    fl::ml::MultitreeMonteCarlo<TableType> *monte_carlo_engine() {
      return monte_carlo_engine_;
    }

    int num_samples() const {
      return num_samples_;
    }

    double average_sample_mean_variance() const {
      double average_sample_mean_variance = 0;
      for(int i = 0; i < mean_variance_pairs_.size(); i++) {
        average_sample_mean_variance +=
          mean_variance_pairs_[i].sample_mean_variance();
      }
      average_sample_mean_variance /= ((double) mean_variance_pairs_.size());
      return average_sample_mean_variance;
    }

    const std::vector< fl::ml::MeanVariancePair > &mean_variance_pairs() const {
      return mean_variance_pairs_;
    }

    std::vector< fl::ml::MeanVariancePair > &mean_variance_pairs() {
      return mean_variance_pairs_;
    }

    void set_num_samples(int num_samples_in) {
      num_samples_ = num_samples_in;
    }

    void add_node(Tree_t *node_in) {
      nodes_.push_back(node_in);
    }

    const std::vector<Tree_t *> &nodes() const {
      return nodes_;
    }

    int get_node_begin(int node_index) const {
      return (*tables_for_variable_arguments_)[node_index].first->get_node_begin(
               nodes_[node_index]);
    }

    int get_node_end(int node_index) const {
      return (*tables_for_variable_arguments_)[node_index].first->get_node_end(
               nodes_[node_index]);
    }

    void SetZero() {
      for(int i = 0; i < mean_variance_pairs_.size(); i++) {
        mean_variance_pairs_[i].SetZero();
      }
    }

    double compute_fraction_num_terms() {

      double fraction = 1;

      // Loop through each variable argument while checking its exclusion
      // against the constant argument indices.
      for(int i = 0; i < tables_for_variable_arguments_->size(); i++) {

        // Compute the num valid arguments for the strata.
        TableType *table = ((*tables_for_variable_arguments_)[i]).first;
        Tree_t *node = nodes_[i];
        int num_valid_values = table->get_node_count(node);
        for(int j = 0;
            j < ((*tables_for_variable_arguments_)[i]).second.size(); j++) {
          int exclude_index = (*constant_arguments_)[j].second;
          if(table->get_node_begin(node) <= exclude_index &&
              exclude_index < table->get_node_end(node)) {
            num_valid_values--;
          }
        }

        fraction *=
          (((double) num_valid_values) /
           ((double)((*tables_for_variable_arguments_)[i]).first->n_entries() -
            ((*tables_for_variable_arguments_)[i]).second.size()));
      }
      return fraction;
    }

    template<typename FunctionType>
    void Init(
      const FunctionType &function_in,
      fl::ml::MultitreeMonteCarlo<TableType> &monte_carlo_engine_in,
      int num_samples_in) {

      num_samples_ = num_samples_in;

      // Allocate the mean variance pair list. This is
      // function-specific.
      function_in.Allocate(&mean_variance_pairs_);
      monte_carlo_engine_ = &monte_carlo_engine_in;
      constant_arguments_ = &(monte_carlo_engine_->constant_arguments());
      tables_for_variable_arguments_ =
        &(monte_carlo_engine_->tables_for_variable_arguments());

      // Clear the mean variance pairs.
      SetZero();
    }

    void AccumulateSample(const std::vector<double> &set_of_results) {
      for(int i = 0; i < set_of_results.size(); i++) {
        mean_variance_pairs_[i].push_back(set_of_results[i]);
      }
    }

    template<typename FunctionType>
    void AccumulateSamples(const FunctionType &function_in) {

      std::vector<int> chosen_variable_arguments(
        tables_for_variable_arguments_->size(), 0);

      for(int i = 0; i < num_samples_; i++) {

        // Generate a random tuple for the variable arguments and call the
        // function using the constant arguments, plus the randomly chosen
        // variable arguments, plus the subproblems.
        std::vector<double> set_of_results;
        ChooseVariableArguments_(&chosen_variable_arguments);
        function_in.Compute(chosen_variable_arguments,
                            *monte_carlo_engine_, &set_of_results);

        // Accumulate the result.
        AccumulateSample(set_of_results);
      }
    }

    template<typename FunctionType>
    bool Split(const FunctionType &function, Stratum *child2) {

      // Among the nodes owned by this stratum, split the one that
      // is the largest.
      int split_node = -1;
      double max_max_distance_within_bound = -1.0;

      for(int i = 0; i < nodes_.size(); i++) {
        TableType *table = ((*tables_for_variable_arguments_)[i]).first;
        double max_distance_within_bound =
          table->get_node_bound(nodes_[i]).MaxDistanceWithinBound();
        if(max_distance_within_bound > max_max_distance_within_bound) {
          split_node = i;
          max_max_distance_within_bound = max_distance_within_bound;
        }
      }

      TableType *split_table =
        (*tables_for_variable_arguments_)[split_node].first;

      if(split_table->node_is_leaf(nodes_[split_node])) {
        typename TableType::template IndexArgs<typename FunctionType::MetricType> index_args;

        index_args.leaf_size =
          split_table->get_node_count(nodes_[split_node]) - 1;
        split_table->SplitNode(index_args, nodes_[split_node]);
      }

      // Create the children and distribute the samples equally.
      if(split_table->node_is_leaf(nodes_[split_node]) == false) {

        int old_num_samples = num_samples_;
        set_num_samples(num_samples_ / 2);
        child2->Init(function, *monte_carlo_engine_,
                     old_num_samples - num_samples_);

        for(int i = 0; i < nodes_.size(); i++) {
          if(i != split_node) {
            child2->add_node(nodes_[i]);
          }
          else {
            Tree_t *split_node_ptr = nodes_[i];
            nodes_[i] = split_table->get_node_left_child(split_node_ptr);
            child2->add_node(split_table->get_node_right_child(split_node_ptr));
          }
        }
        return true;
      }
      else {
        return false;
      }
    }
};
};
};

#endif
