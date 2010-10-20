#ifndef FL_LITE_MLPACK_KCDE_STRATA_H
#define FL_LITE_MLPACK_KCDE_STRATA_H

#include "multitree_monte_carlo.h"
#include "stratum.h"

namespace fl {
namespace ml {

#include <utility>

template<typename TableType>
class Strata {

  public:
    typedef typename TableType::Tree_t Tree_t;

  private:

    std::vector< fl::ml::Stratum<TableType> > strata_;

    std::vector< std::pair<double, double> > global_mean_variance_pairs_;

    int num_samples_;

  private:
    void GatherStatistics_() {

      // Clear the global mean variance pairs.
      for(int i = 0; i < global_mean_variance_pairs_.size(); i++) {
        global_mean_variance_pairs_[i].first = 0;
        global_mean_variance_pairs_[i].second = 0;
      }

      // Loop through each stratum.
      for(int i = 0; i < strata_.size(); i++) {

        const std::vector<fl::ml::MeanVariancePair>
        &mean_variance_pairs = strata_[i].mean_variance_pairs();

        // The number of fraction of terms owned by the strata.
        double fraction = strata_[i].compute_fraction_num_terms();

        for(int j = 0; j < global_mean_variance_pairs_.size(); j++) {
          global_mean_variance_pairs_[j].first +=
            fraction * mean_variance_pairs[j].sample_mean();
          global_mean_variance_pairs_[j].second +=
            fl::math::Sqr(fraction) *
            mean_variance_pairs[j].sample_mean_variance();
        }
      }
    }

  public:

    void SetZero() {

      // Reset the global statistics.
      for(int i = 0; i < global_mean_variance_pairs_.size(); i++) {
        global_mean_variance_pairs_[i].first = 0;
        global_mean_variance_pairs_[i].second = 0;
      }

      // Reset the per-stratum statistics.
      for(int i = 0; i < strata_.size(); i++) {
        strata_[i].SetZero();
      }
    }

    const std::vector< std::pair<double, double> >
    &global_mean_variance_pairs() const {

      return global_mean_variance_pairs_;
    }

    std::vector< std::pair<double, double> > &global_mean_variance_pairs() {
      return global_mean_variance_pairs_;
    }

    const fl::ml::Stratum<TableType> &stratum(int index) const {
      return strata_[index];
    }

    fl::ml::Stratum<TableType> &stratum(int index) {
      return strata_[index];
    }

    int size() const {
      return strata_.size();
    }

    template<typename FunctionType>
    void AccumulateSamples(
      const FunctionType &function_in,
      std::vector< std::pair<double, double> > *summary_mean_variance_pair) {

      // Sample each stratum.
      for(int i = 0; i < strata_.size(); i++) {
        strata_[i].AccumulateSamples(function_in);
      }

      // Combine to produce the global mean variance pairs.
      GatherStatistics_();

      // Produce the summary result. This is function-specific.
      function_in.Summarize(global_mean_variance_pairs_,
                            summary_mean_variance_pair);
    }

    template<typename FunctionType>
    void Init(
      const FunctionType &function_in,
      fl::ml::MultitreeMonteCarlo<TableType> &monte_carlo_engine_in,
      int num_samples_in) {

      strata_.resize(1);
      strata_[0].Init(function_in, monte_carlo_engine_in, num_samples_in);

      for(int i = 0;
          i < monte_carlo_engine_in.tables_for_variable_arguments().size();
          i++) {
        Tree_t *root_node =
          monte_carlo_engine_in.tables_for_variable_arguments()[i].first->
          get_tree();
        strata_[0].add_node(root_node);
      }

      num_samples_ = num_samples_in;

      // Reset global mean variance pairs.
      function_in.Allocate(&global_mean_variance_pairs_);
      SetZero();
    }

    template<typename FunctionType>
    bool Stratify(const FunctionType &function_in) {

      // Choose the worst stratum and split it.
      double worst_strata_score = -1.0;
      int worst_strata_index = -1;
      for(int i = 0; i < strata_.size(); i++) {
        double average_sample_mean_variance =
          strata_[i].average_sample_mean_variance();

        if(average_sample_mean_variance > worst_strata_score) {
          worst_strata_score = average_sample_mean_variance;
          worst_strata_index = i;
        }
      }

      // Push in a blank statum and initialize temporarily.
      strata_.resize(strata_.size() + 1);
      strata_[strata_.size() - 1].Init(function_in,
                                       *(strata_[0].monte_carlo_engine()), 25);
      bool split_success =
        strata_[worst_strata_index].Split(
          function_in, &(strata_[strata_.size() - 1]));

      // If the splitting was failed, then discard.
      if(split_success == false) {
        strata_.resize(strata_.size() - 1);
      }
      else {

        // If success, reset the samples.
        strata_[worst_strata_index].SetZero();
        strata_[strata_.size() - 1].SetZero();
      }
      return split_success;
    }

    void OptimalAllocation() {

      double sum_fraction = 0;

      for(int i = 0; i < strata_.size(); i++) {
        double average_sample_mean_variance =
          strata_[i].average_sample_mean_variance();
        double fraction = strata_[i].compute_fraction_num_terms() *
                          sqrt(average_sample_mean_variance);
        sum_fraction += fraction;
      }
      for(int i = 0; i < strata_.size(); i++) {
        double average_sample_mean_variance =
          strata_[i].average_sample_mean_variance();
        double fraction = strata_[i].compute_fraction_num_terms() *
                          sqrt(average_sample_mean_variance);
        int new_num_samples =
          (int) ceil(fraction / sum_fraction * ((double) num_samples_));

        if(strata_[i].mean_variance_pairs()[0].num_samples() == 0) {
          new_num_samples = 25;
        }
        strata_[i].set_num_samples(new_num_samples);
      }
    }
};
};
};

#endif
