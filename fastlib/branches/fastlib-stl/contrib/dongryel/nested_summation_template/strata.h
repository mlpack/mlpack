#ifndef STRATA_H
#define STRATA_H

#include "fastlib/fastlib.h"
#include <queue>
#include <vector>

class Strata {

 public:

  index_t total_num_stratum;

  index_t total_num_terms;

  ArrayList<DRange> node_list;

  Matrix statistics_for_each_stratum;

  index_t total_num_samples_so_far;

  ArrayList<index_t> output_allocation_for_each_stratum;

  template<typename TreeType>
  class CompareTreeType {
   public:
    bool operator()(TreeType *a, TreeType *b) {
      return a->stat().priority < b->stat().priority;
    }
  };

  /** @brief Takes a root of the tree, and expands up to a given
   *         number of nodes using the number of data points times sum
   *         of per-dimension variance heuristic.
   */
  template<typename TreeType>
  void Init(TreeType *root_in, index_t num_stratum_desired) {

    // Set the total number of terms.
    node_list.Init();
    total_num_terms = root_in->count();

    // The priority queue used for the expansion.
    std::priority_queue<TreeType *, std::vector<TreeType *>,
      CompareTreeType<TreeType> > frontier;
    frontier.push(root_in);
    total_num_stratum = 0;

    // First expand the tree and form the frontier which will become
    // the list of strata.
    do {
      TreeType *popped_node = frontier.top();
      frontier.pop();
      if(popped_node->is_leaf()) {
	node_list.PushBackRaw();
	node_list[node_list.size() - 1].lo = popped_node->begin();
	node_list[node_list.size() - 1].hi = popped_node->end();
	total_num_stratum++;
      }
      else {
	frontier.push(popped_node->left());
	frontier.push(popped_node->right());
      }
      
      // Repeat until the priority queue is not empty and we have
      // still more nodes to expand...
    } while(frontier.size() > 0 &&
	    total_num_stratum < num_stratum_desired &&
	    frontier.size() < num_stratum_desired - total_num_stratum);

    // Fill up the rest of the remaining strata by popping the
    // priority queue.
    index_t remaining_count = num_stratum_desired - total_num_stratum;
    for(index_t c = 0; c < remaining_count; c++) {
      TreeType *popped_node = frontier.top();
      frontier.pop();
      node_list.PushBackRaw();
      node_list[node_list.size() - 1].lo = popped_node->begin();
      node_list[node_list.size() - 1].hi = popped_node->end();
      total_num_stratum++;
    }
    while(!frontier.empty()) {
      frontier.pop();
    }			     

    // The 0th row stores the sum of the samples, the 1st row stores
    // the squared sum for each stratum.
    statistics_for_each_stratum.Init(2, total_num_stratum);
    output_allocation_for_each_stratum.Init(total_num_stratum);
    Reset();
  }

  /** @brief Takes a root of the tree, and expands to such that the
   *         number of tuples in the monochromatic sense in the strata
   *         is at most fraction of the number of tuples formed among
   *         the root node. The expansion is using the number of data
   *         points times sum of per-dimension variance heuristic.
   */
  template<typename TreeType>
  void Init(TreeType *root_in, index_t num_times_replicated,
	    double fraction_desired) {

    // Set the total number of terms.
    node_list.Init();
    total_num_terms = root_in->count();

    // The priority queue used for the expansion.
    std::priority_queue<TreeType *, std::vector<TreeType *>,
      CompareTreeType<TreeType> > frontier;
    frontier.push(root_in);
    total_num_stratum = 0;

    // Compute the n-tuple formed among the points at the root node.
    double total_n_tuples_on_root = 
      math::BinomialCoefficient(root_in->count() - 1, num_times_replicated);
    double total_n_tuples_so_far = total_n_tuples_on_root;

    // First expand the tree and form the frontier which will become
    // the list of strata. This loop might have cancellation issues,
    // since I subtract and add the corrections, but I do this since
    // STL priority queue does not provide iterators...
    do {
      TreeType *popped_node = frontier.top();
      frontier.pop();
      if(popped_node->is_leaf()) {
	popped_node->stat().in_strata = true;
	node_list.PushBackRaw();
	node_list[node_list.size() - 1].lo = popped_node->begin();
	node_list[node_list.size() - 1].hi = popped_node->end();
      }
      else {
	total_n_tuples_so_far -= 
	  math::BinomialCoefficient(popped_node->count() - 1,
				    num_times_replicated);
	total_n_tuples_so_far +=
	  math::BinomialCoefficient(popped_node->left()->count() - 1,
				    num_times_replicated);
	total_n_tuples_so_far +=
	  math::BinomialCoefficient(popped_node->right()->count() - 1,
				    num_times_replicated);
	frontier.push(popped_node->left());
	frontier.push(popped_node->right());
      }

      // Repeat until the priority queue is not empty and we have
      // still more nodes to expand...
    } while(frontier.size() > 0 && total_n_tuples_so_far >
	    fraction_desired * total_n_tuples_on_root);

    // Empty the priority queue.
    while(!frontier.empty()) {
      TreeType *popped_node = frontier.top();
      popped_node->stat().in_strata = true;
      frontier.pop();
      node_list.PushBackRaw();
      node_list[node_list.size() - 1].lo = popped_node->begin();
      node_list[node_list.size() - 1].hi = popped_node->end();
    }    
    total_num_stratum = node_list.size();

    // The 0th row stores the sum of the samples, the 1st row stores
    // the squared sum for each stratum.
    statistics_for_each_stratum.Init(2, total_num_stratum);
    output_allocation_for_each_stratum.Init(total_num_stratum);
    Reset();
  }

  /** @brief Clear all the samples and start over.
   */
  void Reset() {
    for(index_t i = 0; i < total_num_stratum; i++) {
      //num_samples_for_each_stratum = 0;
      output_allocation_for_each_stratum[i] = 0;
    }
    total_num_samples_so_far = 0;
    statistics_for_each_stratum.SetZero();
    //total_samples_to_allocate = 0;    
  }

};

#endif
