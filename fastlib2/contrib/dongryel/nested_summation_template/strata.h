#ifndef STRATA_H
#define STRATA_H

#include "fastlib/fastlib.h"
#include <queue>
#include <vector>

template<typename TreeType>
class Strata {

 public:

  index_t total_num_stratum;

  index_t total_num_terms;

  ArrayList<TreeType *> node_list;

  Matrix statistics_for_each_stratum;

  index_t total_num_samples_so_far;

  ArrayList<index_t> output_allocation_for_each_stratum;

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
  void Init(TreeType *root_in, index_t num_stratum_desired) {

    // Set the total number of terms.
    node_list.Init();
    total_num_terms = root_in->count();

    // The priority queue used for the expansion.
    std::priority_queue<TreeType *, std::vector<TreeType *>,
      CompareTreeType> frontier;
    frontier.push(root_in);
    total_num_stratum = 0;

    // First expand the tree and form the frontier which will become
    // the list of strata.
    do {
      TreeType *popped_node = frontier.pop();
      if(popped_node->is_leaf()) {
	node_list.PushBackRaw();
	node_list[node_list.size() - 1] = popped_node;
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
      node_list.PushBackRaw();
      node_list[node_list.size() - 1] = frontier.pop();
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

  /** @brief Clear all the samples and start over.
   */
  void Reset() {
    for(index_t i = 0; i < totla_num_stratum; i++) {
      num_samples_for_each_stratum = 0;
      output_allocation_for_each_stratum = 0;
    }
    total_num_samples_so_far = 0;
    statistics_for_each_stratum.SetZero();
    total_samples_to_allocate = 0;    
  }

};

#endif
