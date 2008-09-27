#ifndef GEN_HYPERCUBE_TREE_UTIL_H
#define GEN_HYPERCUBE_TREE_UTIL_H

#include "gen_hypercube_tree.h"

namespace GenHypercubeTreeUtil {

  unsigned int FindParent(unsigned int index, index_t dimension) {
    return index >> dimension;
  }

  unsigned int FindLowestDescendant(unsigned int index, index_t dimension) {
    return index << dimension;
  }

  template<typename TStatistic>
  proximity::GenHypercubeTree<TStatistic> *FindNode
  (const ArrayList< ArrayList<proximity::GenHypercubeTree<TStatistic> *> > 
   &nodes_in_each_level, unsigned int node_index, index_t level) {

    const ArrayList<proximity::GenHypercubeTree<TStatistic> *> 
      &candidate_list = nodes_in_each_level[level];
    proximity::GenHypercubeTree<TStatistic> *candidate = NULL;

    for(index_t i = 0; i < candidate_list.size(); i++) {
      if((candidate_list[i])->node_index() == node_index) {
	candidate = candidate_list[i];
	break;
      }
    }
    return candidate;
  }

  template<typename TStatistic>
  void NodeListSetDifference
  (const ArrayList<proximity::GenHypercubeTree<TStatistic> * > &first_list,
   const ArrayList<proximity::GenHypercubeTree<TStatistic> * > &second_list,
   ArrayList<proximity::GenHypercubeTree<TStatistic> * > *set_difference) {
   
    // Initialize the set difference list.
    set_difference->Init();
    
    for(index_t i = 0; i < first_list.size(); i++) {
      
      proximity::GenHypercubeTree<TStatistic> *node_from_first_list = 
	first_list[i];
      bool flag = false;
      
      for(index_t j = 0; j < second_list.size(); j++) {
	proximity::GenHypercubeTree<TStatistic> *node_from_second_list = 
	  second_list[j];

	if(node_from_first_list == node_from_second_list) {
	  flag = true;
	  break;
	}
      }
      if(!flag) {
	set_difference->PushBackCopy(node_from_first_list);
      }
    }
  }

  template<typename TStatistic>
  void FindColleagues(index_t dimension, 
		      proximity::GenHypercubeTree<TStatistic> *node,
		      const 
		      ArrayList
		      < ArrayList<proximity::GenHypercubeTree<TStatistic> * > >
		      &nodes_in_each_level,
		      ArrayList<proximity::GenHypercubeTree<TStatistic> *> 
		      *colleagues) {
    
    // Initialize the colleague node list.
    colleagues->Init();

    ArrayList<proximity::GenHypercubeTree<TStatistic> * > neighbors_of_parent;

    // Find the parent node of the given node.
    proximity::GenHypercubeTree<TStatistic> *parent_node =
      FindNode(nodes_in_each_level, FindParent(node->node_index(), dimension), 
	       node->level() - 1);
    
    // Find the neighbors of the parent node.
    FindNeighborsInAdaptiveGenHypercubeTree
      (nodes_in_each_level, parent_node->node_index(), parent_node->level(),
       dimension, &neighbors_of_parent);
    
    // Find the children of the neighbors of the parent node.
    for(index_t i = 0; i < neighbors_of_parent.size(); i++) {
      
      proximity::GenHypercubeTree<TStatistic> *given_neighbor_of_parent =
	neighbors_of_parent[i];
      for(index_t j = 0; j < given_neighbor_of_parent->num_children(); j++) {
	proximity::GenHypercubeTree<TStatistic> *child_node = 
	  given_neighbor_of_parent->get_child(j);

	// I do not feel like doing bit-shuffling business to optimize
	// this. I'll just compute the node distance.
	double min_distance = 
	  sqrt(node->bound().MinDistanceSq(child_node->bound()));
	
	if(min_distance > 0) {
	  colleagues->PushBackCopy(child_node);
	}
      }
    }
  }

  template<typename TStatistic>
  void RetrieveAdjacentLeafNode
  (const proximity::GenHypercubeTree<TStatistic> *centered_node,
   proximity::GenHypercubeTree<TStatistic> 
   *potentially_fake_leaf_neighbor_node,
   ArrayList<proximity::GenHypercubeTree<TStatistic> > *adjacent_children, 
   ArrayList<proximity::GenHypercubeTree<TStatistic> > 
   *non_adjacent_children) {

    if(potentially_fake_leaf_neighbor_node->is_leaf()) {

      // Compute the minimum distance between the centered node and
      // the current node being considered. I imagine this could be
      // optimized using bit shuffling business, but I am currently
      // not inclined to implementing this.
      double min_distance = 
	sqrt(centered_node->bound().MinDistanceSq
	     (potentially_fake_leaf_neighbor_node->bound()));
      
      // If the minimum distance at least the side length of the
      // reference node being considered, then add to the non-adjacent
      // list. Otherwise, it is adjacent.
      if(min_distance < potentially_fake_leaf_neighbor_node->side_length()) {
	adjacent_children->PushBackCopy(potentially_fake_leaf_neighbor_node);
      }
      else {
	non_adjacent_children->PushBackCopy
	  (potentially_fake_leaf_neighbor_node);
      }
    }
    else {
      for(index_t i = 0; i < 
	    potentially_fake_leaf_neighbor_node->num_children(); i++) {
	
	proximity::GenHypercubeTree<TStatistic> *potential_node =
	  potentially_fake_leaf_neighbor_node->get_child(i);
	RetrieveAdjacentLeafNode(centered_node, potential_node, 
				 adjacent_children, non_adjacent_children);
      }
    }
  }

  template<typename TStatistic>
  void FindAdjacentLeafNode
  (index_t dimension,
   ArrayList< ArrayList<proximity::GenHypercubeTree<TStatistic> *> > 
   *nodes_in_each_level,
   proximity::GenHypercubeTree<TStatistic> *leaf_node,
   ArrayList<proximity::GenHypercubeTree<TStatistic> > *adjacent_children,
   ArrayList<proximity::GenHypercubeTree<TStatistic> > *non_adjacent_children) {
    
    // Initialize the list to be returned.
    adjacent_children->Init();
    non_adjacent_children->Init();

    ArrayList<index_t> neighbor_indices_to_be_filtered;
    neighbor_indices_to_be_filtered.Init();

    // First, find the neighbors of the given leaf node on the same
    // level it is on.
    FindNeighborsInNonAdaptiveGenHypercubeTree
      (leaf_node->node_index(), leaf_node->level(), dimension, 
       neighbor_indices_to_be_filtered);

    // Traverse each fake neighbor and expand to find the real
    // neighboring nodes.
    for(index_t potentially_fake_leaf_neighbor = 0;
	potentially_fake_leaf_neighbor < 
	  neighbor_indices_to_be_filtered.size();
	potentially_fake_leaf_neighbor++) {
      
      proximity::GenHypercubeTree<TStatistic> 
	*potentially_fake_leaf_neighbor_node =
	FindNode(nodes_in_each_level, neighbor_indices_to_be_filtered
		 [potentially_fake_leaf_neighbor], leaf_node->level());
      
      // In this case, we traverse down.
      if(potentially_fake_leaf_neighbor_node != NULL) {
	RetrieveAdjacentLeafNode(leaf_node,
				 potentially_fake_leaf_neighbor_node,
				 adjacent_children, non_adjacent_children);
      }
      // Otherwise, we traverse up to find the parent.
      else {
	
	index_t current_level = leaf_node->level();
	unsigned int potential_node_index = 
	  neighbor_indices_to_be_filtered[potentially_fake_leaf_neighbor];
	proximity::GenHypercubeTree<TStatistic> *potential_candidate = NULL;

	do {
	  
	  potential_node_index = potential_node_index >> dimension;
	  current_level--;
	  potential_candidate = FindNode
	    (nodes_in_each_level, potential_node_index, current_level);

	} while(current_level >= 0 && potential_candidate == NULL);

	// We are guaranteed here that we have a candidate to add.
	DEBUG_ASSERT(potential_candidate != NULL);
	adjacent_children->PushBackCopy(potential_candidate);
      }
    }

  }

  int BitInterleaving(const GenVector<unsigned int> &indices) {

    int result = 0;
    unsigned int offset = 0;
    GenVector<unsigned int> indices_copy;
    indices_copy.Copy(indices);

    do {
      unsigned int sum = 0;
      for(index_t d = 0; d < indices_copy.length(); d++) {
	sum += indices_copy[d];
      }
      if(sum == 0) {
	break;
      }

      for(index_t d = 0; d < indices_copy.length(); d++) {
	result += (indices_copy[d] % 2) << 
	  (indices_copy.length() - d - 1 + offset);
	indices_copy[d] = indices_copy[d] >> 1;
      }
      offset += indices_copy.length();

    } while(true);

    return result;
  }

  void BitDeinterleaving(unsigned int index, unsigned int level,
			 GenVector<unsigned int> &indices) {
    
    for(index_t d = 0; d < indices.length(); d++) {
      indices[d] = 0;
    }
    unsigned int loop = 0;
    while(index > 0 || level > 0) {
      for(index_t d = indices.length() - 1; d >= 0; d--) {
	indices[d] = (1 << loop) * (index % 2) + indices[d];
	index = index >> 1;
      }      
      level--;
      loop++;
    }
  }

  void RecursivelyChooseIndex(const GenVector<unsigned int> &lower_limit,
			      const GenVector<unsigned int> &exclusion_index,
			      const GenVector<unsigned int> &upper_limit,
			      GenVector<unsigned int> &chosen_index, int level,
			      bool valid_combination,
			      ArrayList<unsigned int> &neighbor_indices) {

    if(level < lower_limit.length()) {

      // Choose the lower index.
      chosen_index[level] = lower_limit[level];
      RecursivelyChooseIndex(lower_limit, exclusion_index, upper_limit,
			     chosen_index, level + 1, valid_combination ||
			     (chosen_index[level] != exclusion_index[level]),
			     neighbor_indices);

      // Choose the exclusion index.
      chosen_index[level] = exclusion_index[level];
      RecursivelyChooseIndex(lower_limit, exclusion_index, upper_limit,
			     chosen_index, level + 1, valid_combination ||
			     (chosen_index[level] != exclusion_index[level]),
			     neighbor_indices);

      // Choose the upper index.
      chosen_index[level] = upper_limit[level];
      RecursivelyChooseIndex(lower_limit, exclusion_index, upper_limit,
			     chosen_index, level + 1, valid_combination ||
			     (chosen_index[level] != exclusion_index[level]),
			     neighbor_indices);
    }
    else {

      // If the chosen index is not equal to the exclusion index, then
      // add the node number to the list.
      if(valid_combination) {
	neighbor_indices.PushBackCopy(BitInterleaving(chosen_index));
      }
    }
  }

  void FindNeighborsInNonAdaptiveGenHypercubeTree
  (unsigned int index, index_t level, index_t dimension, 
   ArrayList<unsigned int> *neighbor_indices) {

    // Initialize the neighbor indices.
    neighbor_indices->Init();

    // First, de-interleave the box index.
    GenVector<unsigned int> tmp_vector, lower_limit, upper_limit;
    tmp_vector.Init(dimension);
    lower_limit.Init(dimension);
    upper_limit.Init(dimension);
    BitDeinterleaving(index, level, tmp_vector);
    
    for(index_t d = 0; d < dimension; d++) {
      lower_limit[d] = std::max(tmp_vector[d] - 1, (unsigned int) 0);
      upper_limit[d] = std::min(tmp_vector[d] + 1, 
				(unsigned int) ((1 << level) - 1));
    }

    GenVector<unsigned int> chosen_index;
    chosen_index.Init(dimension);
    RecursivelyChooseIndex(lower_limit, tmp_vector, upper_limit, chosen_index,
			   0, false, *neighbor_indices);
  }

  template<typename TStatistic>
  void FindNeighborsInAdaptiveGenHypercubeTree
  (const ArrayList< ArrayList<proximity::GenHypercubeTree<TStatistic> *> > 
   &nodes_in_each_level,
   unsigned int index, index_t level, index_t dimension, 
   ArrayList<proximity::GenHypercubeTree<TStatistic> * > *neighbor_nodes) {
    
    // Initialize the node list.
    neighbor_nodes->Init();

    ArrayList<unsigned int> unfiltered_neighbor_indices;
    FindNeighborsInNonAdaptiveGenHypercubeTree(index, level, dimension,
					       &unfiltered_neighbor_indices);

    // Now filter the list based on its existence.
    for(index_t i = 0; i < unfiltered_neighbor_indices.size(); i++) {
      proximity::GenHypercubeTree<TStatistic> *node =
	FindNode(nodes_in_each_level, unfiltered_neighbor_indices[i], level);
      
      if(node != NULL) {
	neighbor_nodes->PushBackCopy(node);
      }
    }
  }
};

#endif

