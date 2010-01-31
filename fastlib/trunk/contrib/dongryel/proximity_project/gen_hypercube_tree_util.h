#ifndef GEN_HYPERCUBE_TREE_UTIL_H
#define GEN_HYPERCUBE_TREE_UTIL_H

#include "gen_hypercube_tree.h"
#include "cfmm_tree.h"

namespace GenHypercubeTreeUtil {

  unsigned int FindParent(unsigned int index, index_t dimension) {
    return index >> dimension;
  }

  unsigned int FindLowestDescendant(unsigned int index, index_t dimension) {
    return index << dimension;
  }

  template<typename TreeType>
  TreeType *FindNode
  (const ArrayList< ArrayList<TreeType *> > 
   &nodes_in_each_level, unsigned int node_index, index_t level) {

    const ArrayList<TreeType *> &candidate_list = nodes_in_each_level[level];
    TreeType *candidate = NULL;

    for(index_t i = 0; i < candidate_list.size(); i++) {
      if((candidate_list[i])->node_index() == node_index) {
	candidate = candidate_list[i];
	break;
      }
    }
    return candidate;
  }

  template<typename TreeType>
  void NodeListSetDifference
  (index_t dimension,
   const ArrayList<TreeType * > &first_list,
   const ArrayList<TreeType * > &second_list,
   const ArrayList<TreeType * > &third_list,
   const ArrayList<TreeType * > &filter_list,
   ArrayList<TreeType * > *set_difference) {
   
    // Initialize the set difference list.
    set_difference->Init();
    
    for(index_t i = 0; i < filter_list.size(); i++) {
      
      TreeType *node_from_filter_list = filter_list[i];
      bool flag = false;

      for(index_t j = 0; j < first_list.size(); j++) {
	TreeType *node_from_first_list = first_list[j];

	int level_difference = node_from_first_list->level() -
	  node_from_filter_list->level();
	DEBUG_ASSERT(level_difference >= 0);

	bool is_descendant = (((node_from_first_list->node_index() >> 
				(dimension * level_difference)) ^
			       node_from_filter_list->node_index()) == 0);

	// Filter the node if it is a descendant of the filter
	// candidate.
	if(is_descendant) {
	  flag = true;
	  break;
	}
      }
      if(flag) {
	continue;
      }
      
      for(index_t j = 0; j < second_list.size(); j++) {
	TreeType *node_from_second_list = second_list[j];
	int level_difference = node_from_second_list->level() -
	  node_from_filter_list->level();
	DEBUG_ASSERT(level_difference >= 0);
	bool is_descendant = (((node_from_second_list->node_index() >> 
				(dimension * level_difference)) ^
			       node_from_filter_list->node_index()) == 0);

	if(is_descendant) {
	  flag = true;
	  break;
	}
      }
      if(flag) {
	continue;
      }

      for(index_t j = 0; j < third_list.size(); j++) {
	TreeType *node_from_third_list = third_list[j];
	int level_difference = node_from_third_list->level() -
	  node_from_filter_list->level();
	
	bool is_descendant = (((node_from_third_list->node_index() >> 
				(dimension * level_difference)) ^
			       node_from_filter_list->node_index()) == 0);

	if(is_descendant) {
	  flag = true;
	  break;
	}
      }
      if(flag) {
	continue;
      }

      if(!flag) {
	*(set_difference->PushBackRaw()) = node_from_filter_list;
      }
    }
  }

  template<typename Statistics>
  void FindColleagues
  (index_t dimension, 
   proximity::GenHypercubeTree<Statistics>
   *node, const ArrayList< 
   ArrayList<proximity::GenHypercubeTree<Statistics> * > > 
   &nodes_in_each_level,
   ArrayList< proximity::GenHypercubeTree<Statistics> *> 
   *colleagues) {
    
    // Initialize the colleague node list.
    colleagues->Init();
    
    ArrayList< proximity::GenHypercubeTree<Statistics> * > 
      neighbors_of_parent;
    
    // Find the parent node of the given node.
    proximity::GenHypercubeTree<Statistics> *parent_node =
      FindNode(nodes_in_each_level, 
	       FindParent(node->node_index(), dimension), node->level() - 1);
    
    // Find the neighbors of the parent node.
    FindNeighborsInAdaptiveGenHypercubeTree
      (nodes_in_each_level, parent_node->node_index(), parent_node->level(),
       dimension, &neighbors_of_parent);
      
    // Find the children of the neighbors of the parent node.
    for(index_t i = 0; i < neighbors_of_parent.size(); i++) {
      
      proximity::GenHypercubeTree<Statistics> *given_neighbor_of_parent = 
	neighbors_of_parent[i];
      for(index_t j = 0; j < given_neighbor_of_parent->num_children(); j++) {
	proximity::GenHypercubeTree<Statistics> *child_node = 
	  given_neighbor_of_parent->get_child(j);
	
	// I do not feel like doing bit-shuffling business to optimize
	// this. I'll just compute the node distance.
	double min_distance = 
	  sqrt(node->bound().MinDistanceSq(child_node->bound()));
	
	if(min_distance > 0) {
	  *(colleagues->PushBackRaw()) = child_node;
	}
      }
    }
  }
  
  template<typename TreeType>
  void RetrieveAdjacentLeafNode
  (const TreeType *centered_node, 
   TreeType *potentially_fake_leaf_neighbor_node,
   ArrayList<TreeType *> *adjacent_children, 
   ArrayList<TreeType *> *non_adjacent_children) {
    
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
    if(min_distance >= potentially_fake_leaf_neighbor_node->side_length()) {
      *(non_adjacent_children->PushBackRaw()) =
	potentially_fake_leaf_neighbor_node;
    }
    else {
      
      if(potentially_fake_leaf_neighbor_node->is_leaf()) {
	*(adjacent_children->PushBackRaw()) =
	  potentially_fake_leaf_neighbor_node;
      }
      else {
	for(index_t i = 0; i < 
	      potentially_fake_leaf_neighbor_node->num_children(); i++) {
	  
	  TreeType *potential_node =
	    potentially_fake_leaf_neighbor_node->get_child(i);
	  RetrieveAdjacentLeafNode(centered_node, potential_node, 
				   adjacent_children, 
				   non_adjacent_children);
	}
      }
    }    
  }

  template<typename TreeType>
  void FindAdjacentLeafNode
  (index_t dimension,
   const ArrayList< ArrayList<TreeType *> > &nodes_in_each_level,
   TreeType *leaf_node,
   ArrayList<TreeType *> *adjacent_children,
   ArrayList<TreeType *> *non_adjacent_children) {
    
    // Initialize the list to be returned.
    adjacent_children->Init();
    non_adjacent_children->Init();

    ArrayList<unsigned int> neighbor_indices_to_be_filtered;

    // First, find the neighbors of the given leaf node on the same
    // level it is on.
    FindNeighborsInNonAdaptiveGenHypercubeTree
      (leaf_node->node_index(), leaf_node->level(), dimension, 
       &neighbor_indices_to_be_filtered);
    
    // Traverse each fake neighbor and expand to find the real
    // neighboring nodes.
    for(index_t potentially_fake_leaf_neighbor = 0;
	potentially_fake_leaf_neighbor < 
	  neighbor_indices_to_be_filtered.size();
	potentially_fake_leaf_neighbor++) {
      
      TreeType *potentially_fake_leaf_neighbor_node =
	FindNode(nodes_in_each_level, neighbor_indices_to_be_filtered
		 [potentially_fake_leaf_neighbor], leaf_node->level());
	     
      // In this case, we traverse down.
      if(potentially_fake_leaf_neighbor_node != NULL) {
	RetrieveAdjacentLeafNode
	  (leaf_node, potentially_fake_leaf_neighbor_node,
	   adjacent_children, non_adjacent_children);
      }
      // Otherwise, we traverse up to find the parent.
      else {
	
	index_t current_level = leaf_node->level();
	unsigned int potential_node_index = 
	  neighbor_indices_to_be_filtered[potentially_fake_leaf_neighbor];
	TreeType *potential_candidate = NULL;
	
	do {
	  
	  potential_node_index = potential_node_index >> dimension;
	  current_level--;
	  potential_candidate = FindNode
	    (nodes_in_each_level, potential_node_index, current_level);

	} while(potential_candidate == NULL);

	// We are guaranteed here that we have a non-NULL node, but
	// need to check whether it is actually a leaf or not!
	DEBUG_ASSERT(potential_candidate != NULL);

	if(potential_candidate->is_leaf()) {

	  // Check to see whether there is no duplicate...
	  bool duplicate_flag = false;
	  for(index_t duplicate = 0; duplicate < adjacent_children->size();
	      duplicate++) {
	    if(potential_candidate == (*adjacent_children)[duplicate]) {
	      duplicate_flag = true;
	      break;
	    }
	  }

	  if(!duplicate_flag) {
	    *(adjacent_children->PushBackRaw()) = potential_candidate;
	  }
	}
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

  void RecursivelyChooseIndex
  (const GenVector<unsigned int> &lower_limit,
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
      if(exclusion_index[level] != lower_limit[level]) {
	chosen_index[level] = exclusion_index[level];      
	RecursivelyChooseIndex(lower_limit, exclusion_index, upper_limit,
			       chosen_index, level + 1, valid_combination ||
			       (chosen_index[level] != exclusion_index[level]),
			       neighbor_indices);
      }

      // Choose the upper index.
      if(upper_limit[level] != exclusion_index[level]) {
	chosen_index[level] = upper_limit[level];
	RecursivelyChooseIndex(lower_limit, exclusion_index, upper_limit,
			       chosen_index, level + 1, valid_combination ||
			       (chosen_index[level] != exclusion_index[level]),
			       neighbor_indices);
      }
    }
    else {

      // If the chosen index is not equal to the exclusion index, then
      // add the node number to the list.
      if(valid_combination) {
	*(neighbor_indices.PushBackRaw()) = BitInterleaving(chosen_index);
      }
    }
  }

  void FindNeighborsInNonAdaptiveGenHypercubeTree
  (unsigned int index, unsigned int level, index_t dimension, 
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
      if(tmp_vector[d] > 0) {
	lower_limit[d] = tmp_vector[d] - 1;
      }
      else {
	lower_limit[d] = 0;
      }
      if(tmp_vector[d] < (unsigned int) ((1 << level) - 1)) {
	upper_limit[d] = tmp_vector[d] + 1;
      }
      else {
	upper_limit[d] = (1 << level) - 1;
      }
    }

    GenVector<unsigned int> chosen_index;
    chosen_index.Init(dimension);
    RecursivelyChooseIndex(lower_limit, tmp_vector, upper_limit, chosen_index,
			   0, false, *neighbor_indices);
  }

  template<typename TreeType>
  void FindNeighborsInAdaptiveGenHypercubeTree
  (const ArrayList< ArrayList<TreeType *> > &nodes_in_each_level,
   unsigned int index, index_t level, index_t dimension, 
   ArrayList<TreeType * > *neighbor_nodes) {
    
    // Initialize the node list.
    neighbor_nodes->Init();

    ArrayList<unsigned int> unfiltered_neighbor_indices;
    FindNeighborsInNonAdaptiveGenHypercubeTree(index, level, dimension,
					       &unfiltered_neighbor_indices);

    // Now filter the list based on its existence.
    for(index_t i = 0; i < unfiltered_neighbor_indices.size(); i++) {
      TreeType *node =
	FindNode(nodes_in_each_level, unfiltered_neighbor_indices[i], level);
      
      if(node != NULL) {
	*(neighbor_nodes->PushBackRaw()) = node;
      }
    }
  }

  template<typename TreeType>
  void FindFourthList
  (const ArrayList< ArrayList<TreeType *> > &nodes_in_each_level, 
   unsigned int index, index_t level, index_t dimension,
   const ArrayList<TreeType * > &first_list,
   const ArrayList<TreeType * > &second_list,
   const ArrayList<TreeType * > &third_list,
   ArrayList<TreeType * > *fourth_list) {

    // First, find the parent of the given node.
    ArrayList<TreeType * > neighbors_of_parent;

    // Then, find the neighbors of the parent of the given node.
    FindNeighborsInAdaptiveGenHypercubeTree
      (nodes_in_each_level, index >> dimension, level - 1, dimension,
       &neighbors_of_parent);

    // Filter the list based on the first three lists.
    NodeListSetDifference(dimension, first_list, second_list, third_list, 
			  neighbors_of_parent, fourth_list);
  }

};

#endif
