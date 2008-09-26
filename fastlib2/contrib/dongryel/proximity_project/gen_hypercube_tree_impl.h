#include "fastlib/fastlib.h"

namespace tree_gen_hypercube_tree_private {

  template<typename TStatistic>
  void FindAdjacentChildren(GenHypercubeTree<TStatistic> *leaf_node,
			    ArrayList<unsigned int> &adjacent_children) {
    
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

  unsigned int FindParent(unsigned int index, index_t dimension) {
    return index >> dimension;
  }

  unsigned int FindLowestDescendant(unsigned int index, index_t dimension) {
    return index << dimension;
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
   ArrayList<unsigned int> &neighbor_indices) {

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
			   0, false, neighbor_indices);
  }

  index_t MatrixPartition(index_t particle_set_number, 
			  ArrayList<Matrix *> &matrices, index_t dim, 
			  double splitvalue, index_t first, index_t count, 
			  ArrayList< ArrayList<index_t> > *old_from_new) {
    
    index_t left = first;
    index_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (likely(left <= right) &&
	     matrices[particle_set_number]->get(dim, left) < splitvalue) {
        Vector left_vector;
        matrices[particle_set_number]->MakeColumnVector(left, &left_vector);
        left++;
      }

      while (likely(left <= right) && 
	     matrices[particle_set_number]->get(dim, right) >= splitvalue) {
        Vector right_vector;
        matrices[particle_set_number]->MakeColumnVector(right, &right_vector);
        right--;
      }

      if (unlikely(left > right)) {
        /* left == right + 1 */
        break;
      }

      Vector left_vector;
      Vector right_vector;

      matrices[particle_set_number]->MakeColumnVector(left, &left_vector);
      matrices[particle_set_number]->MakeColumnVector(right, &right_vector);

      left_vector.SwapValues(&right_vector);
      
      if (old_from_new) {
        index_t t = (*old_from_new)[particle_set_number][left];
        (*old_from_new)[particle_set_number][left] = 
	  (*old_from_new)[particle_set_number][right];
        (*old_from_new)[particle_set_number][right] = t;
      }
      
      DEBUG_ASSERT(left <= right);
      right--;
    }
    
    DEBUG_ASSERT(left == right + 1);

    return left;
  }

  template<typename TStatistic>
  bool RecursiveMatrixPartition
  (ArrayList<Matrix *> &matrices,
   GenHypercubeTree<TStatistic> *node, index_t count,
   ArrayList<index_t> &child_begin, ArrayList<index_t> &child_count,
   ArrayList< ArrayList<GenHypercubeTree<TStatistic> *> > *nodes_in_each_level,
   ArrayList< ArrayList<index_t> > *old_from_new, const int level, 
   int recursion_level, unsigned int code) {
    
    if(recursion_level < matrices[0]->n_rows()) {
      const DRange &range_in_this_dimension = node->bound().get
	(recursion_level);
      double split_value = 0.5 * (range_in_this_dimension.lo +
				  range_in_this_dimension.hi);
      
      // Partition based on the current dimension.
      index_t total_left_count = 0;
      index_t total_right_count = 0;

      // Temporary ArrayList for passing in the indices owned by the
      // children.
      ArrayList<index_t> left_child_begin;
      ArrayList<index_t> left_child_count;
      ArrayList<index_t> right_child_begin;
      ArrayList<index_t> right_child_count;
      left_child_begin.Init(matrices.size());
      left_child_count.Init(matrices.size());
      right_child_begin.Init(matrices.size());
      right_child_count.Init(matrices.size());

      // Divide each particle set.
      for(index_t particle_set_number = 0; 
	  particle_set_number < matrices.size(); particle_set_number++) {

	// If there is nothing to divide for the current particle set,
	// then skipt.
	if(child_count[particle_set_number] == 0) {
	  left_child_begin[particle_set_number] = -1;
	  left_child_count[particle_set_number] = 0;
	  right_child_begin[particle_set_number] = -1;
	  right_child_count[particle_set_number] = 0;
	  continue;
	}

	index_t left_count = MatrixPartition(particle_set_number, matrices, 
					     recursion_level, split_value, 
					     child_begin[particle_set_number],
					     child_count[particle_set_number],
					     old_from_new) - 
	  child_begin[particle_set_number];
	index_t right_count = child_count[particle_set_number] - left_count;

	// Divide into two sets.
	left_child_count[particle_set_number] = left_count;
	right_child_count[particle_set_number] = right_count;  
	if(left_count > 0) {
	  left_child_begin[particle_set_number] = 
	    child_begin[particle_set_number];
	}
	else {
	  left_child_begin[particle_set_number] = -1;
	}
	if(right_count > 0) {
	  right_child_begin[particle_set_number] = 
	    child_begin[particle_set_number] + left_count;
	}
	else {
	  right_child_begin[particle_set_number] = -1;
	}
	
	total_left_count += left_count;
	total_right_count += right_count;
      }

      bool left_result = false;
      bool right_result = false;

      if(total_left_count > 0) {
	left_result =
	  RecursiveMatrixPartition
	  (matrices, node, total_left_count, left_child_begin, 
	   left_child_count, nodes_in_each_level, old_from_new, level, 
	   recursion_level + 1, 2 * code);
      }
      if(total_right_count > 0) {
	right_result =
	  RecursiveMatrixPartition
	  (matrices, node, total_right_count, right_child_begin, 
	   right_child_count, nodes_in_each_level, old_from_new, level, 
	   recursion_level + 1, 2 * code + 1);
      }
      
      return left_result || right_result;
    }
    else {

      // Create the child. From the code, also set the bounding cube
      // of half the side length.
      GenHypercubeTree<TStatistic> *new_child =
	node->AllocateNewChild(matrices.size(), 
			       (node->node_index() << 
				matrices[0]->n_rows()) + code);

      // Appropriately set the membership in each particle set.
      for(index_t p = 0; p < matrices.size(); p++) {
	new_child->Init(p, child_begin[p], child_count[p]);
      }

      // Push the newly created child onto the list.
      ((*nodes_in_each_level)[level + 1]).PushBackCopy(new_child);
      new_child->bound().Init(matrices[0]->n_rows());
      
      Vector lower_coord, upper_coord;
      lower_coord.Init(matrices[0]->n_rows());
      upper_coord.Init(matrices[0]->n_rows());

      for(index_t d = matrices[0]->n_rows() - 1; d >= 0; 
	  d--) {
	const DRange &range_in_this_dimension = node->bound().get(d);

	if(code & (1 << d) > 0) {
	  lower_coord[d] = 0.5 * (range_in_this_dimension.lo +
				  range_in_this_dimension.hi);
	  upper_coord[d] = range_in_this_dimension.hi;
	}
	else {
	  lower_coord[d] = range_in_this_dimension.lo;
	  upper_coord[d] = 0.5 * (range_in_this_dimension.lo +
				  range_in_this_dimension.hi);
	}
      }
      new_child->bound() |= lower_coord;
      new_child->bound() |= upper_coord;

      return true;
    }
  }

  template<typename TStatistic>
  void ComputeBoundingHypercube(const ArrayList<Matrix *> &matrices,
				GenHypercubeTree<TStatistic> *node) {

    // Initialize the bound.
    node->bound().Init(matrices[0]->n_rows());

    // Iterate over each point owned by the node and compute its
    // bounding hypercube.
    for(index_t n = 0; n < matrices.size(); n++) {
      for(index_t i = node->begin(n); i < node->end(n); i++) {
	Vector point;
	matrices[n]->MakeColumnVector(i, &point);
	node->bound() |= point;
      }
    }

    // Compute the longest side and correct the maximum coordinate of
    // the bounding box accordingly.
    double max_side_length = 0;
    for(index_t d = 0; d < matrices[0]->n_rows(); d++) {
      const DRange &range_in_this_dimension = node->bound().get(d);
      double side_length = range_in_this_dimension.hi -
	range_in_this_dimension.lo;
      max_side_length = std::max(max_side_length, side_length);
    }
    Vector new_upper_coordinate;
    new_upper_coordinate.Init(matrices[0]->n_rows());
    for(index_t d = 0; d < matrices[0]->n_rows(); d++) {
      const DRange &range_in_this_dimension = node->bound().get(d);
      new_upper_coordinate[d] = range_in_this_dimension.lo + 
	max_side_length;
    } 
    node->bound() |= new_upper_coordinate;
  }

  template<typename TStatistic>
  void SplitGenHypercubeTree
  (ArrayList<Matrix *> &matrices, GenHypercubeTree<TStatistic> *node, 
   index_t leaf_size,
   ArrayList< ArrayList<GenHypercubeTree<TStatistic> *> > *nodes_in_each_level,
   ArrayList< ArrayList<index_t> > *old_from_new, index_t level) {
    
    // Set the level of this node.
    node->set_level(level);

    // If the node is just too small, then do not split.
    if(node->count() <= leaf_size) {
    }
    
    // Otherwise, attempt to split.
    else {
    
      // Ensure that the node list for storing each level is at least
      // the size of the current level + 1.
      nodes_in_each_level->SizeAtLeast(level + 2);
      if(((*nodes_in_each_level)[level + 1]).size() == BIG_BAD_NUMBER) {
	((*nodes_in_each_level)[level + 1]).Init();
      }

      // Recursively split each dimension.
      unsigned int code = 0;
      ArrayList<index_t> child_begin;
      ArrayList<index_t> child_count;
      child_begin.Init(matrices.size());
      child_count.Init(matrices.size());
      for(index_t i = 0; i < matrices.size(); i++) {
	child_begin[i] = node->begin(i);
	child_count[i] = node->count(i);
      }

      bool can_cut = (node->side_length() > DBL_EPSILON) &&
	RecursiveMatrixPartition
	(matrices, node, node->count(), child_begin, child_count,
	 nodes_in_each_level, old_from_new, level, 0, code);

      if(can_cut) {
	for(index_t i = 0; i < node->num_children(); i++) {
	  GenHypercubeTree<TStatistic> *child_node = node->get_child(i);
	  SplitGenHypercubeTree(matrices, child_node, leaf_size, 
				nodes_in_each_level, old_from_new,
				level + 1);
	}
      }
    }
  }
};
