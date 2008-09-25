#include "fastlib/fastlib.h"

namespace tree_gen_hypercube_tree_private {

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

  void FindNeighbors(unsigned int index, index_t level, 
		     index_t dimension, 
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

  index_t MatrixPartition(Matrix& matrix, index_t dim, double splitvalue,
			  index_t first, index_t count, 
			  index_t *old_from_new) {
    
    index_t left = first;
    index_t right = first + count - 1;
    
    /* At any point:
     *
     *   everything < left is correct
     *   everything > right is correct
     */
    for (;;) {
      while (matrix.get(dim, left) < splitvalue && likely(left <= right)) {
        Vector left_vector;
        matrix.MakeColumnVector(left, &left_vector);
        left++;
      }

      while (matrix.get(dim, right) >= splitvalue && likely(left <= right)) {
        Vector right_vector;
        matrix.MakeColumnVector(right, &right_vector);
        right--;
      }

      if (unlikely(left > right)) {
        /* left == right + 1 */
        break;
      }

      Vector left_vector;
      Vector right_vector;

      matrix.MakeColumnVector(left, &left_vector);
      matrix.MakeColumnVector(right, &right_vector);

      left_vector.SwapValues(&right_vector);
      
      if (old_from_new) {
        index_t t = old_from_new[left];
        old_from_new[left] = old_from_new[right];
        old_from_new[right] = t;
      }
      
      DEBUG_ASSERT(left <= right);
      right--;
    }
    
    DEBUG_ASSERT(left == right + 1);

    return left;
  }

  template<typename TStatistic>
  bool RecursiveMatrixPartition
  (Matrix &matrix, GenHypercubeTree<TStatistic> *node, index_t first, 
   index_t count, 
   ArrayList< ArrayList<GenHypercubeTree<TStatistic> *> > *nodes_in_each_level,
   index_t *old_from_new, const int level, int recursion_level, 
   unsigned int code) {

    if(recursion_level < matrix.n_rows()) {
      const DRange &range_in_this_dimension = node->bound().get
	(recursion_level);
      double split_value = 0.5 * (range_in_this_dimension.lo +
				  range_in_this_dimension.hi);
      
      // Partition based on the current dimension.
      index_t left_count = MatrixPartition(matrix, recursion_level, 
					   split_value, first, 
					   count, old_from_new) - first;
      index_t right_count = count - left_count;

      bool left_result = false;
      bool right_result = false;

      if(left_count > 0) {
	left_result =
	  RecursiveMatrixPartition
	  (matrix, node, first, left_count, nodes_in_each_level,
	   old_from_new, level, recursion_level + 1, 2 * code);
      }
      if(right_count > 0) {
	right_result =
	  RecursiveMatrixPartition
	  (matrix, node, first + left_count, right_count, nodes_in_each_level,
	   old_from_new, level, recursion_level + 1, 2 * code + 1);
      }
      
      return left_result || right_result;
    }
    else {

      // Create the child. From the code, also set the bounding cube
      // of half the side length.
      GenHypercubeTree<TStatistic> *new_child = 
	node->set_child(first, count, 
			(node->node_index() << matrix.n_rows()) + code);

      // Push the newly created child onto the list.
      ((*nodes_in_each_level)[level]).PushBackCopy(new_child);
      new_child->bound().Init(matrix.n_rows());
      
      Vector lower_coord, upper_coord;
      lower_coord.Init(matrix.n_rows());
      upper_coord.Init(matrix.n_rows());

      for(index_t d = matrix.n_rows() - 1; d >= 0; d--) {
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
  void ComputeBoundingHypercube(Matrix &matrix, 
				GenHypercubeTree<TStatistic> *node) {

    // Initialize the bound.
    node->bound().Init(matrix.n_rows());

    // Iterate over each point owned by the node and compute its
    // bounding hypercube.
    for(index_t i = node->begin(); i < node->end(); i++) {       
      Vector point;
      matrix.MakeColumnVector(i, &point);
      node->bound() |= point;
    }

    // Compute the longest side and correct the maximum coordinate of
    // the bounding box accordingly.
    double max_side_length = 0;
    for(index_t d = 0; d < matrix.n_rows(); d++) {
      const DRange &range_in_this_dimension = node->bound().get(d);
      double side_length = range_in_this_dimension.hi -
	range_in_this_dimension.lo;
      max_side_length = std::max(max_side_length, side_length);
    }
    Vector new_upper_coordinate;
    new_upper_coordinate.Init(matrix.n_rows());
    for(index_t d = 0; d < matrix.n_rows(); d++) {
      const DRange &range_in_this_dimension = node->bound().get(d);
      new_upper_coordinate[d] = range_in_this_dimension.lo + 
	max_side_length;
    } 
    node->bound() |= new_upper_coordinate;
  }

  template<typename TStatistic>
  void SplitGenHypercubeTree
  (Matrix& matrix, GenHypercubeTree<TStatistic> *node, index_t leaf_size,
   ArrayList< ArrayList<GenHypercubeTree<TStatistic> *> > *nodes_in_each_level,
   index_t *old_from_new, index_t level) {
    
    // Set the level of this node.
    node->set_level(level);

    // If the node is just too small, then do not split.
    if(node->count() <= leaf_size) {
    }
    
    // Otherwise, attempt to split.
    else {
    
      // Ensure that the node list for storing each level is at least
      // the size of the current level + 1.
      nodes_in_each_level->SizeAtLeast(level + 1);
      if(((*nodes_in_each_level)[level]).size() == BIG_BAD_NUMBER) {
	((*nodes_in_each_level)[level]).Init();
      }

      // Temporarily allocate children list.
      node->AllocateChildren(matrix.n_rows());
      
      // Recursively split each dimension.
      unsigned int code = 0;
      bool can_cut = RecursiveMatrixPartition
	(matrix, node, node->begin(), node->count(), nodes_in_each_level,
	 old_from_new, level, 0, code);

      if(can_cut) {
	for(index_t i = 0; i < node->num_children(); i++) {
	  GenHypercubeTree<TStatistic> *child_node = node->get_child(i);
	  SplitGenHypercubeTree(matrix, child_node, leaf_size, 
				nodes_in_each_level, old_from_new,
				level + 1);
	}
      }
    }
  }
};
