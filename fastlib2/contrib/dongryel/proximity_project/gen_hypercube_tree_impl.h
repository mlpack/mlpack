#include "fastlib/fastlib.h"

namespace tree_gen_hypercube_tree_private {

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
  
  bool RecursiveMatrixPartition
  (Matrix &matrix, GenHypercubeTree *node, index_t first, index_t count,
   ArrayList< ArrayList<GenHypercubeTree *> > *nodes_in_each_level,
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
      node->set_child(code, first, count, 
		      (node->node_index() << matrix.n_rows()) + code);
      GenHypercubeTree *new_child = node->get_child(code);

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

  void ComputeBoundingHypercube(Matrix &matrix, GenHypercubeTree *node) {

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

  void SplitGenHypercubeTree
  (Matrix& matrix, GenHypercubeTree *node, index_t leaf_size,
   ArrayList< ArrayList<GenHypercubeTree *> > *nodes_in_each_level,
   index_t *old_from_new, index_t level) {
    
    // Set the level of this node.
    node->set_level(level);
    
    // If the node is just too small, then do not split.
    if(node->count() < leaf_size) {
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
	  GenHypercubeTree *child_node = node->get_child(i);
	  if(child_node != NULL) {
	    SplitGenHypercubeTree(matrix, child_node, leaf_size, 
				  nodes_in_each_level, old_from_new,
				  level + 1);
	  }
	}
      }
      else {
	node->DeleteChildren();
      }
    }
  }
};
