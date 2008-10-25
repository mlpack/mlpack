template<typename MultiTreeProblem>
void MultiTreeDepthFirst<MultiTreeProblem>::Heuristic_
(const ArrayList<Tree *> &nodes, index_t *max_count_among_non_leaf, 
 index_t *split_index) {
  
  for(int start = 0; start < MultiTreeProblem::order; start++) {
    if(!(nodes[start]->is_leaf()) &&
       nodes[start]->count() > *max_count_among_non_leaf) {
      *max_count_among_non_leaf = nodes[start]->count();
      *split_index = start;
    }
  }
}

template<typename MultiTreeProblem>
void MultiTreeDepthFirst<MultiTreeProblem>::MultiTreeDepthFirstBase_
(const ArrayList<Matrix *> &sets, ArrayList<Tree *> &nodes,
 typename MultiTreeProblem::MultiTreeQueryResult &query_results) {

  MultiTreeHelper_<0, MultiTreeProblem::order>::NestedLoop
    (globals_, sets, nodes, query_results);
}

template<typename MultiTreeProblem>
void MultiTreeDepthFirst<MultiTreeProblem>::CopyNodeSet_
(const ArrayList<Tree *> &source_list, ArrayList<Tree *> *destination_list) {

  destination_list->Init(source_list.size());

  for(index_t i = 0; i < MultiTreeProblem::order; i++) {
    (*destination_list)[i] = source_list[i];
  }
}

template<typename MultiTreeProblem>
void MultiTreeDepthFirst<MultiTreeProblem>::MultiTreeDepthFirstCanonical_
(const ArrayList<Matrix *> &sets, ArrayList<Tree *> &nodes,
 typename MultiTreeProblem::MultiTreeQueryResult &query_results) {
  
  if(MultiTreeProblem::ConsiderTupleExact(globals_, nodes)) {
    return;
  }

  // Figure out which ones are non-leaves.
  index_t split_index = -1;
  index_t max_count_among_non_leaf = 0;
  Heuristic_(nodes, &max_count_among_non_leaf, &split_index);
  
  // All leaves, then base case.
  if(split_index < 0) {
    MultiTreeDepthFirstBase_(sets, nodes, query_results);
    return;
  }
  
  // Else, split an internal node and recurse.
  else {
    double new_num_tuples;
    
    // Copy to new nodes list before recursing.
    ArrayList<Tree *> new_nodes;
    CopyNodeSet_(nodes, &new_nodes);
    
    // Push down approximations downward for the node that is to be
    // expanded.
    nodes[split_index]->left()->stat().postponed.ApplyPostponed
      (nodes[split_index]->stat().postponed);
    nodes[split_index]->right()->stat().postponed.ApplyPostponed
      (nodes[split_index]->stat().postponed);
    nodes[split_index]->stat().postponed.SetZero();

    // Recurse to the left.
    new_nodes[split_index] = nodes[split_index]->left();
    new_num_tuples = TotalNumTuples(new_nodes);
    
    // If the current node combination is valid, then recurse.
    if(new_num_tuples > 0) {
      MultiTreeDepthFirstCanonical_(sets, new_nodes, query_results);
    }
    
    // Recurse to the right.
    new_nodes[split_index] = nodes[split_index]->right();
    new_num_tuples = TotalNumTuples(new_nodes);
    
    // If the current node combination is valid, then recurse.
    if(new_num_tuples > 0) {
      MultiTreeDepthFirstCanonical_(sets, new_nodes, query_results);
    }
    
    // Apply the postponed changes for both child nodes.
    typename MultiTreeProblem::MultiTreeQuerySummary tmp_left_child_summary
      (nodes[split_index]->left()->stat().summary);
    tmp_left_child_summary.ApplyPostponed
      (nodes[split_index]->left()->stat().postponed);
    typename MultiTreeProblem::MultiTreeQuerySummary tmp_right_child_summary
      (nodes[split_index]->right()->stat().summary);
    tmp_right_child_summary.ApplyPostponed
      (nodes[split_index]->right()->stat().postponed);

    // Refine statistics after recursing.
    nodes[split_index]->stat().summary.StartReaccumulate();
    nodes[split_index]->stat().summary.Accumulate(tmp_left_child_summary);
    nodes[split_index]->stat().summary.Accumulate(tmp_right_child_summary);
    
    return;
  }
}

template<typename MultiTreeProblem>
void MultiTreeDepthFirst<MultiTreeProblem>::PreProcessTree_(Tree *node) {

  // Reset summary statistics and postponed quantities.
  node->stat().Init(node->bound(), globals_.kernel_aux);
  
  if(!node->is_leaf()) {
    PreProcessTree_(node->left());
    PreProcessTree_(node->right());
  }
}
