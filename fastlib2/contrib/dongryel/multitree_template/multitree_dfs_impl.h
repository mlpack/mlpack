template<typename MultiTreeProblem>
double MultiTreeDepthFirst<MultiTreeProblem>::LeaveOneOutTuplesBase_
(const ArrayList<Tree *> &nodes) {
  
  // Compute the total number of tuples formed among the nodes.
  Tree *current_node = nodes[0];
  int numerator = current_node->count();
  int denominator = 1;
  double total_num_tuples = numerator;
  for(index_t i = 1; i < MultiTreeProblem::order; i++) {
    if(current_node == nodes[i]) {
      if(numerator == 1) {
	total_num_tuples = 0;
	return total_num_tuples;
      }
      else {
	numerator--;
	denominator++;
	total_num_tuples *= ((double) numerator) / ((double) denominator);
      }
    }
    else {
      current_node = nodes[i];
      numerator = current_node->count();
      denominator = 1;
      total_num_tuples *= ((double) numerator) / ((double) denominator);
    }
  }

  for(index_t i = 0; i < MultiTreeProblem::order; i++) {
    int numerator = nodes[i]->count();
    int equal_count = 0;
    for(index_t j = i; j >= 0; j--) {
      if(nodes[j] == nodes[i]) {
	equal_count++;
      }
      else {
	break;
      }
    }
    for(index_t j = i + 1; j < MultiTreeProblem::order; j++) {
      if(nodes[j] == nodes[i]) {
	equal_count++;
      }
      else {
	break;
      }
    }
    total_n_minus_one_tuples_[i] += total_num_tuples /
      ((double) numerator) * ((double) equal_count);
  }
  
  return total_num_tuples;
}

template<typename MultiTreeProblem>
double MultiTreeDepthFirst<MultiTreeProblem>::RecursiveLeaveOneOutTuples_
(ArrayList<Tree *> &nodes, index_t examine_index_start) {
  
  // Test if all the nodes are equal or disjoint.
  bool equal_or_disjoint_flag = true;
  for(index_t i = examine_index_start + 1; i < MultiTreeProblem::order; i++) {

    // If there is a conflict, then return immediately.
    if(nodes[i]->end() <= nodes[i - 1]->begin()) {
      return 0;
    }
    
    // If there is a subsumption, then record the first index that
    // happens so.
    if(first_node_indices_strictly_surround_second_node_indices_
       (nodes[i - 1], nodes[i]) ||
       first_node_indices_strictly_surround_second_node_indices_
       (nodes[i], nodes[i - 1])) {
      examine_index_start = i;
      equal_or_disjoint_flag = false;
    }
  }

  // If everything is either disjoint, or equal, then we can call the
  // base case.
  if(equal_or_disjoint_flag) {
    return LeaveOneOutTuplesBase_(nodes);
  }
  else {
    Tree *node_saved = nodes[examine_index_start];
    nodes[examine_index_start] = node_saved->left();
    double left_count = RecursiveLeaveOneOutTuples_(nodes,
						    examine_index_start);
    nodes[examine_index_start] = node_saved->right();
    double right_count = RecursiveLeaveOneOutTuples_(nodes,
						     examine_index_start);
    nodes[examine_index_start] = node_saved;
    return left_count + right_count;
  }

}

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
 typename MultiTreeProblem::MultiTreeQueryResult &query_results,
 double total_num_tuples) {

  MultiTreeHelper_<0, MultiTreeProblem::order>::NestedLoop
    (globals_, sets, nodes, query_results);

  // Add the postponed information to each point, without causing any
  // duplicate information transmission.
  for(index_t i = 0; i < MultiTreeProblem::order; i++) {
    if(i > 0 && nodes[i] == nodes[i - 1]) {
      continue;
    }
    
    Tree *qnode = nodes[i];

    // Clear the summary statistics of the current query node so that
    // we can refine it to better bounds.
    qnode->stat().summary.StartReaccumulate();

    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Apply postponed to each point.
      query_results.ApplyPostponed(qnode->stat().postponed, q);

      // Refine statistics.
      qnode->stat().summary.Accumulate(query_results, q);

      // Increment the number of (n - 1) tuples pruned.
      query_results.n_pruned[q] += total_n_minus_one_tuples_[i];
    }

    // Clear postponed information.
    qnode->stat().postponed.SetZero();
  }
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
 typename MultiTreeProblem::MultiTreeQueryResult &query_results,
 double total_num_tuples) {
  
  if(MultiTreeProblem::ConsiderTupleExact(globals_, nodes, total_num_tuples,
					  total_n_minus_one_tuples_root_,
					  total_n_minus_one_tuples_)) {
    return;
  }

  // Figure out which ones are non-leaves.
  index_t split_index = -1;
  index_t max_count_among_non_leaf = 0;
  Heuristic_(nodes, &max_count_among_non_leaf, &split_index);
  
  // All leaves, then base case.
  if(split_index < 0) {
    MultiTreeDepthFirstBase_(sets, nodes, query_results, total_num_tuples);
    return;
  }
  
  // Else, split an internal node and recurse.
  else {
    double new_num_tuples;
    
    // Push down approximations downward for the node that is to be
    // expanded.
    nodes[split_index]->left()->stat().postponed.ApplyPostponed
      (nodes[split_index]->stat().postponed);
    nodes[split_index]->right()->stat().postponed.ApplyPostponed
      (nodes[split_index]->stat().postponed);
    nodes[split_index]->stat().postponed.SetZero();

    // Recurse to the left.
    Tree *node_saved = nodes[split_index];
    nodes[split_index] = node_saved->left();
    new_num_tuples = TotalNumTuples(nodes);
    
    // If the current node combination is valid, then recurse.
    if(new_num_tuples > 0) {
      MultiTreeDepthFirstCanonical_(sets, nodes, query_results,
				    new_num_tuples);
    }
    
    // Recurse to the right.
    nodes[split_index] = node_saved->right();
    new_num_tuples = TotalNumTuples(nodes);
    
    // If the current node combination is valid, then recurse.
    if(new_num_tuples > 0) {
      MultiTreeDepthFirstCanonical_(sets, nodes, query_results,
				    new_num_tuples);
    }

    // Put back the saved node back into the list.
    nodes[split_index] = node_saved;
    
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

template<typename MultiTreeProblem>
void MultiTreeDepthFirst<MultiTreeProblem>::PostProcessTree_
(Tree *node, typename MultiTreeProblem::MultiTreeQueryResult &query_results) {
  
  if(node->is_leaf()) {
    for(index_t i = node->begin(); i < node->end(); i++) {
      query_results.ApplyPostponed(node->stat().postponed, i);
      query_results.PostProcess(i);
    }
  }
  else {

    // Push down postponed contributions to the left and the right.
    node->left()->stat().postponed.ApplyPostponed(node->stat().postponed);
    node->right()->stat().postponed.ApplyPostponed(node->stat().postponed);
    
    PostProcessTree_(node->left(), query_results);
    PostProcessTree_(node->right(), query_results);
  }

  node->stat().postponed.SetZero();
}
