/** @file multitree_dfs_impl.h
 *
 *  The implementation of the function templates defined in
 *  multitree_dfs.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

template<typename MultiTreeProblem>
double MultiTreeDepthFirst<MultiTreeProblem>::LeaveOneOutTuplesBase_
(const ArrayList<HybridTree *> &nodes) {
  
  // Compute the total number of tuples formed among the nodes.
  HybridTree *current_node = nodes[0];
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
(ArrayList<HybridTree *> &nodes, index_t examine_index_start) {
  
  // Test if all the nodes are equal or disjoint.
  bool equal_or_disjoint_flag = true;
  for(index_t i = examine_index_start + 1; i < MultiTreeProblem::order; i++) {

    // If there is a conflict, then return immediately.
    if(nodes[i]->end() <= nodes[i - 1]->begin()) {
      return 0;
    }
    
    // If there is a subsumption, then record the first index that
    // happens so.
    if(equal_or_disjoint_flag) {
      if(first_node_indices_strictly_surround_second_node_indices_
	 (nodes[i - 1], nodes[i])) {
	examine_index_start = i - 1;
	equal_or_disjoint_flag = false;
      }
      else if(first_node_indices_strictly_surround_second_node_indices_
	      (nodes[i], nodes[i - 1])) {
	examine_index_start = i;
	equal_or_disjoint_flag = false;
      }
    }
  }

  // If everything is either disjoint, or equal, then we can call the
  // base case.
  if(equal_or_disjoint_flag) {
    return LeaveOneOutTuplesBase_(nodes);
  }
  else {
    HybridTree *node_saved = nodes[examine_index_start];
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
template<bool is_hybrid_node, typename TreeType1, typename TreeType2>
void MultiTreeDepthFirst<MultiTreeProblem>::Heuristic_
(TreeType1 *nd, TreeType2 *nd1, TreeType2 *nd2, TreeType2 **partner1,
 TreeType2 **partner2) {
  
  if(nd == NULL) {
    *partner1 = nd1;
    *partner2 = nd2;
  }
  else {
    
    bool no_conflict_nd1 = (nd1->end() > nd->begin()) || (!is_hybrid_node);
    bool no_conflict_nd2 = (nd2->end() > nd->begin()) || (!is_hybrid_node);
    double d1 = (no_conflict_nd1) ?
      nd->bound().MinDistanceSq(nd1->bound()):DBL_MAX;
    double d2 = (no_conflict_nd2) ?
      nd->bound().MinDistanceSq(nd2->bound()):DBL_MAX;
    
    // Prioritized traversal based on the squared distance bounds.
    if(d1 <= d2) {
      if(no_conflict_nd1) {
	*partner1 = nd1;
      }
      if(no_conflict_nd2) {
	*partner2 = nd2;
      }
    }
    else {
      if(no_conflict_nd2) {
	*partner1 = nd2;
      }
      if(no_conflict_nd1) {
	*partner2 = nd1;
      }
    }
  }
}

template<typename MultiTreeProblem>
void MultiTreeDepthFirst<MultiTreeProblem>::MultiTreeDepthFirstBase_
(const ArrayList<Matrix *> &query_sets,
 const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
 ArrayList<HybridTree *> &hybrid_nodes,
 ArrayList<QueryTree *> &query_nodes,
 ArrayList<ReferenceTree *> &reference_nodes,
 typename MultiTreeProblem::MultiTreeQueryResult &query_results,
 double total_num_tuples) {

  MultiTreeHelper_<0, MultiTreeProblem::num_hybrid_sets>::HybridNodeNestedLoop
    (globals_, query_sets, sets, targets, hybrid_nodes, query_nodes,
     reference_nodes, query_results);

  MultiTreeHelper_<0, MultiTreeProblem::num_query_sets>::QueryNodeNestedLoop
    (globals_, query_sets, sets, targets, hybrid_nodes, query_nodes,
     reference_nodes, query_results);

  // Add the postponed information to each point, without causing any
  // duplicate information transmission for each "hybrid" node.
  for(index_t i = 0; i < MultiTreeProblem::num_hybrid_sets; i++) {
    if(i > 0 && hybrid_nodes[i] == hybrid_nodes[i - 1]) {
      continue;
    }
    
    HybridTree *qnode = hybrid_nodes[i];

    // Clear the summary statistics of the current query node so that
    // we can refine it to better bounds.
    qnode->stat().summary.StartReaccumulate();

    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Apply postponed to each point.
      query_results.ApplyPostponed(qnode->stat().postponed, q);

      // Increment the number of (n - 1) tuples pruned.
      query_results.n_pruned[q] += total_n_minus_one_tuples_[i];

      // Refine statistics.
      qnode->stat().summary.Accumulate(query_results, q);
    }

    // Clear postponed information.
    qnode->stat().postponed.SetZero();

    // Post process function
    qnode->stat().summary.PostAccumulate(globals_, query_results, 
					 qnode->begin(), qnode->count());
  }
  
  // Add the postponed information to each point for each "query"
  // node.
  for(index_t i = 0; i < MultiTreeProblem::num_query_sets; i++) {
    
    QueryTree *qnode = query_nodes[i];

    // Clear the summary statistics of the current query node so that
    // we can refine it to better bounds.
    qnode->stat().summary.StartReaccumulate();

    for(index_t q = qnode->begin(); q < qnode->end(); q++) {

      // Apply postponed to each point.
      query_results.ApplyPostponed(qnode->stat().postponed, q);

      // Increment the number of (n - 1) tuples pruned.
      query_results.UpdatePrunedComponents(reference_nodes, q);

      // Refine statistics.
      qnode->stat().summary.Accumulate(query_results, q);
    }

    // Clear postponed information.
    qnode->stat().postponed.SetZero();

    // Post process function.
    qnode->stat().summary.PostAccumulate(globals_, query_results,
					 qnode->begin(), qnode->count());
  }

}

template<typename MultiTreeProblem>
void MultiTreeDepthFirst<MultiTreeProblem>::MultiTreeDepthFirstCanonical_
(const ArrayList<Matrix *> &query_sets,
 const ArrayList<Matrix *> &sets, const ArrayList<Matrix *> &targets,
 ArrayList<HybridTree *> &hybrid_nodes,
 ArrayList<QueryTree *> &query_nodes,
 ArrayList<ReferenceTree *> &reference_nodes,
 typename MultiTreeProblem::MultiTreeQueryResult &query_results,
 double total_num_tuples) {

  // Declare the delta object so that it can be shared between the
  // exact and the probabilistic pruning methods...
  typename MultiTreeProblem::MultiTreeDelta exact_delta;

  if(MultiTreeProblem::ConsiderTupleExact(globals_, query_results, exact_delta,
					  query_sets, sets, targets,
					  hybrid_nodes, query_nodes,
					  reference_nodes, total_num_tuples,
					  total_n_minus_one_tuples_root_,
					  total_n_minus_one_tuples_)) {
    return;
  }
  else if(MultiTreeProblem::ConsiderTupleProbabilistic
	  (globals_, query_results, exact_delta, query_sets, sets,
	   hybrid_nodes, query_nodes, reference_nodes, total_num_tuples,
	   total_n_minus_one_tuples_root_, total_n_minus_one_tuples_)) {
    return;
  }

  // Recurse to every combination...
  MultiTreeHelper_<0, MultiTreeProblem::num_hybrid_sets>::
    HybridNodeRecursionLoop(query_sets, sets, targets, hybrid_nodes,
			    query_nodes, reference_nodes, total_num_tuples,
			    false, query_results, this);
  return;
}

template<typename MultiTreeProblem>
template<typename Tree>
void MultiTreeDepthFirst<MultiTreeProblem>::PreProcessQueryTree_(Tree *node) {
  
  node->stat().Init(node->bound(), globals_.kernel_aux);

  if(node->is_leaf()) {
  }
  else {
    PreProcessQueryTree_(node->left());
    PreProcessQueryTree_(node->right());
  }
}

template<typename MultiTreeProblem>
template<typename Tree>
void MultiTreeDepthFirst<MultiTreeProblem>::PreProcessQueryTreeMonochromatic_
(Tree *qnode, const typename MultiTreeProblem::MultiTreeQueryResult &results) {

  qnode->stat().summary.StartReaccumulate();

  if(qnode->is_leaf()) {   
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      qnode->stat().summary.Accumulate(results, q);
    }
  }
  else {
    PreProcessQueryTreeMonochromatic_(qnode->left(), results);
    PreProcessQueryTreeMonochromatic_(qnode->right(), results);

    qnode->stat().summary.Accumulate(qnode->left()->stat().summary);
    qnode->stat().summary.Accumulate(qnode->right()->stat().summary);
  }

  // Initialize the precomputed n-tuples...
  qnode->stat().num_precomputed_tuples = 0;
  if(qnode->stat().in_strata) {
    qnode->stat().num_precomputed_tuples = 
      math::BinomialCoefficient(qnode->count() - 1, 
				MultiTreeProblem::order - 1);
  }
  else {

    if(!qnode->is_leaf()) {
      qnode->stat().num_precomputed_tuples = 
	qnode->left()->stat().num_precomputed_tuples +
	qnode->right()->stat().num_precomputed_tuples;
    }
  }
}

template<typename MultiTreeProblem>
template<typename Tree>
void MultiTreeDepthFirst<MultiTreeProblem>::PreProcessReferenceTree_
(Tree *node, index_t reference_tree_index) {

  if(node->is_leaf()) {

    node->stat().PostInit(node->bound(), globals_.kernel_aux, sets_, targets_,
			  node->begin(), node->count());
  }
  else {
    PreProcessReferenceTree_(node->left(), reference_tree_index);
    PreProcessReferenceTree_(node->right(), reference_tree_index);

    node->stat().PostInit(node->bound(), globals_.kernel_aux, sets_, targets_,
			  node->begin(), node->count(), node->left()->stat(),
			  node->right()->stat());
  }
}

template<typename MultiTreeProblem>
template<typename Tree>
void MultiTreeDepthFirst<MultiTreeProblem>::PostProcessTree_
(const Matrix &qset, Tree *node, 
 typename MultiTreeProblem::MultiTreeQueryResult &query_results) {
  
  if(node->is_leaf()) {
    for(index_t i = node->begin(); i < node->end(); i++) {
      query_results.FinalPush(qset, node->stat(), i);
      query_results.PostProcess(globals_, i);
    }
  }
  else {

    // Push down postponed contributions to the left and the right.
    node->stat().FinalPush(node->left()->stat());
    node->stat().FinalPush(node->right()->stat());
    
    PostProcessTree_(qset, node->left(), query_results);
    PostProcessTree_(qset, node->right(), query_results);
  }

  node->stat().postponed.SetZero();
}
