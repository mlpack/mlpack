/**
 * @file allnn.h
 *
 * This file contains the definition of the AllNN class for dual-tree or naive
 * all-nearest-neighbors computation.
 * 
 * @see allnn_main.cc
 */

#include "allnn.h"
#include "fastlib/fx/io.h"

using namespace mlpack;
using namespace mlpack::allnn;

// We are calling an advanced constructor of arma::mat which allows us to use
// the same memory area as another matrix if desired (for aliasing).  For this
// constructor, the queries matrix is the same as the references matrix.
AllNN::AllNN(arma::mat& references_in, bool alias_matrix, bool naive) :
    references_(references_in.memptr(), references_in.n_rows,
        references_in.n_cols, !alias_matrix),
    queries_(references_.memptr(), references_.n_rows, references_.n_cols,
        false),
    naive_(naive),
    number_of_prunes_(0) {

  // We can't call out to the more complex constructor, but that will change
  // with C++0x.  Then, this constructor copypasta can be eliminated!
  
  /*
   * A bit of a trick so we can still use BaseCase_: we'll expand
   * the leaf size so that our trees only have one node.
   */
  if(IO::checkValue("leaf_size"))
    leaf_size_ = IO::getValue<int>("leaf_size");
  else
    leaf_size_ = 20;
  
  if(naive)
    leaf_size_ = references_.n_cols;

  // Need to move to a new timer system
  //fx_timer_start(module_, "tree_building");
  IO::startTimer("allnn/tree_building");

  reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_,
      leaf_size_, old_from_new_references_);
  query_tree_ = reference_tree_;
  // Make another alias
  old_from_new_queries_ = arma::Col<index_t>(old_from_new_references_.memptr(),
      old_from_new_references_.n_elem, false, true);

  //fx_timer_stop(module_, "tree_building");
  IO::stopTimer("allnn/tree_building");
  
  /* Ready the list of nearest neighbor candidates to be filled. */
  neighbor_indices_.set_size(queries_.n_cols);

  /* Ready the vector of upper bound nn distances for use. */
  neighbor_distances_.set_size(queries_.n_cols);
  neighbor_distances_.fill(DBL_MAX);
}

AllNN::AllNN(arma::mat& queries_in, arma::mat& references_in,
             bool alias_matrix, bool naive) :
    references_(references_in.memptr(), references_in.n_rows,
        references_in.n_cols, !alias_matrix),
    queries_(queries_in.memptr(), queries_in.n_rows, queries_in.n_cols,
        !alias_matrix),
    naive_(naive),
    number_of_prunes_(0) {
  
  /*
   * A bit of a trick so we can still use BaseCase_: we'll expand
   * the leaf size so that our trees only have one node.
   */
  if(IO::checkValue("leaf_size"))
    leaf_size_ = IO::getValue<int>("leaf_size");
  else
    leaf_size_ = 20;
  
  if(naive)
    leaf_size_ = max(queries_.n_cols, references_.n_cols);
  
  // Need to move to a new timer system
  //fx_timer_start(module_, "tree_building");
  IO::startTimer("tree_building");

  reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_,
      leaf_size_, old_from_new_references_);
  query_tree_ = reference_tree_;

  //fx_timer_stop(module_, "tree_building");
  IO::stopTimer("tree_building");
  
  /* Ready the list of nearest neighbor candidates to be filled. */
  neighbor_indices_.set_size(queries_.n_cols);

  /* Ready the vector of upper bound nn distances for use. */
  neighbor_distances_.set_size(queries_.n_cols);
  neighbor_distances_.fill(DBL_MAX);
}

// Note that we don't delete the fx module; it's managed externally.
AllNN::~AllNN() {
  if (reference_tree_ != query_tree_)
    delete reference_tree_;
  if (query_tree_ != NULL)
    delete query_tree_;
}

/**
 * Computes the minimum squared distance between the bounding boxes
 * of two nodes.
 */
double AllNN::MinNodeDistSq_(TreeType* query_node, TreeType* reference_node) {
  return query_node->bound().MinDistanceSq(reference_node->bound());
}


/**
 * Performs exhaustive computation between two nodes.
 *
 * Note that naive also makes use of this function.
 */
void AllNN::GNPBaseCase_(TreeType* query_node, TreeType* reference_node) {
  /* Make sure we didn't try to split children */
  DEBUG_ASSERT(query_node != NULL);
  DEBUG_ASSERT(reference_node != NULL);

  /* Make sure we should be in the base case */
  DEBUG_WARN_IF(!query_node->is_leaf());
  DEBUG_WARN_IF(!reference_node->is_leaf());

  /* Used to find the query node's new upper bound */
  double max_nearest_neighbor_distance = -1.0;

  /* Loop over all query-reference pairs */

  // Trees don't store their points, but instead give index ranges.
  // To make this feasible, they have to rearrange their input
  // matrices, which is why we were sure to make copies.
  for (index_t query_index = query_node->begin();
      query_index < query_node->end(); query_index++) {

    arma::vec query_point = queries_.unsafe_col(query_index);
    double distance_to_hrect = 
      reference_node->bound().MinDistanceSq(query_point);

    /* Try to prune one last time */
    if (distance_to_hrect < neighbor_distances_[query_index]) {
      for (index_t reference_index = reference_node->begin();
          reference_index < reference_node->end(); reference_index++) {

        arma::vec reference_point = references_.unsafe_col(reference_index);
        if (reference_node != query_node || reference_index != query_index) {
          // BLAS can perform many vectors ops more quickly than C/C++.
          double distance = la::DistanceSqEuclidean(query_point,
              reference_point);

          /* Record points found to be closer than the best so far */
          if (distance < neighbor_distances_[query_index]) {
            neighbor_distances_[query_index] = distance;
            neighbor_indices_[query_index] = reference_index;
          }
        }
      } /* for reference_index */
    } /* for prune test */

    /* Find the upper bound nn distance for this node */
    if (neighbor_distances_[query_index] > max_nearest_neighbor_distance) {
      max_nearest_neighbor_distance = neighbor_distances_[query_index];
    }

  } /* for query_index */

  /* Update the upper bound nn distance for the node */
  query_node->stat().set_max_distance_so_far(max_nearest_neighbor_distance);

} /* GNPBaseCase_ */


/**
 * Performs one node-node comparison in the GNP algorithm and
 * recurses upon all child combinations if no prune.
 */
void AllNN::GNPRecursion_(TreeType* query_node, TreeType* reference_node,
    double lower_bound_distance) {

  /* Make sure we didn't try to split children */
  DEBUG_ASSERT(query_node != NULL);
  DEBUG_ASSERT(reference_node != NULL);

  // The following asserts equality of two doubles and prints their
  // values if it fails.  Note that this *isn't* a particularly fast
  // debug check, though; it negates the benefit of passing ahead a
  // precomputed distance entirely.  That's why we have --mode=fast.

  /* Make sure the precomputed bounding information is correct */
  DEBUG_SAME_DOUBLE(lower_bound_distance,
      MinNodeDistSq_(query_node, reference_node));

  if (lower_bound_distance > query_node->stat().max_distance_so_far()) {

    /*
     * A reference node with lower-bound distance greater than this
     * query node's upper-bound nearest neighbor distance cannot
     * contribute a reference closer than any of the queries'
     * current neighbors, hence prune
     */
    number_of_prunes_++;

  } else if (query_node->is_leaf() && reference_node->is_leaf()) {

    /* Cannot further split leaves, so process exhaustively */
    GNPBaseCase_(query_node, reference_node);

  } else if (query_node->is_leaf()) {

    /* Query node's a leaf, but we can split references */
    double left_distance =
      MinNodeDistSq_(query_node, reference_node->left());
    double right_distance =
      MinNodeDistSq_(query_node, reference_node->right());

    /*
     * Nearer reference node more likely to contribute neighbors
     * (and thus tighten bounds), so visit it first
     */
    if (left_distance < right_distance) {
      // GNP part
      GNPRecursion_(query_node, reference_node->left(), left_distance);
      // GNP part
      GNPRecursion_(query_node, reference_node->right(), right_distance);
    } else {
      // GNP part
      GNPRecursion_(query_node, reference_node->right(), right_distance);
      // Prefetching directives
      // GNP part
      GNPRecursion_(query_node, reference_node->left(), left_distance);
    }

  } else if (reference_node->is_leaf()) {

    /* Reference node's a leaf, but we can split queries */
    double left_distance =
      MinNodeDistSq_(query_node->left(), reference_node);
    double right_distance =
      MinNodeDistSq_(query_node->right(), reference_node);

    /* Order of recursion does not matter */
    // GNP part 
    GNPRecursion_(query_node->left(), reference_node, left_distance);
    GNPRecursion_(query_node->right(), reference_node, right_distance);

    /* Update upper bound nn distance base new child bounds */
    query_node->stat().set_max_distance_so_far(
        max(query_node->left()->stat().max_distance_so_far(),
          query_node->right()->stat().max_distance_so_far()));

  } else {

    /*
     * Neither node is a leaf, so split both
     *
     * The order we process the query node's children doesn't
     * matter, but for each we should visit their nearer reference
     * node first.
     */

    double left_distance =
      MinNodeDistSq_(query_node->left(), reference_node->left());
    double right_distance =
      MinNodeDistSq_(query_node->left(), reference_node->right());

    if (left_distance < right_distance) {
      // GNP part        
      GNPRecursion_(query_node->left(),
          reference_node->left(), left_distance);
      // GNP part         
      GNPRecursion_(query_node->left(),
          reference_node->right(), right_distance);
    } else {
      // GNP part          
      GNPRecursion_(query_node->left(),
          reference_node->right(), right_distance);
      // GNP part         
      GNPRecursion_(query_node->left(),
          reference_node->left(), left_distance);
    }

    left_distance =
      MinNodeDistSq_(query_node->right(), reference_node->left());
    right_distance =
      MinNodeDistSq_(query_node->right(), reference_node->right());

    if (left_distance < right_distance) {
      GNPRecursion_(query_node->right(),
          reference_node->left(), left_distance);
      GNPRecursion_(query_node->right(),
          reference_node->right(), right_distance);
    } else {
      // GNP part         
      GNPRecursion_(query_node->right(),
          reference_node->right(), right_distance);
      // GNP part          
      GNPRecursion_(query_node->right(),
          reference_node->left(), left_distance);
    }
    /* Update upper bound nn distance base new child bounds */
    query_node->stat().set_max_distance_so_far(
        max(query_node->left()->stat().max_distance_so_far(),
          query_node->right()->stat().max_distance_so_far()));

  }

} /* GNPRecursion_ */

////////// Public Functions ////////////////////////////////////////

/**
 * Computes the nearest neighbors and stores them in results if
 * provided.  Overloaded version provided if you don't have any results you
 * are interested in.
 */
void AllNN::ComputeNeighbors(arma::vec& distances) {
  if(naive_) {
    //fx_timer_start(module_, "naive_time");
    IO::startTimer("allnn/naive/naive_time");
    
    /* BaseCase_ on the roots is equivalent to naive */
    GNPBaseCase_(query_tree_, reference_tree_);
    
    IO::stopTimer("allnn/naive/naive_time");
    //fx_timer_stop(module_, "naive_time");
  } else {
    //fx_timer_start(module_, "dual_tree_computation");
    IO::startTimer("allnn/dual_tree_computation");
    /* Start recursion on the roots of either tree */
    GNPRecursion_(query_tree_, reference_tree_,
        MinNodeDistSq_(query_tree_, reference_tree_));

    //fx_timer_stop(module_, "dual_tree_computation");
    IO::stopTimer("allnn/dual_tree_computation");
    
    // Save the total number of prunes to the FASTexec module; this
    // will printed after calling fx_done or can be read back later.
    //fx_result_int(module_, "number_of_prunes", number_of_prunes_);
  }
}
void AllNN::ComputeNeighbors(arma::vec& distances, arma::Col<index_t>& results) {
  ComputeNeighbors(distances);
  EmitResults(distances, results);
} /* ComputeNeighbors */

/**
 * Initialize and fill an arma::vec of results.
 */
void AllNN::EmitResults(arma::vec& distances, arma::Col<index_t>& results) {
  results.set_size(neighbor_indices_.n_elem);
  distances.set_size(neighbor_distances_.n_elem);

  /* Map the indices back from how they have been permuted. */
  for (index_t i = 0; i < neighbor_indices_.n_elem; i++) {
    results[old_from_new_queries_[i]] =
      old_from_new_references_[neighbor_indices_[i]];
    distances[old_from_new_references_[i]] = neighbor_distances_[i];
  }

} /* EmitResults */

void AllNN::loadDocumentation() {
	const char* rootNode = "allnn";
	IO::add(rootNode, "Performs dual-tree all-nearest-neighbors computation.");
	
	IO::add<int>("leaf_size", "The maximum number of points to store at a leaf.", rootNode);
	IO::add("tree_building", "Time spend building the kd-tree.", rootNode);
	IO::add("dual_tree_computation", "Time spent computing the nearest neighbors.", rootNode);
	IO::add("number_of_prunes", "Total node-pairs found to be too far to matter.", rootNode);
	
	const char* naive = "allnn/naive";
	IO::add(naive, "Performs naive all-nearest-neighbors computation.");
	
	IO::add("naive_time", "Time spent performing the naive computation.", naive); 
}
