/**
 * @file allknn.h
 *
 * Defines AllNN class to perform all-nearest-neighbors on two specified 
 * data sets.
 */

// inclusion guards, please add them to your .h files
#ifndef ALLKNN_H
#define ALLKNN_H

// We need to include fastlib.  If you want to use fastlib, 
// you need to have this line in addition to
// the deplibs section of your build.py
#include <fastlib/fastlib.h>
#include <vector>
#include <string>
/**
 * Forward declaration for the tester class
 */
class TestAllkNN;
/**
* Performs all-nearest-neighbors.  This class will build the trees and 
* perform the recursive  computation.
*/
class AllkNNkdTree {
  // Declare the tester class as a friend class so that it has access
  // to the private members of the class
  friend class TestAllkNN;
  
  //////////////////////////// Nested Classes ///////////////////////////////////////////////
  /**
  * Extra data for each node in the tree.  For all nearest neighbors, 
  * each node only
  * needs its upper bound on its nearest neighbor distances.  
  */
  class QueryStat {
    
    // Defines many useful things for a class, including a pretty 
    // printer and copy constructor
    OT_DEF_BASIC(QueryStat) {
      // Include this line for all non-pointer members
      // There are other versions for arrays and pointers, see base/otrav.h
      OT_MY_OBJECT(max_distance_so_far_); 
    } // OT_DEF_BASIC
    
   private:
    
    /**
    * The upper bound on the node's nearest neighbor distances.
     */
    double max_distance_so_far_;
    
   public:
    double max_distance_so_far() {
      return max_distance_so_far_; 
    } 
    
    void set_max_distance_so_far(double new_dist) {
      max_distance_so_far_ = new_dist; 
    } 
    
    // In addition to any member variables for the statistic, all stat 
    // classes need two Init 
    // functions, one for leaves and one for non-leaves. 
    
    /**
     * Initialization function used in tree-building when initializing 
     * a leaf node.  For allnn, needs no additional information 
     * at the time of tree building.  
     */
    void Init(const Matrix& matrix, index_t start, index_t count) {
      // The bound starts at infinity
      max_distance_so_far_ = DBL_MAX;
    } 
    
    /**
     * Initialization function used in tree-building when initializing a non-leaf node.  For other algorithms,
     * node statistics can be built using information from the children.  
     */
    void Init(const Matrix& matrix, index_t start, index_t count, 
        const QueryStat& left, const QueryStat& right) {
      // For allnn, non-leaves can be initialized in the same way as leaves
      Init(matrix, start, count);
    } 
    
  };

  //////////////////////////// END of Nested Classes ///////////////////////////////////////////////
  
  // TreeType are BinarySpaceTrees where the data are bounded by 
  // Euclidean bounding boxes, the data are stored in a Matrix, 
  // and each node has a QueryStat for its bound.
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, QueryStat> TreeType;
   
  
  /////////////////////////////// Members //////////////////////////////////////////////////
 private:
  // These will store our data sets.
  Matrix queries_;
  Matrix references_;
  // Pointers to the roots of the two trees.
  TreeType* query_tree_;
  std::vector<TreeType*> query_tree_vec_;  // use it for single tree, a vector of single-point-trees
  TreeType* reference_tree_;
  // The total number of prunes.
  index_t number_of_prunes_;
  // A permutation of the indices for tree building.
  ArrayList<index_t> old_from_new_queries_;
  ArrayList<index_t> old_from_new_references_;
  // The number of points in a leaf
  index_t leaf_size_;
  // The distance to the candidate nearest neighbor for each query
  Vector neighbor_distances_;
  // The indices of the candidate nearest neighbor for each query
  ArrayList<index_t> neighbor_indices_;
  // number of nearest neighbrs
  index_t knns_; 
  // The module containing the parameters for this computation. 
  struct datanode* module_;
  // Query index for single tree
  index_t query_index_single_;
  // The mode of trees: dual tree==2, single tree==1
  int tree_dual_single_;

  
  /////////////////////////////// Constructors /////////////////////////////////////////////
  
  // Add this at the beginning of a class to prevent accidentally calling the copy constructor
  FORBID_ACCIDENTAL_COPIES(AllkNNkdTree);
  

 public:
  index_t ct_dist_comp;
  /**
  * Constructors are generally very simple in FASTlib; most of the work is done by Init().  This is only
  * responsible for ensuring that the object is ready to be destroyed safely.  
  */
  AllkNNkdTree() {
    query_tree_vec_.clear();
    query_tree_ = NULL;
    reference_tree_ = NULL;
  } 
  
  /**
  * The tree is the only member we are responsible for deleting.  The others will take care of themselves.  
  */
  ~AllkNNkdTree() {
    for (std::vector<TreeType*>::iterator i = query_tree_vec_.begin(); i < query_tree_vec_.end(); i++) {
      if (*i != NULL) {
	delete *i;
      }
    }
    query_tree_vec_.clear();
    if (query_tree_ != NULL) {
      delete query_tree_;
    }
    if (reference_tree_ != NULL) {
      delete reference_tree_;
    }
  } 
    
  void Init(int tree_dual_single) {
    tree_dual_single_ = tree_dual_single;
    ct_dist_comp = 0;
  }
      
 /////////////////////////////// Helper Functions ///////////////////////////////////////////////////
  
  /**
   * Computes the minimum squared distance between the bounding boxes of two nodes
   */
  double MinNodeDistSq_ (TreeType* query_node, TreeType* reference_node) {
    // node->bound() gives us the DHrectBound class for the node
    // It has a function MinDistanceSq which takes another DHrectBound
    return query_node->bound().MinDistanceSq(reference_node->bound());
  } 
  
  
  /**
   * Performs exhaustive computation between two leaves.  
   */
  void ComputeBaseCase_(TreeType* query_node, TreeType* reference_node) {
    index_t ind=0;
    // Check that the pointers are not NULL
    DEBUG_ASSERT(query_node != NULL);
    DEBUG_ASSERT(reference_node != NULL);
    // Check that we really should be in the base case
    DEBUG_WARN_IF(!query_node->is_leaf());
    DEBUG_WARN_IF(!reference_node->is_leaf());
    
    // Used to find the query node's new upper bound
    double query_max_neighbor_distance = -1.0;
    std::vector<std::pair<double, index_t> > neighbors(knns_);
    // node->begin() is the index of the first point in the node, 
    // node->end is one past the last index
    for (index_t query_index = query_node->begin(); query_index < query_node->end(); query_index++) {
      // Get the query point from the matrix
      Vector query_point;
      if (tree_dual_single_ == 2) {// for dual tree
	queries_.MakeColumnVector(query_index, &query_point);
	ind = query_index*knns_;
      }
      else if (tree_dual_single_ == 1) {// for single tree
	queries_.MakeColumnVector(query_index_single_, &query_point);
	ind = query_index_single_*knns_;
      }
      // queries_.MakeColumnVector(query_index, &query_point);
      
      for(index_t i=0; i<knns_; i++) {
        neighbors[i]=std::make_pair(neighbor_distances_[ind+i], neighbor_indices_[ind+i]);
      }
      // We'll do the same for the references
      for (index_t reference_index = reference_node->begin(); reference_index < reference_node->end(); reference_index++) {
	// Confirm that points do not identify themselves as neighbors in the monochromatic case
        if (likely(reference_node != query_node || reference_index != query_index)) {
	  Vector reference_point;
	  references_.MakeColumnVector(reference_index, &reference_point);
	  // We'll use lapack to find the distance between the two vectors
	  double distance = la::DistanceSqEuclidean(query_point, reference_point);
	  ct_dist_comp++;
	  // If the reference point is closer than the current candidate, 
	  // we'll update the candidate
	  if (distance < neighbor_distances_[ind+knns_-1])
	    neighbors.push_back(std::make_pair(distance, reference_index));
	}
      } // for reference_index
      // if ((index_t)neighbors.size()>knns_) {
      std::sort(neighbors.begin(), neighbors.end());
      for(index_t i=0; i<knns_; i++) {
        neighbor_distances_[ind+i] = neighbors[i].first;
        neighbor_indices_[ind+i]  = neighbors[i].second;
      }
      neighbors.resize(knns_);
      // We need to find the upper bound distance for this query node
      if (neighbor_distances_[ind+knns_-1] > query_max_neighbor_distance)
        query_max_neighbor_distance = neighbor_distances_[ind+knns_-1]; 
      
    } // for query_index 
    // Update the upper bound for the query_node
    query_node->stat().set_max_distance_so_far(query_max_neighbor_distance);
         
  } // ComputeBaseCase_
  
  
  /**
   * The recursive function for dual tree
   */
  void DualTreeNeighborsRecursion_(TreeType* query_node, TreeType* reference_node, double lower_bound_distance) {
    // A DEBUG statement with no predefined message
    DEBUG_ASSERT(query_node != NULL);
    // A DEBUG statement with a predefined message
    DEBUG_ASSERT_MSG(reference_node != NULL, "reference node is null");
    // Make sure the bounding information is correct
    //DEBUG_ASSERT(lower_bound_distance == MinNodeDistSq_(query_node, reference_node));

    ////////////////////////////////
    //// Begin Dual-Tree kNN Search
    ////////////////////////////////
    
    if (lower_bound_distance > query_node->stat().max_distance_so_far()) {
      // Pruned by distance
      number_of_prunes_++;
    }
    // node->is_leaf() works as one would expect
    else if (query_node->is_leaf() && reference_node->is_leaf()) {
      // Base Case
      ComputeBaseCase_(query_node, reference_node);
    }
    // Only query is a leaf
    else if (query_node->is_leaf()) {
      double left_distance = MinNodeDistSq_(query_node, reference_node->left());
      double right_distance = MinNodeDistSq_(query_node, reference_node->right());
      ct_dist_comp += 2;
      /* For Depth First Search, we order the computations by distance  */
      if (left_distance < right_distance) {
	// DFS
        DualTreeNeighborsRecursion_(query_node, reference_node->left(), left_distance);
	// backtracking
        DualTreeNeighborsRecursion_(query_node, reference_node->right(), right_distance);
      }
      else {
	// DFS
        DualTreeNeighborsRecursion_(query_node, reference_node->right(), right_distance);
	// backtracking
        DualTreeNeighborsRecursion_(query_node, reference_node->left(), left_distance);
      }
      
    }
    // Only reference is a leaf 
    else if (reference_node->is_leaf()) {
      double left_distance = MinNodeDistSq_(query_node->left(), reference_node);
      double right_distance = MinNodeDistSq_(query_node->right(), reference_node);
      ct_dist_comp += 2;
      DualTreeNeighborsRecursion_(query_node->left(), reference_node, left_distance);
      DualTreeNeighborsRecursion_(query_node->right(), reference_node, right_distance);
      // We need to update the upper bound based on the new upper bounds of the children
      query_node->stat().set_max_distance_so_far(
          max(query_node->left()->stat().max_distance_so_far(),
              query_node->right()->stat().max_distance_so_far()));
    } 
    // Neither reference nor query node is a leaf. Recurse on both as above
    else {
      double left_distance = MinNodeDistSq_(query_node->left(), reference_node->left());
      double right_distance = MinNodeDistSq_(query_node->left(), reference_node->right());
      ct_dist_comp += 2;
      if (left_distance < right_distance) {
	// DFS
        DualTreeNeighborsRecursion_(query_node->left(), reference_node->left(), left_distance);
	// backtracking
        DualTreeNeighborsRecursion_(query_node->left(), reference_node->right(), right_distance);
      }
      else {
	// DFS
        DualTreeNeighborsRecursion_(query_node->left(), reference_node->right(), right_distance);
	// backtracking
        DualTreeNeighborsRecursion_(query_node->left(), reference_node->left(), left_distance);
      }
      left_distance = MinNodeDistSq_(query_node->right(), reference_node->left());
      right_distance = MinNodeDistSq_(query_node->right(), reference_node->right());
      ct_dist_comp += 2;
      
      if (left_distance < right_distance) {
	// DFS
        DualTreeNeighborsRecursion_(query_node->right(), reference_node->left(), left_distance);
	// backtracking
        DualTreeNeighborsRecursion_(query_node->right(), reference_node->right(), right_distance);
      }
      else {
	// DFS
        DualTreeNeighborsRecursion_(query_node->right(), reference_node->right(), right_distance);
	// backtracking
        DualTreeNeighborsRecursion_(query_node->right(), reference_node->left(), left_distance);
      }
      
      // Update the upper bound as above
      query_node->stat().set_max_distance_so_far(
          max(query_node->left()->stat().max_distance_so_far(),
              query_node->right()->stat().max_distance_so_far()));
      
    }
    
  }
  

  ////////////////////////////////// Public Functions ////////////////////////////////////////////////
  

  /**
   * Initializes the AllNN structure for brute force NN search
   * This means that we simply ignore all the tree building.
   */
  void BruteForceInit(const Matrix& queries_in, const Matrix& references_in, index_t knns){
    
    queries_.Copy(queries_in);
    references_.Copy(references_in);
    knns_=knns;
    
    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());
    
    neighbor_indices_.Init(queries_.n_cols()*knns_);
    neighbor_distances_.Init(queries_.n_cols()*knns_);
    neighbor_distances_.SetAll(DBL_MAX);
    
    // Set leaf_size_ to be large enough that each tree has only one node
    leaf_size_ = max(queries_.n_cols(), references_.n_cols());
    
    query_tree_ = tree::MakeKdTreeMidpoint<TreeType>(queries_, leaf_size_, &old_from_new_queries_, NULL);
    reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_, leaf_size_, &old_from_new_references_, NULL);
  }

  /******************************************
   * Does the brute force k-nearest neighbors search and stores them in results
   *****************************************/
  void BruteForceAllkNN(ArrayList<index_t>* resulting_neighbors, ArrayList<double>*  distances) {
    resulting_neighbors->Init(neighbor_indices_.size());
    distances->Init(neighbor_distances_.length());
    
    ComputeBaseCase_(query_tree_, reference_tree_);
    // map the indices back from how they have been permuted
    for (index_t i = 0; i < neighbor_indices_.size(); i++) {
      (*resulting_neighbors)[old_from_new_references_[i/knns_]*knns_+ i%knns_]= old_from_new_references_[neighbor_indices_[i]];
      (*distances)[old_from_new_references_[i/knns_]*knns_+ i%knns_]= neighbor_distances_[i];
    }
  }


  /**
  * Setup the class and build the trees.  Note: we are initializing with const references to prevent 
  * local copies of the data.
  */
    void kdTreeInit(const Matrix& queries_in, const Matrix& references_in, index_t leaf_size, index_t knns) {
    // track the number of prunes
    number_of_prunes_ = 0;
    
    // Make sure the leaf size is valid
    leaf_size_ = leaf_size;
    DEBUG_ASSERT(leaf_size_ > 0);
    
    // Make sure the knns is valid
    knns_ = knns;
    DEBUG_ASSERT(knns_ > 0);
    // Copy the matrices to the class members since they will be rearranged.  
    queries_.Copy(queries_in);
    references_.Copy(references_in);
    //queries_train_.Copy(queries_train_in);
    
    // The data sets need to have the same number of points
    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());
    
    // Initialize the list of nearest neighbor candidates
    neighbor_indices_.Init(queries_.n_cols() * knns_);
    
    // Initialize the vector of upper bounds for each point.  
    neighbor_distances_.Init(queries_.n_cols() * knns_);
    neighbor_distances_.SetAll(DBL_MAX);

    // This call makes each tree from a matrix, leaf size, and two arrays 
    // that record the permutation of the data points
    // Instead of NULL, it is possible to specify an array new_from_old_
    // construct dual trees for query and reference data
    if (tree_dual_single_ == 2 ) {
      query_tree_ = tree::MakeKdTreeMidpoint<TreeType>(queries_, leaf_size_, &old_from_new_queries_, NULL);
      reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_, leaf_size_, &old_from_new_references_, NULL);
    }
    // construct single tree for reference data, while store query data in a vector of single point trees
    else if (tree_dual_single_ == 1 ) {
      for (index_t i = 0; i < queries_in.n_cols(); i++) {
	Matrix query_point;
	queries_.MakeColumnSlice(i, 1, &query_point);
	TreeType *single_point_tree = tree::MakeKdTreeMidpoint<TreeType>(query_point, leaf_size_, 
          &old_from_new_queries_, NULL);

	query_tree_vec_.push_back(single_point_tree);
	old_from_new_queries_.Renew();
      }
      reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_, leaf_size_, &old_from_new_references_, NULL);
    }
    else {
      NOTIFY("Unknown tree mode! Usage: 2:dual tree, 1:single tree.");
      exit(1);
    }

  }



  /******************************************
   * Computes the kd-tree based k-nearest neighbors search and stores them in results
   *****************************************/
  void kdTreeAllkNN(ArrayList<index_t>* resulting_neighbors, ArrayList<double>* distances) {
    // initialize the results list before filling it
    resulting_neighbors->Init(neighbor_indices_.size());
    distances->Init(neighbor_distances_.length());

    // Start on the root of each tree
    if (tree_dual_single_ == 2) { // use dual tree (reference and query trees) to do all-k-NN search
      printf("Dual kd Tree all-k-NN search.\n");

      printf("Query data provided.\n");
      // dual tree NNS
      DualTreeNeighborsRecursion_(query_tree_, reference_tree_, MinNodeDistSq_(query_tree_, reference_tree_));
      // get results
      for (index_t i = 0; i < neighbor_indices_.size(); i++) {
	(*resulting_neighbors)[old_from_new_queries_[i/knns_]*knns_+ i%knns_]= old_from_new_references_[neighbor_indices_[i]];
	(*distances)[old_from_new_queries_[i/knns_]*knns_+ i%knns_]= neighbor_distances_[i];
      }
    }
    else if (tree_dual_single_ == 1) { // use single tree (reference tree only) to do all-k-NN search
      printf("Single kd Tree all-k-NN search\n");
      query_index_single_ = 0;

      printf("Query data provided.\n");
      // single tree NNS
      for (std::vector<TreeType*>::iterator ct = query_tree_vec_.begin(); ct < query_tree_vec_.end(); ++ct, ++query_index_single_) {
	//DualTreeNeighborsRecursion_(*ct, reference_tree_, MinNodeDistSq_(*ct, reference_tree_));
	DualTreeNeighborsRecursion_(*ct, reference_tree_, MinNodeDistSq_(*ct, reference_tree_));
      }
      // get results	
      for (index_t i = 0; i < neighbor_indices_.size(); i++) {
	index_t q = i/knns_;
	(*resulting_neighbors)[q*knns_+ i%knns_]= old_from_new_references_[neighbor_indices_[i]];
	(*distances)[q*knns_+ i%knns_] = neighbor_distances_[i];
      }
    }
    else {
      printf("Unknown tree mode! Usage: 2:dual tree, 1:single tree\n");
      exit(1);
    }
    
  }

  void Destruct() {
    for (std::vector<TreeType*>::iterator i = query_tree_vec_.begin(); i < query_tree_vec_.end(); i++) {
      if (*i != NULL) {
	delete *i;
      }
    }
    if (query_tree_ != NULL) {
      delete query_tree_;
    }
    if (reference_tree_ != NULL) {
      delete reference_tree_;
    }
    queries_.Destruct();
    references_.Destruct();
    old_from_new_queries_.Renew();
    old_from_new_references_.Renew();
    neighbor_distances_.Destruct();
    neighbor_indices_.Renew();
  }
   
}; //class AllNN


#endif
