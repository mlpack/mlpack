/**
 * @file allkfn.h
 *
 * Defines AllNN class to perform all-nearest-neighbors on two specified 
 * data sets.
 */

// inclusion guards, please add them to your .h files
#ifndef ALLKFN_H
#define ALLKFN_H

// We need to include fastlib.  If you want to use fastlib, 
// you need to have this line in addition to
// the deplibs section of your build.py
#include <fastlib/fastlib.h>
#include <vector>
#include <functional>
/**
 * Forward declaration for the tester class
 */
class TestAllkFN;
/**
* Performs all-furthest-neighbors.  This class will build the trees and 
* perform the recursive  computation.
*/
class AllkFN {
  // Declare the tester class as a friend class so that it has access
  // to the private members of the class
  friend class TestAllkNFN;
  
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
      OT_MY_OBJECT(min_distance_so_far_); 
    } // OT_DEF_BASIC
    
   private:
    
    /**
    * The lower bound on the node's nearest furthest distances.
     */
    double min_distance_so_far_;
    
   public:
    
    double min_distance_so_far() {
      return min_distance_so_far_; 
    } 
    
    
    void set_min_distance_so_far(double new_dist) {
      min_distance_so_far_ = new_dist; 
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
      // The bound starts at zero
      min_distance_so_far_ = 0;
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
    
  }; //class AllkFNStat  
  
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
  index_t kfns_; 
   // The module containing the parameters for this computation. 
  struct datanode* module_;
  
  
  /////////////////////////////// Constructors /////////////////////////////////////////////
  
  // Add this at the beginning of a class to prevent accidentally calling the copy constructor
  FORBID_ACCIDENTAL_COPIES(AllkFN);
  

 public:

  /**
  * Constructors are generally very simple in FASTlib; most of the work is done by Init().  This is only
  * responsible for ensuring that the object is ready to be destroyed safely.  
  */
  AllkFN() {
    query_tree_ = NULL;
    reference_tree_ = NULL;
  } 
  
  /**
  * The tree is the only member we are responsible for deleting.  The others will take care of themselves.  
  */
  ~AllkFN() {
    if (query_tree_ != NULL) {
      delete query_tree_;
    }
    if (reference_tree_ != NULL) {
      delete reference_tree_;
    }
  } 
    
      
 /////////////////////////////// Helper Functions ///////////////////////////////////////////////////
  
  /**
   * Computes the maximum squared distance between the bounding boxes of two nodes
   */
  double MaxNodeDistSq_ (TreeType* query_node, TreeType* reference_node) {
    // node->bound() gives us the DHrectBound class for the node
    // It has a function MinDistanceSq which takes another DHrectBound
    return query_node->bound().MaxDistanceSq(reference_node->bound());
  } 
  
  
  /**
   * Performs exhaustive computation between two leaves.  
   */
  void ComputeBaseCase_(TreeType* query_node, TreeType* reference_node) {
   
    // DEBUG statements should be used frequently, since they incur no overhead
    // when compiled in fast mode
    
    // Check that the pointers are not NULL
    DEBUG_ASSERT(query_node != NULL);
    DEBUG_ASSERT(reference_node != NULL);
    // Check that we really should be in the base case
    DEBUG_WARN_IF(!query_node->is_leaf());
    DEBUG_WARN_IF(!reference_node->is_leaf());
    
    // Used to find the query node's new upper bound
    double query_min_neighbor_distance = DBL_MAX;
    std::vector<std::pair<double, index_t> > neighbors(kfns_);
    // node->begin() is the index of the first point in the node, 
    // node->end is one past the last index
    for (index_t query_index = query_node->begin(); 
         query_index < query_node->end(); query_index++) {
       
      // Get the query point from the matrix
      Vector query_point;
      queries_.MakeColumnVector(query_index, &query_point);
      
      index_t ind = query_index*kfns_;
      for(index_t i=0; i<kfns_; i++) {
        neighbors[i]=std::make_pair(neighbor_distances_[ind+i],
                                    neighbor_indices_[ind+i]);
      }
      // We'll do the same for the references
      for (index_t reference_index = reference_node->begin(); 
           reference_index < reference_node->end(); reference_index++) {

	      // Confirm that points do not identify themselves as neighbors
	      // in the monochromatic case
        if (likely(reference_node != query_node ||
		        reference_index != query_index)) {
	        Vector reference_point;
	        references_.MakeColumnVector(reference_index, &reference_point);
	        // We'll use lapack to find the distance between the two vectors
	        double distance =
	        la::DistanceSqEuclidean(query_point, reference_point);
	        // If the reference point is closer than the current candidate, 
	        // we'll update the candidate
	        if (distance > neighbor_distances_[ind+kfns_-1]) {
	          neighbors.push_back(std::make_pair(distance, reference_index));
          }
	      }
	    } // for reference_index
      std::sort(neighbors.begin(), neighbors.end(),
         std::greater<std::pair<double, index_t> >());
      for(index_t i=0; i<kfns_; i++) {
        neighbor_distances_[ind+i] = neighbors[i].first;
        neighbor_indices_[ind+i]  = neighbors[i].second;
      }
      neighbors.resize(kfns_);
      // We need to find the lower bound distance for this query node
      if (neighbor_distances_[ind+kfns_-1] < query_min_neighbor_distance) {
        query_min_neighbor_distance = neighbor_distances_[ind+kfns_-1]; 
      }
    }
    // for query_index 
    // Update the lower bound for the query_node
    query_node->stat().set_min_distance_so_far(query_min_neighbor_distance);
  } // ComputeBaseCase_
  
  
  /**
   * The recursive function
   */
  void ComputeNeighborsRecursion_ (TreeType* query_node, 
      TreeType* reference_node, 
      double higher_bound_distance) {
   
    // DEBUG statements should be used frequently, 
    // either with or without messages 
    
    // A DEBUG statement with no predefined message
    DEBUG_ASSERT(query_node != NULL);
    // A DEBUG statement with a predefined message
    DEBUG_ASSERT_MSG(reference_node != NULL, "reference node is null");
//    // Make sure the bounding information is correct
//    DEBUG_ASSERT(higher_bound_distance == MaxNodeDistSq_(query_node, 
//        reference_node));
    
    if (higher_bound_distance < query_node->stat().min_distance_so_far()) {
      // Pruned by distance
      number_of_prunes_++;
    }
    // node->is_leaf() works as one would expect
    else if (query_node->is_leaf() && reference_node->is_leaf()) {
      // Base Case
      ComputeBaseCase_(query_node, reference_node);
    }
    else if (query_node->is_leaf()) {
      // Only query is a leaf
      
      // We'll order the computation by distance 
      double left_distance = MaxNodeDistSq_(query_node, reference_node->left());
      double right_distance = MaxNodeDistSq_(query_node, reference_node->right());
      
      if (left_distance > right_distance) {
        ComputeNeighborsRecursion_(query_node, reference_node->left(), 
            left_distance);
        ComputeNeighborsRecursion_(query_node, reference_node->right(), 
            right_distance);
      }
      else {
        ComputeNeighborsRecursion_(query_node, reference_node->right(), 
            right_distance);
        ComputeNeighborsRecursion_(query_node, reference_node->left(), 
            left_distance);
      }
      
    }
    
    else if (reference_node->is_leaf()) {
      // Only reference is a leaf 
      
      double left_distance = MaxNodeDistSq_(query_node->left(), reference_node);
      double right_distance = MaxNodeDistSq_(query_node->right(), reference_node);
      
      ComputeNeighborsRecursion_(query_node->left(), reference_node, 
          left_distance);
      ComputeNeighborsRecursion_(query_node->right(), reference_node, 
          right_distance);
      
      // We need to update the upper bound based on the new upper bounds of 
      // the children
      query_node->stat().set_min_distance_so_far(
          min(query_node->left()->stat().min_distance_so_far(),
              query_node->right()->stat().min_distance_so_far()));
    } else {
      // Recurse on both as above
      
      double left_distance = MaxNodeDistSq_(query_node->left(), 
          reference_node->left());
      double right_distance = MaxNodeDistSq_(query_node->left(), 
          reference_node->right());
      
      if (left_distance > right_distance) {
        ComputeNeighborsRecursion_(query_node->left(), reference_node->left(), 
            left_distance);
        ComputeNeighborsRecursion_(query_node->left(), reference_node->right(), 
            right_distance);
      }
      else {
        ComputeNeighborsRecursion_(query_node->left(), reference_node->right(), 
            right_distance);
        ComputeNeighborsRecursion_(query_node->left(), reference_node->left(), 
            left_distance);
      }
      left_distance = MaxNodeDistSq_(query_node->right(), reference_node->left());
      right_distance = MaxNodeDistSq_(query_node->right(), 
          reference_node->right());
      
      if (left_distance > right_distance) {
        ComputeNeighborsRecursion_(query_node->right(), reference_node->left(), 
            left_distance);
        ComputeNeighborsRecursion_(query_node->right(), reference_node->right(), 
            right_distance);
      }
      else {
        ComputeNeighborsRecursion_(query_node->right(), reference_node->right(), 
            right_distance);
        ComputeNeighborsRecursion_(query_node->right(), reference_node->left(), 
            left_distance);
      }
      
      // Update the upper bound as above
      query_node->stat().set_min_distance_so_far(
          min(query_node->left()->stat().min_distance_so_far(),
              query_node->right()->stat().min_distance_so_far()));
      
    }
    
  } // ComputeNeighborsRecursion_
  
  
   
  
  ////////////////////////////////// Public Functions ////////////////////////////////////////////////
  
  /**
  * Setup the class and build the trees.  Note: we are initializing with const references to prevent 
  * local copies of the data.
  */

   void Init(const Matrix& queries_in, const Matrix& references_in, struct datanode* module_in) {
    
    // set the module
    module_ = module_in;
    
    // track the number of prunes
    number_of_prunes_ = 0;
    
    // Get the leaf size from the module
    leaf_size_ = fx_param_int(module_, "leaf_size", 20);
    // Make sure the leaf size is valid
    DEBUG_ASSERT(leaf_size_ > 0);
    
    // Copy the matrices to the class members since they will be rearranged.  
    queries_.Copy(queries_in);
    references_.Copy(references_in);
    
    // The data sets need to have the same number of points
    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());
    
		// K-nearest neighbors initialization
		kfns_ = fx_param_int(module_, "kfns", 1);
  
    // Initialize the list of nearest neighbor candidates
    neighbor_indices_.Init(queries_.n_cols() * kfns_);
    
		// Initialize the vector of upper bounds for each point.  
    neighbor_distances_.Init(queries_.n_cols() * kfns_);
    neighbor_distances_.SetAll(0);

    // We'll time tree building
    fx_timer_start(module_, "tree_building");

    // This call makes each tree from a matrix, leaf size, and two arrays 
		// that record the permutation of the data points
    // Instead of NULL, it is possible to specify an array new_from_old_
    query_tree_ = tree::MakeKdTreeMidpoint<TreeType>(queries_, leaf_size_, 
				&old_from_new_queries_, NULL);
    reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_, 
				leaf_size_, &old_from_new_references_, NULL);
    
    // Stop the timer we started above
    fx_timer_stop(module_, "tree_building");

  } // Init

  /** Use this if you want to run allknn it on a single dataset 
   * the query tree and reference tree are the same
   */
  void Init(const Matrix& references_in, struct datanode* module_in) {
     
    // set the module
    module_ = module_in;
    
    // track the number of prunes
    number_of_prunes_ = 0;
    
    // Get the leaf size from the module
    leaf_size_ = fx_param_int(module_, "leaf_size", 20);
    // Make sure the leaf size is valid
    DEBUG_ASSERT(leaf_size_ > 0);
    
    // Copy the matrices to the class members since they will be rearranged.  
    references_.Copy(references_in);
    queries_.Alias(references_);    
		// K-nearest neighbors initialization
		kfns_ = fx_param_int(module_, "kfns", 1);
  
    // Initialize the list of nearest neighbor candidates
    neighbor_indices_.Init(references_.n_cols() * kfns_);
    
		// Initialize the vector of upper bounds for each point.  
    neighbor_distances_.Init(references_.n_cols() * kfns_);
    neighbor_distances_.SetAll(0.0);

    // We'll time tree building
    fx_timer_start(module_, "tree_building");

    // This call makes each tree from a matrix, leaf size, and two arrays 
		// that record the permutation of the data points
    // Instead of NULL, it is possible to specify an array new_from_old_
    query_tree_ = NULL;
    reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_, 
				leaf_size_, &old_from_new_references_, NULL);
    
    // Stop the timer we started above
    fx_timer_stop(module_, "tree_building");

  }
  void Init(const Matrix& queries_in, const Matrix& references_in, 
      index_t leaf_size, index_t kfns) {
    
    // track the number of prunes
    number_of_prunes_ = 0;
    
    // Make sure the leaf size is valid
    leaf_size_ = leaf_size;
    DEBUG_ASSERT(leaf_size_ > 0);
    
    // Make sure the kfns is valid
    kfns_ = kfns;
    DEBUG_ASSERT(kfns_ > 0);
    // Copy the matrices to the class members since they will be rearranged.  
    queries_.Copy(queries_in);
    references_.Copy(references_in);
    
    // The data sets need to have the same number of points
    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());
    
  
    // Initialize the list of nearest neighbor candidates
    neighbor_indices_.Init(queries_.n_cols() * kfns_);
    
    // Initialize the vector of upper bounds for each point.  
    neighbor_distances_.Init(queries_.n_cols() * kfns_);
    neighbor_distances_.SetAll(0);


    // This call makes each tree from a matrix, leaf size, and two arrays 
    // that record the permutation of the data points
    // Instead of NULL, it is possible to specify an array new_from_old_
    query_tree_ = tree::MakeKdTreeMidpoint<TreeType>(queries_, leaf_size_, 
        &old_from_new_queries_, NULL);
    reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_, 
        leaf_size_, &old_from_new_references_, NULL);

  } // Init

  void Init(const Matrix& references_in, index_t leaf_size, index_t kfns) {
    // track the number of prunes
    number_of_prunes_ = 0;
    
    // Make sure the leaf size is valid
    leaf_size_ = leaf_size;
    DEBUG_ASSERT(leaf_size_ > 0);
    
    // Make sure the kfns is valid
    kfns_ = kfns;
    DEBUG_ASSERT(kfns_ > 0);
    // Copy the matrices to the class members since they will be rearranged.  
    references_.Copy(references_in);
    queries_.Alias(references_); 
  
    // Initialize the list of nearest neighbor candidates
    neighbor_indices_.Init(references_.n_cols() * kfns_);
    
    // Initialize the vector of upper bounds for each point.  
    neighbor_distances_.Init(references_.n_cols() * kfns_);
    neighbor_distances_.SetAll(0.0);


    // This call makes each tree from a matrix, leaf size, and two arrays 
    // that record the permutation of the data points
    // Instead of NULL, it is possible to specify an array new_from_old_
    query_tree_ = NULL;
    reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(references_, 
        leaf_size_, &old_from_new_references_, NULL);
   // This is an annoying feature of fastlib
    old_from_new_queries_.Init();
  }

  void Destruct() {
    queries_.Destruct();
    references_.Destruct();
    old_from_new_queries_.Renew();
    old_from_new_references_.Renew();
    neighbor_distances_.Destruct();
    neighbor_indices_.Renew();
    if (query_tree_ != NULL) {
      delete query_tree_;
      query_tree_=NULL;
    }
    if (reference_tree_ != NULL) {
      delete reference_tree_;
      reference_tree_=NULL;
    }
  } 
  
  /**
   * Initializes the AllNN structure for naive computation.  
   * This means that we simply ignore the tree building.
   */
  void InitNaive(const Matrix& queries_in, 
      const Matrix& references_in, index_t kfns){
    
    queries_.Copy(queries_in);
    references_.Copy(references_in);
    kfns_=kfns;
    
    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());
    
    neighbor_indices_.Init(queries_.n_cols()*kfns_);
    neighbor_distances_.Init(queries_.n_cols()*kfns_);
    neighbor_distances_.SetAll(0.0);
    
    // The only difference is that we set leaf_size_ to be large enough 
    // that each tree has only one node
    leaf_size_ = max(queries_.n_cols(), references_.n_cols());
    
    query_tree_ = tree::MakeKdTreeMidpoint<TreeType>(queries_, 
        leaf_size_, &old_from_new_queries_, NULL);
    reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(
        references_, leaf_size_, &old_from_new_references_, NULL);
        
  } // InitNaive
  
   void InitNaive(const Matrix& references_in, index_t kfns){
    
    references_.Copy(references_in);
    queries_.Alias(references_);
    kfns_=kfns;
    
    neighbor_indices_.Init(references_.n_cols()*kfns_);
    neighbor_distances_.Init(references_.n_cols()*kfns_);
    neighbor_distances_.SetAll(0.0);
    
    // The only difference is that we set leaf_size_ to be large enough 
    // that each tree has only one node
    leaf_size_ = references_.n_cols();
    
    query_tree_ = NULL;
    reference_tree_ = tree::MakeKdTreeMidpoint<TreeType>(
        references_, leaf_size_, &old_from_new_references_, NULL);
    // This is an annoying feature of fastlib
    old_from_new_queries_.Init();
  } // InitNaive
  
  /**
   * Computes the nearest neighbors and stores them in *results
   */
  void ComputeNeighbors(ArrayList<index_t>* resulting_neighbors,
                        ArrayList<double>* distances) {
    
    // Start on the root of each tree
    if (query_tree_!=NULL) {
      ComputeNeighborsRecursion_(query_tree_, reference_tree_, 
          MaxNodeDistSq_(query_tree_, reference_tree_));
    } else {
       ComputeNeighborsRecursion_(reference_tree_, reference_tree_, 
          MaxNodeDistSq_(reference_tree_, reference_tree_));
    }
    
    // We need to initialize the results list before filling it
    resulting_neighbors->Init(neighbor_indices_.size());
    distances->Init(neighbor_distances_.length());
    // We need to map the indices back from how they have 
    // been permuted
    if (query_tree_ != NULL) {
      for (index_t i = 0; i < neighbor_indices_.size(); i++) {
        (*resulting_neighbors)[
          old_from_new_queries_[i/kfns_]*kfns_+ i%kfns_] = 
          old_from_new_references_[neighbor_indices_[i]];
        (*distances)[
          old_from_new_queries_[i/kfns_]*kfns_+ i%kfns_] = 
          neighbor_distances_[i];
      }
    } else {
      for (index_t i = 0; i < neighbor_indices_.size(); i++) {
        (*resulting_neighbors)[
          old_from_new_references_[i/kfns_]*kfns_+ i%kfns_] = 
          old_from_new_references_[neighbor_indices_[i]];
        (*distances)[
          old_from_new_references_[i/kfns_]*kfns_+ i%kfns_] = 
          neighbor_distances_[i];
      }
    }
  } // ComputeNeighbors
  
  
  /**
   * Does the entire computation naively
   */
  void ComputeNaive(ArrayList<index_t>* resulting_neighbors,
                    ArrayList<double>*  distances) {
    if (query_tree_!=NULL) {
      ComputeBaseCase_(query_tree_, reference_tree_);
    } else {
       ComputeBaseCase_(reference_tree_, reference_tree_);
    }

    // The same code as above
    resulting_neighbors->Init(neighbor_indices_.size());
    distances->Init(neighbor_distances_.length());
    // We need to map the indices back from how they have 
    // been permuted
    for (index_t i = 0; i < neighbor_indices_.size(); i++) {
      (*resulting_neighbors)[
        old_from_new_references_[i/kfns_]*kfns_+ i%kfns_] = 
        old_from_new_references_[neighbor_indices_[i]];
      (*distances)[
        old_from_new_references_[i/kfns_]*kfns_+ i%kfns_] = 
        neighbor_distances_[i];

    }
  }
   
}; //class AllNN


#endif
// end inclusion guards
