/**
 * @file exact_max_ip.h
 *
 */

#ifndef EXACT_MAX_IP_H
#define EXACT_MAX_IP_H

#include <assert.h>
#include <fastlib/fastlib.h>
#include <vector>
#include <armadillo>
#include "general_spacetree.h"
#include "gen_metric_tree.h"

using namespace mlpack;

PARAM_MODULE("maxip", "Parameters for the class that "
	     "builds a tree on the reference set and "
	     "searches for the maximum inner product "
	     "by the branch-and-bound method.");

PARAM_INT("knns", "The number of top innner products required",
	  "maxip", 1);
PARAM_DOUBLE("tau", "The rank error in terms of the \% of "
	     "reference set size", "maxip", 1.0);
PARAM_DOUBLE("alpha", "The error probability", "maxip", 0.95);
PARAM_INT("leaf_size", "The leaf size for the ball-tree", 
	  "maxip", 20);

PARAM_FLAG("angle_prune", "The flag to trigger the tighter"
	   " pruning using the angles as well", "maxip");


//   {"tree_building", FX_TIMER, FX_CUSTOM, NULL,
//    " The timer to record the time taken to build" 
//    " the query and the reference tree.\n"},
//   {"tree_building_approx", FX_TIMER, FX_CUSTOM, NULL,
//    " The timer to record the time taken to build" 
//    " the query and the reference tree for InitApprox.\n"},
//   {"computing_sample_sizes", FX_TIMER, FX_CUSTOM, NULL,
//    " The timer to compute the sample sizes.\n"},


/**
 * Performs all-nearest-neighbors.  This class will build the trees and 
 * perform the recursive  computation.
 */
class MaxIP {
  
//   //////////////////////////// Nested Classes /////////////////////////
//   class QueryStat {
//   private:
//     index_t test_;

//   public:
//     index_t test() { return test_; }

//     void Init() {
//       test_ = 0;
//     }
//   }; // QueryStat

  // TreeType are BinarySpaceTrees where the data are bounded by 
  // Euclidean bounding boxes, the data are stored in a Matrix, 
  // and each node has a QueryStat for its bound.
  // typedef GeneralBinarySpaceTree<DBallBound< kernel::LMetric<2>, arma::vec>, arma::mat> TreeType;
  typedef GeneralBinarySpaceTree<DBallBound<>, arma::mat> TreeType;
   
  
  /////////////////////////////// Members ////////////////////////////
private:
  // These will store our data sets.
  arma::mat queries_;
  arma::mat references_;
  // This will store the query index for the single tree run
  size_t query_;
  double query_norm_;
  // Pointers to the roots of the two trees.
  TreeType* reference_tree_;
  // The total number of prunes.
  size_t number_of_prunes_;
  // A permutation of the indices for tree building.
  // ArrayList<size_t> old_from_new_queries_;
  arma::Col<size_t> old_from_new_references_;
  // The number of points in a leaf
  size_t leaf_size_;
  // The distance to the candidate nearest neighbor for each query
  arma::vec max_ips_;
  // The indices of the candidate nearest neighbor for each query
  arma::Col<size_t> max_ip_indices_;
  // number of nearest neighbrs
  size_t knns_; 
  // The total number of distance computations
  size_t distance_computations_;
  // The total number of split decisions
  size_t split_decisions_;


  /////////////////////////////// Constructors ////////////////////////
  
  // Add this at the beginning of a class to prevent accidentally
  // calling the copy constructor
  // FORBID_ACCIDENTAL_COPIES(MaxIP);
  
public:
  /**
   * Constructors are generally very simple in FASTlib;
   * most of the work is done by Init().  This is only
   * responsible for ensuring that the object is ready
   * to be destroyed safely.  
   */
  MaxIP() {
    reference_tree_ = NULL;
    //query_trees_.clear();
  } 
  
  /**
   * The tree is the only member we are responsible for deleting.
   * The others will take care of themselves.  
   */
  ~MaxIP() {
    if (reference_tree_ != NULL) {
      delete reference_tree_;
    }
  }
    
  /////////////////////////// Helper Functions //////////////////////
  
private:
  /**
   * Computes the minimum squared distance between the
   * bounding boxes of two nodes
   */
  double MaxNodeIP_(TreeType* reference_node) {
    // CHECK IF IT ACTUALLY WORKS


    // counting the split decisions 
    split_decisions_++;

    // compute maximum possible inner product 
    // between a point and a ball in terms of 
    // the ball's center and radius
    arma::vec q = queries_.col(query_);
    arma::vec centroid = reference_node->bound().center();
    double c_norm = arma::norm(centroid, 2);

    double rad = std::sqrt(reference_node->bound().radius());

    double max_cos_qr = 1.0;

    if (IO::HasParam("maxip/angle_prune")) { 
      // tighter bound of \max_{r \in B_p^R} <q,r> 
      //    = |q| \max_{r \in B_p^R} |r| cos <qr 
      //    \leq |q| \max_{r \in B_p^R} |r| \max_{r \in B_p^R} cos <qr 
      //    \leq |q| (|p|+R) if <qp \leq \max_r <pr
      //    \leq |q| (|p|+R) cos( <qp - \max_r <pr ) otherwise

      if (rad <= c_norm) {
	double cos_qp = arma::dot(q, centroid) 
	  / (query_norm_ * c_norm);
	double sin_qp = std::sqrt(1 - cos_qp * cos_qp);

	double max_sin_pr = rad / c_norm;
	double min_cos_pr = std::sqrt(1 - max_sin_pr * max_sin_pr);

	if (min_cos_pr > cos_qp) { // <qp \geq \max_r <pr
	  // cos( <qp - <pr ) = cos <qp * cos <pr + sin <qp * sin <pr
	  double cos_qp_max_pr = (cos_qp * min_cos_pr) 
	    + (sin_qp * max_sin_pr);

	  max_cos_qr = std::max(cos_qp_max_pr, 0.0);
	}
      }

    }

    // Otherwise :
    // simple bound of \max_{r \in B_p^R} <q,r> 
    //    = |q| \max_{r \in B_p^R} |r| cos <qr 
    //    \leq |q| \max_{r \in B_p^R} |r| \leq |q| (|p|+R)

    return (query_norm_ * (c_norm + rad) * max_cos_qr);
  } 


  /**
   * Performs exhaustive computation between two leaves.  
   */
  void ComputeBaseCase_(TreeType* reference_node) {
   
    // Check that the pointers are not NULL
    assert(reference_node != NULL);
    // assert(reference_node->is_leaf());
    assert(query_ >= 0);
    assert(query_ < queries_.n_cols);

//     // Check that we really should be in the base case
//     DEBUG_WARN_IF(!reference_node->is_leaf());

    // Used to find the query node's new upper bound
    // query_min_ip = DBL_MAX;
    
    std::vector<std::pair<double, size_t> > candidates(knns_);

    // Get the query point from the matrix
    arma::vec q = queries_.col(query_); 

    size_t ind = query_*knns_;
    for(size_t i = 0; i < knns_; i++)
      candidates[i] = std::make_pair(max_ips_(ind+i),
				     max_ip_indices_(ind+i));
    
    // We'll do the same for the references
    for (size_t reference_index = reference_node->begin(); 
	 reference_index < reference_node->end(); reference_index++) {

      // Confirm that points do not identify themselves as neighbors
      // in the monochromatic case
      arma::vec rpoint = references_.col(reference_index);

      // We'll use arma to find the inner product of the two vectors
      double ip = arma::dot(q, rpoint);
      // If the reference point is greater than the current candidate, 
      // we'll update the candidate
      if (ip > max_ips_(ind+knns_-1)) {
	candidates.push_back(std::make_pair(ip, reference_index));
      }
    } // for reference_index

    std::sort(candidates.begin(), candidates.end());
    std::reverse(candidates.begin(), candidates.end());
    for(size_t i = 0; i < knns_; i++) {
      max_ips_(ind+i) = candidates[i].first;
      max_ip_indices_(ind+i) = candidates[i].second;
    }
    candidates.clear();

    // for now the query lower bounds are accessed from 
    // the variable 'max_ips_(query_ * knns_ + knns_ - 1)'
    
    // We need to find the upper bound distance for this query node
    // if (max_ips_(ind+knns_-1) < query_min_ip) {
    //   query_min_ip = max_ips_(ind+knns_-1); 
    // }

    // Update the upper bound for the query_node
    // query_node->stat().set_max_distance_so_far(query_max_neighbor_distance);
    distance_computations_ 
      += reference_node->end() - reference_node->begin();
         
  } // ComputeBaseCase_
  
  
  /**
   * The recursive function
   */
  void ComputeNeighborsRecursion_(TreeType* reference_node, 
				  double upper_bound_ip) {

    //    assert(query_node != NULL);
    assert(reference_node != NULL);
    assert(upper_bound_ip == MaxNodeIP_(reference_node));

    // just checking for the single tree version
    // assert(query_node->end()
    //	 - query_node->begin() == 1);
    
    if (upper_bound_ip < max_ips_((query_*knns_) + knns_ -1)) { 
      // Pruned by distance
      number_of_prunes_++;
//       IO::Info << "Current best: " << max_ips_((query_ * knns_) + knns_ -1)
// 	       << std::endl << "Node best: " << upper_bound_ip << std::endl;
    }
//     // node->is_leaf() works as one would expect
//     else if (query_node->is_leaf() && reference_node->is_leaf()) {
//       // Base Case
//     //  ComputeBaseCase_(query_node, reference_node);
//     }
//     else if (query_node->is_leaf()) {
//       // Only query is a leaf
      
//       // We'll order the computation by distance 
//       double left_distance = MinNodeDistSq_(query_node,
// 					    reference_node->left());
//       double right_distance = MinNodeDistSq_(query_node,
// 					     reference_node->right());
      
//       if (left_distance < right_distance) {
//         ComputeNeighborsRecursion_(query_node, reference_node->left(), 
// 				   left_distance);
//         ComputeNeighborsRecursion_(query_node, reference_node->right(), 
// 				   right_distance);
//       }
//       else {
//         ComputeNeighborsRecursion_(query_node, reference_node->right(), 
// 				   right_distance);
//         ComputeNeighborsRecursion_(query_node, reference_node->left(), 
// 				   left_distance);
//       }
//     }
    
    else if (reference_node->is_leaf()) {
      // base case for the single tree case
      ComputeBaseCase_(reference_node);

//       // Only reference is a leaf 
//       double left_distance
// 	= MinNodeDistSq_(query_node->left(), reference_node);
//       double right_distance
// 	= MinNodeDistSq_(query_node->right(), reference_node);
      
//       ComputeNeighborsRecursion_(query_node->left(), reference_node, 
// 				 left_distance);
//       ComputeNeighborsRecursion_(query_node->right(), reference_node, 
// 				 right_distance);
      
//       // We need to update the upper bound based on the new upper bounds of 
//       // the children
//       query_node->stat().set_max_distance_so_far(max(query_node->left()->stat().max_distance_so_far(),
// 						     query_node->right()->stat().max_distance_so_far()));
    } else {
      // Recurse on both as above
      double left_ip = MaxNodeIP_(reference_node->left());
      double right_ip = MaxNodeIP_(reference_node->right());

      if (left_ip > right_ip) {
	ComputeNeighborsRecursion_(reference_node->left(), 
				   left_ip);
	ComputeNeighborsRecursion_(reference_node->right(),
				   right_ip);
      } else {
	ComputeNeighborsRecursion_(reference_node->right(),
				   right_ip);
	ComputeNeighborsRecursion_(reference_node->left(), 
				   left_ip);
      }
    }      
//       double left_distance = MinNodeDistSq_(query_node->left(), 
// 					    reference_node->left());
//       double right_distance = MinNodeDistSq_(query_node->left(), 
// 					     reference_node->right());
      
//       if (left_distance < right_distance) {
//         ComputeNeighborsRecursion_(query_node->left(),
// 				   reference_node->left(), 
// 				   left_distance);
//         ComputeNeighborsRecursion_(query_node->left(),
// 				   reference_node->right(), 
// 				   right_distance);
//       } else {
//         ComputeNeighborsRecursion_(query_node->left(),
// 				   reference_node->right(), 
// 				   right_distance);
//         ComputeNeighborsRecursion_(query_node->left(),
// 				   reference_node->left(), 
// 				   left_distance);
//       }

//       left_distance = MinNodeDistSq_(query_node->right(),
// 				     reference_node->left());
//       right_distance = MinNodeDistSq_(query_node->right(), 
// 				      reference_node->right());
      
//       if (left_distance < right_distance) {
//         ComputeNeighborsRecursion_(query_node->right(),
// 				   reference_node->left(), 
// 				   left_distance);
//         ComputeNeighborsRecursion_(query_node->right(),
// 				   reference_node->right(), 
// 				   right_distance);
//       } else {
//         ComputeNeighborsRecursion_(query_node->right(),
// 				   reference_node->right(), 
// 				   right_distance);
//         ComputeNeighborsRecursion_(query_node->right(),
// 				   reference_node->left(), 
// 				   left_distance);
//       }
      
//       // Update the upper bound as above
//       query_node->stat().set_max_distance_so_far(max(query_node->left()->stat().max_distance_so_far(),
// 						     query_node->right()->stat().max_distance_so_far()));
      
//    }
  } // ComputeNeighborsRecursion_
  

  /////////////// Public Functions ////////////////////
public:
  /**
   * Setup the class and build the trees.
   * Note: we are initializing with const references to prevent 
   * local copies of the data.
   */
  void Init(const arma::mat& queries_in,
	    const arma::mat& references_in) {
    
    
    // track the number of prunes and computations
    number_of_prunes_ = 0;
    distance_computations_ = 0;
    split_decisions_ = 0;
    
    // Get the leaf size from the module
    leaf_size_ = IO::GetParam<int>("maxip/leaf_size");
    // Make sure the leaf size is valid
    assert(leaf_size_ > 0);
    
    // Copy the matrices to the class members since they will be rearranged.  
    queries_ = queries_in;
    references_ = references_in;
    
    // The data sets need to have the same number of points
    assert(queries_.n_rows == references_.n_rows);
    
    // K-nearest neighbors initialization
    knns_ = IO::GetParam<int>("maxip/knns");
  
    // Initialize the list of nearest neighbor candidates
    max_ip_indices_ 
      = -1 * arma::ones<arma::Col<size_t> >(queries_.n_cols * knns_, 1);
    
    // Initialize the vector of upper bounds for each point.
    // We do not consider negative values for inner products.
    max_ips_ = 0.0 * arma::ones<arma::vec>(queries_.n_cols * knns_, 1);

    // We'll time tree building
    IO::StartTimer("tree_building");

    // This call makes each tree from a matrix, leaf size, and two arrays 
    // that record the permutation of the data points
    // Instead of NULL, it is possible to specify an array new_from_old_

//     // Here we need to change the query tree into N single-point
//     // query trees
//     for (size_t i = 0; i < queries_.n_cols(); i++) {
//       Matrix query;
//       queries_.MakeColumnSlice(i, 1, &query);
//       TreeType *single_point_tree
// 	= tree::MakeKdTreeMidpoint<TreeType>(query,
// 					     leaf_size_, 
// 					     &old_from_new_queries_,
// 					     NULL);
//       query_trees_.push_back(single_point_tree);
//       old_from_new_queries_.Renew();
//     }

    reference_tree_
      = proximity::MakeGenMetricTree<TreeType>(references_, 
					       leaf_size_,
					       &old_from_new_references_,
					       NULL);
    
    // Stop the timer we started above
    IO::StopTimer("tree_building");

//     // initializing the sample_sizes_
//     sample_sizes_.Init();
  } // Init

  void Destruct() {
//     for (std::vector<TreeType*>::iterator it = query_trees_.begin();
// 	 it < query_trees_.end(); it++) {
//       if (*it != NULL) {
// 	delete *it;
//       }
//     }
    if (reference_tree_ != NULL) {
      delete reference_tree_;
    }
//     queries_.Destruct();
//     references_.Destruct();
//     old_from_new_queries_.Renew();
//     old_from_new_references_.Renew();
//     neighbor_distances_.Destruct();
//     neighbor_indices_.Renew();

//     sample_sizes_.Renew();
  }

  /**
   * Initializes the AllNN structure for naive computation.  
   * This means that we simply ignore the tree building.
   */
  void InitNaive(const arma::mat& queries_in, 
		 const arma::mat& references_in) {
    
    queries_ = queries_in;
    references_ = references_in;
    
    // track the number of prunes and computations
    number_of_prunes_ = 0;
    distance_computations_ = 0;
    split_decisions_ = 0;
    
    // The data sets need to have the same number of dimensions
    assert(queries_.n_rows == references_.n_rows);
    
    // K-nearest neighbors initialization
    knns_ = IO::GetParam<int>("maxip/knns");
  
    // Initialize the list of nearest neighbor candidates
    max_ip_indices_
      = -1 * arma::ones<arma::Col<size_t> >(queries_.n_cols * knns_, 1);
    
    // Initialize the vector of upper bounds for each point.
    // We do not consider negative values for inner products.
    max_ips_ = 0.0 * arma::ones<arma::vec>(queries_.n_cols * knns_, 1);

    // The only difference is that we set leaf_size_ to be large enough 
    // that each tree has only one node
    leaf_size_ = std::max(queries_.n_cols, references_.n_cols);

    // We'll time tree building
    IO::StartTimer("tree_building");


    // Here we need to change the query tree into N single-point
    // query trees
//     for (size_t i = 0; i < queries_.n_cols(); i++) {
//       Matrix query;
//       queries_.MakeColumnSlice(i, 1, &query);
//       TreeType *single_point_tree
// 	= tree::MakeKdTreeMidpoint<TreeType>(query,
// 					     leaf_size_, 
// 					     &old_from_new_queries_,
// 					     NULL);
//       query_trees_.push_back(single_point_tree);
//       old_from_new_queries_.Renew();
//     }
    
    reference_tree_
      = proximity::MakeGenMetricTree<TreeType>(references_, 
					       leaf_size_,
					       &old_from_new_references_,
					       NULL);

    // Stop the timer we started above
    IO::StopTimer("tree_building");
    
//     // initialiazing the sample_sizes_
//     sample_sizes_.Init();
  } // InitNaive
  
  
  /**
   * Computes the nearest neighbors and stores them in *results
   */
  void ComputeNeighbors(arma::Col<size_t>* resulting_neighbors,
                        arma::vec* ips) {

    // Start on the root of each tree
    // the index of the query in the queries_ matrix
    // query_ = 0;
    // assert((size_t)query_trees_.size() == queries_.n_cols());
//     for (std::vector<TreeType*>::iterator query_tree = query_trees_.begin();
// 	 query_tree < query_trees_.end(); ++query_tree, ++query_) {

    for (query_ = 0; query_ < queries_.n_cols; ++query_) {

      query_norm_ = arma::norm(queries_.col(query_), 2);
//       double query_norm_test = std::sqrt(arma::dot(queries_.col(query_), 
// 						   queries_.col(query_)));

//       IO::Info << "Arma val: " << query_norm_ << std::endl
// 	       << "Crude val: " << query_norm_test << std::endl;

      ComputeNeighborsRecursion_(reference_tree_, 
				 MaxNodeIP_(reference_tree_));
    }

    resulting_neighbors->set_size(max_ips_.n_elem);
    ips->set_size(max_ips_.n_elem);

    for (size_t i = 0; i < max_ips_.n_elem; i++) {
      size_t query = i/knns_;
      (*resulting_neighbors)(query*knns_+ i%knns_)
	= old_from_new_references_(max_ip_indices_(i));
      (*ips)(query*knns_+ i%knns_) = max_ips_(i);
    }

    IO::Info << "Tree-based Search - Number of prunes: " 
	     << number_of_prunes_ << std::endl;
    IO::Info << "\t \t Avg. # of DC: " 
	     << (double) distance_computations_ 
      / (double) queries_.n_cols << std::endl;
    IO::Info << "\t \t Avg. # of SD: " 
	     << (double) split_decisions_ 
      / (double) queries_.n_cols << std::endl;


//     NOTIFY("Tdc = %zu"d, Tmc = %zu"d, adc = %lg, amc = %lg",
// 	   dc, mc, (float)dc/(float)query_trees_.size(), 
// 	   (float)mc/(float)query_trees_.size());
  } // ComputeNeighbors
  
  
  /**
   * Does the entire computation naively
   */
  void ComputeNaive(arma::Col<size_t>* resulting_neighbors,
                        arma::vec* ips) {
//   void ComputeNaive(ArrayList<size_t>* resulting_neighbors,
//                     ArrayList<double>*  distances) {


    for (query_ = 0; query_ < queries_.n_cols; ++query_) {
      ComputeBaseCase_(reference_tree_);
    }

    resulting_neighbors->set_size(max_ips_.n_elem);
    ips->set_size(max_ips_.n_elem);

    for (size_t i = 0; i < max_ips_.n_elem; i++) {
      size_t query = i/knns_;
      (*resulting_neighbors)(query*knns_+ i%knns_)
	= old_from_new_references_(max_ip_indices_(i));
      (*ips)(query*knns_+ i%knns_) = max_ips_(i);
    }
    
    IO::Info << "Brute-force Search - Number of prunes: " 
	     << number_of_prunes_ << std::endl;
    IO::Info << "\t \t Avg. # of DC: " 
	     << (double) distance_computations_ 
      / (double) queries_.n_cols << std::endl;
    IO::Info << "\t \t Avg. # of SD: " 
	     << (double) split_decisions_ 
      / (double) queries_.n_cols << std::endl;
//     // Start on the root of each tree
//     // the index of the query in the queries_ matrix
//     query_ = 0;
//     assert((size_t)query_trees_.size() == queries_.n_cols());
//     for (std::vector<TreeType*>::iterator query_tree = query_trees_.begin();
// 	 query_tree < query_trees_.end(); ++query_tree, ++query_) {

//       ComputeBaseCase_(*query_tree, reference_tree_);
//     }
//     for (size_t i = 0; i < neighbor_indices_.size(); i++) {
//       size_t query = i/knns_;
//       (*resulting_neighbors)[query*knns_+ i%knns_]
// 	= old_from_new_references_[neighbor_indices_[i]];
//       (*distances)[query*knns_+ i%knns_] = neighbor_distances_[i];
//     }
  }

}; //class AllkNN

#endif

