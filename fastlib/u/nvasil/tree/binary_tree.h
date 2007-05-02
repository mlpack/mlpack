/*
 * =====================================================================================
 * 
 *       Filename:  tree.h
 * 
 *    Description:  A generic multidimensional binary tree. Currently tested under 
 *                  kd-nodes and ball-nodes
 * 
 *        Version:  2.0
 *        Created:  02/09/2007 08:25:15 PM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  This material is property of Georgia Tech Fastlab-ESP Lab, 
 *                  and it is not for distribution
 *         
 * =====================================================================================
 */



#ifndef BINARY_TREE_H_
#define BINARY_TREE_H_
#include <stdio.h>
#include <string>
#include <errno.h>
#include <string.h>
#include <vector> 
#include <list>
#include "u/nvasil/loki/Typelist.h"
#include "fastlib/fastlib.h"
#include "show_progress.h"
using namespace std;
template<typename TYPELIST, bool diagnostic>
class BinaryTree {
 public:
	// For testing purposes only
  friend class BinaryTreeTest;
	typedef typename Loki::TL::TypeAt<TYPELIST, 0>::Result Precision_t;
	typedef typename Loki::TL::TypeAt<TYPELIST, 1>::Result Allocator_t;
  typedef typename Loki::TL::TypeAt<TYPELIST, 2>::Result Metric_t;
	typedef typename Loki::TL::TypeAt<TYPELIST, 3>::Result BoundingBox_t;
	typedef typename Loki::TL::TypeAt<TYPELIST, 4>::Result NodeCachedStatistics_t;
	typedef typename Loki::TL::TypeAr<TYPELIST, 5>::Result Pivot_t;
	typedef Allocator_t::template ArrayPtr<Precision_t> Array_t;
  typedef Node<TYPELIST, diagnostic> Node_t;
	typedef Allocator_t::template Ptr<Node> NodePtr_t;
	typedef Point<Precision_t, Allocator_t> Point_t;
	typedef Node_t::Result Result_t;
	typedef BinaryTree<TYPELIST, bool> BinaryTree_t;
  typedef Pivot_t::PivotInfo PivotInfo_t;	
	class OutPutAllocator {
	 public: 
		OutPutAllocator() {
			num_=0;
		}
		void set_ptr(Result_t *ptr) {
		  ptr_=ptr;
		}
	  Result_t *get_ptr() {
		  return ptr_;
		}
		Result_t *Allocate(int32 num_of_points, int32 range) {
		  Result_t *result=ptr_+num_;
			num_+=range*num_of_points;
		  return result;	
		}
	 private:
		Result_t *ptr_;
		IDPRECISION num_;

	};
	BinaryTree();
  ~BinaryTree();
	void Init(BinaryDataset &data);
  // Call this function to build Depth first a tree
  void BuildDepthFirst();
	void BuildDepthFirst(Node_ptr &ptr, Pivot_t *pivot);
  void BuildBreadthFirst();
	void BuildBreadthFirst(list<pair<Node_ptr_ptr, Pivot_t *> > &fifo);
   // Builds tree k depth first. It builds all the  subtrees depth first up to k level
	void BuildKDepthFirst();
  template<typename POINTTYPE, typename NEIGHBORTYPE>
  void NearestNeighbor(POINTTYPE &test_point,
                       vector<pair<Precision_t, Point_t> > *nearest_point,
                       NEIGHBORTYPE range);

	// This is the core function doing the recursion, Use that only if you want
	// to start the search from a particular node and not the parent
	template<typename POINTTYPE, typename NEIGHBORTYPE>
  void NearestNeighbor(Node_ptr ptr,
                       POINTTYPE &test_point,
                       vector<pair<Precision_t, Point_t> > *nearest_point,
                       NEIGHBORTYPE range,
                       bool &found);
  
  // This is the duall tree nearest neighbors method, again it works
  // for all cases k nearest/ range nearest 
  template<typename NEIGHBORTYPE>
  void AllNearestNeighbors(Node_ptr query, 
                           NEIGHBORTYPE range);                    
  template<typename NEIGHBORTYPE>
  void AllNearestNeighbors(Node_ptr query, 
                           Node_ptr reference,
                           NEIGHBORTYPE range, 
                           PRECISION distance);
	void InitAllKNearestNeighborOutput(string file, int32 knns);
	void CloseAllKNearestNeighborOutput(int32 knns);
	void InitAllKNearestNeighborOutput(Node_ptr ptr, 
		                                int32 knns);
	void InitAllRangeNearestNeighborOutput(string file);
	void CloseAllRangeNearestNeighborOutput(int32 range);
	void InitAllRangeNearestNeighborOutput(Node_ptr ptr, 
		                                int32 range);


	// Print the tree depth first
	void Print();
  void RecursivePrint(Node_ptr ptr);  
  // Resets the counters of the tree that keep the statistics of search
	void ResetCounters() {
    computations_.Reset();
  }                     
  string Statistics();
  string Computations();  
  void set_log_file(const string &log_file);
  int32 get_current_level() {
    return current_level_;
  };
  uint64 get_num_of_points(){
  	return num_of_points_;
  }
  Node_ptr get_parent() {
  	return parent_;
  }
	void set_discriminator(PointIdentityDiscriminator<IDPRECISION> *disc) {
	  discriminator_.reset(disc);
	}
	void set_max_points_on_leaf(index_t max_point_on_leaf) {
		max_points_on_leaf_=max_points_on_leaf;
	}
	index_t get_max_points_on_leaf() {
	  return max_points_on_leaf();
	}
 private:
	// Maximum number of points on a leaf
  index_t  max_points_on_leaf_;
  // Parent/Root
	Node_ptr parent_;
  // Source of data
	BinaryDataset data_;
  // Total number of points on the tree
	index_t num_of_points_;
	// Number of Leafs on the tree
	index_t num_of_leafs_;
	// Number of nodes (incuding leafs)
	index_t node_id_;
	// Current level of tree while we build it
  index_t current_level_;
	// Maximum depth of the tree
  index_t max_depth_;
	// Minimum depth of the tree
	index_t min_depth_;
  // Dimensionality of points
	int32 dimension_;
  // Total number of points visited during search
	index_t total_nodes_visited_;
	// Structure for keeping statistics on the comparisons and distances computed
	// during search
	ComputationsCounter<diagnostic> computations_;
	// total number of nodes visited
	index_t total_points_visited_;
  // used for visualization of progress during tree build
	ShowProgress progress_;
  bool log_progress_;
	// Output file for All nearest neighbors
  OutPutAllocator all_nn_out_;
  FILE *log_file_ptr_;
	string log_file_;
	// This is usefull for our timit experiments
  PointIdentityDiscriminator	discriminator_;
	Pivot_t pivoter_;
};
#include "binary_tree_impl.h"
#endif /*BINARY_TREE_H_*/
