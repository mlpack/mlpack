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
#include <limits>
#include <vector> 
#include <list>
#include "fastlib/fastlib.h"
#include "u/nvasil/loki/static_check.h"
#include "u/nvasil/tree/node.h"
#include "u/nvasil/tree/knn_node.h"
#include "u/nvasil/tree/show_progress.h"
using namespace std;
template<typename TYPELIST, bool diagnostic>
class BinaryTree {
 public: 
	typedef typename TYPELIST::Node_t      Node_t; 
	typedef typename Node_t::Precision_t   Precision_t;
	typedef typename Node_t::Allocator_t   Allocator_t;
  typedef typename Node_t::Metric_t      Metric_t;
	typedef typename Node_t::BoundingBox_t BoundingBox_t;
	typedef typename Node_t::NodeCachedStatistics_t NodeCachedStatistics_t;
	typedef typename Node_t::PointIdDiscriminator_t PointIdDiscriminator_t;
	typedef typename TYPELIST::Pivot_t Pivot_t;
	typedef typename Allocator_t::template ArrayPtr<Precision_t> Array_t;
	typedef typename Allocator_t::template Ptr<Node_t> NodePtr_t;
	typedef typename Allocator_t::template Ptr<NodePtr_t> NodePtrPtr_t;
	typedef Point<Precision_t, Allocator_t> Point_t;
	typedef typename Node_t::NNResult Result_t;
	typedef BinaryTree<TYPELIST, diagnostic> BinaryTree_t;
  typedef typename Pivot_t::PivotInfo PivotInfo_t;	
  // For testing purposes only
  template<typename, bool> friend class BinaryTreeTest;

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
		Result_t *Allocate(int32 num_of_points, int32 knns) {
		  Result_t *result=ptr_+num_;
			num_+=knns*num_of_points;
		  return result;	
		}
	 private:
		Result_t *ptr_;
		index_t num_;
	};
	BinaryTree(){}
  ~BinaryTree();
	void Init(BinaryDataset<Precision_t> *data);
	void Destruct() {}
  // Call this function to build Depth first a tree
	void BuildDepthFirst();
	void BuildDepthFirst(NodePtr_t ptr, PivotInfo_t *pivot);
  void BuildBreadthFirst();
	void BuildBreadthFirst(
			list<pair<NodePtrPtr_t, PivotInfo_t *> > &fifo);
   // Builds tree k depth first. It builds all the  subtrees depth first up to k level
	void BuildKDepthFirst();
  template<typename POINTTYPE, typename NEIGHBORTYPE>
  void NearestNeighbor(POINTTYPE test_point,
      vector<pair<Precision_t, Point_t> > *nearest_point,
      NEIGHBORTYPE range);

	// This is the core function doing the recursion, Use that only if you want
	// to start the search from a particular node and not the parent
	template<typename POINTTYPE, typename NEIGHBORTYPE>
  void NearestNeighbor(NodePtr_t ptr,
     POINTTYPE &test_point,
     vector<pair<Precision_t, Point_t> > *nearest_point,
     NEIGHBORTYPE range,
     bool &found);
  
  // This is the duall tree nearest neighbors method, again it works
  // for all cases k nearest/ range nearest 
  template<typename NEIGHBORTYPE>
  void AllNearestNeighbors(NodePtr_t query, 
                           NEIGHBORTYPE range);                    
  template<typename NEIGHBORTYPE>
  void AllNearestNeighbors(NodePtr_t query, 
                           NodePtr_t reference,
                           NEIGHBORTYPE range, 
                           Precision_t distance);
	void InitAllKNearestNeighborOutput(string file, int32 knns);
	void CloseAllKNearestNeighborOutput(int32 knns);
	void InitAllKNearestNeighborOutput(NodePtr_t ptr, 
		                                int32 knns);
	void InitAllRangeNearestNeighborOutput(string file);
  void InitAllRangeNearestNeighborOutput(NodePtr_t ptr,
		                                     FILE *fp);
	void CloseAllRangeNearestNeighborOutput();
	void CollectKNearestNeighborWithMMAP(string file);
	void CollectKNearestNeighbor(NodePtr_t ptr, 
		                           typename Node_t::NNResult *out);
  void CollectKNearestNeighborWithFwrite(string file); 
	void CollectKNearestNeighbor(NodePtr_t ptr, FILE *out);


	// Print the tree depth first
	void Print();
  void RecursivePrint(NodePtr_t ptr);  
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
  NodePtr_t get_parent() {
  	return parent_;
  }
	void set_discriminator(PointIdDiscriminator_t *disc) {
	  discriminator_.reset(disc);
	}
	void set_max_points_on_leaf(index_t max_points_on_leaf) {
		max_points_on_leaf_=max_points_on_leaf;
	}
	index_t get_max_points_on_leaf() {
	  return max_points_on_leaf_;
	}
	void set_knns(index_t knns) {
	  knns_=knns;
	}
 private:
	// Maximum number of points on a leaf
  index_t  max_points_on_leaf_;
  // Parent/Root
	NodePtr_t parent_;
  // Source of data
	BinaryDataset<Precision_t> *data_;
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
  PointIdDiscriminator_t discriminator_;
	// Does all the partitioning for the tree
	Pivot_t pivoter_;
	// This is a value for the knns set ahead when we want to build the tre
	// specifically for knns
	index_t knns_;
  
	template<typename NODETYPE>
	struct NodeInitializerTrait {
    static void Init(NodePtr_t ptr) {
		  
		}
	  static const bool IsItGoodForRangeNN=true;
		static const bool IsItGoodForKnnInitialization= true;
	};
	
		
};

template<>
template<typename TYPELIST, bool diagnostic>
struct BinaryTree<TYPELIST, diagnostic>::NodeInitializerTrait<KnnNode<TYPELIST, diagnostic> > {
  static void Init(BinaryTree<TYPELIST, diagnostic>::NodePtr_t ptr) {
		ptr->set_kneighbors(BinaryTree<TYPELIST, diagnostic>::knns_);
  } 
  static const bool IsItGoodForRangeNN=false;
  static const bool IsItGoodForKnnInitialization=false;
};

#include "binary_tree_impl.h"
#endif /*BINARY_TREE_H_*/
