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



#ifndef TREE_H_
#define TREE_H_
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string>
#include <math.h>
#include <errno.h>
#include <string.h>
#include <list>
#include <boost/scoped_ptr.hpp>
#include <boost/scoped_array.hpp>
#include "base/basic_types.h"
#include "pivot_policy.h"
#include "show_progress.h"
#include "point_identity_discriminator.h"
// The tree operates on points that have precision PRECISION and a unique identifier
// of type IDPRECISION. The trees can use any type of memory managment defined by 
// ALLOCATOR. ALLOCATOR can be disk-based, use cache or it can be distributed. It
// would be prefered to be a singleton structure although this is necessary. 
// The minimum requirments are to implement the following types:
// ALLOCATOR::Ptr<typename T> It has to be assignable, copyable
// ALLOCATOR::ArrayPtr<typename T>  
// ALLOCATOR::malloc<typename T>()
// ALLOCATOR::malloc<typename T>(size_t size)
// ALLOCATOR::calloc<typename T>(size_t size, T initial_value)
// For more information about the  requirments of the ALLOCATOR look at documentation. 
// NODE is the class that implements node and leafs on the tree. It is required
// to have the same PRECISION and IDPRECISION, ALLOCATOR as the tree that's why it is
// passed as a templage parameter. See documentation for the functions that NODE
// should implement. It is recomended that you derive NODE from class Node as described
//  in node.h, see also kdnode.h as an example
// The bool diagnostic template parameter is provided as a parameter for the logging
// procedures on the tree
template<typename PRECISION, 
	       typename IDPRECISION, 
				 typename ALLOCATOR,
				 bool diagnostic, 
  template<typename PRECISION, 
	         typename IDPRECISION,
					 typename ALLOCATOR,
					 bool diagnostic> class NODE>
class Tree {
 public:
	// For testing purposes only
  friend class TreeTest;
	// Some quick typedef definitions for making code more readable
  typedef NODE<PRECISION, IDPRECISION, ALLOCATOR, diagnostic> Node_t;
	typedef typename Node_t::Result Result_t;
	typedef typename Node_t::BoundingBox_t BoundingBox_t;
  // PivotPolicy is necessary for the tree building. For more inforamtion look at 
	// Andrew Moore's paper on kd-trees
	typedef PivotPolicy<PRECISION, 
					            IDPRECISION, 
											ALLOCATOR, Node_t, diagnostic> Policy_t;
	typedef typename Policy_t::Pivot_t Pivot_t;
	typedef typename ALLOCATOR::template Ptr<Node_t> Node_ptr;	
  typedef typename ALLOCATOR::template Ptr<typename 
		       ALLOCATOR::template Ptr<Node_t> >  Node_ptr_ptr;
	typedef Tree<PRECISION, IDPRECISION, ALLOCATOR, diagnostic, NODE> Tree_t;
	class OutPutAllocator {
	 public: 
		OutPutAllocator() {
			num_=0;
		}
		void set_ptr(typename Node_t::Result *ptr) {
		  ptr_=ptr;
		}
		typename Node_t::Result *get_ptr() {
		  return ptr_;
		}
		typename Node_t::Result *Allocate(int32 num_of_points, int32 range) {
		  typename Node_t::Result *result=ptr_+num_;
			num_+=range*num_of_points;
		  return result;	
		}
	 private:
		typename Node_t::Result *ptr_;
		IDPRECISION num_;

	};
	// Constructor, sets some  tree parameters
	// Data<PRECISION, IDPRECISION> *data is the source of data, it is not destroyed
	// after the construction of the tree. It is a stream of points with their identifier
	// value [ PRECISION PRECISION ..... IDPRECISION]
	// dimension is the dimensionality of the data points
	// num_of_points is the number of data_points
  Tree(DataReader<PRECISION, IDPRECISION> *data, 
			 int32 dimension, IDPRECISION num_of_points);
	// Destructor
  ~Tree();
  // Call this function to build Depth first a tree in a serial way
  void SerialBuildDepthFirst();
	// This is the function for the recursion of SerialBuildFirst
	// It builds the subtree starting from ptr, based on the information
	// provided by Pivot_t *pivot
  void SerialBuildDepthFirst(Node_ptr &ptr, Pivot_t *pivot);
  // Builds the tree depth first in a parallel way (not implemented yet)
	void ParallelBuildDepthFirst();
  void ParallelBuildDepthFirst(Node_ptr &ptr, Pivot_t *pivot);
  //	Call this function to build the tree breadth first
  void SerialBuildBreadthFirst();
	// Core function for breadth first build
  void SerialBuildBreadthFirst(list<pair<Node_ptr_ptr, Pivot_t *> > &fifo);
  // Not implemented yet
	void ParallelBuildBreadthFirst();
  void ParallelBuildBreadthFirst(list<pair<Node_ptr_ptr, Pivot_t> > &fifo);                         
  // Builds tree k depth first. It builds all the  subtrees depth first up to k level
	void SerialBuildKDepthFirst();
  void ParallelBuildKDepthFirst();
  // This function will return any of the nearest neighbors types
  // k nearest, range nearest or just nearest, depending on how you
  // call it. It seaches starting from the parent
	// If  RETURNTYPE is a point then it will return the nearest neighbor
	// IF  RETURNTYPE is a vector then it will return 
	// a) The range nearest neighbors if NEIGHBORTYPE is PRECISION
	// b) the k nearest neigbors if NERIGBORTYPE is int32
	// We recommend that when you call it you do explicit template parameter
	// definition. If you leave it on the compiler you might accidently
	// get the wrong results. It is much more error prone 
  template<typename POINTTYPE, typename RETURNTYPE, typename NEIGHBORTYPE>
  void NearestNeighbor(POINTTYPE &test_point,
                       RETURNTYPE *nearest_point,
                       PRECISION  *distance,
                       NEIGHBORTYPE range);

	// This is the core function doing the recursion, Use that only if you want
	// to start the search from a particular node and not the parent
	template<typename POINTTYPE, typename RETURNTYPE, typename NEIGHBORTYPE>
  void NearestNeighbor(Node_ptr ptr,
                       POINTTYPE &test_point,
                       RETURNTYPE *nearest_point,
                       PRECISION   *distance,
                       NEIGHBORTYPE range,
                       bool &found);
  
  // This is the duall tree nearest neighbors method, again it works
  // for all cases k nearest/ range nearest 
  template<typename NEIGHBORTYPE >
  void AllNearestNeighbors(Node_ptr query, 
                           Node_ptr reference,
                           NEIGHBORTYPE range);                    
  template<typename NEIGHBORTYPE >
  void AllNearestNeighbors(Node_ptr query, 
                           Node_ptr reference,
                           NEIGHBORTYPE range, 
                           PRECISION distance);
	// After you run AllNearestNeighbors, run this one to dump the nearest neigbors on
	// a file
  void PrintNeighbors(string filename);                                                                                       
	void PrintNeighborsRecursive(Node_ptr ptr, FILE *fp); 
	// These are being used by All nn for efficient output of the data
	// it turns out that most of the time is spent in collecting the 
	// results for output, while this one is writing directly on the output
  void InitAllKNearestNeighborOutput(string file, int32 range);
	void CloseAllKNearestNeighborOutput(int32 range);
	void InitAllKNearestNeighborOutput(Node_ptr ptr, 
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
 private:
	// Maximum number of points on a leaf
  IDPRECISION max_points_on_leaf_;
  // Parent/Root
	Node_ptr parent_;
  // Source of data
	DataReader<PRECISION, IDPRECISION> *data_;
  // Total number of points on the tree
	IDPRECISION num_of_points_;
	// Number of Leafs on the tree
	IDPRECISION num_of_leafs_;
	// Number of nodes (incuding leafs)
	IDPRECISION node_id_;
	// Current level of tree while we build it
  IDPRECISION current_level_;
	// Maximum depth of the tree
  IDPRECISION max_depth_;
	// Minimum depth of the tree
	IDPRECISION min_depth_;
  // Dimensionality of points
	int32 dimension_;
  // Total number of points visited during search
	IDPRECISION total_nodes_visited_;
	// Structure for keeping statistics on the comparisons and distances computed
	// during search
	ComputationsCounter<diagnostic> computations_;
	// total number of nodes visited
	IDPRECISION total_points_visited_;
  // used for visualization of progress during tree build
	ShowProgress progress_;
  bool log_progress_;
	// Output file for All nearest neighbors
  OutPutAllocator all_nn_out_;
  FILE *log_file_ptr_;
	string log_file_;
	// This is usefull for our timit experiments
  boost::scoped_ptr<PointIdentityDiscriminator<IDPRECISION> >
		discriminator_;
};
#include "tree_impl.h"
#endif /*TREE_H_*/
