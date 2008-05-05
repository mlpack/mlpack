/**
 * @file all_centroid_knn.h
 *
 * Defines AllNN class to perform all-nearest-neighbors on two specified between 
 * the centroids of data sets. Instead of using the points it chooses the centroid of
 * the nodes at a predifined level
 */

#ifndef ALL_CENTROID_KNN_H
#define ALL_CENTROIND_KNN_H

// We need to include fastlib.  If you want to use fastlib, 
// you need to have this line in addition to
// the deplibs section of your build.py
#include <fastlib/fastlib.h>
#include "mlpack/allknn/allknn.h"
/**
 * Forward declaration for the tester class
 */
class TestAllCentroidkNN;
/**
* Performs all-nearest-neighbors.  This class will build the trees and 
* perform the recursive  computation.
*/
class AllCentroidkNN {
  // Declare the tester class as a friend class so that it has access
  // to the private members of the class
  friend class TestAllCentroidkNN;
  
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
      OT_MY_OBJECT(centroid_id_);
    } // OT_DEF_BASIC
    
   private:
    /**
    * The upper bound on the node's nearest neighbor distances.
     */
    index_t centroid_id_;
    
   public:
    index_t centroid_id() {
      return centroind_id_;
    }
    void set_centroid_id(index_t centroid_id) {
      centroid_id_=centroid_id;
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
      centroid_id_=AllCentroidkNN::centroid_counter_;
      AllCentroidkNN::centroid_counter_++;
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
    
  }; //class AllNNStat  
  
  // TreeType are BinarySpaceTrees where the data are bounded by 
  // Euclidean bounding boxes, the data are stored in a Matrix, 
  // and each node has a QueryStat for its bound.
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, QueryStat> TreeType;
   
  
  /////////////////////////////// Members //////////////////////////////////////////////////
 private:
  // These will store our data sets.
  Matrix points_;
  // Pointers to the roots of the two trees.
  TreeType* tree_for_centroids_;
 // A permutation of the indices for tree building.
  ArrayList<index_t> old_from_new_points_;
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
 
 /////////////////////////////// Helper Functions ///////////////////////////////////////////////////
  void ComputeCentroidsRecursion_(TreeType *node, Matrix *centroids) {
    Vector vec;
    index_t centroid_id=node->stat().centroid_id();
    centroids->MakeVector(centroid_id, &vec);
    node->bound().CalculateMidpoint(&vec);	
    if (!nodes->IsLeaf()) {
      ComputeCentroidsRecursion_(node->left(), centroids);
      ComputeCentroidsRecursion_(node->right(), centroids);
    }
  }
 
  void RetrieveCentroidsRecursion_(TreeType *node, index_t level,
      index_t parent_centroid_id,  
      GenVector<index_t> *centroid_ids, Matrix *features) { 
    
    index_t centroid_id=node->stat().centroid_id();
    if (level==0 || node->IsLeaf()) {
      centroid_ids->PushBackCopy(centroid_id);
      memcpy(features->GetColumnPtr(centroid_id), 
          features->GetColumnPtr(parent_centroid_id), sizeof(double)*dimension_);
      return;
    }
    if (!node->IsLeaf()) {
      level--;
      RetrieveCentroidsRecursion_(node->left(), level, centroid_id, centroid_ids);
      RetrieveCentroidsRecursion_(node->right(), level, centroid_id, centroid_ids);
    }
  }

  void RetrieveCentroidsRecursion_(TreeType *node, double range, 
      index_t parent_centroid_id,  
      GenVector<index_t> *centroid_ids) { 
    
    index_t centroid_id=node->stat().centroid_id();
    if (node.bounds().CalculateMaxDistanceSq()<range || node->IsLeaf()) {
      centroid_ids->PushBackCopy(centroid_id);
      memcpy(features->GetColumnPtr(centroid_id), 
          features->GetColumnPtr(parent_centroid_id), sizeof(double)*dimension_);
    }
    if (!data_tree->IsLeaf()) {
      RetrieveCentroidsRecursion_(data_tree->left(), range, centroid_id, centroid_ids);
      RetrieveCentroidsRecursion_(data_tree->right(), range, centroid_id, centroid_ids);
    }
  } 

  void FromCentroidsToPointsRecurse_(TreeType *node, Matrix &centroid_features, 
      Matrix *point_features) {
    if (node->IsLeaf()) {
      for(index_t i=node->begin(); i<node->end() ;i++) {
        index_t ind=old_from_new_points(i);
        memcpy(point_features->GetColumnPtr(ind), 
            centroid_features.GetColumnPtr(node->stat().centroid_id()), 
            dimension_* sizeof(double));
      }
    } else {
      FromCentroidsToPointsRecurse_(node->left(), centroid_features, 
          point_features);
      FromCentroidsToPointsRecurse_(node->right(), centroid_features, 
          point_features); 
    }
  }

  void FromCentroidsToPointsRecurse_(TreeType *node, Matrix &centroid_features, 
      index_t level, Matrix *point_features) {
    if (level==0 || node->IsLeaf()) {
      for(index_t i=node->begin(); i<node->end() ;i++) {
        index_t ind=old_from_new_points(i);
        memcpy(point_features->GetColumnPtr(ind), 
            centroid_features.GetColumnPtr(node->stat().centroid_id()), 
            dimension_* sizeof(double));
      }
    } else {
      level--;
      FromCentroidsToPointsRecurse_(node->left(), centroid_features, 
          level, point_features);
      FromCentroidsToPointsRecurse_(node->right(), centroid_features, 
          level, point_features); 
    }
  }

  /////////////////////////////// Constructors /////////////////////////////////////////////
  // Add this at the beginning of a class to prevent accidentally calling the copy constructor
  FORBID_ACCIDENTAL_COPIES(AllCentroidkNN);
  

 public:
  static index_t centroid_counter_; 
  /**
  * Constructors are generally very simple in FASTlib; most of the work is done by Init().  This is only
  * responsible for ensuring that the object is ready to be destroyed safely.  
  */
  AllCentroidkNN() {
    tree_for_centroids_ = NULL;
  } 
  
  /**
  * The tree is the only member we are responsible for deleting.  The others will take care of themselves.  
  */
  ~AllCentroidkNN() {
    if (tree_for_centroid!=NULL) {
      delete tree_for_centroid_;
    }
  } 
    
      
  
  
 ////////////////////////////////// Public Functions ////////////////////////////////////////////////
  
  /** Use this if you want to run allknn it on a single dataset 
   * the query tree and reference tree are the same
   */
  void Init(const Matrix& points, struct datanode* module_in) {
     
    // set the module
    module_ = module_in;
    
    points_.Copy(points);
    // Get the leaf size from the module
    leaf_size_ = fx_param_int(module_, "leaf_size", 20);
    // Make sure the leaf size is valid
    DEBUG_ASSERT(leaf_size_ > 0);
    // We'll time tree building
    fx_timer_start(module_, "tree_building");

    // This call makes each tree from a matrix, leaf size, and two arrays 
		// that record the permutation of the data points
    // Instead of NULL, it is possible to specify an array new_from_old_
    tree_for_centroid_ = tree::MakeKdTreeMidpoint<TreeType>(points_, leaf_size_, 
				&old_from_new_points_, NULL);
   
    // Stop the timer we started above
    fx_timer_stop(module_, "tree_building");

  }
  
  void InitCentroidList(
      ArrayList<FEATURE> > *centroids) {
    centroids->Init(centroid_counter_);
  }

  void InitCentroidList(
      ArrayList<Vector> *centroids) {
    centroids->Init(dimension_, centroid_counter_);
  }

  void ComputeCentroids(TreeType *data_tree, Matrix *centroids) {
    ComputeCentroidsRecursion_(data_tree, centroids); 
  }

  void RetrieveCentroids(TreeType *data_tree, index_t level, 
      GenVector<index_t> *centroid_ids, Matrix *features) {
    centroid_ids->Init();
    RetrieveCentroidsRecursion_(data_tree, level, 0, centroid_ids, features);
  }

  void RetrieveCentroids(TreeType *data_tree, double range, 
      GenVector<index_t> *centroid_ids, Matrix *features) {
    centroid_ids->Init();
    RetrieveCentroidsRecursion_(data_tree, range, 0, centroid_ids, features);
  }

  void AllkCentroids(Matrix &centroids, GenVector<index_t> &centroid_ids,
      ArrayList<index_t> *resulting_neighbors, 
      GenVector<double> *distances) {
  
    AllkNN allknn_;
    Matrix centroid_data;
    centroid_data.Init(dimension_, centroid_ids.size());
    for(index_t i=0; i<centroid_ids.size(); i++) {
      memcpy(centroid_data.GetColumnPtr(i), 
          centroids.GetColumnPtr(centroid_ids[i]), dimension_*sizeof(double));
    }
    datanode *module=fx_submodule(module_, "/allknn", "allknn");
    allknn_.Init(centroid_data, module);
    allknn_.ComputeNeighbors(resulting_neighbors, distances);
  }
  
  void FromCentroidsToPoints(Matrix &centroid_features, 
      Matrix *point_features) {
    FromCentroidsToPointsRecurse_(data_tree_, centroid_features, point_features);
  }
   
}; //class AllNN


#endif
