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
      OT_MY_OBJECT(is_it_visited_);
    } // OT_DEF_BASIC
    
   private:
    /**
    * The upper bound on the node's nearest neighbor distances.
     */
    index_t centroid_id_;
    bool is_it_visited_;
   public:
    index_t centroid_id() {
      return centroid_id_;
    }
    void set_centroid_id(index_t centroid_id) {
      centroid_id_=centroid_id;
    }
    void set_is_it_visited(bool flag) {
      is_it_visited_=flag;
    }
    bool is_it_visited() {
      return is_it_visited_;
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
      is_it_visited_=false;
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
 // typedef BinarySpaceTree<DBallBound<2>, Matrix, QueryStat> TreeType; 
  
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
    // number of nearest neighbrs
  index_t knns_; 
   // The module containing the parameters for this computation. 
  struct datanode* module_;
  index_t dimension_;
  index_t tree_max_depth_;
  index_t tree_min_depth_;

 /////////////////////////////// Helper Functions ///////////////////////////////////////////////////
  void ComputeCentroidsRecursion_(TreeType *node, Matrix *centroids) {
    index_t centroid_id=node->stat().centroid_id();
    Vector vec;
    centroids->MakeColumnVector(centroid_id, &vec);
    node->bound().CalculateMidpointOverwrite(&vec);	
    // index_t ind=math::RandInt(node->begin(), node->end());
    //centroids->CopyColumnFromMat(centroid_id, ind, points_);
    if (!node->is_leaf()) {
      ComputeCentroidsRecursion_(node->left(), centroids);
      ComputeCentroidsRecursion_(node->right(), centroids);
    }
  }
 
  void RetrieveCentroidsRecursion_(TreeType *node, index_t level,
      index_t parent_centroid_id,  
      ArrayList<index_t> *centroid_ids, Matrix *features) { 
    
    index_t centroid_id=node->stat().centroid_id();
    if (level==0 || node->is_leaf()) {
      centroid_ids->PushBackCopy(centroid_id);
      if (node->stat().is_it_visited()==false) {
        features->CopyColumnFromMat(centroid_id, parent_centroid_id, *features);
        node->stat().set_is_it_visited(true);
      }
      return;
    }
    if (!node->is_leaf()) {
      level--;
      RetrieveCentroidsRecursion_(node->left(), level, centroid_id, 
          centroid_ids, features);
      RetrieveCentroidsRecursion_(node->right(), level, centroid_id, centroid_ids, 
          features);
    }
  }

  void RetrieveCentroidsRecursion_(TreeType *node, double range, 
      index_t parent_centroid_id,  
      ArrayList<index_t> *centroid_ids, Matrix *features) { 
    
    index_t centroid_id=node->stat().centroid_id();
    if (node->bound().CalculateMaxDistanceSq()<range || node->is_leaf()) {
      centroid_ids->PushBackCopy(centroid_id);
      if (node->stat().is_it_visited()==false) {
        features->CopyColumnFromMat(centroid_id, parent_centroid_id, *features);
      }
   }
    if (!node->is_leaf()) {
      RetrieveCentroidsRecursion_(node->left(), range, centroid_id, centroid_ids,
          features);
      RetrieveCentroidsRecursion_(node->right(), range, centroid_id, centroid_ids,
          features);
    }
  } 

  void FromCentroidsToPointsRecurse_(TreeType *node, Matrix &centroid_features, 
      Matrix *point_features) {
    if (node->is_leaf()) {
      for(index_t i=node->begin(); i<node->end() ;i++) {
        index_t ind=old_from_new_points_[i];
        point_features->CopyColumnFromMat(ind, node->stat().centroid_id(), centroid_features);
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
    if (level==0 || node->is_leaf()) {
      Vector perturb;
      perturb.Init(centroid_features.n_rows());
      for(index_t i=node->begin(); i<node->end() ;i++) {
        perturb.SetAll(math::Random(0.0, 1.0));
        index_t ind=old_from_new_points_[i];
        point_features->CopyColumnFromMat(ind, node->stat().centroid_id(), centroid_features);
        la::AddTo(perturb.length(), perturb.ptr(), point_features->GetColumnPtr(ind));
      }
    } else {
      level--;
      FromCentroidsToPointsRecurse_(node->left(), centroid_features, 
          level, point_features);
      FromCentroidsToPointsRecurse_(node->right(), centroid_features, 
          level, point_features); 
    }
  }
  
  void ComputeDepth_(TreeType *node, index_t level) {
    if (node->is_leaf()) {
      if (level>tree_max_depth_) {
        tree_max_depth_=level;
      }
      if (level<tree_min_depth_) {
        tree_min_depth_=level;
      }
    } else {
      level++;
      ComputeDepth_(node->left(), level);
      ComputeDepth_(node->right(), level);
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
    if (tree_for_centroids_!=NULL) {
      delete tree_for_centroids_;
    }
  } 
    
      
  
  
 ////////////////////////////////// Public Functions ////////////////////////////////////////////////
  
  /** Use this if you want to run allknn it on a single dataset 
   * the query tree and reference tree are the same
   */
  void Init(const Matrix& points, struct datanode* module_in) {
     
    // set the module
    module_ = module_in;
    dimension_=points.n_rows(); 
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
    NOTIFY("Building the tree...");
    tree_for_centroids_ = tree::MakeKdTreeMidpoint<TreeType>(points_, leaf_size_, 
				&old_from_new_points_, NULL);
    // Stop the timer we started above
    NOTIFY("Tree built...");
    fx_timer_stop(module_, "tree_building");
    NOTIFY("Computing the minimum and the maximum depth of the tree..");
    tree_max_depth_=0;
    tree_min_depth_=RAND_MAX;
    ComputeDepth_(tree_for_centroids_, 0);
    NOTIFY("Tree max_depth:%i min_depth:%i \n", tree_max_depth_, tree_min_depth_);
  }

/*  
  void InitCentroidList(ArrayList<FEATURE> > *centroids) {
    centroids->Init(centroid_counter_);
  }
*/
  void Destruct() {
    points_.Destruct();
   if (tree_for_centroids_!=NULL) {
     delete tree_for_centroids_;
     tree_for_centroids_=NULL;
   };
   old_from_new_points_.Destruct();
  } 
  void ComputeCentroids(Matrix *centroids) {
    centroids->Init(dimension_, centroid_counter_);
    ComputeCentroidsRecursion_(tree_for_centroids_, centroids); 
  }

  void RetrieveCentroids(index_t level, ArrayList<index_t> *centroid_ids, 
      Matrix *features) {
    centroid_ids->Init();
    RetrieveCentroidsRecursion_(tree_for_centroids_, level, 0, centroid_ids, features);
  }

  void RetrieveCentroids(double range, ArrayList<index_t> *centroid_ids, 
      Matrix *features) {
    centroid_ids->Init();
    RetrieveCentroidsRecursion_(tree_for_centroids_, range, 0, centroid_ids, features);
  }

  void AllkCentroids(Matrix &centroids, ArrayList<index_t> &centroid_ids,
      ArrayList<index_t> *resulting_neighbors, 
      ArrayList<double> *distances) {
  
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
    FromCentroidsToPointsRecurse_(tree_for_centroids_, centroid_features, point_features);
  }
  
  void FromCentroidsToPointsRecurse(Matrix &centroid_features, 
      index_t level, Matrix *point_features) {
    point_features->Init(centroid_features.n_rows(), points_.n_cols());
    FromCentroidsToPointsRecurse_(tree_for_centroids_, centroid_features, 
       level, point_features); 
  }
 
  void FromCentroidsToPoints1(Matrix &centroids, Matrix &points, 
      Matrix &centroid_features, ArrayList<index_t> &centroid_ids,
      Matrix *point_features) {
    AllkNN allknn;
    index_t knns=5;
    allknn.Init(points, centroids, 20, knns);
    ArrayList<index_t> resulting_neighbors;
    ArrayList<double> distances;
    allknn.ComputeNeighbors(&resulting_neighbors, &distances);
    point_features->Init(centroid_features.n_rows(), points.n_cols());
    for(index_t i=0; i<points.n_cols(); i++)  {
      double  norm_const=0.0;
      Vector vec;
      point_features->MakeColumnVector(i, &vec);
      vec.SetAll(0.0);
      for(index_t j=0; j<knns; j++) {
         norm_const+=distances[i*knns+j];
         index_t ind=resulting_neighbors[i*knns_+j];
         Vector vec1;
         centroid_features.MakeColumnVector(centroid_ids[ind], &vec1);
         la::AddExpert(distances[i*knns+j], vec1, &vec);
      }
      la::Scale(1.0/norm_const, &vec);
    }
    
  } 
  
  
  index_t tree_max_depth() {
    return tree_max_depth_;
  }

  index_t tree_min_depth() {
    return tree_min_depth_;
  }
 
}; //class AllCentroidkNN 

index_t AllCentroidkNN::centroid_counter_=0; 


#endif
