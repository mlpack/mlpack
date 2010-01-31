/*
 *  geomst2.h
 *  
 *
 *  Created by William March on 12/4/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GEOMST2_H
#define GEOMST2_H

#include "fastlib/fastlib.h"
#include "emst_cover.h"
#include "mlpack/emst/union_find.h"

const fx_entry_doc geomst_entries[] = {
{"MST_computation", FX_TIMER, FX_CUSTOM, NULL, 
"Total time required to compute the MST.\n"},
{"total_length", FX_RESULT, FX_DOUBLE, NULL, 
"The total length of the MST.\n"},
{"number_of_points", FX_RESULT, FX_INT, NULL,
"The number of points in the data set.\n"},
{"dimension", FX_RESULT, FX_INT, NULL,
"The dimensionality of the data.\n"},
{"tree_building", FX_TIMER, FX_CUSTOM, NULL,
"Time taken to construct the cover tree.\n"},
{"finding_pairs", FX_TIMER, FX_CUSTOM, NULL,
  "Time taken finding the closest pair of points.\n"},
{"number_of_pairs", FX_RESULT, FX_INT, NULL,
  "The number of pairs in the WSPD.\n"},
{"number_of_bcp_computations", FX_RESULT, FX_INT, NULL,
  "The number of closest pairs computed.\n"},
FX_ENTRY_DOC_DONE
};

const fx_submodule_doc geomst_submodules[] = {
FX_SUBMODULE_DOC_DONE
};

const fx_module_doc geomst_doc = {
geomst_entries, geomst_submodules,
"Algorithm module for DualTreeBoruvka on a cover tree.\n"
};



class GeoMST2 {
  
private:
  
  class GeoMSTStat {
    
  private:
    
    // The farthest apart two points in the node can be
    double width_;
    
  public:
    
    void Init(const Matrix& dataset, index_t start, index_t count) {
      
      width_ = -1.0;
      
    } // leaf init
    
    void Init(const Matrix& dataset, index_t start, index_t count,
              const GeoMSTStat& left_stat, const GeoMSTStat& right_stat) {
      
      Init(dataset, start, count);
      
    } // non leaf Init()
    
    double max_dist() const {
      return width_;
    }
    
    void set_max_dist(double max_dist) {
      width_ = max_dist;
    }
    
  }; // stat class
  
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, GeoMSTStat> GeoMSTTree;
  
  class NodePair {
    
  private:
    
    GeoMSTTree* node1_;
    GeoMSTTree* node2_;
    
    // this is the minimum distance between the nodes before the closest pair 
    // has been found and is the distance between this pair after
    double dist_;
    
    // the closest pair from the two nodes
    index_t pt1_;
    index_t pt2_;
    
    bool bcp_done_;
    
  public:
    
    void Init(GeoMSTTree* n1, GeoMSTTree* n2, double dist) {
      
      node1_ = n1;
      node2_ = n2;
      
      dist_ = dist;
      
      pt1_ = -1;
      pt2_ = -1;
      
      bcp_done_ = false;
      
    }
    
    GeoMSTTree* node1() {
      return node1_;
    }

    GeoMSTTree* node2() {
      return node2_;
    }
    
    index_t pt1() const {
      return pt1_;
    }
    
    index_t pt2() const {
      return pt2_;
    }
    
    void set_points(index_t p1, index_t p2) {
      pt1_ = p1;
      pt2_ = p2;
    }
    
    double dist() const {
      return dist_;
    }
    
    void set_dist(double new_dist) {
      
      DEBUG_ASSERT(new_dist >= 0.0);
      
      dist_ = new_dist;
      
    }
    
    void set_pair(index_t point1, index_t point2, double dist) {
      
      pt1_ = point1;
      pt2_ = point2;
      dist_ = dist;
      bcp_done_ = true;
      
    }
    
    bool PairDone() {
      return bcp_done_;
    }
    
  }; // class NodePair
  
  /////////// Variables ////////////////
  
  GeoMSTTree* tree_;
  ArrayList<EdgePair> edges_;
  index_t number_of_points_;
  index_t number_of_edges_;
  fx_module* mod_;
  UnionFind connections_;
  Matrix data_points_;
  index_t number_of_bcp_;
  
  ArrayList<NodePair> wspd_;
  
  ArrayList<index_t> old_from_new_permutation_;
  
  double total_dist_;
  
  MinHeap<double, NodePair*> heap_;
  
  
  
  ////////////// Functions //////////////////
  
  void WellSeparated_(GeoMSTTree* node1, GeoMSTTree* node2) {
    
    double min_dist = node1->bound().MinDistanceSq(node2->bound());
    min_dist = sqrt(min_dist);
    
    double max1 = node1->stat().max_dist();
    double max2 = node2->stat().max_dist();
    
    if (min_dist > (2.0 * max(max1, max2))) {
      
      // add the pair to the decomposition
      
      NodePair new_pair;
      new_pair.Init(node1, node2, min_dist);
      
      wspd_.PushBackCopy(new_pair);
      
    }
    else {
      
      if (max1 >= max2) {
        
        WellSeparated_(node1->left(), node2);
        WellSeparated_(node1->right(), node2);
        
      }
      else {
        
        WellSeparated_(node1, node2->left());
        WellSeparated_(node1, node2->right());
        
      }
      
    }
    
  } // WellSeparated_()
  
  void FindPairs_(GeoMSTTree* node) {
    
    if (!(node->is_leaf())) {
      
      FindPairs_(node->left());
      FindPairs_(node->right());
      
      WellSeparated_(node->left(), node->right());
      
    }
    
  } // FindPairs_()
  
  void AddEdge_(index_t e1, index_t e2, double distance) {
    
    //EdgePair edge;
    DEBUG_ASSERT_MSG((e1 != e2), 
                     "Indices are equal in DualTreeBoruvka.add_edge(%d, %d, %f)\n", 
                     e1, e2, distance);
    
    DEBUG_ASSERT_MSG((distance >= 0.0), 
                     "Negative distance input in DualTreeBoruvka.add_edge(%d, %d, %f)\n", 
                     e1, e2, distance);
    
    if (e1 < e2) {
      edges_[number_of_edges_].Init(e1, e2, distance);
    }
    else {
      edges_[number_of_edges_].Init(e2, e1, distance);
    }
    
    number_of_edges_++;
    total_dist_ += distance;
    
  } // AddEdge_
  
  void FindClosestPair_(GeoMSTTree* n1, GeoMSTTree* n2, index_t* pt1, 
                        index_t* pt2, double* dist) {
    
    
    if (n1->is_leaf() && n2->is_leaf()) {
      
      DEBUG_ASSERT(n1->count() == 1);
      DEBUG_ASSERT(n2->count() == 1);

      DEBUG_ASSERT(n1->begin() != n2->begin());

      Vector vec1, vec2;
      data_points_.MakeColumnVector(n1->begin(), &vec1);
      data_points_.MakeColumnVector(n2->begin(), &vec2);
      
      double this_dist = sqrt(la::DistanceSqEuclidean(vec1, vec2));
      
      if (this_dist < *dist) {
       
        *pt1 = n1->begin();
        *pt2 = n2->begin();
        
        *dist = this_dist;
        
      }
      
    } 
    else {
     
      if (n1->stat().max_dist() > n2->stat().max_dist()) {
        
        double dist1 = sqrt(n1->left()->bound().MinDistanceSq(n2->bound()));
        double dist2 = sqrt(n1->right()->bound().MinDistanceSq(n2->bound()));
        
        if (dist1 < dist2) {
          
          if (dist1 < *dist) {
          
            FindClosestPair_(n1->left(), n2, pt1, pt2, dist);
            
          }
          
          if (dist2 < *dist) {

            FindClosestPair_(n1->right(), n2, pt1, pt2, dist);

          }
          
        }
        else {
          
          if (dist2 < *dist) {
            
            FindClosestPair_(n1->right(), n2, pt1, pt2, dist);
            
          }
          
          if (dist1 < *dist) {
            
            FindClosestPair_(n1->left(), n2, pt1, pt2, dist);
            
          }
          
        }
        
      }
      else {
        
        // the second node is fatter
        
        double dist1 = sqrt(n1->bound().MinDistanceSq(n2->left()->bound()));
        double dist2 = sqrt(n1->bound().MinDistanceSq(n2->right()->bound()));
        
        if (dist1 < dist2) {
          
          if (dist1 < *dist) {
            
            FindClosestPair_(n1, n2->left(), pt1, pt2, dist);
            
          }
          
          if (dist2 < *dist) {
            
            FindClosestPair_(n1, n2->right(), pt1, pt2, dist);
            
          }
          
        }
        else {
          
          if (dist2 < *dist) {
            
            FindClosestPair_(n1, n2->right(), pt1, pt2, dist);
            
          }
          
          if (dist1 < *dist) {
            
            FindClosestPair_(n1, n2->left(), pt1, pt2, dist);
            
          }
          
        }
        
      }
      
    }
    
  } // FindClosestPair_()
  
  
  struct SortEdgesHelper_ {
    bool operator() (const EdgePair& pairA, const EdgePair& pairB) {
      return (pairA.distance() > pairB.distance());
    }
  } SortFun;
  
  void SortEdges_() {
    
    std::sort(edges_.begin(), edges_.end(), SortFun);
    
  } // SortEdges_()
  
  
  void EmitResults_(Matrix* results) {
    
    SortEdges_();
    
    DEBUG_ASSERT(number_of_edges_ == number_of_points_ - 1);
    results->Init(3, number_of_edges_);
    
    for (index_t i = 0; i < (number_of_points_ - 1); i++) {
      
      edges_[i].set_lesser_index(old_from_new_permutation_[edges_[i]
                                                           .lesser_index()]);
      
      edges_[i].set_greater_index(old_from_new_permutation_[edges_[i]
                                                            .greater_index()]);
      
      results->set(0, i, edges_[i].lesser_index());
      results->set(1, i, edges_[i].greater_index());
      results->set(2, i, sqrt(edges_[i].distance()));
      
    }
  
  } // EmitResults_
  
  void ComputeWidths_(GeoMSTTree* tree) {
    
    if (tree->is_leaf()) {
     
      DEBUG_ASSERT(tree->count() == 1);
      
      tree->stat().set_max_dist(0.0);
      
    }
    else {
      
      tree->stat().set_max_dist(sqrt(tree->bound().CalculateMaxDistanceSq()));
      ComputeWidths_(tree->left());
      ComputeWidths_(tree->right());
      
    }
    
  } // ComputeWidths_
  
  
  
  
public:
  
  void Init(const Matrix& data, fx_module* mod) {
    
    number_of_edges_ = 0;
    data_points_.Copy(data);
    mod_ = mod;
    
    fx_timer_start(mod_, "tree_building");
    
    tree_ = tree::MakeKdTreeMidpoint<GeoMSTTree>
    (data_points_, 1, &old_from_new_permutation_, NULL);
    
    ComputeWidths_(tree_);

    fx_timer_stop(mod_, "tree_building");
    
    number_of_points_ = data_points_.n_cols();
    edges_.Init(number_of_points_ - 1);
    connections_.Init(number_of_points_);
    
    total_dist_ = 0.0;
    
    wspd_.Init(0);
    
    heap_.Init();
    
    number_of_bcp_ = 0;
    
  } // Init
  
  void ComputeMST(Matrix* results) {
    
    fx_timer_start(mod_, "MST_computation");
    
    fx_timer_start(mod_, "finding_pairs");
    FindPairs_(tree_);
    fx_timer_stop(mod_, "finding_pairs");
    
    // make sheap, read from it, add edges as found
    
    // add some stuff to the heap
    for (index_t i = 0; i < wspd_.size(); i++) {
      
      heap_.Put(wspd_[i].dist(), &(wspd_[i]));
      
    } // add all pairs
    
    while (number_of_edges_ < number_of_points_ - 1) { 
      
      if (!(heap_.is_empty())) {
        
        NodePair* this_pair = heap_.Pop();
        
        if (this_pair->PairDone()) {
          
          if (connections_.Find(this_pair->pt1()) != 
              connections_.Find(this_pair->pt2())) {
            
            // point is safe to add
            connections_.Union(this_pair->pt1(), this_pair->pt2());
            
            AddEdge_(this_pair->pt1(), this_pair->pt2(), this_pair->dist());
            
          }
          else {
            
            // they're already connected
            
          }
          
        }
        else {
         
          // find the closest pair and re-insert in the heap
          
          double cp_dist = DBL_MAX;
          index_t point1, point2;
          FindClosestPair_(this_pair->node1(), this_pair->node2(), &point1, 
                           &point2, &cp_dist);
          number_of_bcp_++;
          
          this_pair->set_pair(point1, point2, cp_dist);
          
          heap_.Put(cp_dist, this_pair);
          
        }
        
      } // heap not empty
      else {
        
        FATAL("Heap exhausted prematurely.\n");
        
      } // heap empty
      
    } // while tree not finished
    
    fx_timer_stop(mod_, "MST_computation");

    
    EmitResults_(results);
    
    fx_result_double(mod_, "total_length", total_dist_);
    fx_result_int(mod_, "number_of_points", data_points_.n_cols());
    fx_result_int(mod_, "number_of_pairs", wspd_.size());
    fx_result_int(mod_, "number_of_bcp_computations", number_of_bcp_);
    
  } // ComputeMST
  
  
}; // class



#endif