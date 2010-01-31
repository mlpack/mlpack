/**
* @file dtb_cover.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Contains an implementation of the DualTreeBoruvka algorithm for finding a 
 * Euclidean Minimum Spanning Tree using a cover tree.  
 */

#ifndef DTB_COVER_H
#define DTB_COVER_H

#include "emst_cover.h"
#include "cover_tree.h"
#include "ctree.h"
#include "mlpack/emst/union_find.h"

const fx_entry_doc dtb_cover_entries[] = {
{"MST_computation", FX_TIMER, FX_CUSTOM, NULL, 
  "Total time required to compute the MST.\n"},
{"total_squared_length", FX_RESULT, FX_DOUBLE, NULL, 
  "The total length of the MST.\n"},
{"number_of_points", FX_RESULT, FX_INT, NULL,
  "The number of points in the data set.\n"},
{"dimension", FX_RESULT, FX_INT, NULL,
  "The dimensionality of the data.\n"},
{"number_of_loops", FX_RESULT, FX_INT, NULL,
  "The number of iterations required to find the MST.\n"},
{"do_naive", FX_PARAM, FX_BOOL, NULL,
  "Not currently supported, do not use.\n"},
{"base", FX_PARAM, FX_DOUBLE, NULL,
  "The base distance for the levels of the cover tree.  2 in the paper,\n"
  "defaults to 1.3 here.\n"},
{"prune_factor", FX_PARAM, FX_DOUBLE, NULL,
  "Loosens the bounds for debugging.  Defaults to 0.\n"},
{"tree_building", FX_TIMER, FX_CUSTOM, NULL,
  "Time taken to construct the cover tree.\n"},
FX_ENTRY_DOC_DONE
};

const fx_submodule_doc dtb_cover_submodules[] = {
{"tree", &tree_construction_doc,
"Tree construction.\n"},
FX_SUBMODULE_DOC_DONE
};

const fx_module_doc dtb_cover_doc = {
  dtb_cover_entries, dtb_cover_submodules,
  "Algorithm module for DualTreeBoruvka on a cover tree.\n"
};



/**
* A Stat class for use with fastlib's trees.  This one only stores two values.
 *
 * @param max_neighbor_distance The upper bound on the distance to the nearest 
 * neighbor of any point in this node.
 *
 * @param component_membership The index of the component that all points in 
 * this node belong to.  This is the same index returned by UnionFind for all
 * points in this node.  If points in this node are in different components, 
 * this value will be negative.  
 */
class DTBCoverStat {
  
private:
  
  //double max_candidate_distance_;
  index_t component_membership_;
  //index_t candidate_ref_;
  //double distance_to_qnode_;
  ArrayList<double> distance_to_qnode_;
  
public:
    
  bool not_connected(const DTBCoverStat& other) {
    
    if (component_membership_ < 0) {
      return true;
    }
    else {
      return (component_membership_ != other.component_membership());
    }
    
  }

  void set_component_membership(index_t membership) {
    component_membership_ = membership;
  }
  
  index_t component_membership() const {
    return component_membership_; 
  }

  void set_distance_to_qnode(double dist) {
    //printf("Setting qnode distance to %g\n", dist);
    //distance_to_qnode_ = dist;
    distance_to_qnode_.PushBackCopy(dist);
  }
  
  double distance_to_qnode() const {
    //return distance_to_qnode_;
    return distance_to_qnode_.back();
  }
  
  void pop_last_distance() {
    distance_to_qnode_.PopBack();
  }
  
  ArrayList<double>& distances() {
    return distance_to_qnode_;
  }
  
  /** 
    * A generic initializer.
    */
  void Init() {
    
    //set_max_neighbor_distance(DBL_MAX);
    set_component_membership(-1);
    //distance_to_qnode_ = -1.0;
    distance_to_qnode_.Init(0);
    
  }
  
  /**
    * An initializer for leaves.
   */
  void Init(const Matrix& dataset, index_t start, index_t count) {
    
    if (count == 1) {
      set_component_membership(start);
      //distance_to_qnode_ = -1.0;
      //set_max_neighbor_distance(DBL_MAX);
      distance_to_qnode_.Init(0);
    }
    else {
      Init();
    }
    
  }
  
  /**
    * An initializer for non-leaves.  Simply calls the leaf initializer.
   */
  void Init(const Matrix& dataset, index_t start, index_t count,
            const DTBCoverStat& left_stat, const DTBCoverStat& right_stat) {
    
    Init(dataset, start, count);
    
  }
  
}; // class DTBCoverStat


/**
 * Performs the MST calculation using the Dual-Tree Boruvka algorithm.
 */
class DualCoverTreeBoruvka {

  FORBID_ACCIDENTAL_COPIES(DualCoverTreeBoruvka);
  
 public:
  
  // For now, everything is in Euclidean space
  static const index_t metric = 2;

  //typedef BinarySpaceTree<DHrectBound<metric>, Matrix, DTBCoverStat> DTBTree;
  typedef CoverTreeNode<DTBCoverStat, double> DTBTree;
  
  //////// Member Variables /////////////////////
  
 private:
  
  index_t number_of_edges_;
  ArrayList<EdgePair> edges_;
  index_t number_of_points_;
  UnionFind connections_;
  struct datanode* module_;
  Matrix data_points_;
  double base_;
  
  // edges to be added
  ArrayList<index_t> neighbors_in_component_;
  ArrayList<index_t> neighbors_out_component_;
  ArrayList<double> neighbors_distances_;
  
  // bounds - these are indexed by component index
  ArrayList<double> candidate_dists_;
  // this stores the component of the candidate reference
  ArrayList<index_t> candidate_refs_;
  
  // used to eliminate pruning in debugging
  double prune_factor_;
  
  // output info
  double total_dist_;
  index_t number_of_loops_;
  /*
  index_t number_distance_prunes_;
  index_t number_component_prunes_;
  index_t number_leaf_computations_;
  index_t number_q_recursions_;
  index_t number_r_recursions_;
  index_t number_both_recursions_;
  */
  int do_naive_;
  
  //bool do_depth_first_;
  
  DTBTree* tree_;
   
  
////////////////// Constructors ////////////////////////
  
 public:

  DualCoverTreeBoruvka() {}
  
  ~DualCoverTreeBoruvka() {
    if (tree_ != NULL) {
      delete tree_; 
    }
  }
  
  
  ////////////////////////// Private Functions ////////////////////
 private:
    
  /**
  * Adds a single edge to the edge list
   */
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
    
  } // AddEdge_
  
  
  /**
   * Adds all the edges found in one iteration to the list of neighbors.
   */
  void AddAllEdges_() {
    
    for (index_t i = 0; i < number_of_points_; i++) {
      index_t component_i = connections_.Find(i);
      
      index_t in_edge_i = neighbors_in_component_[component_i];
      index_t out_edge_i = neighbors_out_component_[component_i];
      if (connections_.Find(in_edge_i) != connections_.Find(out_edge_i)) {
        double dist = neighbors_distances_[component_i];
        total_dist_ = total_dist_ + dist;
        AddEdge_(in_edge_i, out_edge_i, dist);
        connections_.Union(in_edge_i, out_edge_i);
      }
    }
    
  } // AddAllEdges_
  
  inline double compare_(DTBTree** p, DTBTree** q) {
    return ((*p)->stat().distance_to_qnode() - (*q)->stat().distance_to_qnode());
  } // compare_
  
  inline void swap_(DTBTree** p, DTBTree** q) {
    DTBTree* temp = *p;
    *p = *q;
    *q = temp;
  } // swap_()
  
  void SortReferences_(ArrayList<DTBTree*> *cover_set) {
    
    if (cover_set->size() <= 1) {
      // already sorted
      return;
    }
    
    register DTBTree **begin = cover_set->begin();
    DTBTree **end = &(cover_set->back());
    DTBTree **right = end;
    DTBTree **left;
    
    while (right > begin) {
      
      DTBTree **mid = begin + ((end - begin) >> 1);
      
      if (compare_(mid, begin) < 0.0) {
        swap_(mid, begin);
      }
      if (compare_(end, mid) < 0.0) {
        swap_(mid, end);
        if (compare_(mid, begin) < 0.0) {
          swap_(mid, begin);
        }
      }
      
      left = begin + 1;
      right = end - 1;
      
      do {
        
        while (compare_(left, mid) < 0.0) {
          left++;
        }
        
        while (compare_(mid, right) < 0.0) {
          right--;
        }
        
        if (left < right) {
          swap_(left, right);
          if (mid == left) {
            mid = right;
          }
          else if (mid == right) {
            mid = left;
          }
          left++;
          right--;
        }
        else if (left == right) {
          left ++;
          right--;
          break;
        }
      } while (left <= right);
      
      end = right;
    }
    
  } // SortReferences_
  
  /**
   * Determines if the bound for the parent is also valid for the child
   * Is not valid if the child is connected to the candidate for the parent
   */
  bool ValidBound_(index_t parent, index_t child) {
    
    return !(child == candidate_refs_[parent]);
    
  }
  
  void reset_leaf_nodes_(ArrayList<DTBTree*> *leaves) {
    
    DTBTree** begin = leaves->begin();
    DTBTree** end = leaves->end();
    
    for (; begin != end; begin++) {
      (*begin)->stat().pop_last_distance();
    }
    
  } // reset_leaf_nodes ()
  
  void reset_cover_sets_(ArrayList<ArrayList<DTBTree*> > *cover_set, 
                         index_t current_scale, index_t max_scale) {
    
    for (index_t i = current_scale; i <= max_scale; i++) {
      
      DTBTree** begin = (*cover_set)[i].begin();
      DTBTree** end = (*cover_set)[i].end();

      for (; begin != end; begin++) {
        
        (*begin)->stat().pop_last_distance();
        
      }
      
    } 
    
  } // reset_cover_sets()
  

  void ComputeBaseCase_(DTBTree* query, ArrayList<DTBTree*> *leaves) {
    
    //printf("Computing Base Case for query: %d\n", query->point());
    
    if (query->num_of_children() > 0) {
      
      DTBTree** child = query->children()->begin();
      ComputeBaseCase_(*child, leaves);
      
      DTBTree** child_end = query->children()->end();
      
      for (++child; child != child_end; child++) {
        
        ArrayList<DTBTree*> new_leaves;
        
        CopyLeafNodes_(*child, leaves, &new_leaves);
        
        ComputeBaseCase_(*child, &new_leaves);
        
        reset_leaf_nodes_(&new_leaves);
        
      } // iterate over children
      
    } // query not leaf
    else {
      
      //printf("query: %d\n", query->point());
      
      index_t query_comp = connections_.Find(query->point());
      
      for (index_t i = 0; i < leaves->size(); i++) {
        
        DTBTree* leaf = (*leaves)[i];
        
        index_t ref_comp = connections_.Find(leaf->point());
        
        //printf("Considering query: %d, leaf: %d\n", query->point(), leaf->point());
        /*
        if (query->point() == 0 || query->point() == 12) {
          printf("Found them\n");
        }
         */
        
        if (query_comp != ref_comp) {
          
#ifdef DEBUG
          // check that the correct distance is stored in the stat
      
          Vector q_vec, r_vec;
          data_points_.MakeColumnVector(query->point(), &q_vec);
          data_points_.MakeColumnVector(leaf->point(), &r_vec);
          
          double real_dist = sqrt(la::DistanceSqEuclidean(q_vec, r_vec));
          
          DEBUG_APPROX_DOUBLE(real_dist, leaf->stat().distance_to_qnode(), 10e-5);
          
#endif
          
          if (leaf->stat().distance_to_qnode() <= candidate_dists_[query_comp]) {
          //if (real_dist <= candidate_dists_[query_comp]) {  
          
            //candidate_dists_[query_comp] = real_dist;
            candidate_dists_[query_comp] = leaf->stat().distance_to_qnode();
            
            /*
            // this could be the new bound for the other component as well
            if (candidate_dists_[query_comp] < candidate_dists_[ref_comp]) {
              candidate_dists_[ref_comp] = candidate_dists_[query_comp];
              candidate_refs_[ref_comp] = query_comp;
            }
             */
            
            candidate_refs_[query_comp] = ref_comp;
            
            
            /*
            if (leaf->stat().distance_to_qnode() == 0.0) {
              printf("Adding zero distance\n");
            }
             */
            // added this check in debugging, doesn't help
            if (leaf->stat().distance_to_qnode() <= neighbors_distances_[query_comp]) {

              neighbors_distances_[query_comp] = leaf->stat().distance_to_qnode();
              //neighbors_distances_[query_comp] = real_dist;
              neighbors_in_component_[query_comp] = query->point();
              neighbors_out_component_[query_comp] = leaf->point();
              
              // can add a reverse bound here
              // this bound could work for the reference component
              if (neighbors_distances_[query_comp] < neighbors_distances_[ref_comp]) {
                neighbors_distances_[ref_comp] = neighbors_distances_[query_comp];
                neighbors_in_component_[ref_comp] = leaf->point();
                neighbors_out_component_[ref_comp] = query->point();
                
                // PROBLEM: can this prevent children of the reference points
                // from finding neighbors?
                candidate_dists_[ref_comp] = neighbors_distances_[ref_comp];
                
              }
            
            } // is it the edge to be added? 
          
          } // is it the new candidate
          
        } // is connected?
        
      } // iterate over reference leaves
      
    } // query is leaf
    
  } // ComputeBaseCase_
  
 
  void DescendRefSet_(DTBTree* query, ArrayList<ArrayList<DTBTree*> > *cover,
                      ArrayList<DTBTree*> *leaf_nodes, index_t current_scale, 
                      index_t *max_scale) {
    
    DTBTree** begin = (*cover)[current_scale].begin();
    DTBTree** end = (*cover)[current_scale].end();
    
    index_t query_comp = connections_.Find(query->point());
    DEBUG_ASSERT(query_comp == connections_.Find(query->point()));
    double query_bound = candidate_dists_[query_comp];
    index_t ref_comp = candidate_refs_[query_comp];
    
    Vector q_vec;
    data_points_.MakeColumnVector(query->point(), &q_vec);
    
    ArrayList<DTBTree*> ref_children;
    ref_children.Init(0);
    
    // fill in the reference children
    for (; begin != end; begin++) {
      
      // This is just a warning in Pari's code
      //DEBUG_ASSERT(current_scale == (*begin)->scale_depth());
      
#ifdef DEBUG
      
      Vector q_vec, r_vec;
      data_points_.MakeColumnVector(query->point(), &q_vec);
      data_points_.MakeColumnVector((*begin)->point(), &r_vec);
      
      double real_dist = sqrt(la::DistanceSqEuclidean(q_vec, r_vec));
      
      DEBUG_APPROX_DOUBLE(real_dist, (*begin)->stat().distance_to_qnode(), 10e-5);
      
#endif
      
      // added check for query's grandchild
      if ((*begin)->stat().distance_to_qnode() <= query_bound 
                         + (*begin)->max_dist_to_grandchild() 
                         + (2.0 * query->max_dist_to_grandchild())
                         + prune_factor_) {
      
        // iterate over the children of this member of the reference set
        DTBTree** child = (*begin)->children()->begin();
        DTBTree** child_end = (*begin)->children()->end();
        for (; child != child_end; child++) {
          
          if (query->stat().not_connected((*child)->stat())) {
            
            Vector r_vec;
            data_points_.MakeColumnVector((*child)->point(), &r_vec);
            
            double dist = sqrt(la::DistanceSqEuclidean(q_vec, r_vec));
            double dist_bound = dist;
            
            // the upper bound needs to be streched 
            
            // tried deleting this because points weren't all finding neighbors
            
            // IMPORTANT: need to keep the bound for this query from being set 
            // to itself
            // the problem case is when the reference considered here (*child)
            // is a leaf, and (*child)->point() == query->point(), but the 
            // query is not connected to all of its children.  
            
            if (query_comp == connections_.Find((*child)->point())) {
              if (unlikely((*child)->is_leaf())) {
                dist_bound = DBL_MAX;
              }
              // need to account for the possibility that the ref and all 
              // its descendants are connected to the query and thus not 
              // count any of those descendants toward the upper bound
              else if (unlikely(query_comp == (*child)->stat().component_membership())) {
                dist_bound = DBL_MAX;
              }
              // the query is connected to the reference, but not to all of the 
              // reference's children
              // this means that the query's component has a neighbor within 
              // the distance to the reference's children
              // PROBLEM: what about descendants of the query that aren't 
              // connected to the query?
              // the reference could be very close to its children, which
              // might lead this query to prune all of its potential neighbors, 
              // including itself
              // This would leave the descendants of the query without any 
              // neighbors
              // SOLN: only apply this when the query is connected to all of its
              // descendants
              // Have to make sure that the query and reference are far enough apart
              else if ((query_comp == query->stat().component_membership())
                       && (dist > query->max_dist_to_grandchild()
                                  + 2.0 * (*child)->max_dist_to_grandchild()) 
                       && ((*child)->max_dist_to_grandchild() < dist_bound)) {
                dist_bound = (*child)->max_dist_to_grandchild();
                //printf("New prune.\n");
              }
              else {
                dist_bound += (*child)->max_dist_to_grandchild();
                /*
                if ((*child)->max_dist_to_grandchild()) {
                  
                }
                 */
              }
            } // do we need the extra 2^i
            
            /*
            if (dist_bound == 0.0) {
              printf("dist bound of 0\n");
            }
             */
            
            // is the dist the new candidate to be the minimum? 
            if (dist_bound < query_bound) {
              query_bound = dist_bound;
              ref_comp = connections_.Find((*child)->point());
            } // is this the new d?
            
            //printf("query_bound: %g, dist_bound: %g\n", query_bound, dist_bound);
            
            
            if (dist <= query_bound + (*child)->max_dist_to_grandchild() 
                                   + (2.0 * query->max_dist_to_grandchild())
                                    + prune_factor_) {
             
              ref_children.PushBackCopy(*child);
              (*child)->stat().set_distance_to_qnode(dist);
              
            } // this child may make it to the next cover set
            
          } // is the query connected to this child?
          
        } // iterate over children of this reference
        
        
      } // does this reference still count?
      
    } // fill in the reference children
    
    candidate_dists_[query_comp] = query_bound;
    candidate_refs_[query_comp] = ref_comp;
    
    /*
    // this could be the new bound for the other component as well
    if (candidate_dists_[query_comp] < candidate_dists_[ref_comp]) {
      candidate_dists_[ref_comp] = candidate_dists_[query_comp];
      candidate_refs_[ref_comp] = query_comp;
    }
     */
    
    
    if (ref_children.size() > 0) {
      
      begin = ref_children.begin();
      end = ref_children.end();
      
      for (; begin != end; begin++) {
        
#ifdef DEBUG
        
        Vector q_vec, r_vec;
        data_points_.MakeColumnVector(query->point(), &q_vec);
        data_points_.MakeColumnVector((*begin)->point(), &r_vec);
        
        double real_dist = sqrt(la::DistanceSqEuclidean(q_vec, r_vec));
        
        DEBUG_ASSERT(real_dist == (*begin)->stat().distance_to_qnode());
        
#endif
        
        if ((*begin)->stat().distance_to_qnode() <= query_bound 
              + (*begin)->max_dist_to_grandchild() 
              + (2.0 * query->max_dist_to_grandchild())
              + prune_factor_) {
          
          if ((*begin)->num_of_children() > 0) {
            
            // update the max scale
            if (*max_scale < (*begin)->scale_depth()) {
              *max_scale = (*begin)->scale_depth();
            } // update max scale
            
            (*cover)[(*begin)->scale_depth()].PushBackCopy(*begin);
            
          } // if this node is not a leaf
          else {
            
            leaf_nodes->PushBackCopy(*begin);
            DEBUG_ASSERT((*begin)->scale_depth() == 100);
            
          } // node is a leaf
          
        } // does it meet the distance bound?
        
      } // iterate over reference children
      
    } // if there are any children at this scale
    
    // What if there aren't any valid children?
    // This can't happen, since one of the points in the reference cover set
    // is the candidate (or the parent of the candidate)
    
  } // DescendRefSet_()
  
  /**
   *
   */
  void CopyLeafNodes_(DTBTree* query, ArrayList<DTBTree*>* old_leaf, 
                      ArrayList<DTBTree*> *new_leaf) {
    
    new_leaf->Init(0);
    
    DTBTree** begin = old_leaf->begin();
    DTBTree** end = old_leaf->end();
    
    Vector q_vec;
    data_points_.MakeColumnVector(query->point(), &q_vec);
    
    index_t q_comp = connections_.Find(query->point());
    
    double upper_bound = candidate_dists_[q_comp];
    
    
    for (; begin != end; begin++) {
      
      // check if this leaf still works
      if (query->stat().not_connected((*begin)->stat())) {
        
        Vector r_vec;
        data_points_.MakeColumnVector((*begin)->point(), &r_vec);
        
        double dist = sqrt(la::DistanceSqEuclidean(q_vec, r_vec));
        
        if (dist <= upper_bound + (2.0 * query->max_dist_to_grandchild())
                    + prune_factor_) {

          // need to make sure this is a valid bound for the query
          // we may have to keep the leaf for some of the query's children
          // while the query itself can't use the leaf because they're already
          // connected
          if (q_comp != connections_.Find((*begin)->point())) {
            if (dist < upper_bound) {
              upper_bound = dist;
            }
          }
          
          (*begin)->stat().set_distance_to_qnode(dist);
          new_leaf->PushBackCopy(*begin);
          
        } // check distances
        
      } // are they fully connected?
      
    } // iterate over leaves
    
    candidate_dists_[q_comp] = upper_bound;
    
  } // CopyLeafNodes_()
  
  /**
   *
   */
  void CopyCoverSets_(DTBTree* query, 
                      ArrayList<ArrayList<DTBTree*> >* old_cover,
                      ArrayList<ArrayList<DTBTree*> >* new_cover, 
                      index_t current_scale, index_t max_scale) {
    
    
    new_cover->Init(101);
    for (index_t i = 0; i < 101; i++) {
      (*new_cover)[i].Init(0);
    }
    
    Vector q_vec;
    data_points_.MakeColumnVector(query->point(), &q_vec);
    
    index_t q_comp = connections_.Find(query->point());
    
    double upper_bound = candidate_dists_[q_comp];
    //index_t cand_ref = candidate_refs_[q_comp];
    
    for (index_t scale = current_scale; scale <= max_scale; scale++) {
      
      DTBTree** begin = (*old_cover)[scale].begin();
      DTBTree** end = (*old_cover)[scale].end();
      
      for (; begin != end; begin++) {
        
        // are they fully connected?
        if (query->stat().not_connected((*begin)->stat())) {
          
          Vector r_vec;
          data_points_.MakeColumnVector((*begin)->point(), &r_vec);
          
          double dist = sqrt(la::DistanceSqEuclidean(q_vec, r_vec));
          double dist_bound = dist;
          
          /*
          if (q_comp == connections_.Find((*begin)->point())) {
            dist_bound += (*begin)->max_dist_to_grandchild();
            DEBUG_ASSERT((*begin)->num_of_children() > 0);
          } 
          */
          // need this for same reason as above
          if (q_comp == connections_.Find((*begin)->point())) {
            if (unlikely((*begin)->is_leaf())) {
              dist_bound = DBL_MAX;
            }
            // need to account for the possibility that the ref and all 
            // its descendants are connected to the query and thus not 
            // count any of those descendants toward the upper bound
            else if (unlikely(q_comp == (*begin)->stat().component_membership())) {
              dist_bound = DBL_MAX;
            }
            else {
              dist_bound += (*begin)->max_dist_to_grandchild();
              /*
               if ((*child)->max_dist_to_grandchild()) {
               
               }
               */
            }
          } // do we need the extra 2^i
          
          if (dist_bound < upper_bound) {
            upper_bound = dist_bound;
          }
          
          
          if (dist <= upper_bound + (*begin)->max_dist_to_grandchild() 
                                  + (2.0 * query->max_dist_to_grandchild())
                                  + prune_factor_) {
            
            (*begin)->stat().set_distance_to_qnode(dist);
            (*new_cover)[scale].PushBackCopy(*begin);
          
          } // distance check
          
        } // are they fully connected?
        
      } // iterate over the nodes at this scale
      
    } // iterate over the scales
    
    
    candidate_dists_[q_comp] = upper_bound;
    
  } // CopyCoverSets_()
  
  
  /**
   * The expansion pattern in the algorithm proof
   *
   * ref_cover[i][j] is the jth node at scale level i
   */
  void HybridExpansion_(DTBTree* query, 
                        ArrayList<ArrayList<DTBTree*> > *ref_cover, 
                        ArrayList<DTBTree*> *leaf_nodes, index_t current_scale,
                        index_t max_scale) {
    
    //index_t query_comp_index = connections_.Find(query->point());
    
    if (current_scale > max_scale) {
      // base case
      
      //printf("Computing base case.\n");
      
      // What about the remaining references in the cover set?
      // There shouldn't be any because of the max_scale
      ComputeBaseCase_(query, leaf_nodes);
      
    } // base case
    else if ((query->scale_depth() < current_scale)
             && (query->scale_depth() != 100)) {
      // descend query tree
      
      //printf("Query Tree Descend, query = %d\n", query->point());
      
      DTBTree** child = query->children()->begin();
      DTBTree** child_end = query->children()->end();
      
      //double query_bound = candidate_dists_[query_comp_index];
      
      for (++child; child != child_end; child++) {
        
        ArrayList<DTBTree*> new_leaf;
        ArrayList<ArrayList<DTBTree*> > new_cover;
        
        //index_t child_comp_index = connections_.Find((*child)->point());
        
        // I don't think I need to pass bounds down
        // if the child doesn't already have a bound, then it will be able to 
        // find one right away
        // The best available reference at the current scale will immediately be 
        // available, since we're doing the breadth-first expansion on 
        // references
        
        CopyLeafNodes_(*child, leaf_nodes, &new_leaf);
        CopyCoverSets_(*child, ref_cover, &new_cover, current_scale, max_scale);
        
        // do the recursion
        
        HybridExpansion_(*child, &new_cover, &new_leaf, current_scale, 
                         max_scale);
        
        
        reset_leaf_nodes_(&new_leaf);
        reset_cover_sets_(&new_cover, current_scale, max_scale);
        
      } // iterate over children 
      
      // do the self-child
      
      HybridExpansion_(query->child(0), ref_cover, leaf_nodes, current_scale, 
                       max_scale);
      
    } // descend query
    else {
      // descend references 
      
      //printf("Reference Tree Descend, query = %d\n", query->point());
      
      index_t new_max = max_scale;
      DescendRefSet_(query, ref_cover, leaf_nodes, current_scale, &new_max);
      index_t new_current = current_scale + 1;
      HybridExpansion_(query, ref_cover, leaf_nodes, new_current, new_max);
      
    } // descend references
    
  } // HybridExpansion_()
  
  /**
    * Computes the nearest neighbor of each point in each iteration 
   * of the algorithm
   */
  void ComputeNeighbors_() {
    /*
    if (do_depth_first_) {
      FATAL("Depth first no longer supported.\n");
      //DepthFirst_(tree_, tree_);
    }
    else {
     */ 
     ArrayList<ArrayList<DTBTree*> > cover;
      cover.Init(101);
      for (index_t i = 0; i < 101; i++) {
        cover[i].Init(0);
      }
      ArrayList<DTBTree*> leaves;
      leaves.Init(0);
      
      tree_->stat().set_distance_to_qnode(0.0);
      
      
      cover[0].PushBackCopy(tree_);
      
      // Current scale is set to 0 here, but the scale of the root seems to be 
      // 1 sometimes?
      HybridExpansion_(tree_, &cover, &leaves, 0, 0);
      
    //}
  } // ComputeNeighbors_
  
  /**
   *
   */
  
  struct EdgeSortHelper_ {
  
    bool operator() (const EdgePair& pairA, const EdgePair& pairB) {
      return (pairA.distance() > pairB.distance());
    }
    
  } SortFun;
  
  /**
   *
   */
  void SortEdges_() {
    
    std::sort(edges_.begin(), edges_.end(), SortFun);
    
  } // SortEdges_()
  
  
  /**
   * Sort the edges and output them.
   */
  void EmitResults_(Matrix* results) {
    
    SortEdges_();
    
    DEBUG_ASSERT(number_of_edges_ == number_of_points_ - 1);
    results->Init(3, number_of_edges_);
    
    for (index_t i = 0; i < number_of_edges_; i++) {
      results->set(0, i, edges_[i].lesser_index());
      results->set(1, i, edges_[i].greater_index());
      results->set(2, i, edges_[i].distance());
    }
  
  } // EmitResults_
  
  
  
  /**
    * This function resets the values in the nodes of the tree
   * nearest neighbor distance, check for fully connected nodes
   */
  void CleanupHelper_(DTBTree* tree) {
    
    // descend children first
    if (!(tree->is_leaf())) {
      
      // iterate over children
      CleanupHelper_(tree->child(0));
      index_t comp = tree->child(0)->stat().component_membership();
      
      for (index_t i = 1; i < tree->num_of_children(); i++) {
        
        CleanupHelper_(tree->child(i));
        index_t this_comp = tree->child(i)->stat().component_membership();
        if (comp != this_comp) {
          comp = -1;
        }
        
      } // iterate over children
      
      tree->stat().set_component_membership(comp);
      
    } // descend children
    else {
      //printf("Setting leaf's component membership\n");
      tree->stat().set_component_membership(connections_.Find(tree->point()));
    }
     
    // leaf should already have the right component membership
    
    tree->stat().distances().Clear();
        
  } // CleanupHelper_
  
  /**
    * The values stored in the tree must be reset on each iteration.  
   */
  void Cleanup_() {
    
    //printf("Cleanup called\n");
    
    for (index_t i = 0; i < number_of_points_; i++) {
      
      neighbors_distances_[i] = DBL_MAX;
      DEBUG_ONLY(neighbors_in_component_[i] = -2);
      DEBUG_ONLY(neighbors_out_component_[i] = BIG_BAD_NUMBER);
      
      candidate_dists_[i] = DBL_MAX;
      candidate_refs_[i] = -1;
      
    }
    
    if (!do_naive_) {
      CleanupHelper_(tree_);
    }
  } // Cleanup()
  
  /**
    * Format and output the results
   */
  void OutputResults_() {
    
    //VERBOSE_ONLY(ot::Print(edges));
    
    fx_result_double(module_, "total_squared_length", total_dist_);
    fx_result_int(module_, "number_of_points", number_of_points_);
    fx_result_int(module_, "dimension", data_points_.n_rows());
    fx_result_int(module_, "number_of_loops", number_of_loops_);
    
    
    /*
    fx_format_result(module_, "number_distance_prunes", 
                     "%d", number_distance_prunes_);
    fx_format_result(module_, "number_component_prunes", 
                     "%d", number_component_prunes_);
    fx_format_result(module_, "number_leaf_computations", 
                     "%d", number_leaf_computations_);
    fx_format_result(module_, "number_q_recursions", 
                     "%d", number_q_recursions_);
    fx_format_result(module_, "number_r_recursions", 
                     "%d", number_r_recursions_);
    fx_format_result(module_, "number_both_recursions", 
                     "%d", number_both_recursions_);
    */
    
  } // OutputResults_
  
  /////////// Public Functions ///////////////////
  
 public: 
    
  index_t number_of_edges() {
    return number_of_edges_;
  }

  
  /**
   * Takes in a reference to the data set and a module.  Copies the data, 
   * builds the tree, and initializes all of the member variables.
   *
   */
  void Init(const Matrix& data, struct datanode* mod) {
    
    number_of_edges_ = 0;
    data_points_.Copy(data);
    module_ = mod;
    
    do_naive_ = fx_param_exists(module_, "do_naive");
    
    if (!do_naive_) {
      
      fx_timer_start(module_, "tree_building");
      
      fx_module* tree_mod = fx_submodule(module_, "tree");
      
      base_ = fx_param_double(module_, "base", 1.3);

      tree_ = ctree::MakeCoverTree<DTBTree, double>
          (data_points_, base_, tree_mod);
      
      fx_timer_stop(module_, "tree_building");
      
      //ctree::PrintTree(tree_);
      
    }
    else {
      tree_ = NULL; 
    }
    
    number_of_points_ = data_points_.n_cols();
    edges_.Init(number_of_points_-1);
    connections_.Init(number_of_points_);
    
    neighbors_in_component_.Init(number_of_points_);
    neighbors_out_component_.Init(number_of_points_);
    neighbors_distances_.Init(number_of_points_);    
    
    candidate_dists_.Init(number_of_points_);
    candidate_refs_.Init(number_of_points_);
    
    for (index_t i = 0; i < number_of_points_; i++) {
      candidate_dists_[i] = DBL_MAX;
      candidate_refs_[i] = -1;
      neighbors_in_component_[i] = -1;
      neighbors_out_component_[i] = -1;
      neighbors_distances_[i] = DBL_MAX;
    }
    
    // query and reference roots are the same
    tree_->stat().set_distance_to_qnode(0.0);
    
    total_dist_ = 0.0;
    number_of_loops_ = 0;
    /*
    number_distance_prunes_ = 0;
    number_component_prunes_ = 0;
    number_leaf_computations_ = 0;
    number_q_recursions_ = 0;
    number_r_recursions_ = 0;
    number_both_recursions_ = 0;
    */
    //do_depth_first_ = fx_param_bool(module_, "depth_first", false);
    
    prune_factor_ = fx_param_double(module_, "prune_factor", 0.0);
    
    Cleanup_();
    
  } // Init
    
    
  /**
   * Call this function after Init.  It will iteratively find the nearest 
   * neighbor of each component until the MST is complete.
   */
  void ComputeMST(Matrix* results) {
    
    fx_timer_start(module_, "MST_computation");
    
    while (number_of_edges_ < (number_of_points_ - 1)) {
      printf("=== Finding Neighbors ===\n");
      ComputeNeighbors_();
      
      printf("=== Adding Edges ===\n");
      AddAllEdges_();
      
      number_of_loops_++;
      
      Cleanup_();
    
      VERBOSE_ONLY(printf("number_of_loops = %d\n", number_of_loops_));
    }
    
    fx_timer_stop(module_, "MST_computation");
    
    if (results != NULL) {
     
      EmitResults_(results);
      
    }
    
    
    OutputResults_();
    
  } // ComputeMST
  
}; //class DualTreeBoruvka

#endif // inclusion guards
