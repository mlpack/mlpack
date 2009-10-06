/**
* @file dtb_cover.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Contains an implementation of the DualTreeBoruvka algorithm for finding a 
 * Euclidean Minimum Spanning Tree.  
 */

#ifndef DTB_COVER_H
#define DTB_COVER_H

#include "emst_cover.h"
#include "cover_tree.h"
#include "ctree.h"
#include "mlpack/emst/union_find.h"

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
class DTBStat {
  
private:
  
  //double max_candidate_distance_;
  index_t component_membership_;
  //index_t candidate_ref_;
  double distance_to_qnode_;
  
public:
    
  bool not_connected(const DTBStat& other) {
    
    if (component_membership_ < 0) {
      return true;
    }
    else {
      return (component_membership_ != other.component_membership());
    }
    
  }
  
  /*
  void set_max_candidate_distance(double distance) {
    max_candidate_distance_ = distance;
  }

  double max_candidate_distance() {
    return max_candidate_distance_;
  }
   */
  
  void set_component_membership(index_t membership) {
    component_membership_ = membership;
  }
  
  index_t component_membership() const {
    return component_membership_; 
  }
  /*
  index_t candidate_ref() const {
    return candidate_ref_;
  }
  
  void set_candidate_ref(index_t ref) {
    candidate_ref_ = ref;
  }
  */
  
  void set_distance_to_qnode(double dist) {
    distance_to_qnode_ = dist;
  }
  
  double distance_to_qnode() const {
    return distance_to_qnode_;
  }
  
  /** 
    * A generic initializer.
    */
  void Init() {
    
    //set_max_neighbor_distance(DBL_MAX);
    set_component_membership(-1);
    
  }
  
  /**
    * An initializer for leaves.
   */
  void Init(const Matrix& dataset, index_t start, index_t count) {
    
    if (count == 1) {
      set_component_membership(start);
      //set_max_neighbor_distance(DBL_MAX);
    }
    else {
      Init();
    }
    
  }
  
  /**
    * An initializer for non-leaves.  Simply calls the leaf initializer.
   */
  void Init(const Matrix& dataset, index_t start, index_t count,
            const DTBStat& left_stat, const DTBStat& right_stat) {
    
    Init(dataset, start, count);
    
  }
  
}; // class DTBStat


/**
 * Performs the MST calculation using the Dual-Tree Boruvka algorithm.
 */
class DualCoverTreeBoruvka {

  FORBID_ACCIDENTAL_COPIES(DualCoverTreeBoruvka);
  
 public:
  
  // For now, everything is in Euclidean space
  static const index_t metric = 2;

  //typedef BinarySpaceTree<DHrectBound<metric>, Matrix, DTBStat> DTBTree;
  typedef CoverTreeNode<DTBStat, double> DTBTree;
  
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
  
  // output info
  double total_dist_;
  index_t number_of_loops_;
  index_t number_distance_prunes_;
  index_t number_component_prunes_;
  index_t number_leaf_computations_;
  index_t number_q_recursions_;
  index_t number_r_recursions_;
  index_t number_both_recursions_;
  
  int do_naive_;
  
  bool do_depth_first_;
  
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
   * Finds the component of the candidate nearest neighbor for the given query
   *
   * Needs to check both arrays of neighbors, take the one that doesn't contain 
   * the query, and go with it.
   * Also needs a way to handle the non-existence of a candidate
   */
  /*
  index_t FindCandidateComp_(index_t query_comp) {
    
    index_t ret_comp;
    
    index_t in_comp = connections_.Find(neighbors_in_component_[query_comp]);
    index_t out_comp = connections_.Find(neighbors_out_component_[query_comp]);
    
    DEBUG_ASSERT(in_comp != out_comp);
    
    if (in_comp == query_comp) {
      ret_comp = out_comp;
    }
    else {
      ret_comp = in_comp;
    }
    
    return ret_comp;
    
  }
   */
  
  /**
   * Determines if the bound for the parent is also valid for the child
   * Is not valid if the child is connected to the candidate for the parent
   */
  bool ValidBound_(index_t parent, index_t child) {
    
    return !(child == candidate_refs_[parent]);
    
  }
  

  void ComputeBaseCase_(DTBTree* query, ArrayList<DTBTree*> *leaves) {
    
    if (query->num_of_children() > 0) {
      
      DTBTree** child = query->children()->begin();
      ComputeBaseCase_(*child, leaves);
      
      DTBTree** child_end = query->children()->end();
      
      for (++child; child != child_end; child++) {
        
        ArrayList<DTBTree*> new_leaves;
        
        CopyLeafNodes_(*child, leaves, &new_leaves);
        
        ComputeBaseCase_(*child, &new_leaves);
        
      } // iterate over children
      
    } // query not leaf
    else {
      
      index_t query_comp = connections_.Find(query->point());
      
      for (index_t i = 0; i < leaves->size(); i++) {
        
        DTBTree* leaf = (*leaves)[i];
        
        index_t ref_comp = connections_.Find(leaf->point());
        
        if (query_comp != ref_comp) {
          
#ifdef DEBUG
      
          Vector q_vec, r_vec;
          data_points_.MakeColumnVector(query->point(), &q_vec);
          data_points_.MakeColumnVector(leaf->point(), &r_vec);
          
          double real_dist = sqrt(la::DistanceSqEuclidean(q_vec, r_vec));
          
          DEBUG_APPROX_DOUBLE(real_dist, leaf->stat().distance_to_qnode(), 10e-5);
          
#endif
          
          if (leaf->stat().distance_to_qnode() < candidate_dists_[query_comp]) {
            
            candidate_dists_[query_comp] = leaf->stat().distance_to_qnode();
            candidate_refs_[query_comp] = ref_comp;
            
            neighbors_distances_[query_comp] = leaf->stat().distance_to_qnode();
            neighbors_in_component_[query_comp] = query->point();
            neighbors_out_component_[query_comp] = leaf->point();
            
          } // is it the new candidate
          
        } // is connected?
        
      } // iterate over reference leaves
      
    } // query is leaf
    
  } // ComputeBaseCase_
  
  
  /**
    * Handles the recursive calls to find the nearest neighbors in an iteration
   */
  void DepthFirst_(DTBTree *query_node, DTBTree *reference_node) {
   
    index_t query_comp_index = connections_.Find(query_node->point());
    index_t ref_comp_index = connections_.Find(reference_node->point());
    
    Vector q_vec, r_vec;
    data_points_.MakeColumnVector(query_node->point(), &q_vec);
    data_points_.MakeColumnVector(reference_node->point(), &r_vec);
    
    
    if (query_node->is_leaf() && reference_node->is_leaf()) {
      // base case
      
      // are they connected?
      if (query_comp_index != ref_comp_index) {
        
        double dist = sqrt(la::DistanceSqEuclidean(q_vec, r_vec));
        
        // is it close enough to be the new neighbor
        if (dist < candidate_dists_[query_comp_index]) {
          
          DEBUG_ASSERT(dist < neighbors_distances_[query_comp_index]);
          
          candidate_dists_[query_comp_index] = dist;
          candidate_refs_[query_comp_index] = ref_comp_index;
          neighbors_distances_[query_comp_index] = dist;
          neighbors_in_component_[query_comp_index] = query_node->point();
          neighbors_out_component_[query_comp_index] = reference_node->point();
          
        } // close enough to be new neighbor
        
      } // not connected
      
    }
    else if (reference_node->is_leaf()) {
      // only recurse on queries
      
      DTBTree** q_child = query_node->children()->begin();
      
      DepthFirst_(*q_child, reference_node);
      
      DTBTree** q_end = query_node->children()->end();
      
      // iterate over query children
      for (++q_child; q_child != q_end; q_child++) {
        
        index_t q_child_index = connections_.Find((*q_child)->point());
        
        Vector q_child_vec;
        data_points_.MakeColumnVector((*q_child)->point(), &q_child_vec);
        
        // pass the bound down to the query child
        if (ValidBound_(query_comp_index, q_child_index)) {
          
          // is it close enough to count
          if (candidate_dists_[query_comp_index] 
              < candidate_dists_[q_child_index]) {
            candidate_dists_[q_child_index] = candidate_dists_[query_comp_index];
            candidate_refs_[q_child_index] = candidate_refs_[query_comp_index];
          } // close enough
          
        } // can the bound be passed down due to connections?
        
        double query_upper_bound = candidate_dists_[q_child_index];
        query_upper_bound += 2.0 * (*q_child)->max_dist_to_grandchild();
        
        double ref_lower_bound = reference_node->stat().distance_to_qnode();
        ref_lower_bound -= (*q_child)->dist_to_parent();
        
        // preliminary distance check
        if (ref_lower_bound <= query_upper_bound) {
          
          double dist = sqrt(la::DistanceSqEuclidean(q_child_vec, r_vec));
          
          // real distance check
          if (dist <= query_upper_bound) {
            
            // update bound?
            if (dist < candidate_dists_[q_child_index]) {
              
              // is the bound valid
              index_t reference_comp_index = connections_.Find(reference_node->point());
              if (q_child_index != reference_comp_index) {
                candidate_dists_[q_child_index] = dist;
                candidate_refs_[q_child_index] = reference_comp_index;
              } // updating bound if valid
              
            } // is the bound close enough to update?
            
            reference_node->stat().set_distance_to_qnode(dist);
            DepthFirst_(*q_child, reference_node);
            
            // TODO: the allnn code pops the bound here, why?
            
          } // real distance check
          
        } // preliminary distance check
        
      } // iterate over query children
      
    } // only recurse on queries
    else {
      // recurse on both
      
      ArrayList<DTBTree*> ref_set;
      ref_set.Init(0);
      
      double dist_lower_bd = reference_node->stat().distance_to_qnode();
      dist_lower_bd = dist_lower_bd - query_node->max_dist_to_grandchild();
      dist_lower_bd = dist_lower_bd - reference_node->max_dist_to_grandchild();
      double cand_upper_bd = candidate_dists_[query_comp_index];
      cand_upper_bd += query_node->max_dist_to_grandchild();
      
      double query_upper = candidate_dists_[query_comp_index];
      query_upper += 2.0 * query_node->max_dist_to_grandchild();
      
      double ref_lower = reference_node->stat().distance_to_qnode();
      ref_lower -= reference_node->max_dist_to_grandchild();
      
      // recheck the pruning criterion
      if (ref_lower <= query_upper) {
        
        DTBTree** r_child = reference_node->children()->begin();
        
        // check that they're not connected
        if (query_node->stat().not_connected((*r_child)->stat())) {
        
          ref_lower = reference_node->stat().distance_to_qnode() 
                      - (*r_child)->max_dist_to_grandchild();
          
          // check the reference self-child
          if (ref_lower <= query_upper) {
            (*r_child)->stat().set_distance_to_qnode(reference_node->stat().distance_to_qnode());
            ref_set.PushBackCopy(*r_child);
          }
        } // check for connection
        
        DTBTree** r_end = reference_node->children()->end();
        
        // iterate over other reference children
        for (++r_child; r_child != r_end; r_child++) {
        
          // check for connection
          if (query_node->stat().not_connected((*r_child)->stat())) {
          
            // d(q, r_c) >= d(q, r) - d(r, r_c) 
            // this is a first approx
            ref_lower = reference_node->stat().distance_to_qnode();
            ref_lower -= (*r_child)->dist_to_parent();
            ref_lower -= (*r_child)->max_dist_to_grandchild();
            
            if (ref_lower <= query_upper) {
              
              Vector r_child_vec;
              data_points_.MakeColumnVector((*r_child)->point(), &r_child_vec);
              
              double dist = sqrt(la::DistanceSqEuclidean(q_vec, r_child_vec));
              
              ref_lower = dist - (*r_child)->max_dist_to_grandchild();
              
              // check true lower bound
              if (ref_lower <= query_upper) {
                
                // is this dist the new candidate
                if (dist < candidate_dists_[query_comp_index]) {
                  
                  // is the candidate valid for the query point
                  index_t r_child_comp_index = connections_.Find((*r_child)->point());
                  if (r_child_comp_index != query_comp_index) {
                    
                    candidate_dists_[query_comp_index] = dist;
                    candidate_refs_[query_comp_index] = r_child_comp_index;
                    
                  } // is the candidate valid for the query point
                  
                }// is this dist the new candidate
                
                (*r_child)->stat().set_distance_to_qnode(dist);
                ref_set.PushBackCopy(*r_child);
                
              } // check true lower bound
              
            } // check approx bound
              
          } // check connection
          
        } // iterate over reference children
        
      } // intial prune check
      
      // did we prune them all?
      if (ref_set.size() == 0){
        return; 
      }
      
      SortReferences_(&ref_set);
      
      DTBTree** r_begin = ref_set.begin();
      DTBTree** r_end = ref_set.end();
      
      if (query_node->is_leaf()) {
        
        //iterate over references
        for (DTBTree** begin = r_begin; begin != r_end; begin++) {
          
          // NOTE: the nodes were already checked to see if they're connected
          
          // close enough?
          if ((*begin)->stat().distance_to_qnode() 
              - (*begin)->max_dist_to_grandchild() 
              <= candidate_dists_[query_comp_index]) {
            
            DepthFirst_(query_node, *begin);
            
          } // passed distance prune
          
        } // iterating over references
        
      } // query is leaf
      else {
        // query isn't leaf
        
        DTBTree** q_child = query_node->children()->begin();
        DTBTree** q_end = query_node->children()->end();
        
        // descend all the self children, since their bounds and connectedness
        // were checked above
        for (DTBTree** begin = r_begin; begin != r_end; begin++) {
          
          DepthFirst_(*q_child, *begin);
          
        } // iterate over all references with query self-child
        
        // iterate over query children
        for (++q_child; q_child != q_end; q_child++) {
          
          index_t q_child_index = connections_.Find((*q_child)->point());
          
          Vector q_child_vec;
          data_points_.MakeColumnVector((*q_child)->point(), &q_child_vec);
          
          // pass the bound down to the query child
          if (ValidBound_(query_comp_index, q_child_index)) {
            
            // is it close enough to count
            if (candidate_dists_[query_comp_index] 
                < candidate_dists_[q_child_index]) {
              candidate_dists_[q_child_index] = candidate_dists_[query_comp_index];
              candidate_refs_[q_child_index] = candidate_refs_[query_comp_index];
            } // close enough
            
          } // can the bound be passed down due to connections?
          
          // iterate over references
          for (DTBTree** begin = r_begin; begin != r_end; begin++) {
            
            // check if the nodes are connected
            if ((*q_child)->stat().not_connected((*begin)->stat())) {
            
              double query_upper_bd = candidate_dists_[q_child_index];
              query_upper_bd += 2.0 * (*q_child)->max_dist_to_grandchild();
              
              double ref_lower_bd = (*begin)->stat().distance_to_qnode();
              ref_lower_bd -= (*begin)->max_dist_to_grandchild();
              // haven't yet computed the actual distance between this query
              // and reference - this is a lower bound using the
              // distance to qnode above
              ref_lower_bd -= (*q_child)->dist_to_parent();
              
              // preliminary distance check
              if (ref_lower_bd <= query_upper_bd) {
                
                Vector r_child_vec;
                data_points_.MakeColumnVector((*begin)->point(), &r_child_vec);
                
                double dist = sqrt(la::DistanceSqEuclidean(q_child_vec, r_child_vec));
                
                ref_lower_bd = dist - (*begin)->max_dist_to_grandchild();
                
                // real dist check
                if (ref_lower_bd <= query_upper_bd) {
                  
                  // only update the bound if it is a valid connection
                  index_t r_child_comp_index = connections_.Find((*begin)->point());
                  if (q_child_index != r_child_comp_index) {
                    // is this the new upper bound?
                    if (dist < candidate_dists_[q_child_index]) {
                      candidate_dists_[q_child_index] = dist;
                      candidate_refs_[q_child_index] = r_child_comp_index;
                    } // set new upper bound
                  } // is the new bound valid through non-connection?
                  
                  (*begin)->stat().set_distance_to_qnode(dist);
                  DepthFirst_(*q_child, *begin);
                  
                  // TODO: There's a pop here in the NN code, why?
                  
                } // real dist check
                
              } // preliminary distance check
                
            } // not connected
            
          } // iterate over references
          
        } // iterate over query non-self children
        
      } // query not leaf
      
    } // recurse on both
  
  } // DepthFirst_
  
  void DescendRefSet_(DTBTree* query, ArrayList<ArrayList<DTBTree*> > *cover,
                      ArrayList<DTBTree*> *leaf_nodes, index_t current_scale, 
                      index_t *max_scale) {
    
    DTBTree** begin = (*cover)[current_scale].begin();
    DTBTree** end = (*cover)[current_scale].end();
    
    index_t query_comp = connections_.Find(query->point());
    double query_bound = candidate_dists_[query_comp];
    
    Vector q_vec;
    data_points_.MakeColumnVector(query->point(), &q_vec);
    
    ArrayList<DTBTree*> ref_children;
    ref_children.Init(0);
    
    // fill in the reference children
    for (; begin != end; begin++) {
      
      DEBUG_ASSERT(current_scale == (*begin)->scale_depth());
      
      if (query_bound <= (*begin)->stat().distance_to_qnode() 
                         + (*begin)->max_dist_to_grandchild()) {
      
        // iterate over the children of this member of the reference set
        DTBTree** child = (*begin)->children()->begin();
        DTBTree** child_end = (*begin)->children()->end();
        for (; child != child_end; child++) {
          
          if (query->stat().not_connected((*child)->stat())) {
            
            Vector r_vec;
            data_points_.MakeColumnVector((*child)->point(), &r_vec);
            
            double dist = sqrt(la::DistanceSqEuclidean(q_vec, r_vec));
            (*child)->stat().set_distance_to_qnode(dist);
            double dist_bound = dist;
            
            // the upper bound needs to be streched 
            if (query_comp == connections_.Find((*child)->point())) {
              dist_bound += (*child)->max_dist_to_grandchild();
            } // do we need the extra 2^i
            
            // is the dist the new candidate to be the minimum? 
            if (dist_bound < query_bound) {
              query_bound = dist_bound;
            } // is this the new d?
            
            if (dist <= query_bound + (*child)->max_dist_to_grandchild() 
                                   + query->max_dist_to_grandchild()) {
             
              ref_children.PushBackCopy(*child);
              
            } // this child may make it to the next cover set
            
          } // is the query connected to this child?
          
        } // iterate over children of this reference
        
        
      } // does this reference still count?
      
    } // fill in the reference children
    
    // TODO: what about the candidate point?  I need to know it to pass the 
    // bounds down the tree
    candidate_dists_[query_comp] = query_bound;
    
    if (ref_children.size() > 0) {
      
      begin = ref_children.begin();
      end = ref_children.end();
      
      for (; begin != end; begin++) {
        
        if ((*begin)->stat().distance_to_qnode() <= query_bound 
              + (*begin)->max_dist_to_grandchild() 
              + query->max_dist_to_grandchild()) {
          
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
    
    // TODO: what if there aren't any valid children?
    // need to make sure the leaves get examined in the base case
    
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
        
        // don't need this case because the ref is a leaf
        //if (q_comp == connections_.Find((*begin)->point())) {
        //  dist_bound += (*begin)->max_dist_to_grandchild();
        //}
        
        if (dist <= upper_bound + query->max_dist_to_grandchild()) {

          if (dist < upper_bound) {
            upper_bound = dist;
          }
          
          (*begin)->stat().set_distance_to_qnode(dist);
          new_leaf->PushBackCopy(*begin);
          
        } // check distances
        
      } // are they fully connected?
      
    } // iterate over leaves
    
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
          if (q_comp == connections_.Find((*begin)->point())) {
            dist_bound += (*begin)->max_dist_to_grandchild();
          } 
          
          if (dist_bound < upper_bound) {
            upper_bound = dist_bound;
          }
          
          // do the distances work? 
          if (dist <= upper_bound + (*begin)->max_dist_to_grandchild() 
                                  + query->max_dist_to_grandchild()) {
            
            (*begin)->stat().set_distance_to_qnode(dist);
            (*new_cover)[scale].PushBackCopy(*begin);
            
          } // distance check
          
        } // are they fully connected?
        
      } // iterate over the nodes at this scale
      
    } // iterate over the scales
    
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
      
      ComputeBaseCase_(query, leaf_nodes);
      
    } // base case
    else if ((query->scale_depth() <= current_scale)
             && (query->scale_depth() != 100)) {
      // descend query tree
      
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
        
        // clean out the new lists somehow
        // the other code cleans out the distances to the qnodes
        // this shouldn't matter since it's depth-first here, right?
        
      } // iterate over children 
      
      // do the self-child
      
      HybridExpansion_(query->child(0), ref_cover, leaf_nodes, current_scale, 
                       max_scale);
      
    } // descend query
    else {
      // descend references 
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
    if (do_depth_first_) {
      DepthFirst_(tree_, tree_);
    }
    else {
      ArrayList<ArrayList<DTBTree*> > cover;
      cover.Init(101);
      for (index_t i = 0; i < 101; i++) {
        cover[i].Init(0);
      }
      ArrayList<DTBTree*> leaves;
      leaves.Init(0);
      
      tree_->stat().set_distance_to_qnode(0.0);
      
      cover[0].PushBackCopy(tree_);
      
      HybridExpansion_(tree_, &cover, &leaves, 0, 0);
      
    }
  } // ComputeNeighbors_
  
  
  /**
    * Unpermute the edge list and output it to results
   *
   * TODO: Make this sort the edge list by distance as well for hierarchical
   * clusterings.
   */
  void EmitResults_(Matrix* results) {
    
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
     
    // leaf should already have the right component membership
        
  } // CleanupHelper_
  
  /**
    * The values stored in the tree must be reset on each iteration.  
   */
  void Cleanup_() {
    
    for (index_t i = 0; i < number_of_points_; i++) {
      
      neighbors_distances_[i] = DBL_MAX;
      DEBUG_ONLY(neighbors_in_component_[i] = BIG_BAD_NUMBER);
      DEBUG_ONLY(neighbors_out_component_[i] = BIG_BAD_NUMBER);
      
      candidate_dists_[i] = DBL_MAX;
      candidate_refs_[i] = -1;
      
    }
    number_of_loops_++;
    
    if (!do_naive_) {
      CleanupHelper_(tree_);
    }
  } // Cleanup()
  
  /**
    * Format and output the results
   */
  void OutputResults_() {
    
    //VERBOSE_ONLY(ot::Print(edges));
    
    fx_format_result(module_, "total_squared_length", "%f", total_dist_);
    fx_format_result(module_, "number_of_points", "%d", number_of_points_);
    fx_format_result(module_, "dimension", "%d", data_points_.n_rows());
    fx_format_result(module_, "number_of_loops", "%d", number_of_loops_);
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
   * This module will be checked for the optional parameters "leaf_size" and 
   * "do_naive".  
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
    number_distance_prunes_ = 0;
    number_component_prunes_ = 0;
    number_leaf_computations_ = 0;
    number_q_recursions_ = 0;
    number_r_recursions_ = 0;
    number_both_recursions_ = 0;
    
    do_depth_first_ = fx_param_bool(module_, "depth_first", false);
    
  } // Init
    
    
  /**
   * Call this function after Init.  It will iteratively find the nearest 
   * neighbor of each component until the MST is complete.
   */
  void ComputeMST(Matrix* results) {
    
    fx_timer_start(module_, "MST_computation");
    
    while (number_of_edges_ < (number_of_points_ - 1)) {
      ComputeNeighbors_();
      
      AddAllEdges_();
      
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
