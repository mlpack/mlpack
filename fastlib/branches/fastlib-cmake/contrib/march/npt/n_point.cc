/**
 *  n_point.cc
 *  
 *
 *  Created by William March on 2/24/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "n_point.h"

int NPointAlg::NChooseR_(int n, int r) {
  
  DEBUG_ASSERT(n >= 0);
  DEBUG_ASSERT(r >= 0);
  
  if(n < r) return 0;
  
  int divisor = 1;
  int multiplier = n;
  
  int answer = 1;
  
  while (divisor <= r) {
    
    answer = (answer * multiplier) / divisor;
    
    multiplier--;
    divisor++;
    
  } 
  
  return answer;
  
} // NChooseR


int NPointAlg::CountTuples_(ArrayList<NPointNode*>& nodes) {
  
  // counts[i] = j means that there are j copies of node i
  // negative value means that this node is counted elsewhere
  ArrayList<int> counts;
  counts.Init(tuple_size_);
  
  for (index_t i = 0; i < tuple_size_; i++) {
    counts[i] = 1;
  } // initialize counts
  
  // check all pairs of nodes
  for (index_t i = 0; i < tuple_size_; i++) {
    
    NPointNode* node_i = nodes[i];
    
    for (index_t j = i+1; j < tuple_size_; j++) {
      
      NPointNode* node_j = nodes[j];
      
      // do they overlap
      if ((node_i->begin() >= node_j->end()) || 
          (node_j->begin() >= node_i->end())) {
        
        // don't need to do anything
        
      } // they definitely don't overlap
      else if ((node_i->begin() == node_j->begin()) &&
               (node_i->end() == node_j->end())) {
        
        // add 1, set j to -1
        counts[i] += 1;
        counts[j] = -1;
        
      } // the are identical
      else if (node_j->count() > node_i->count()) {
        
        // split j and recurse
        nodes[j] = node_j->left();
        int left_count = CountTuples_(nodes);
        nodes[j] = node_j->right();
        int right_count = CountTuples_(nodes);
        
        nodes[j] = node_j;
        return left_count + right_count;
        
      } // j is bigger
      else {
        
        // split i and recurse
        nodes[i] = node_i->left();
        int left_count = CountTuples_(nodes);
        nodes[i] = node_i->right();
        int right_count = CountTuples_(nodes);
        
        nodes[i] = node_i;
        return left_count + right_count;
        
      } // i is bigger
      
    } // for j
  } // for i
  
  // we didn't recurse, so now count them up

  int total_count = 1;
  
  for (index_t i = 0; i < tuple_size_; i++) {
    
    if (counts[i] > 0) {
      
      NPointNode* node_i = nodes[i];
      int size = node_i->count();
      
      // error here: I have a repeated node of size 1
      total_count *= NChooseR_(size, counts[i]);
      
    }
    
  } // count the sizes
  
  return total_count;
  
  
} // CountTuples_()


// returns true if the indices violate the symmetry requirement
bool NPointAlg::PointsViolateSymmetry_(index_t ind1, index_t ind2) {
  return (ind2 <= ind1);
} // PointsViolateSymmetry_()

int NPointAlg::BaseCaseHelper_(ArrayList<ArrayList<index_t> >& point_sets,
                                  ArrayList<bool>& permutation_ok,
                                  ArrayList<index_t>& points_in_tuple,
                                  double* weighted_result, int k) {
  
  int result = 0;
  
  ArrayList<bool> permutation_ok_copy;
  permutation_ok_copy.InitCopy(permutation_ok);
  
  ArrayList<index_t> k_rows;
  k_rows.InitAlias(point_sets[k]);
  
  bool bad_symmetry = false;
  
  // loop over possible points for the kth member of the tuple
  for (index_t i = 0; !bad_symmetry && i < k_rows.size(); i++) {
    
    index_t point_index_i = k_rows[i];
    
    bool this_point_works = true;

    Vector point_i;
    data_points_.MakeColumnVector(point_index_i, &point_i);
    
    // TODO: is this too inefficient?
    permutation_ok_copy.Clear();
    permutation_ok_copy.AppendCopy(permutation_ok);
    
    // loop over previously chosen points to see if there is a conflict
    // IMPORTANT: I'm assuming here that once a point violates symmetry,
    // all the other points will too
    // This means I'm assuming that the lists of points are in a continuous 
    // order
    for (index_t j = 0; !bad_symmetry && this_point_works && j < k; j++) {
      
      index_t point_index_j = points_in_tuple[j];
      
      // need to swap indices here, j should come before i
      bad_symmetry = PointsViolateSymmetry_(point_index_j, point_index_i);
      
      // don't compute the distances if we don't have to
      if (!bad_symmetry) {
        Vector point_j;
        data_points_.MakeColumnVector(point_index_j, &point_j);
        
        double point_dist_sq = la::DistanceSqEuclidean(point_i, point_j);
        
        //printf("Testing point pair (%d, %d)\n", j, k);
        this_point_works = matcher_.TestPointPair(point_dist_sq, j, k, 
                                                  permutation_ok_copy);
        
      } // compute the distances and check the matcher
      
    } // for j
    
    /*
    printf("Considering point %d in position %d.  bad_symmetry: %d, works: %d\n",
           point_index_i, k, bad_symmetry, this_point_works);
    */
    
    // now, if the point passed, we put it in place and recurse
    if (this_point_works && !bad_symmetry) {
      
      points_in_tuple[k] = point_index_i;
      
      // base case of the recursion
      if (k == tuple_size_ - 1) {
        
        result++;
        
        // take care of weights
        double these_weights = 1.0;
        
        for (index_t a = 0; a < num_tuples_; a++) {
          these_weights *= data_weights_[points_in_tuple[a]];
        } // for a
        
        (*weighted_result) = (*weighted_result) + these_weights;
        
      } // base case
      else {
        
        result += BaseCaseHelper_(point_sets, permutation_ok_copy, 
                                  points_in_tuple, weighted_result, k+1);
        
      } // recurse
      
      DEBUG_ONLY(points_in_tuple[k] = -1);
      
    } // did the point work
    
  } // for i
  
  return result;
  
} // BaseCaseHelper_()


int NPointAlg::BaseCase_(ArrayList<ArrayList<index_t> >& point_sets, 
                         double* weighted_result) {

  ArrayList<bool> permutation_ok;
  permutation_ok.Init(matcher_.num_permutations());
  for (index_t i = 0; i < matcher_.num_permutations(); i++) {
    permutation_ok[i] = true;
  } // for i
  
  ArrayList<index_t> points_in_tuple;
  points_in_tuple.Init(tuple_size_);
  
#ifdef DEBUG
  for (index_t j = 0; j < tuple_size_; j++) {
    points_in_tuple[j] = -1;
  } // for j
#endif
  
  int result = BaseCaseHelper_(point_sets, permutation_ok, points_in_tuple, 
                               weighted_result, 0);
  
  return result;
  
} // BaseCase_()


// Computes the heuristic priority of the node list
// IMPORTANT: lower number means higher priority
// Make the heuristic express how likely we are to be able to prune vs. how 
// likely we are to not have to bother because we can ignore this part of the 
// computation
double NPointAlg::HybridHeuristic_(ArrayList<NPointNode*>& nodes) {

  // for now, really dumb heuristic, just split the potentially largest tuples
  return -1.0 * CountTuples_(nodes);
  
} // HybridHeuristic_()

int NPointAlg::HybridExpansion_() {
  
  int num_tuples = 0;
  
  ArrayList<ArrayList<NPointNode*> > tuple_list;
  tuple_list.Init();
  
  ArrayList<NPointNode*> first_list;
  first_list.Init(tuple_size_);
  for (int i = 0; i < tuple_size_; i++) {
    first_list[i] = tree_;
  }
  
  tuple_list.PushBackCopy(first_list);
  
  upper_bound_ = CountTuples_(first_list);
  
  // loop until we can terminate
  while ((double)abs(upper_bound_ - num_tuples) / (double)num_tuples 
         >= error_tolerance_) {
    
    DEBUG_ASSERT(num_tuples >= 0);
    DEBUG_ASSERT(upper_bound_ >= num_tuples);
    
    int new_upper_bound = 0;
    
    //printf("upper_bound: %d, num_tuples: %d\n", upper_bound_, num_tuples);
    
    ArrayList<ArrayList<NPointNode*> > new_tuple_list;
    new_tuple_list.Init();
    
    // split the entire list
    for (index_t i = 0; i < tuple_list.size(); i++) {
      
      ArrayList<NPointNode*>& nodes = tuple_list[i];
      
      bool all_leaves = true;
      index_t split_ind;
      int split_size = -1;
      
      for (index_t i = 0; i < tuple_size_; i++) {
        
        if (!nodes[i]->is_leaf()) {
          all_leaves = false;
          
          // choose node to split
          // for now, splitting largest
          if (nodes[i]->count() > split_size) {
            split_ind = i;
            split_size = nodes[i]->count();
          }
          
        }
        
      } // check for leaves and splitting
      
      // is it a base case?
      if (all_leaves) {
        
        ArrayList<ArrayList<index_t> > point_sets;
        point_sets.Init(tuple_size_);
        for (index_t i = 0; i < tuple_size_; i++) {
          point_sets[i].Init(nodes[i]->count());
          
          for (index_t j = 0; j < nodes[i]->count(); j++) {
            point_sets[i][j] = j + nodes[i]->begin();
          } // for j
          
        } // for i
        
        double this_weighted_result;
        
        int num_tuples_here = BaseCase_(point_sets, &this_weighted_result);
        
        num_tuples += num_tuples_here;
        //new_upper_bound += num_tuples_here;
        
        //printf("new_upper_bound: %d, num_tuples: %d\n", new_upper_bound, num_tuples);
        
      }
      else {
        
        ArrayList<NPointNode*> new_nodes;
        NPointNode* split_node = nodes[split_ind];
        
        nodes[split_ind] = split_node->left();
        
        int left_status = CheckNodeList_(nodes);
        // TODO: add subsume check
        if (left_status == EXCLUDE) {

          num_exclusion_prunes_++;
          
        }
        else if (left_status == BAD_SYMMETRY) {
          
          // do nothing
          
        }
        else {
          
          new_tuple_list.PushBackCopy(nodes);
          new_upper_bound += CountTuples_(nodes);
          //printf("new_upper_bound: %d, num_tuples: %d\n", new_upper_bound, num_tuples);
          
        } // what to do with left child?
        
        new_nodes.InitCopy(nodes);
        new_nodes[split_ind] = split_node->right();
        
        int right_status = CheckNodeList_(new_nodes);
        
        if (right_status == EXCLUDE) {
          
          num_exclusion_prunes_++;
          
        }
        else if (right_status == BAD_SYMMETRY) {
          
          // do nothing
          
        }
        else {
          
          new_tuple_list.PushBackCopy(new_nodes);
          new_upper_bound += CountTuples_(new_nodes);
          //printf("new_upper_bound: %d, num_tuples: %d\n", new_upper_bound, num_tuples);
          
        } // what to do with right child?
        
      } // not all leaves
      
    } // iterate over all tuples in the list

    // copy the list over and repeat
    //tuple_list.Destruct();
    //tuple_list.InitSteal(new_tuple_list.begin(), new_tuple_list.size());
    tuple_list.Swap(&new_tuple_list);
    upper_bound_ = new_upper_bound + num_tuples;
    
  } // main loop
  
  return num_tuples;
  
  
} // HybridExpansion_()


// Determines if the list of nodes violates the matcher
int NPointAlg::CheckNodeList_(ArrayList<NPointNode*>& nodes) {
  
  int return_status = INCONCLUSIVE;
  
  ArrayList<int> permutation_ok;
  permutation_ok.Init(matcher_.num_permutations());
  for (index_t i = 0; i < permutation_ok.size(); i++) {
    permutation_ok[i] = SUBSUME;
  }
  
  for (index_t i = 0; i < tuple_size_; i++) {
    
    NPointNode* node_i = nodes[i];
    
    for (index_t j = i+1; j < tuple_size_; j++) {
      
      NPointNode* node_j = nodes[j];
      
      // TODO: would it be more efficient to make this check outside, without
      // possibly doing some permutation checks with the matcher
      // Another possibility would be to check this before making a recursive 
      // call -> this is closer to the way my HF code works
      
      // check if the nodes are in the right order, if not, return 0
      //if (node_j->stat().node_index() < node_i->stat().node_index()) {
      if (node_j->end() <= node_i->begin()) {  
        
        //printf("Returning for violated symmetry.\n\n");
        return BAD_SYMMETRY;
        
      }
      
      int status = matcher_.TestHrectPair(node_i->bound(), node_j->bound(), i, 
                                          j, permutation_ok);
      
      // IMPORTANT: can't exit here, messes up the tracking of bounds
      if (status == EXCLUDE) {
        // we should be able to prune
        
        return_status = EXCLUDE;
        
      } // are we able to exclude this n-tuple?
      
      
    } // looping over other nodes (j)
    
  } // looping over nodes in the tuple (i)
  
  // TODO: check if it is really subsume here
  
  return return_status;
  
  
} //CheckNodeList_()


// TODO: make this handle weighted results, should be easy
int NPointAlg::DepthFirstRecursion_(ArrayList<NPointNode*>& nodes, 
                                    index_t previous_split) {
  
  //printf("Depth first recursion on nodes: \n");
  //printf("Node1: [%d, %d); Node2: [%d, %d)\n", nodes[0]->begin(),
  //       nodes[0]->end(), nodes[1]->begin(), nodes[1]->end());
  
  DEBUG_ASSERT(nodes.size() == tuple_size_);
  
  bool all_leaves = true;
  
  int num_tuples_here = 0;
  
  index_t split_index = -1;
  int split_count = -1;
  
  // main loop
  for (index_t i = 0; i < tuple_size_; i++) {
    
    NPointNode* node_i = nodes[i];
    
    if (!(node_i->is_leaf())) {
      all_leaves = false;
    
      // is this the largest non-leaf?
      if (node_i->count() > split_count) {
        split_index = i;
        split_count = node_i->count();
      }
      
    } // is this a leaf?

  } // looping over nodes in the tuple (i)
  
  int status = CheckNodeList_(nodes);
  
  if (status == EXCLUDE) {
    num_exclusion_prunes_++;
    num_tuples_here = 0;
    return num_tuples_here;
  }
  else if (status == BAD_SYMMETRY) {
    num_tuples_here = 0;
    return num_tuples_here;
  }
  // TODO: add subsuming prunes
  
  // recurse
  if (all_leaves) {
    // call the base case
    // TODO: is it worth doing this after the prune checks?  
    // i.e. if it is an all leaf case, should we check this in the beginning and call the 
    // base case before trying to prune, since the base case will make 
    // basically the same checks anyway
    
    double this_weighted_result = 0.0;
    
    // fill in the array of point indices
    ArrayList<ArrayList<index_t> > point_sets;
    point_sets.Init(tuple_size_);
    for (index_t i = 0; i < tuple_size_; i++) {
      point_sets[i].Init(nodes[i]->count());
      
      for (index_t j = 0; j < nodes[i]->count(); j++) {
        point_sets[i][j] = j + nodes[i]->begin();
      } // for j
      
    } // for i
    
    //printf("Doing BaseCase on pair (%d, %d)\n", point_sets[0][0], point_sets[1][0]);
    num_tuples_here = BaseCase_(point_sets, &this_weighted_result);
    //printf("tuples in this base case: %d\n\n", num_tuples_here);
    
  } // base case
  else {
    
    // TODO: be clever about which pairs I have to test here
    // I should only have to check the n pairs that involve one of the new 
    // nodes, since all the rest can't have changed
    // I think this will require storing more details about what caused a 
    // permutation to be rejected before
    // This is referenced somewhere in the auton code, npt3.c?
    
    // For now, the heuristic is to split the largest non-leaf node
    NPointNode* split_node = nodes[split_index];
    
    //printf("Recursing on tuple %d\n", split_index);
    
    nodes[split_index] = split_node->left();
    num_tuples_here += DepthFirstRecursion_(nodes, split_index);
    nodes[split_index] = split_node->right();
    num_tuples_here += DepthFirstRecursion_(nodes, split_index);
    
    nodes[split_index] = split_node;
  
    //printf("Returning after recursive calls.\n\n");
    
  } // not base case
  
  return num_tuples_here;
  
} // DepthFirstRecursion_()




