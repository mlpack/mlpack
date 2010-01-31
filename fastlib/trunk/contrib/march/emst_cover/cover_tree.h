/**
 * @file cover_tree.h
 * 
 * This file defines a cover tree node. 
 *
 */

#ifndef COVER_TREE_NODE_H
#define COVER_TREE_NODE_H

#include <fastlib/fastlib.h>

template<class TStatistic, typename T>
class CoverTreeNode {

 public:
  typedef TStatistic Statistic;

 private:
  // the point in the dataset
  index_t point_;

  // the distance to the farthest descendant
  T max_dist_to_grandchild_;

  // distance to the parent
  T dist_to_parent_;

  // the list of children, the first one being the 
  // self child
  ArrayList<CoverTreeNode*> children_;

  // the number of children
  index_t num_of_children_;

  // depth of the node in terms of scale
  index_t scale_depth_; 

  // the cached statistic
  Statistic stat_;

 public:

  CoverTreeNode() {
    children_.Init(0);
    stat_.Init();
  }

  ~CoverTreeNode() {
    for (index_t i = 0; i < num_of_children_; i++) {
      delete children_[i];
    }
  }

  // setters
  void set_point(index_t point) {
    point_ = point;
  }

  void set_max_dist_to_grandchild(double dist) {
    max_dist_to_grandchild_ = dist;
    return;
  }

  void set_dist_to_parent(double dist) {
    dist_to_parent_ = dist;
    return;
  }

  void set_num_of_children(index_t n) {
     num_of_children_ = n;
    return;
  }

  void set_scale_depth(index_t scale_depth) {
    scale_depth_ = scale_depth;
    return;
  }

  index_t point() {
    return point_;
  }

  // getters
  T max_dist_to_grandchild() {
    return max_dist_to_grandchild_;
  }

  T dist_to_parent() {
    return dist_to_parent_;
  }

  ArrayList<CoverTreeNode*> *children() {
    return &children_;
  }

  CoverTreeNode *child(index_t i) {
    return children_[i];
  }

  index_t num_of_children() {
    return num_of_children_;
  }

  index_t scale_depth() {
    return scale_depth_;
  }

  Statistic& stat() {
    return stat_;
  }

  void MakeNode(index_t point) {
    point_ = point;
    return;
  }

  void MakeNode(index_t point, T max_dist, 
		index_t scale_depth, 
		ArrayList<CoverTreeNode*> *children) {

    point_ = point;
    max_dist_to_grandchild_ = max_dist;
    scale_depth_ = scale_depth;
    children_.Renew();
    children_.InitSteal(children);
    num_of_children_ = children_.size();
    return;
  }

  void MakeLeafNode(index_t point) {
    
    point_ = point;
    max_dist_to_grandchild_ = 0.0;
    dist_to_parent_ = 0.0;
    num_of_children_ = 0;
    scale_depth_ = 100;
    return;
  }

  // helper function
  bool is_leaf() {
    return num_of_children_ == 0;
  }

};

#endif
