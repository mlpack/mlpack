#ifndef COVER_TREE_NODE_GC_H
#define COVER_TREE_NODE_GC_H

#include <fastlib/fastlib.h>

template<class TStatistic, typename T>
class CoverTreeNode {

 public:
  typedef TStatistic Statistic;

 private:
  size_t point_;
  T max_dist_to_grandchild_;
  T dist_to_parent_;
  ArrayList<CoverTreeNode*> children_;
  size_t num_of_children_;
  size_t scale_depth_; // depth of the node in terms of scale
  Statistic stat_;

 public:

  CoverTreeNode() {
    children_.Init(0);
    stat_.Init();
  }

  ~CoverTreeNode() {
    for (size_t i = 0; i < num_of_children_; i++) {
      delete children_[i];
    }
  }

  void set_point(size_t point) {
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

  void set_num_of_children(size_t n) {
     num_of_children_ = n;
    return;
  }

  void set_scale_depth(size_t scale_depth) {
    scale_depth_ = scale_depth;
    return;
  }

  size_t point() {
    return point_;
  }

  T max_dist_to_grandchild() {
    return max_dist_to_grandchild_;
  }

  T dist_to_parent() {
    return dist_to_parent_;
  }

  ArrayList<CoverTreeNode*> *children() {
    return &children_;
  }

  CoverTreeNode *child(size_t i) {
    return children_[i];
  }

  size_t num_of_children() {
    return num_of_children_;
  }

  size_t scale_depth() {
    return scale_depth_;
  }

  Statistic& stat() {
    return stat_;
  }

  void MakeNode(size_t point) {
    point_ = point;
    //stat_.Init();
    return;
  }

  void MakeNode(size_t point, T max_dist, 
		size_t scale_depth, 
		ArrayList<CoverTreeNode*> *children) {

    point_ = point;
    max_dist_to_grandchild_ = max_dist;
    scale_depth_ = scale_depth;
    children_.Renew();
    children_.InitSteal(children);
    num_of_children_ = children_.size();
    //stat_.Init();
    
    return;
  }

  void MakeLeafNode(size_t point) {
    
    point_ = point;
    max_dist_to_grandchild_ = 0.0;
    dist_to_parent_ = 0.0;
    num_of_children_ = 0;
    scale_depth_ = 100;
    //stat_.Init();

    return;
  }

  bool is_leaf() {
    return num_of_children_ == 0;
  }

};

#endif
