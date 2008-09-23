/*
 * =====================================================================================
 * 
 *       Filename:  splitter.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  09/17/2008 09:17:39 AM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef SPLITTER_H_
#define SPLITTER_H_
#include "fastlib/fastlib.h"
#include "gop_nmf.h"



class SimpleSplitter {
 public:
  void Init() {
 
  }
  
  success_t Split(Matrix &lower_bound, Matrix &upper_bound, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound) {
    double widest_range=0;
    index_t widest_range_i=-1;
    index_t widest_range_j=-1;
    for(index_t i=0; i<lower_bound.n_rows(); i++) {
      for(index_t j=0; j<lower_bound.n_cols(); j++) {
        DEBUG_ASSERT(upper_bound.get(i, j) >= lower_bound.get(i,j));
        if (upper_bound.get(i, j) - lower_bound.get(i, j) > widest_range) {
          widest_range=upper_bound.get(i, j) - lower_bound.get(i, j);
          widest_range_i=i;
          widest_range_j=j;
        }
      }
    }
    left_lower_bound->Copy(lower_bound);
    left_upper_bound->Copy(upper_bound);
    right_lower_bound->Copy(lower_bound);
    right_upper_bound->Copy(upper_bound);
    double split_value=(upper_bound.get(widest_range_i, widest_range_j)
          +lower_bound.get(widest_range_i, widest_range_j))/2;

    left_upper_bound->set(widest_range_i, widest_range_j, split_value);
    right_lower_bound->set(widest_range_i, widest_range_j, split_value); 
    return SUCCESS_PASS;
  }
  
  success_t ChangeState(SolutionPack &solution) {
    return SUCCESS_PASS;
  }
  bool CanSplitMore() {
    return false;
  }
};

/* *****************************************************
 * *****************************************************
 * This version preserves the tree on w and the tree on h
 */
class TreeOnWandTreeOnHSplitter {
 public: 
  typedef BinarySpaceTree<DHrectBound<2>, Matrix> TreeType;
  void Init(fx_module *module, 
            Matrix &data_points) {
    module_=module;
    w_leaf_size_=fx_param_int(module_, "w_leaf_size", 1);
    h_leaf_size_=fx_param_int(module_, "h_leaf_size", 1);
    w_offset_=fx_param_int_req(module_, "w_offset");
    h_offset_=fx_param_int_req(module_, "h_offset");
    h_length_=fx_param_int_req(module_, "h_length");
    minimum_interval_length_=fx_param_double(module_, "minimum_interval_length", 1e-2);
    
    DEBUG_ASSERT(w_leaf_size_ > 0);
    DEBUG_ASSERT(h_leaf_size_ > 0);
    Matrix w_points;
    w_points.Copy(data_points);
    w_tree_ = tree::MakeKdTreeMidpoint<TreeType>(w_points, 
                                                 w_leaf_size_, 
				                                         &w_old_from_new_points_, 
                                                 NULL);
    w_current_depth_=0;
    ComputeTreeDepth_(w_tree_, &w_tree_max_depth_);
    GoDownToDepth_(w_tree_, 
                   w_current_depth_, 
                   &w_point_nodes_);

    Matrix h_points;
    la::TransposeInit(data_points, &h_points);
    h_tree_ = tree::MakeKdTreeMidpoint<TreeType>(h_points, 
                                                 h_leaf_size_, 
				                                         &h_old_from_new_points_, 
                                                 NULL);
    h_current_depth_=0;
    ComputeTreeDepth_(h_tree_, &h_tree_max_depth_);
    GoDownToDepth_(h_tree_, 
                   h_current_depth_, 
                   &h_point_nodes_);

    w_too_small_to_be_split_=false;
    h_too_small_to_be_split_=false;
    split_on_w_=true;
    split_on_h_=false;
    rehash_w_=true;
  }
  
  success_t Split(Matrix &lower_bound, Matrix &upper_bound, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound) {
    if (split_on_w_==true && w_too_small_to_be_split_ == false) {
      if (SpitMatrix_(lower_bound, 
                  upper_bound, 
                  w_old_from_new_points_, 
                  w_point_nodes_,
                  w_offset_, 
                  left_lower_bound, left_upper_bound,
                  right_lower_bound, right_upper_bound) == SUCCESS_FAIL) {
        
        w_too_small_to_be_split_=true;
      }
      if (h_too_small_to_be_split_==false) {
        split_on_w_=false;
      } else {
        split_on_w_=true;
      }
    } else { 
      if (split_on_w_==false && h_too_small_to_be_split_ ==false) {
        NOTIFY("Split on h");
        if (SpitMatrix_(lower_bound, 
                        upper_bound, 
                        h_old_from_new_points_, 
                        h_point_nodes_,
                        h_offset_, 
                        left_lower_bound, left_upper_bound,
                        right_lower_bound, right_upper_bound)==SUCCESS_FAIL) {
        
          h_too_small_to_be_split_=true;
        }
        if (w_too_small_to_be_split_==false) {
          split_on_w_=true;
        } else {
          split_on_w_=false;
        }
      }
    }
    if (w_too_small_to_be_split_ && h_too_small_to_be_split_) {
        return SUCCESS_FAIL;
    }
    return SUCCESS_PASS; 
  }
  
  
  // if it returns false it means that no further optimization can be done
  success_t ChangeState(SolutionPack &solution) {
      Matrix opt_points;
    opt_points.Alias(solution.solution_);

    w_too_small_to_be_split_=false;
    h_too_small_to_be_split_=false;
//    if (rehash_w_==false){
      rehash_w_=true;
      h_current_depth_++;
      if (h_current_depth_ < h_tree_max_depth_) {
        h_point_nodes_.Renew();
        GoDownToDepth_(h_tree_, h_current_depth_, &h_point_nodes_);
      }
//    } else {
      rehash_w_=false;
      w_current_depth_++;
      if (w_current_depth_ < w_tree_max_depth_) {
        w_point_nodes_.Renew();
        GoDownToDepth_(w_tree_, w_current_depth_, &w_point_nodes_);
      }
//    }
    return SUCCESS_PASS;
  }

  bool CanSplitMore() {
    if (w_current_depth_<w_tree_max_depth_ || h_current_depth_<h_tree_max_depth_) {
      return true;
    } else {
      return false;
    }
  }
  
 private:
  fx_module *module_;
  index_t w_offset_;
  index_t h_offset_; 
  TreeType *w_tree_;
  TreeType *h_tree_;
  index_t w_leaf_size_;
  index_t h_leaf_size_;
  ArrayList<index_t> w_old_from_new_points_;
  ArrayList<index_t> h_old_from_new_points_;
  index_t w_current_depth_;
  index_t h_current_depth_;
  index_t w_tree_max_depth_;
  index_t h_tree_max_depth_;
  index_t h_length_;
  // here we have the grouping of points 
  ArrayList<std::pair<index_t, index_t> >  w_point_nodes_;
  ArrayList<std::pair<index_t, index_t> >  h_point_nodes_;
  bool split_on_w_;
  bool split_on_h_;
  bool rehash_w_; 
  bool w_too_small_to_be_split_; 
  bool h_too_small_to_be_split_;
  double minimum_interval_length_;
  
  void ComputeTreeDepth_(TreeType *tree_node, index_t *depth) {
    *depth=-1;
    ComputeTreeDepthRecursion_(tree_node, -1, depth);
  }

  void ComputeTreeDepthRecursion_(TreeType *tree_node, 
                                  index_t depth, 
                                  index_t *max_depth) {
     
     depth+=1; 
     if (tree_node->is_leaf()==false) {
       ComputeTreeDepthRecursion_(tree_node->left(), 
                                   depth,
                                   max_depth);
       
       ComputeTreeDepthRecursion_(tree_node->right(), 
                                  depth,
                                  max_depth);
     } else {
       if (depth>*max_depth) {
         *max_depth=depth;
       }
     }
    
  }
  
  void GoDownToDepth_(TreeType *tree_node, 
                      index_t desired_depth, 
                      ArrayList<std::pair<index_t, index_t> > *point_nodes) {
    point_nodes->Init();
    GoDownToDepthRecursion_(tree_node, 
                            -1,
                            desired_depth,
                            point_nodes);    
  }
 
  void GoDownToDepthRecursion_(TreeType *tree_node, 
                               index_t current_depth,
                               index_t desired_depth, 
                               ArrayList<std::pair<index_t, index_t> > *point_nodes) {

    current_depth+=1;
    if (tree_node->is_leaf()==false 
        && current_depth<desired_depth) {
      GoDownToDepthRecursion_(tree_node->left(),
                              current_depth,
                              desired_depth, 
                              point_nodes);
      GoDownToDepthRecursion_(tree_node->right(),
                              current_depth,
                              desired_depth, 
                              point_nodes);

      
    } else {
      std::pair<index_t, index_t> *ptr;
      ptr = point_nodes->PushBackRaw();
      ptr->first=tree_node->begin();
      ptr->second=tree_node->end();
    }
    
  }
 
  success_t SpitMatrix_(Matrix &lower_bound, Matrix &upper_bound, 
          ArrayList<index_t> &old_from_new_points, 
          ArrayList<std::pair<index_t, index_t> > &point_nodes,
          index_t offset, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound) {
   
    double widest_range=0;
    index_t widest_range_i=-1;
    index_t widest_range_j=-1;
    for(index_t i=0; i< point_nodes.size(); i++) {
      index_t p=offset+old_from_new_points[point_nodes[i].first];
      for(index_t j=0; j<lower_bound.n_rows(); j++) {
        if (upper_bound.get(j, p) - lower_bound.get(j, p) > widest_range) {
          widest_range=upper_bound.get(j, p) - lower_bound.get(j, p);
          widest_range_i=i;
          widest_range_j=j;
        }
      }
    }
      
    left_lower_bound->Copy(lower_bound);
    left_upper_bound->Copy(upper_bound);
    right_lower_bound->Copy(lower_bound);
    right_upper_bound->Copy(upper_bound);
    index_t p=offset+old_from_new_points[point_nodes[widest_range_i].first];
    double split_value=(upper_bound.get(widest_range_j, p)
        +lower_bound.get(widest_range_j, p))/2; 
    double interval_length=split_value - lower_bound.get(widest_range_j, p);
    if (interval_length < minimum_interval_length_) {
      return SUCCESS_FAIL;
    }
    
    NOTIFY("interval length:%lg w_depth:%i h_depth:%i", 
        interval_length, w_current_depth_, h_current_depth_);
    for(index_t i=point_nodes[widest_range_i].first; 
        i<point_nodes[widest_range_i].second; i++) {
      left_upper_bound->set(widest_range_j, offset+old_from_new_points[i], split_value);
      right_lower_bound->set(widest_range_j, offset+old_from_new_points[i], split_value); 
    }
    return SUCCESS_PASS;
  }
};

/* *****************************************************
 * *****************************************************
 * This version preserves the tree on w and builds a  tree on h
 */
class TreeOnWandBuildTreeOnHSplitter {
 public: 
  typedef BinarySpaceTree<DHrectBound<2>, Matrix> TreeType;
  void Init(fx_module *module, 
            Matrix &data_points) {
    module_=module;
    w_leaf_size_=fx_param_int(module_, "w_leaf_size", 1);
    h_leaf_size_=fx_param_int(module_, "h_leaf_size", 1);
    w_offset_=fx_param_int_req(module_, "w_offset");
    h_offset_=fx_param_int_req(module_, "h_offset");
    h_length_=fx_param_int_req(module_, "h_length");
    minimum_interval_length_=fx_param_double(module_, "minimum_interval_length", 1e-2);
    
    DEBUG_ASSERT(w_leaf_size_ > 0);
    DEBUG_ASSERT(h_leaf_size_ > 0);
    Matrix w_points;
    w_points.Copy(data_points);
    w_tree_ = tree::MakeKdTreeMidpoint<TreeType>(w_points, 
                                                 w_leaf_size_, 
				                                         &w_old_from_new_points_, 
                                                 NULL);
    w_current_depth_=0;
    ComputeTreeDepth_(w_tree_, &w_tree_max_depth_);
    GoDownToDepth_(w_tree_, 
                   w_current_depth_, 
                   &w_point_nodes_);

    h_tree_ = NULL;
    h_current_depth_=0;
    h_point_nodes_.Init();
    h_point_nodes_.PushBackCopy(std::make_pair(0, data_points.n_rows()));
    h_old_from_new_points_.Init();
    for(index_t i=0; i<data_points.n_rows(); i++) {
      h_old_from_new_points_.PushBackCopy(i);
    } 
    w_too_small_to_be_split_=false;
    h_too_small_to_be_split_=false;
    split_on_w_=true;
    split_on_h_=false;
    rehash_w_=true;
  }
  
  success_t Split(Matrix &lower_bound, Matrix &upper_bound, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound) {
    if (split_on_w_==true && w_too_small_to_be_split_ == false) {
      if (SpitMatrix_(lower_bound, 
                  upper_bound, 
                  w_old_from_new_points_, 
                  w_point_nodes_,
                  w_offset_, 
                  left_lower_bound, left_upper_bound,
                  right_lower_bound, right_upper_bound) == SUCCESS_FAIL) {
        
        w_too_small_to_be_split_=true;
      }
      if (h_too_small_to_be_split_==false) {
        split_on_w_=false;
      } else {
        split_on_w_=true;
      }
    } else { 
      if (split_on_w_==false && h_too_small_to_be_split_ ==false) {
        NOTIFY("Split on h");
        if (SpitMatrix_(lower_bound, 
                        upper_bound, 
                        h_old_from_new_points_, 
                        h_point_nodes_,
                        h_offset_, 
                        left_lower_bound, left_upper_bound,
                        right_lower_bound, right_upper_bound)==SUCCESS_FAIL) {
        
          h_too_small_to_be_split_=true;
        }
        if (w_too_small_to_be_split_==false) {
          split_on_w_=true;
        } else {
          split_on_w_=false;
        }
      }
    }
    if (w_too_small_to_be_split_ && h_too_small_to_be_split_) {
        return SUCCESS_FAIL;
    }
    return SUCCESS_PASS; 
  }
  
  
  // if it returns false it means that no further optimization can be done
  success_t ChangeState(SolutionPack &solution) {
      Matrix opt_points;
    opt_points.Alias(solution.solution_);

    w_too_small_to_be_split_=false;
    h_too_small_to_be_split_=false;
   // if (rehash_w_==false){
      rehash_w_=true;
      h_current_depth_++;
      if (h_tree_!=NULL) {
        delete h_tree_;
      }
      h_old_from_new_points_.Renew();
      Matrix h_points;
      h_points.Copy(solution.solution_.GetColumnPtr(h_offset_), 
                    solution.solution_.n_rows(), h_length_);
      for(index_t i=0; i<h_points.n_rows(); i++) {
        for(index_t j=0; j<h_points.n_cols(); j++) {
          h_points.set(i, j, exp(h_points.get(i, j)));
        }
      }
      h_tree_=tree::MakeKdTreeMidpoint<TreeType>(h_points, 
                                                 h_leaf_size_, 
				                                         &h_old_from_new_points_, 
                                                 NULL);

      ComputeTreeDepth_(h_tree_, &h_tree_max_depth_);
      if (h_current_depth_ <= h_tree_max_depth_) {
        h_point_nodes_.Renew();
        GoDownToDepth_(h_tree_, h_current_depth_, &h_point_nodes_);
      }
   // } else {
      rehash_w_=false;
      w_current_depth_++;
      if (w_current_depth_ <=w_tree_max_depth_) {
        w_point_nodes_.Renew();
        GoDownToDepth_(w_tree_, w_current_depth_, &w_point_nodes_);
      }
   // }
    return SUCCESS_PASS;
  }

  bool CanSplitMore() {
    if (w_current_depth_<w_tree_max_depth_ || h_current_depth_<h_tree_max_depth_) {
      return true;
    } else {
      return false;
    }
  }
  
 private:
  fx_module *module_;
  index_t w_offset_;
  index_t h_offset_; 
  TreeType *w_tree_;
  TreeType *h_tree_;
  index_t w_leaf_size_;
  index_t h_leaf_size_;
  ArrayList<index_t> w_old_from_new_points_;
  ArrayList<index_t> h_old_from_new_points_;
  index_t w_current_depth_;
  index_t h_current_depth_;
  index_t w_tree_max_depth_;
  index_t h_tree_max_depth_;
  index_t h_length_;
  // here we have the grouping of points 
  ArrayList<std::pair<index_t, index_t> >  w_point_nodes_;
  ArrayList<std::pair<index_t, index_t> >  h_point_nodes_;
  bool split_on_w_;
  bool split_on_h_;
  bool rehash_w_; 
  bool w_too_small_to_be_split_; 
  bool h_too_small_to_be_split_;
  double minimum_interval_length_;
  
  void ComputeTreeDepth_(TreeType *tree_node, index_t *depth) {
    *depth=-1;
    ComputeTreeDepthRecursion_(tree_node, -1, depth);
  }

  void ComputeTreeDepthRecursion_(TreeType *tree_node, 
                                  index_t depth, 
                                  index_t *max_depth) {
     
     depth+=1; 
     if (tree_node->is_leaf()==false) {
       ComputeTreeDepthRecursion_(tree_node->left(), 
                                   depth,
                                   max_depth);
       
       ComputeTreeDepthRecursion_(tree_node->right(), 
                                  depth,
                                  max_depth);
     } else {
       if (depth>*max_depth) {
         *max_depth=depth;
       }
     }
    
  }
  
  void GoDownToDepth_(TreeType *tree_node, 
                      index_t desired_depth, 
                      ArrayList<std::pair<index_t, index_t> > *point_nodes) {
    point_nodes->Init();
    GoDownToDepthRecursion_(tree_node, 
                            -1,
                            desired_depth,
                            point_nodes);    
  }
 
  void GoDownToDepthRecursion_(TreeType *tree_node, 
                               index_t current_depth,
                               index_t desired_depth, 
                               ArrayList<std::pair<index_t, index_t> > *point_nodes) {

    current_depth+=1;
    if (tree_node->is_leaf()==false 
        && current_depth<desired_depth) {
      GoDownToDepthRecursion_(tree_node->left(),
                              current_depth,
                              desired_depth, 
                              point_nodes);
      GoDownToDepthRecursion_(tree_node->right(),
                              current_depth,
                              desired_depth, 
                              point_nodes);

      
    } else {
      std::pair<index_t, index_t> *ptr;
      ptr = point_nodes->PushBackRaw();
      ptr->first=tree_node->begin();
      ptr->second=tree_node->end();
    }
    
  }
 
  success_t SpitMatrix_(Matrix &lower_bound, Matrix &upper_bound, 
          ArrayList<index_t> &old_from_new_points, 
          ArrayList<std::pair<index_t, index_t> > &point_nodes,
          index_t offset, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound) {
   
    double widest_range=0;
    index_t widest_range_i=-1;
    index_t widest_range_j=-1;
    for(index_t i=0; i< point_nodes.size(); i++) {
      index_t p=offset+old_from_new_points[point_nodes[i].first];
      for(index_t j=0; j<lower_bound.n_rows(); j++) {
        if (upper_bound.get(j, p) - lower_bound.get(j, p) > widest_range) {
          widest_range=upper_bound.get(j, p) - lower_bound.get(j, p);
          widest_range_i=i;
          widest_range_j=j;
        }
      }
    }
      
    left_lower_bound->Copy(lower_bound);
    left_upper_bound->Copy(upper_bound);
    right_lower_bound->Copy(lower_bound);
    right_upper_bound->Copy(upper_bound);
    index_t p=offset+old_from_new_points[point_nodes[widest_range_i].first];
    double split_value=(upper_bound.get(widest_range_j, p)
        +lower_bound.get(widest_range_j, p))/2; 
    double interval_length=split_value - lower_bound.get(widest_range_j, p);
    if (interval_length < minimum_interval_length_) {
      return SUCCESS_FAIL;
    }
    
    NOTIFY("interval length:%lg w_depth:%i h_depth:%i", 
        interval_length, w_current_depth_, h_current_depth_);
    for(index_t i=point_nodes[widest_range_i].first; 
        i<point_nodes[widest_range_i].second; i++) {
      left_upper_bound->set(widest_range_j, offset+old_from_new_points[i], split_value);
      right_lower_bound->set(widest_range_j, offset+old_from_new_points[i], split_value); 
    }
    return SUCCESS_PASS;
  }
};

/* *****************************************************
 * *****************************************************
 * This version builds a  tree on w and builds a  tree on h
 */
class BuildTreeOnWandBuildTreeOnHSplitter {
 public: 
  typedef BinarySpaceTree<DHrectBound<2>, Matrix> TreeType;
  void Init(fx_module *module, 
            Matrix &data_points) {
    module_=module;
    w_leaf_size_=fx_param_int(module_, "w_leaf_size", 1);
    h_leaf_size_=fx_param_int(module_, "h_leaf_size", 1);
    w_offset_=fx_param_int_req(module_, "w_offset");
    h_offset_=fx_param_int_req(module_, "h_offset");
    h_length_=fx_param_int_req(module_, "h_length");
    minimum_interval_length_=fx_param_double(module_, "minimum_interval_length", 1e-2);
    
    DEBUG_ASSERT(w_leaf_size_ > 0);
    DEBUG_ASSERT(h_leaf_size_ > 0);
    w_tree_ = NULL;
    w_current_depth_=0;
    w_point_nodes_.Init();
    w_point_nodes_.PushBackCopy(std::make_pair(0, data_points.n_cols()));
    w_old_from_new_points_.Init();
    for(index_t i=0; i<data_points.n_cols(); i++) {
     w_old_from_new_points_.PushBackCopy(i);
    } 
    
    h_tree_ = NULL;
    h_current_depth_=0;
    h_point_nodes_.Init();
    h_point_nodes_.PushBackCopy(std::make_pair(0, data_points.n_rows()));
    h_old_from_new_points_.Init();
    for(index_t i=0; i<data_points.n_rows(); i++) {
      h_old_from_new_points_.PushBackCopy(i);
    } 
    w_too_small_to_be_split_=false;
    h_too_small_to_be_split_=false;
    split_on_w_=true;
    split_on_h_=false;
    rehash_w_=true;
  }
  
  success_t Split(Matrix &lower_bound, Matrix &upper_bound, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound) {
    if (split_on_w_==true && w_too_small_to_be_split_ == false) {
      if (SpitMatrix_(lower_bound, 
                  upper_bound, 
                  w_old_from_new_points_, 
                  w_point_nodes_,
                  w_offset_, 
                  left_lower_bound, left_upper_bound,
                  right_lower_bound, right_upper_bound) == SUCCESS_FAIL) {
        
        w_too_small_to_be_split_=true;
      }
      if (h_too_small_to_be_split_==false) {
        split_on_w_=false;
      } else {
        split_on_w_=true;
      }
    } else { 
      if (split_on_w_==false && h_too_small_to_be_split_ ==false) {
        NOTIFY("Split on h");
        if (SpitMatrix_(lower_bound, 
                        upper_bound, 
                        h_old_from_new_points_, 
                        h_point_nodes_,
                        h_offset_, 
                        left_lower_bound, left_upper_bound,
                        right_lower_bound, right_upper_bound)==SUCCESS_FAIL) {
        
          h_too_small_to_be_split_=true;
        }
        if (w_too_small_to_be_split_==false) {
          split_on_w_=true;
        } else {
          split_on_w_=false;
        }
      }
    }
    if (w_too_small_to_be_split_ && h_too_small_to_be_split_) {
        return SUCCESS_FAIL;
    }
    return SUCCESS_PASS; 
  }
  
  
  // if it returns false it means that no further optimization can be done
  success_t ChangeState(SolutionPack &solution) {
    w_too_small_to_be_split_=false;
    h_too_small_to_be_split_=false;
   // if (rehash_w_==false){
      rehash_w_=true;
      h_current_depth_++;
      if (h_tree_!=NULL) {
        delete h_tree_;
      }
      h_old_from_new_points_.Renew();
      Matrix h_points;
      h_points.Copy(solution.solution_.GetColumnPtr(h_offset_), 
                    solution.solution_.n_rows(), h_length_);
      for(index_t i=0; i<h_points.n_rows(); i++) {
        for(index_t j=0; j<h_points.n_cols(); j++) {
          h_points.set(i, j, exp(h_points.get(i, j)));
        }
      }
     
      h_tree_=tree::MakeKdTreeMidpoint<TreeType>(h_points, 
                                                 h_leaf_size_, 
				                                         &h_old_from_new_points_, 
                                                 NULL);
      
      ComputeTreeDepth_(h_tree_, &h_tree_max_depth_);
      if (h_current_depth_ <= h_tree_max_depth_) {
        h_point_nodes_.Renew();
        GoDownToDepth_(h_tree_, h_current_depth_, &h_point_nodes_);
      }
   // } else {
      rehash_w_=false;
      w_current_depth_++;
      if (w_tree_!=NULL) {
        delete w_tree_;
      }
      w_old_from_new_points_.Renew();
      Matrix w_points;
      w_points.Copy(solution.solution_.GetColumnPtr(w_offset_), 
                    solution.solution_.n_rows(), solution.solution_.n_cols()-h_length_);
      for(index_t i=0; i<w_points.n_rows(); i++) {
        for(index_t j=0; j<w_points.n_cols(); j++) {
          w_points.set(i, j, exp(w_points.get(i, j)));
        }
      }
      
      w_tree_=tree::MakeKdTreeMidpoint<TreeType>(w_points, 
                                                 w_leaf_size_, 
				                                         &w_old_from_new_points_, 
                                                 NULL);
      
      ComputeTreeDepth_(w_tree_, &w_tree_max_depth_);
      if (w_current_depth_ <= w_tree_max_depth_) {
        w_point_nodes_.Renew();
        GoDownToDepth_(w_tree_, w_current_depth_, &w_point_nodes_);
      }
    // }
    return SUCCESS_PASS;
  }

  bool CanSplitMore() {
    if (w_current_depth_<w_tree_max_depth_ || h_current_depth_<h_tree_max_depth_) {
      return true;
    } else {
      return false;
    }
  }
  
 private:
  fx_module *module_;
  index_t w_offset_;
  index_t h_offset_; 
  TreeType *w_tree_;
  TreeType *h_tree_;
  index_t w_leaf_size_;
  index_t h_leaf_size_;
  ArrayList<index_t> w_old_from_new_points_;
  ArrayList<index_t> h_old_from_new_points_;
  index_t w_current_depth_;
  index_t h_current_depth_;
  index_t w_tree_max_depth_;
  index_t h_tree_max_depth_;
  index_t h_length_;
  // here we have the grouping of points 
  ArrayList<std::pair<index_t, index_t> >  w_point_nodes_;
  ArrayList<std::pair<index_t, index_t> >  h_point_nodes_;
  bool split_on_w_;
  bool split_on_h_;
  bool rehash_w_; 
  bool w_too_small_to_be_split_; 
  bool h_too_small_to_be_split_;
  double minimum_interval_length_;
  
  void ComputeTreeDepth_(TreeType *tree_node, index_t *depth) {
    *depth=-1;
    ComputeTreeDepthRecursion_(tree_node, -1, depth);
  }

  void ComputeTreeDepthRecursion_(TreeType *tree_node, 
                                  index_t depth, 
                                  index_t *max_depth) {
     
     depth+=1; 
     if (tree_node->is_leaf()==false) {
       ComputeTreeDepthRecursion_(tree_node->left(), 
                                   depth,
                                   max_depth);
       
       ComputeTreeDepthRecursion_(tree_node->right(), 
                                  depth,
                                  max_depth);
     } else {
       if (depth>*max_depth) {
         *max_depth=depth;
       }
     }
    
  }
  
  void GoDownToDepth_(TreeType *tree_node, 
                      index_t desired_depth, 
                      ArrayList<std::pair<index_t, index_t> > *point_nodes) {
    point_nodes->Init();
    GoDownToDepthRecursion_(tree_node, 
                            -1,
                            desired_depth,
                            point_nodes);    
  }
 
  void GoDownToDepthRecursion_(TreeType *tree_node, 
                               index_t current_depth,
                               index_t desired_depth, 
                               ArrayList<std::pair<index_t, index_t> > *point_nodes) {

    current_depth+=1;
    if (tree_node->is_leaf()==false 
        && current_depth<desired_depth) {
      GoDownToDepthRecursion_(tree_node->left(),
                              current_depth,
                              desired_depth, 
                              point_nodes);
      GoDownToDepthRecursion_(tree_node->right(),
                              current_depth,
                              desired_depth, 
                              point_nodes);

      
    } else {
      std::pair<index_t, index_t> *ptr;
      ptr = point_nodes->PushBackRaw();
      ptr->first=tree_node->begin();
      ptr->second=tree_node->end();
    }
    
  }
 
  success_t SpitMatrix_(Matrix &lower_bound, Matrix &upper_bound, 
          ArrayList<index_t> &old_from_new_points, 
          ArrayList<std::pair<index_t, index_t> > &point_nodes,
          index_t offset, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound) {
   
    double widest_range=0;
    index_t widest_range_i=-1;
    index_t widest_range_j=-1;
    for(index_t i=0; i< point_nodes.size(); i++) {
      index_t p=offset+old_from_new_points[point_nodes[i].first];
      for(index_t j=0; j<lower_bound.n_rows(); j++) {
        if (upper_bound.get(j, p) - lower_bound.get(j, p) > widest_range) {
          widest_range=upper_bound.get(j, p) - lower_bound.get(j, p);
          widest_range_i=i;
          widest_range_j=j;
        }
      }
    }
      
    left_lower_bound->Copy(lower_bound);
    left_upper_bound->Copy(upper_bound);
    right_lower_bound->Copy(lower_bound);
    right_upper_bound->Copy(upper_bound);
    index_t p=offset+old_from_new_points[point_nodes[widest_range_i].first];
    double split_value=(upper_bound.get(widest_range_j, p)
        +lower_bound.get(widest_range_j, p))/2; 
    double interval_length=split_value - lower_bound.get(widest_range_j, p);
    if (interval_length < minimum_interval_length_) {
      return SUCCESS_FAIL;
    }
    
    NOTIFY("interval length:%lg w_depth:%i h_depth:%i", 
        interval_length, w_current_depth_, h_current_depth_);
    for(index_t i=point_nodes[widest_range_i].first; 
        i<point_nodes[widest_range_i].second; i++) {
      left_upper_bound->set(widest_range_j, offset+old_from_new_points[i], split_value);
      right_lower_bound->set(widest_range_j, offset+old_from_new_points[i], split_value); 
    }
    return SUCCESS_PASS;
  }
};


#endif
