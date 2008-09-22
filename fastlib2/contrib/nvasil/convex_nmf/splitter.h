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
  
  void Split(Matrix &lower_bound, Matrix &upper_bound, 
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
  }
  
  void ChangeState(SolutionPack &solution) {
  }
};

/*
 * This version preserves the tree on w and it builds a tree on h on the fly
 */
class TreeOnWandHSplitter {
 public: 
  typedef BinarySpaceTree<DHrectBound<2>, Matrix> TreeType;
 FILE *fp_; 
  void Init(fx_module *module, 
            Matrix &w_points) {
    module_=module;
    w_leaf_size_=fx_param_int(module_, "w_leaf_size", 1);
    h_leaf_size_=fx_param_int(module_, "h_leaf_size", 1);
    w_offset_=fx_param_int_req(module_, "w_offset");
    h_offset_=fx_param_int_req(module_, "h_offset");
    h_length_=fx_param_int_req(module_, "h_length");
    
    DEBUG_ASSERT(w_leaf_size_ > 0);
    DEBUG_ASSERT(h_leaf_size_ > 0);
    Matrix data_points;
    data_points.Copy(w_points);
    w_tree_ = tree::MakeKdTreeMidpoint<TreeType>(data_points, 
                                                 w_leaf_size_, 
				                                         &w_old_from_new_points_, 
                                                 NULL);
    w_current_depth_=0;
    h_current_depth_=0;
 
    ComputeTreeDepth_(w_tree_, &w_tree_max_depth_);
    GoDownToDepth_(w_tree_, 
                   w_current_depth_, 
                   &w_point_nodes_);

    h_old_from_new_points_.Init();
    for(index_t i=0; i<h_length_; i++) {
      h_old_from_new_points_.PushBackCopy(i);
    }
    h_point_nodes_.Init();
    h_point_nodes_.PushBackCopy(std::make_pair(0, h_length_));
    h_tree_=NULL;   
    split_on_w_=true;
    rehash_w_=false;
    fp_=fopen("temp", "w");
  }
  void Split(Matrix &lower_bound, Matrix &upper_bound, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound) {
    if (split_on_w_==true) {
      SpitMatrix_(lower_bound, 
                  upper_bound, 
                  w_old_from_new_points_, 
                  w_point_nodes_,
                  w_offset_, 
                  left_lower_bound, left_upper_bound,
                  right_lower_bound, right_upper_bound);
      for(index_t i=40; i<lower_bound.n_cols(); i++) {
       fprintf(fp_, "%lg ", lower_bound.get(0,i));
      }
      fprintf(fp_, "\n");
      for(index_t i=40; i<upper_bound.n_cols(); i++) {
       fprintf(fp_, "%lg ", upper_bound.get(0,i));
      }
      fprintf(fp_, "\n\n");

      split_on_w_=false;
    } else {
      NOTIFY("Split on h");
      SpitMatrix_(lower_bound, 
                  upper_bound, 
                  h_old_from_new_points_, 
                  h_point_nodes_,
                  h_offset_, 
                  left_lower_bound, left_upper_bound,
                  right_lower_bound, right_upper_bound);
      split_on_w_=true;
    }
    
  }
  // if it returns false it means that no further optimization can be done
  bool ChangeState(SolutionPack &solution) {
    Matrix opt_points;
    opt_points.Alias(solution.solution_);
    if (rehash_w_==false) {
      rehash_w_=true;
      //split_on_w_=false;
      // Make nodes for h
      Matrix data_points;
      data_points.Copy(opt_points.GetColumnPtr(h_offset_), opt_points.n_rows(), 
         h_length_);
      // you have to delete the tree first!!!!
      if (h_tree_!=NULL) {
        delete h_tree_;
      }
      h_old_from_new_points_.Renew();
      h_tree_ = tree::MakeKdTreeMidpoint<TreeType>(data_points, h_leaf_size_, 
				&h_old_from_new_points_, NULL);
      ComputeTreeDepth_(h_tree_, &h_tree_max_depth_);
      if (h_current_depth_>h_tree_max_depth_) {
        return false;
      }
      h_point_nodes_.Renew();
      h_current_depth_++;
      GoDownToDepth_(h_tree_, h_current_depth_, &h_point_nodes_);
      return true;
    } else {
      rehash_w_=false;
     // split_on_w_=true;
      w_point_nodes_.Renew();
      w_current_depth_++;
      if (w_current_depth_ >w_tree_max_depth_) {
        return false;
      }
      w_point_nodes_.Renew();
      GoDownToDepth_(w_tree_, w_current_depth_, &w_point_nodes_);
      return true;
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
  bool rehash_w_;  
  
  void ComputeTreeDepth_(TreeType *tree_node, index_t *depth) {
    *depth=0;
    ComputeTreeDepthRecursion_(tree_node, 0, depth);
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
                            0,
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
 
  void SpitMatrix_(Matrix &lower_bound, Matrix &upper_bound, 
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
    NOTIFY("interval length:%lg", split_value - lower_bound.get(widest_range_j, p));
    for(index_t i=point_nodes[widest_range_i].first; 
        i<point_nodes[widest_range_i].second; i++) {
      left_upper_bound->set(widest_range_j, offset+old_from_new_points[i], split_value);
      right_lower_bound->set(widest_range_j, offset+old_from_new_points[i], split_value); 
    }
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


    split_on_w_=true;
    rehash_w_=false;
  }
  void Split(Matrix &lower_bound, Matrix &upper_bound, 
          Matrix *left_lower_bound, Matrix *left_upper_bound,
          Matrix *right_lower_bound, Matrix *right_upper_bound) {
    if (split_on_w_==true) {
      SpitMatrix_(lower_bound, 
                  upper_bound, 
                  w_old_from_new_points_, 
                  w_point_nodes_,
                  w_offset_, 
                  left_lower_bound, left_upper_bound,
                  right_lower_bound, right_upper_bound);
      split_on_w_=false;
    } else {
      NOTIFY("Split on h");
      SpitMatrix_(lower_bound, 
                  upper_bound, 
                  h_old_from_new_points_, 
                  h_point_nodes_,
                  h_offset_, 
                  left_lower_bound, left_upper_bound,
                  right_lower_bound, right_upper_bound);
      split_on_w_=true;
    }
    
  }
  // if it returns false it means that no further optimization can be done
  bool ChangeState(SolutionPack &solution) {
    Matrix opt_points;
    opt_points.Alias(solution.solution_);
//    if (rehash_w_==false) {
      rehash_w_=true;
      h_point_nodes_.Renew();
      h_current_depth_++;
      if (h_current_depth_ >h_tree_max_depth_) {
        return false;
      }
      h_point_nodes_.Renew();
      GoDownToDepth_(h_tree_, h_current_depth_, &h_point_nodes_);
//      return true;
//    } else {
      rehash_w_=false;
      w_point_nodes_.Renew();
      w_current_depth_++;
      if (w_current_depth_ >w_tree_max_depth_) {
        return false;
      }
      w_point_nodes_.Renew();
      GoDownToDepth_(w_tree_, w_current_depth_, &w_point_nodes_);
      return true;
//    } 
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
  bool rehash_w_;  
  
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
 
  void SpitMatrix_(Matrix &lower_bound, Matrix &upper_bound, 
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
    NOTIFY("interval length:%lg", split_value - lower_bound.get(widest_range_j, p));
    for(index_t i=point_nodes[widest_range_i].first; 
        i<point_nodes[widest_range_i].second; i++) {
      left_upper_bound->set(widest_range_j, offset+old_from_new_points[i], split_value);
      right_lower_bound->set(widest_range_j, offset+old_from_new_points[i], split_value); 
    }
  }
};


#endif
