/**
 * @file square_tree.h
 *
 * @author Bill March
 *
 */
 
#ifndef SQUARE_TREE_H
#define SQUARE_TREE_H

/**
 * Implements the square tree idea for handling $N^2$ queries
 *
 * TODO: Needs to be templatized
 */
template<class QueryTree1, class QueryTree2, class SquareTreeStat>
class SquareTree {

 private:

  QueryTree1* query1_;
  QueryTree2* query2_;
  
  SquareTree* left_left_child_;
  SquareTree* left_right_child_;
  SquareTree* right_left_child_;
  SquareTree* right_right_child_;

  bool query1_leaf_;
  bool query2_leaf_;
  
  // Can have a templatized stat class here to make this generic
  SquareTreeStat* stat_;


 public:

  void Init(QueryTree1* query1_root, QueryTree2* query2_root) {
  
    query1_ = query1_root;
    query2_ = query2_root;
    
    if (query1_->is_leaf() && query2_->is_leaf()) {
    
      query1_leaf_ = true;
      query2_leaf = true;
      
      left_left_child_ = NULL;
      left_right_child_ = NULL;
      right_left_child_ = NULL;
      right_right_child_ = NULL;
      
      stat_ = new SquareTreeStat();
      stat_.Init();
      
    }
    else if (query1_->is_leaf()) {
    
    
    
    }
    else if(query2_->is_leaf()) {
    
    }
    else {
    
      left_left_child_ = new SquareTree();
      left_right_child_ = new SquareTree();
      right_left_child_ = new SquareTree();
      right_right_child_ = new SquareTree();
      
      query1_leaf_ = false;
      query2_leaf_ = false;
      
      left_left_child_.Init(query1_root->left(), query2_root->left());
      left_right_child_.Init(query1_root->left(), query2_root->right());
      right_left_child_.Init(query1_root->right(), query2_root->left());
      right_right_child_.Init(query1_root->right(), query2_root->right());
      
      stat_ = new SquareTreeStat();
      stat_.Init();
    
    }
  
  } // Init()

}; // class SquareTree





#endif