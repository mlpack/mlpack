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
 *
 * Assuming that if one node is a leaf, then it always goes left and right is 
 * null
 */
template<class QueryTree1, class QueryTree2, class SquareTreeStat>
class SquareTree {

  friend class SquareTreeTester;

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
  SquareTreeStat stat_;


 public:

  void Init(QueryTree1* query1_root, QueryTree2* query2_root) {
  
    query1_ = query1_root;
    query2_ = query2_root;
    
    if (query1_->is_leaf() && query2_->is_leaf()) {
    
      query1_leaf_ = true;
      query2_leaf_ = true;
      
      left_left_child_ = NULL;
      left_right_child_ = NULL;
      right_left_child_ = NULL;
      right_right_child_ = NULL;
      
      stat_.Init(left_left_child_, left_right_child_, right_left_child_, 
                 right_right_child_);
      
    }
    else if (query1_->is_leaf()) {
    
      query1_leaf_ = true;
      query2_leaf_ = false;
      
      left_left_child_ = new SquareTree();
      left_right_child_ = new SquareTree();
      
      left_left_child_.Init(query1_root, query2_root->left());
      left_right_child_.Init(query1_root, query2_root->right());
      
      right_left_child_ = NULL;
      right_right_child_ = NULL;
      
      stat_.Init(left_left_child_, left_right_child_, right_left_child_, 
                 right_right_child_);
      
    }
    else if(query2_->is_leaf()) {
    
      query1_leaf_ = false;
      query2_leaf_ = true;
      
      left_left_child_ = new SquareTree();
      right_left_child_ = new SquareTree();
      
      left_right_child_ = NULL;
      right_right_child_ = NULL;
      
      left_left_child_.Init(query1_root->left(), query2_root);
      right_left_child_.Init(query1_root->right(), query2_root);
    
    
      stat_.Init(left_left_child_, left_right_child_, right_left_child_, 
                 right_right_child_);
      
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
      
      stat_.Init(left_left_child_, left_right_child_, right_left_child_, 
                 right_right_child_);
      
    }
  
  } // Init()
  
  QueryTree1* query1() {
    return query1_;
  }
  
  QueryTree2* query2() {
    return query2_;
  }
  
  bool is_leaf() {
    return (query1_leaf_ && query2_leaf_);
  }
  
  SquareTree* left_left_child() {
    return left_left_child_;
  }
  
  SquareTree* left_right_child() {
    return left_right_child_;
  }
  
  SquareTree* right_left_child() {
    return right_left_child_;
  }
  
  SquareTree* right_right_child() {
    return right_right_child_;
  }
  

}; // class SquareTree





#endif