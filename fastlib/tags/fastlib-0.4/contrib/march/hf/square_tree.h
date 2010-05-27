/**
 * @file square_tree.h
 *
 * @author Bill March
 *
 */
 
#ifndef SQUARE_TREE_H
#define SQUARE_TREE_H

/**
* The stat class for the square tree
 */
class SquareIntegralStat {
  
private:
  
  double density_upper_bound_;
  
  double density_lower_bound_;
  
  // These are bounds on the computation that's actually been done
  // i.e. the minimum and maximum total actually returned from approximations
  // and base cases
  // i.e. the minimum and maximum values actually written to the Coulomb/
  // exchange matrix
  // They start at 0, since there has been no computation
  double entry_lower_bound_;
  double entry_upper_bound_;
  
  // The number of reference PAIRS that haven't been accounted for by either
  // approximation or a base case
  index_t remaining_references_;
  
  // The value of an approximation performed on this node
  // If no approximation has been done, this value is 0
  double approximation_val_;
  
  
public:
  
  void Init(const SquareIntegralStat& left, const SquareIntegralStat& right) {
    
    density_upper_bound_ = max(left.density_upper_bound(), 
                               right.density_upper_bound());
    
    density_lower_bound_ = min(left.density_lower_bound(), 
                               right.density_lower_bound());
    
    
    entry_lower_bound_ = 0.0;
    entry_upper_bound_ = 0.0;

    approximation_val_ = 0.0;
    
    remaining_references_ = left.remaining_references();
    
  } // void Init (2 children)
  
  void Init(index_t start1, index_t end1, index_t start2, index_t end2, 
            index_t num_funs) {
    
    /*
    double min_density = DBL_MAX;
    double max_density = -DBL_MAX;
    
    for (index_t i = start1; i < end1; i++) {
      
      for (index_t j = start2; j < end2; j++) {
        
        double this_density = density.get(i, j);
        if (this_density < min_density) {
          min_density = this_density;
        }
        if (this_density > max_density) {
          max_density = this_density;
        }
        
      } // j
      
    } // i
    
    density_upper_bound_ = max_density;
    density_lower_bound_ = min_density;
    DEBUG_ASSERT(density_upper_bound_ > -DBL_MAX);
    DEBUG_ASSERT(density_lower_bound_ < DBL_MAX);
    */
    
    density_upper_bound_ = DBL_MAX;
    density_lower_bound_ = -DBL_MAX;
    
    
    entry_upper_bound_ = 0.0;
    entry_lower_bound_ = 0.0;

    approximation_val_ = 0.0;
    
    remaining_references_ = num_funs * num_funs;
    
  } // void Init(leaf)
  
  void set_density_upper_bound(double bound) {
    density_upper_bound_ = bound;
  }
  
  double density_upper_bound() const {
    return density_upper_bound_;
  }
  
  void set_density_lower_bound(double bound) {
    density_lower_bound_ = bound;
  }
  
  double density_lower_bound() const {
    return density_lower_bound_;
  }
  
  void set_entry_lower_bound(double bound) {
    entry_lower_bound_ = bound;
  }
  
  double entry_lower_bound() const {
    return entry_lower_bound_;
  }
  
  void set_entry_upper_bound(double bound) {
    entry_upper_bound_ = bound;
  } 
  
  double entry_upper_bound() const {
    return entry_upper_bound_;
  }
  
  void set_remaining_references(index_t ref) {
    DEBUG_ASSERT_MSG(ref >= 0, "Negative remaining references\n");
    remaining_references_ = ref;
  }
  
  index_t remaining_references() const {
    return remaining_references_;
  }
  
  void set_approximation_val(double val) {
    approximation_val_ = val;
  }
  
  double approximation_val() const {
    return approximation_val_;
  }
  
  
}; // class SquareIntegralStat


/**
 * Implements the square tree idea for handling $N^2$ queries
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
  
  /*
  SquareTree* left_left_child_;
  SquareTree* left_right_child_;
  SquareTree* right_left_child_;
  SquareTree* right_right_child_;
  */

  SquareTree* left_child_;
  SquareTree* right_child_;

  /*
  bool query1_leaf_;
  bool query2_leaf_;
  */
  
  // Can have a templatized stat class here to make this generic
  SquareTreeStat stat_;


 public:

  /**
   *  Init function for only two children at each level
   *
   *  It's important that I only create children where the index of q1  is 
   *  greater than that of q2
   */
  void Init(QueryTree1* query1_root, QueryTree2* query2_root, 
            index_t num_funs) {
  
    query1_ = query1_root;
    query2_ = query2_root;
  
    index_t q1_height = query1_->stat().height();
    index_t q2_height = query2_->stat().height();
    
    DEBUG_ASSERT(query1_root->end() > query2_root->begin());
  
    if (q1_height == 0 && q2_height == 0) {
    
      left_child_ = NULL;
      right_child_ = NULL;
      
      stat_.Init(query1_->begin(), query1_->end(), query2_->begin(), 
                 query2_->end(), num_funs);
    
    }
    // I'm assuming that query1_ will always have two significant children
    // it's fine to split q1, since this will only increase it's index
    else if (q1_height == q2_height) {
    
      left_child_ = new SquareTree();
      right_child_ = new SquareTree();
      
      DEBUG_ASSERT(query1_->right()->end() > query2_->begin());
      DEBUG_ASSERT(query1_->left()->end() > query2_->begin());
      
      left_child_->Init(query1_->left(), query2_, num_funs);
      right_child_->Init(query1_->right(), query2_, num_funs);
      
      stat_.Init(left_child_->stat(), right_child_->stat());
    
    }
    else if (q2_height > q1_height) {
      // Both children of q2 count
      // it's fine to split if we know that q2's larger child will still be less
      // than q1
      if (query1_->end() > query2_->right()->begin()) {
               
        DEBUG_ASSERT(query1_->end() > query2_->left()->begin());
                     
        left_child_ = new SquareTree();
        right_child_ = new SquareTree();
        
        left_child_->Init(query1_, query2_->left(), num_funs);
        right_child_->Init(query1_, query2_->right(), num_funs);
        
        stat_.Init(left_child_->stat(), right_child_->stat());
      
      }
      // Can't go farther down query2
      // i.e. query2 has height one, and one child doesn't count 
      // so, this is a leaf
      else if (query2_->left()->stat().height() == 0) {
      
        left_child_ = NULL;
        right_child_ = NULL;
        
        query2_ = query2_->left();
        DEBUG_ASSERT(query2_->is_leaf());
        DEBUG_ASSERT(query1_->end() > query2_->begin());
        
        stat_.Init(query1_->begin(), query1_->end(), query2_->begin(), 
                   query2_->end(), num_funs);
        
      }
      // Idea: since query2 isn't necessary, go farther down query2
      else {
      
        left_child_ = new SquareTree();
        right_child_ = new SquareTree();
        
        query2_ = query2_->left();
        
        DEBUG_ASSERT(query1_->end() > query2_->left()->begin());
        DEBUG_ASSERT(query1_->end() > query2_->right()->begin());
                     
        left_child_->Init(query1_, query2_->left(), num_funs);
        right_child_->Init(query1_, query2_->right(), num_funs);
        
        stat_.Init(left_child_->stat(), right_child_->stat());
      
      }
    } // q2 higher
    else {
      // fine to split q1
      if (query1_->left()->end() > query2_->begin()) {
      
        DEBUG_ASSERT(query1_->right()->end() > query2_->begin());
        
        left_child_ = new SquareTree();
        right_child_ = new SquareTree();
        
        left_child_->Init(query1_->left(), query2_, num_funs);
        right_child_->Init(query1_->right(), query2_, num_funs);
        
        stat_.Init(left_child_->stat(), right_child_->stat());        
      
      }
      // q1 is too low to split twice
      else if (query1_->right()->stat().height() == 0) {
      
        left_child_ = NULL;
        right_child_ = NULL;
        
        query1_ = query1_->right();
        DEBUG_ASSERT(query1_->is_leaf());
        DEBUG_ASSERT(query1_->end() > query2_->begin());
        
        stat_.Init(query1_->begin(), query1_->end(), query2_->begin(), 
                   query2_->end(), num_funs);
              
      }
      // q1 is split twice
      else {
      
        left_child_ = new SquareTree();
        right_child_ = new SquareTree();
        
        query1_ = query1_->right();
        
        DEBUG_ASSERT(query1_->left()->end() > query2_->begin());
        DEBUG_ASSERT(query1_->right()->end() > query2_->begin());
        
        left_child_->Init(query1_->left(), query2_, num_funs);
        right_child_->Init(query1_->right(), query2_, num_funs);
        
        stat_.Init(left_child_->stat(), right_child_->stat());  
      
      }
    
    } // q1 higher 

  /*    
    DEBUG_ASSERT(stat_.density_upper_bound() < DBL_MAX);
    DEBUG_ASSERT(stat_.density_lower_bound() > -DBL_MAX);
  */
  
  } // Init() (two-children)
  
  QueryTree1* query1() {
    return query1_;
  }
  
  QueryTree2* query2() {
    return query2_;
  }
  
  bool is_leaf() {
    return (!left_child_);
  }

  SquareTree* left() {
    return left_child_;
  }
  
  SquareTree* right() {
    return right_child_;
  }
  
  const SquareTreeStat& stat() const {
    return stat_;
  }
  
  SquareTreeStat& stat() {
    return stat_;
  }
  
  void Print() {
  
    printf("query1: %d to %d: %d total\t\t query2: %d to %d: %d total\n",
           query1_->begin(), query1_->end(), query1_->count(),
           query2_->begin(), query2_->end(), query2_->count());
    printf("density_upper_bound: %g, density_lower_bound: %g\n\n", 
           stat_.density_upper_bound(), stat_.density_lower_bound());
    
    if (left_child_ != NULL) {
      left_child_->Print();
    }
    if (right_child_ != NULL) {
      right_child_->Print();
    } 
    
  } // Print()
  
  
  

}; // class SquareTree





#endif
