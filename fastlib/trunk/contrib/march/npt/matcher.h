/*
 *  matcher.h
 *  
 *
 *  Created by William March on 2/23/10.
 *
 *
 *  Stores the upper and lower bounds on each pair in the n-tuple.
 *  Can also evaluate whether a set of points or nodes violates these 
 *  conditions.
 */

// TODO: where should I keep the list of acceptable permutations?  
// I think it should just be a vector passed down in the recursion

class Matcher {
  
private:
  
  class Permutations {
    
  private:
    
    GenMatrix<index_t> permutation_indices_;
    
    int num_perms_;
    
  public:
    
    void Init(index_t n) {
      
      num_perms_ = factorials(n);
      permutation_indices_.Init(num_perms_, n);
      
      // TODO: fill in the permutations

      
      
    } // Init()
    
    int num_perms() {
      return num_perms_;
    }
    
    index_t GetPermutation(index_t perm_index, index_t pt_index) {
      
      return permutation_indices_.get(perm_index, pt_index);
      
    } // GetPermutation
    
  }; // class Permutations
  
  // these are symmetric n \times n matrices
  // the diagonals are not defined, since they are never accessed.
  // for a tuple to work, we need L_{i,j} \leq d(x_i, x_j) \leq H_{i,j}
  // for all pairs (x_i, x_j) in the tuple (under some permutation)
  // these are the squared bounds to prevent square roots in the code
  Matrix lower_bounds_sqr_;
  Matrix upper_bounds_sqr_;
  
  Permutations perms_;
  
  int tuple_size_;
  int num_permutations_;
  
  ///////////////////// functions ///////////////////////
  
  /**
   *
   */
  index_t GetPermutationIndex_(index_t perm_index, index_t pt_index) {
    
    return perms_.GetPermutation(perm_index, pt_index);
    
  } // GetPermutation
  
  /**
   *
   */
  bool CheckDistances_(double dist_sq, index_t ind1, index_t ind2);

  
public:
  
  /**
   *
   */
  void Init(const Matrix& lower, const Matrix& upper, int tuple_size) {
    
    lower_bounds_.Copy(lower);
    upper_bounds_.Copy(upper);
    tuple_size_ = tuple_size;
    
    // need to set them to be the squares of the values read in 
    for (index_t i = 0; i < tuple_size_; i++) {
      for (index_t j = 0; j < i; j++) {
        
        double new_low = lower_bounds_.get(i, j) * lower_bounds_.get(i, j);
        double new_hi = upper_bounds_.get(i, j) * upper_bounds_.get(i, j);
        
        lower_bounds_.set(i, j, new_low);
        lower_bounds_.set(j, i, new_low);
        
        upper_bound_.set(i, j, new_hi);
        upper_bound_.set(j, i, new_hi);
        
      } // for j
    } // for i
    
    perms_.Init(tuple_size_);
    num_permutations_ = perms_.num_perms();
    
  } // Init()
  
  
  /**
   *
   */
  bool TestPointPair(double dist_sq, index_t tuple_index_1, 
                     index_t tuple_index_2, ArrayList<bool> permutation_ok);
  
  
  // TODO: how will I know what the nodes are inside this function?
  // I think the Auton code just passes the hrects
  /**
   *
   */
  bool TestNodePair();
  
}; // Matcher

