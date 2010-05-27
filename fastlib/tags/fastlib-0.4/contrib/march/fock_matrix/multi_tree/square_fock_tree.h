#ifndef SQUARE_FOCK_TREE_H
#define SQUARE_FOCK_TREE_H

#include "contrib/march/fock_matrix/fock_impl/eri.h"

/**
 * The square tree, necessary for the multi tree fock code.
 */
template<class QueryTree>
class SquareFockTree {

 private:

  /**
   * Stat class for the square tree
   */
  class SquareFockStat {
  
   private:
    
    double density_upper_bound_;
    double density_lower_bound_;
    
    // These are bounds on the computation that's actually been done
    // i.e. the minimum and maximum total actually returned from approximations
    // and base cases
    // i.e. the minimum and maximum values actually written to the Coulomb/
    // exchange matrix
    double entry_lower_bound_;
    double entry_upper_bound_;
    
    // can't just use the entry_lower_bound for relative error prunes because
    // the entries can be positive or negative.  
    double relative_error_bound_;
    
    // The number of reference PAIRS that haven't been accounted for by either
    // approximation or a base case
    index_t remaining_references_;
    
    // The value of an approximation performed on this node
    // If no approximation has been done, this value is 0
    double approximation_val_;
    
    double remaining_epsilon_;
    
    // the K_1 value in the integral
    // will only be useful for the coulomb computation
    double max_gpt_factor_;
    double min_gpt_factor_;
    
    double max_schwartz_factor_;
    double min_schwartz_factor_;
    
    // bounds on sum of exponents
    double max_gamma_;
    double min_gamma_;
    
    
   public:
      
    void Init(const SquareFockStat& left, const SquareFockStat& right) {
      
      density_upper_bound_ = max(left.density_upper_bound(), 
                                 right.density_upper_bound());
      
      density_lower_bound_ = min(left.density_lower_bound(), 
                                 right.density_lower_bound());
      
      
      entry_lower_bound_ = -DBL_MAX;
      entry_upper_bound_ = DBL_MAX;
      relative_error_bound_ = 0.0;
      
      approximation_val_ = 0.0;
      
      remaining_references_ = left.remaining_references();
      DEBUG_ASSERT(remaining_references_ == right.remaining_references());
     
    } // void Init (2 children)
  
    void Init(index_t start1, index_t end1, index_t start2, index_t end2, 
              index_t num_funs) {
      
      density_upper_bound_ = DBL_MAX;
      density_lower_bound_ = -DBL_MAX;
      
      entry_lower_bound_ = -DBL_MAX;
      entry_upper_bound_ = DBL_MAX;
      relative_error_bound_ = 0.0;
      
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
    
    // The lower bound for relative error needs to account for the possibility 
    // of the upper and lower bounds on the value of the result being either
    // positive or negative.  
    void set_relative_error_bound(double min_entry, double max_entry) {
      
      DEBUG_ASSERT(max_entry >= min_entry);
      
      if (max_entry < 0.0) {
        relative_error_bound_ = fabs(max_entry);
      }
      else if(min_entry > 0.0) {
        relative_error_bound_ = min_entry;
      }
      else { 
        relative_error_bound_ = min(max_entry, fabs(min_entry));
      }
      
    }
    
    double relative_error_bound() const {
      return relative_error_bound_;
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
    
    void set_remaining_epsilon(double eps) {
      remaining_epsilon_ = eps;
    }
    
    double remaining_epsilon() const {
      return remaining_epsilon_;
    }
    
    double max_gpt_factor() const {
      return max_gpt_factor_;
    }
    
    void set_max_gpt_factor(double fac) {
      max_gpt_factor_ = fac;
    }
    
    double min_gpt_factor() const {
      return min_gpt_factor_;
    }
    
    void set_min_gpt_factor(double fac) {
      min_gpt_factor_ = fac;
    }
    
    double max_gamma() const {
      return max_gamma_;
    }
    
    void set_max_gamma(double gam) {
      max_gamma_ = gam;
    }

    double min_gamma() const {
      return min_gamma_;
    }
    
    void set_min_gamma(double gam) {
      min_gamma_ = gam;
    }
    
    double max_schwartz_factor() {
      return max_schwartz_factor_;
    } 
    
    void set_max_schwartz_factor(double max_s) {
      max_schwartz_factor_ = max_s;
    }

    double min_schwartz_factor() {
      return min_schwartz_factor_;
    } 

    void set_min_schwartz_factor(double min_s) {
      min_schwartz_factor_ = min_s;
    }
    
    void Print() {
      
      printf("\t Density bounds: (%g, %g)\n", density_lower_bound_, 
             density_upper_bound_);
      printf("\t Entry bounds: (%g, %g)\n", entry_lower_bound_, 
             entry_upper_bound_);
      printf("Remaining epsilon: %g\n", remaining_epsilon_);
      printf("Remaining references: %d\n", remaining_references_);
      
    }
    
    
  }; // class SquareFockStat
  
  ////////////////// Member variables //////////////////////////
  
 private:
  
  QueryTree* query1_;
  QueryTree* query2_;
  
  SquareFockTree* left_child_;
  SquareFockTree* right_child_;
  
  SquareFockStat stat_;
  
  DHrectBound<2> bound_;
  
  
 public:
    
    /**
    *  Init function for only two children at each level
     *
     *  It's important that I only create children where the index of q1  is 
     *  greater than that of q2
     */
    void Init(QueryTree* query1_root, QueryTree* query2_root, 
              index_t num_funs, const Matrix& points, const Vector& exp) {
      
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
        
        left_child_ = new SquareFockTree<QueryTree>();
        right_child_ = new SquareFockTree<QueryTree>();
        
        DEBUG_ASSERT(query1_->right()->end() > query2_->begin());
        DEBUG_ASSERT(query1_->left()->end() > query2_->begin());
        
        left_child_->Init(query1_->left(), query2_, num_funs, points, exp);
        right_child_->Init(query1_->right(), query2_, num_funs, points, exp);
        
        stat_.Init(left_child_->stat(), right_child_->stat());
        
      }
      else if (q2_height > q1_height) {
        // Both children of q2 count
        // it's fine to split if we know that q2's larger child will still be less
        // than q1
        if (query1_->end() > query2_->right()->begin()) {
          
          DEBUG_ASSERT(query1_->end() > query2_->left()->begin());
          
          left_child_ = new SquareFockTree<QueryTree>();
          right_child_ = new SquareFockTree<QueryTree>();
          
          left_child_->Init(query1_, query2_->left(), num_funs, points, exp);
          right_child_->Init(query1_, query2_->right(), num_funs, points, exp);
          
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
          
          left_child_ = new SquareFockTree<QueryTree>();
          right_child_ = new SquareFockTree<QueryTree>();
          
          query2_ = query2_->left();
          
          DEBUG_ASSERT(query1_->end() > query2_->left()->begin());
          DEBUG_ASSERT(query1_->end() > query2_->right()->begin());
          
          left_child_->Init(query1_, query2_->left(), num_funs, points, exp);
          right_child_->Init(query1_, query2_->right(), num_funs, points, exp);
          
          stat_.Init(left_child_->stat(), right_child_->stat());
          
        }
      } // q2 higher
      else {
        // fine to split q1
        if (query1_->left()->end() > query2_->begin()) {
          
          DEBUG_ASSERT(query1_->right()->end() > query2_->begin());
          
          left_child_ = new SquareFockTree<QueryTree>();
          right_child_ = new SquareFockTree<QueryTree>();
          
          left_child_->Init(query1_->left(), query2_, num_funs, points, exp);
          right_child_->Init(query1_->right(), query2_, num_funs, points, exp);
          
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
          
          left_child_ = new SquareFockTree<QueryTree>();
          right_child_ = new SquareFockTree<QueryTree>();
          
          query1_ = query1_->right();
          
          DEBUG_ASSERT(query1_->left()->end() > query2_->begin());
          DEBUG_ASSERT(query1_->right()->end() > query2_->begin());
          
          left_child_->Init(query1_->left(), query2_, num_funs, points, exp);
          right_child_->Init(query1_->right(), query2_, num_funs, points, exp);
          
          stat_.Init(left_child_->stat(), right_child_->stat());  
          
        }
        
      } // q1 higher 
      
            /*    
        DEBUG_ASSERT(stat_.density_upper_bound() < DBL_MAX);
      DEBUG_ASSERT(stat_.density_lower_bound() > -DBL_MAX);
      */
      
      SetStatsAndBounds(points, exp);
      
  } // Init() (two-children)
  
  
  // I think this will need to be called from Init if I'm going to prune the 
  // square tree with a shell pair cutoff
  void SetStatsAndBounds(const Matrix& points, const Vector& exponents) {
    // set stats and bounds
    // leaf
    if (left_child_ == NULL) {
      
      double min_dist = query1_->bound().MinDistanceSq(query2_->bound());
      double max_fac = eri::IntegralGPTFactor(query1_->stat().min_bandwidth(), 
                                              query2_->stat().min_bandwidth(), 
                                              min_dist);
      
      double max_dist = query1_->bound().MaxDistanceSq(query2_->bound());
      double min_fac = eri::IntegralGPTFactor(query1_->stat().max_bandwidth(), 
                                              query2_->stat().max_bandwidth(), 
                                              max_dist);
      
      stat_.set_max_gpt_factor(max_fac);
      stat_.set_min_gpt_factor(min_fac);
      
      double max_schwartz = eri::DistanceIntegral(query1_->stat().min_bandwidth(),
                                                  query2_->stat().min_bandwidth(),
                                                  query1_->stat().min_bandwidth(),
                                                  query2_->stat().min_bandwidth(),
                                                  min_dist, min_dist, 0.0);
      max_schwartz *= query1_->stat().max_normalization();
      max_schwartz *= query1_->stat().max_normalization();
      max_schwartz *= query2_->stat().max_normalization();
      max_schwartz *= query2_->stat().max_normalization();

      double min_schwartz = eri::DistanceIntegral(query1_->stat().max_bandwidth(),
                                                  query2_->stat().max_bandwidth(),
                                                  query1_->stat().max_bandwidth(),
                                                  query2_->stat().max_bandwidth(),
                                                  max_dist, max_dist, 0.0);      
      min_schwartz *= query1_->stat().min_normalization();
      min_schwartz *= query1_->stat().min_normalization();
      min_schwartz *= query2_->stat().min_normalization();
      min_schwartz *= query2_->stat().min_normalization();
      
      
      stat_.set_max_schwartz_factor(max_schwartz);
      stat_.set_min_schwartz_factor(min_schwartz);
      
      /*bound_.WeightedAverageBoxesInit(query1_->stat().min_bandwidth(), 
        query1_->stat().max_bandwidth(), 
        query1_->bound(), 
        query2_->stat().min_bandwidth(), 
        query2_->stat().max_bandwidth(), 
        query2_->bound());

  */
      
      // set bound exhaustively
      // need access to bandwidths to do this correctly
      bound_.Init(3);
      
      for (index_t i = query1_->begin(); i < query1_->end(); i++) {
        
        Vector i_vec;
        i_vec.Copy(points.GetColumnPtr(i), 3);
        la::Scale(exponents[i], &i_vec);
        
        for (index_t j = query2_->begin(); j < query2_->end(); j++) {
          
          Vector j_vec;
          j_vec.Copy(points.GetColumnPtr(j), 3);
          la::Scale(exponents[j], &j_vec);     
          
          Vector bound_vec;
          la::AddInit(i_vec, j_vec, &bound_vec);
          la::Scale(1/(exponents[i] + exponents[j]), &bound_vec);
          
          /*
          i_vec.PrintDebug("i_vec");
          j_vec.PrintDebug("j_vec");
          printf("exp[%d]: %g, exp[%d]: %g\n", i, exponents[i], j, exponents[j]);
          bound_vec.PrintDebug("bound_vec");
          printf("\n\n");
          */
          
          bound_|=bound_vec;
          
        }
          
      }
        
      stat_.set_max_gamma(query1_->stat().max_bandwidth() 
                          + query2_->stat().max_bandwidth());
      stat_.set_min_gamma(query1_->stat().min_bandwidth() 
                          + query2_->stat().min_bandwidth());
      
    }
    // non-leaf
    else {
      
      //left_child_->SetStatsAndBounds(points, exponents);
      //right_child_->SetStatsAndBounds(points, exponents);
      
      stat_.set_max_gpt_factor(max(left_child_->stat().max_gpt_factor(), 
                                   right_child_->stat().max_gpt_factor()));
      stat_.set_min_gpt_factor(min(left_child_->stat().min_gpt_factor(), 
                                   right_child_->stat().min_gpt_factor()));
      
      bound_.Init(3);
      bound_|=left_child_->bound();
      bound_|=right_child_->bound();   
      
      stat_.set_max_gamma(max(left_child_->stat().max_gamma(), 
                              right_child_->stat().max_gamma()));
      stat_.set_min_gamma(min(left_child_->stat().min_gamma(), 
                              right_child_->stat().min_gamma()));
                              
      stat_.set_max_schwartz_factor(max(left_child_->stat().max_schwartz_factor(),
                                        right_child_->stat().max_schwartz_factor()));
      stat_.set_min_schwartz_factor(min(left_child_->stat().min_schwartz_factor(),
                                        right_child_->stat().min_schwartz_factor()));
      
    }
  }   
    
  
  QueryTree* query1() {
    return query1_;
  }
  
  QueryTree* query2() {
    return query2_;
  }
  
  bool is_leaf() {
    return (!left_child_);
  }
  
  SquareFockTree* left() {
    return left_child_;
  }
  
  SquareFockTree* right() {
    return right_child_;
  }
  
  const SquareFockStat& stat() const {
    return stat_;
  }
  
  SquareFockStat& stat() {
    return stat_;
  }
  
  const DHrectBound<2>& bound() const {
    return bound_;
  }
  
  DHrectBound<2>& bound() {
    return bound_;
  }
  
  void Print() {
    
    printf("query1: %d to %d: %d total\t\t query2: %d to %d: %d total\n",
           query1_->begin(), query1_->end(), query1_->count(),
           query2_->begin(), query2_->end(), query2_->count());
    stat_.Print();
    printf("\n");
    
    /*
    printf("density_upper_bound: %g, density_lower_bound: %g\n\n", 
           stat_.density_upper_bound(), stat_.density_lower_bound());
    printf();
    */
     
    if (left_child_ != NULL) {
      left_child_->Print();
    }
    if (right_child_ != NULL) {
      right_child_->Print();
    } 
    
  } // Print()
  

}; // class SquareFockTree




#endif