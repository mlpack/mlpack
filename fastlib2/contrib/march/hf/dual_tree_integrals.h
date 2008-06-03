/**
 * @file dual_tree_integrals.h
 *
 * @author Bill March (march@gatech.edu)
 *
 * Code for computing two electron integrals with a dual-tree approximation
 */

#ifndef DUAL_TREE_INTEGRALS_H
#define DUAL_TREE_INTEGRALS_H

#include <fastlib/fastlib.h>
#include "square_tree.h"

/**
 * Algorithm class for computing two-electron integrals.
 *
 * This object should still exist between SCF iterations, rather than being 
 * reconstructed each time.
 */
class DualTreeIntegrals {
  
  FORBID_ACCIDENTAL_COPIES(DualTreeIntegrals);
  
  friend class DualTreeIntegralsTest;
  friend class FockMatrixTest;
  
 public:
  DualTreeIntegrals() {}
  
  ~DualTreeIntegrals() {}
  
 private:
  /**
   * Stat class for tree building.  
   *
   * Maybe something for allocating error.  
   *
   * Should also include density matrix bounds.
   */
  class IntegralStat {
   
   private:
   
    // The node's index in a pre-order depth-first traversal of the tree
    index_t node_index_;
    
    // I might need some notion of bandwidths
    double min_bandwidth_;
    double max_bandwidth_;
    
    index_t height_;
    
    double density_upper_bound_;
    double density_lower_bound_;
    
   public:
    void Init() {
      
      node_index_ = -1;
      min_bandwidth_ = 0.0;
      max_bandwidth_ = DBL_MAX;
      
    } // Init
    
    void Init(const Matrix& matrix, index_t start, index_t count) {
      
      Init();
      height_ = 0;
      
    } // Init (leaves)
    
    void Init(const Matrix& matrix, index_t start, index_t count, 
              const IntegralStat& left, const IntegralStat& right) {
    
      Init();
      height_ = max(left.height(), right.height()) + 1;
      
    } // Init (non-leaves)
    
    index_t height() const {
      return height_;
    } // height()
    
    void set_height(index_t new_height) {
      height_ = new_height;
    } // set_height
    
    
    index_t node_index() const {
      return node_index_;
    } // node_index
    
    void set_node_index(index_t new_index) {
      node_index_ =  new_index;
    } // set_node_index
    
    double min_bandwidth() const {
      return min_bandwidth_; 
    } // min_bandwdith
    
    void set_min_bandwidth(double new_min) {
      min_bandwidth_ = new_min;
    } //set_min_bandwidth
    
    void set_max_bandwidth(double new_max) {
      max_bandwidth_ = new_max;
    } // set_max_bandwidth
    
    double max_bandwidth() const {
      return max_bandwidth_;
    } // max_bandwidth_
    
    void set_density_upper_bound(double upper_bound) {
      density_upper_bound_ = upper_bound;
    } // set_density_upper_bound_()
    
    double density_upper_bound() const {
      return density_upper_bound_;
    } // density_upper_bound()

    void set_density_lower_bound(double lower_bound) {
        density_lower_bound_ = lower_bound;
      } // set_density_lower_bound_()
    
    double density_lower_bound() const {
      return density_lower_bound_;
    } // density_lower_bound()
      
    
  }; // class IntegralStat
  
  
 public:
  // This assumes identical bandwidth small Gaussians
  // Otherwise, I'll need something other than a Matrix
  // I should also consider something better than bounding boxes
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, IntegralStat> IntegralTree; 
  
  typedef SquareTree<IntegralTree, IntegralTree, SquareIntegralStat> 
          SquareIntegralTree;
  
 private:
    
  // The tree 
  IntegralTree* tree_;
  
  // the square tree
  SquareIntegralTree* square_tree_;
  
  // The centers of the identical width, spherical Gaussian basis functions
  Matrix centers_;
  
  // The fx module
  struct datanode* module_;
  
  // The common bandwidth of all the basis functions
  double bandwidth_;
  
  // The number of times an approximation is invoked
  int coulomb_approximations_;
  int exchange_approximations_;
  
  // The number of times the base case is called
  int coulomb_base_cases_;
  int exchange_base_cases_;
  
  // The value eps where the returned integrals are within 1-eps of the true 
  // value
  double epsilon_;
  
  // the amount of error allocated to the coulomb matrix
  double epsilon_coulomb_;
  
  // the amount of error allocated to the exchange matrix
  double epsilon_exchange_;
  
  double epsilon_absolute_;
  double epsilon_coulomb_absolute_;
  double epsilon_exchange_absolute_;
  
  // Values that can be guaranteed to be less than this value are pruned by the 
  // absolute criterion
  double hybrid_cutoff_;
  
  // The return values are stored here
  // total_integrals_.ref(i, j) is the fock matrix entry i, j
  Matrix fock_matrix_;
  
  // The exchange contribution
  Matrix exchange_matrix_;
  
  // The coulomb contribution
  Matrix coulomb_matrix_;
  
  // The density matrix, will be input for now
  Matrix density_matrix_;
  
  // The total number of basis functions
  index_t number_of_basis_functions_;
  
  index_t traversal_index_;
  
  // Stores the permutation used in tree-building
  ArrayList<index_t> old_from_new_centers_;
  
  // Size of leaves in the tree
  int leaf_size_;
  
  // The normalization constant to the fourth power
  double normalization_constant_fourth_;
  
  index_t num_absolute_prunes_;
  index_t num_relative_prunes_;
  

  /////////////////////// Functions ///////////////////////////////////
  
  /**
   * Does a preorder traversal of the tree to set remaining statistics in the
   * nodes.  
   *
   * TODO: this can be used to compute density matrix bounds when I include them
   */
  void PreOrderTraversal_(IntegralTree* this_node) {
    
    traversal_index_ = traversal_index_ + 1;
    
    if (this_node->is_leaf()) {
     
      this_node->stat().set_node_index(traversal_index_);
      
    }
    else {
      
      this_node->stat().set_node_index(traversal_index_);
      
      PreOrderTraversal_(this_node->left());
      PreOrderTraversal_(this_node->right());
      
         
    }
   
    
  } // PreOrderTraversal  
  
  
  double ComputeUpperBound_() {
  
    return 0.0;
    
  } // ComputeUpperBound_()
  
  double ComputeLowerBound_() {
  
    return 0.0;
  
  } // ComputeLowerBound_()
  
  index_t CountOnDiagonal_(SquareIntegralTree* rho_sigma);
    
  /**
   * Determines if the integral among the four nodes can be approximated
   *
   * TODO: This function is wrong, needs to be updated
   */
  bool CanApproximateCoulomb_(SquareIntegralTree* mu_nu, 
                              SquareIntegralTree* rho_sigma, 
                              double* approximate_value) { 
                              
    IntegralTree* mu = mu_nu->query1();
    IntegralTree* nu = mu_nu->query2();
    IntegralTree* rho = rho_sigma->query1();
    IntegralTree* sigma = rho_sigma->query2();
    
    bool can_prune = false;
    
    double rho_sigma_min_dist = rho->bound().MinDistanceSq(sigma->bound());
    double rho_sigma_max_dist = rho->bound().MaxDistanceSq(sigma->bound());
    //double rho_sigma_mid_dist = rho->bound().MidDistanceSq(sigma->bound());
    
    double mu_nu_min_dist = mu->bound().MinDistanceSq(nu->bound());
    double mu_nu_max_dist = mu->bound().MaxDistanceSq(nu->bound());
    //double mu_nu_mid_dist = mu->bound().MidDistanceSq(nu->bound());
    
    
    DHrectBound<2> mu_nu_ave;
    mu_nu_ave.AverageBoxesInit(mu->bound(), nu->bound());
    DHrectBound<2> rho_sigma_ave;
    rho_sigma_ave.AverageBoxesInit(rho->bound(), sigma->bound());
    //ot::Print(mu_nu_ave);
    //ot::Print(rho_sigma_ave);
    
    double four_way_min_dist = mu_nu_ave.MinDistanceSq(rho_sigma_ave);
    double four_way_max_dist = mu_nu_ave.MaxDistanceSq(rho_sigma_ave);
    //double four_way_mid_dist = mu_nu_ave.MidDistanceSq(rho_sigma_ave);
    //printf("four_way_min: %g\n", four_way_min_dist);
    
    double up_bound = ComputeSingleIntegral_(mu_nu_min_dist, rho_sigma_min_dist, 
                                             four_way_min_dist);
    double low_bound = ComputeSingleIntegral_(mu_nu_max_dist, 
                                              rho_sigma_max_dist, 
                                              four_way_max_dist);
    
    // Need to account for the change in bounds if the density matrix entries 
    // are negative.  If the density lower bound is negative, then the lower 
    // bound becomes the largest integral times the density lower bound, 
    // instead of the smallest integral and vice versa.  
    if (rho_sigma->stat().density_upper_bound() >= 0.0) {
      up_bound = up_bound * rho_sigma->stat().density_upper_bound();
    }
    else {
      up_bound = low_bound * rho_sigma->stat().density_upper_bound();
    }
    if (rho_sigma->stat().density_lower_bound() >= 0.0) {
      low_bound = low_bound * rho_sigma->stat().density_lower_bound();
    }
    else {
      low_bound = up_bound * rho_sigma->stat().density_lower_bound();
    }
    
    
    
    DEBUG_ASSERT(up_bound >= low_bound);
    
    
    Vector mu_center;
    mu->bound().CalculateMidpoint(&mu_center);
    Vector nu_center;
    nu->bound().CalculateMidpoint(&nu_center);
    Vector rho_center;
    rho->bound().CalculateMidpoint(&rho_center);
    Vector sigma_center;
    sigma->bound().CalculateMidpoint(&sigma_center);
    
    
    double approx_val = ComputeSingleIntegral_(mu_center, nu_center, rho_center, 
                                               sigma_center);
                                               
    // Need to make this work with on diagonal non-square boxes
    if (RectangleOnDiagonal_(rho, sigma)) {
      index_t on_diagonal = CountOnDiagonal_(rho_sigma);
      index_t off_diagonal = (rho->count() * sigma->count()) - on_diagonal;
      up_bound = (on_diagonal * up_bound) + (2 * off_diagonal * up_bound);
      // These two were missing, I think they needed to be here
      low_bound = (on_diagonal * low_bound) + (2 * off_diagonal * low_bound);
      approx_val = (on_diagonal * approx_val) + (2 * off_diagonal * approx_val);
    }
    else if (rho != sigma) {
      up_bound = 2 * up_bound * rho->count() * sigma->count();
      low_bound = 2 * low_bound * rho->count() * sigma->count();
      approx_val = 2 * approx_val * rho->count() * sigma->count();
    }
    else {
      up_bound = up_bound * rho->count() * sigma->count();
      low_bound = low_bound * rho->count() * sigma->count();
      approx_val = approx_val * rho->count() * sigma->count();
    }
        
    
    
    
     // I'm not sure this is right, but it will work for now
    approx_val = approx_val * 0.5 * (rho_sigma->stat().density_upper_bound() + 
                                     rho_sigma->stat().density_lower_bound());
    
    
    
    // This is absolute error, I'll need to get the lower bound out of the 
    // mu_nu square tree to do relative
    double my_allowed_error = epsilon_coulomb_ * rho->count() * sigma->count()
        / (number_of_basis_functions_ * number_of_basis_functions_);
   // printf("my_allowed_error: %g\n", my_allowed_error);
    
    // The total error I'm incurring is the max error for one integral times 
    // the number of approximations I'm making
    DEBUG_ASSERT(up_bound >= approx_val);
    DEBUG_ASSERT(approx_val >= low_bound);
    double my_max_error = max((up_bound - approx_val), 
                              (approx_val - low_bound));
    
    // if using absolute error, then existing code is fine
    
    bool below_cutoff;
    
    // For hybrid error 
    // assuming epsilon coulomb is the relative error tolerance
    if ((fabs(mu_nu->stat().entry_upper_bound()) <= hybrid_cutoff_) && 
        (fabs(mu_nu->stat().entry_lower_bound()) <= hybrid_cutoff_)) {
      
      // set my allowed_error to be the absolute bound
      my_allowed_error = my_allowed_error * epsilon_coulomb_absolute_ / 
                         epsilon_coulomb_;
                         
      below_cutoff = true;
      
    }
    else {
      my_allowed_error = my_allowed_error * mu_nu->stat().entry_lower_bound();
      below_cutoff = false;
    }
    
    // For relative error
    /*my_allowed_error = my_allowed_error * mu_nu->stat().entry_lower_bound();*/

    DEBUG_ONLY(*approximate_value = BIG_BAD_NUMBER);

    if (my_max_error < my_allowed_error) {
      
      //DEBUG_ASSERT(my_max_error < epsilon_);
      
      can_prune = true;
      
      *approximate_value = approx_val;
      //printf("approx_val = %g\n", *approximate_value);
      
      if (below_cutoff) {
        num_absolute_prunes_++;
      }
      else {
        num_relative_prunes_++;
      }
      
    }
    
    
    return can_prune;
    
  } // CanApproximateCoulomb_
  
  
  /**
   * Determines if the integral among the four nodes can be approximated
   *
   * Needs to be updated, I don't think I have to worry about overcounting
   */
  bool CanApproximateExchange_(SquareIntegralTree* mu_nu, 
                              SquareIntegralTree* rho_sigma, 
                              double* approximate_value) { 
    
    
    IntegralTree* mu = mu_nu->query1();
    IntegralTree* nu = mu_nu->query2();
    IntegralTree* rho = rho_sigma->query1();
    IntegralTree* sigma = rho_sigma->query2();
    
    // If it is possible to prune here, that will still be true for the two 
    // children, where we won't have to try to handle the near symmetry
    if (RectangleOnDiagonal_(rho, sigma)) {
      DEBUG_ONLY(*approximate_value = BIG_BAD_NUMBER);
      return false;
    }
    
    
    bool can_prune = false;
    
    double mu_rho_min_dist = mu->bound().MinDistanceSq(rho->bound());
    double mu_rho_max_dist = mu->bound().MaxDistanceSq(rho->bound());
    
    double nu_sigma_min_dist = nu->bound().MinDistanceSq(sigma->bound());
    double nu_sigma_max_dist = nu->bound().MaxDistanceSq(sigma->bound());
    
    DHrectBound<2> mu_rho_ave;
    mu_rho_ave.AverageBoxesInit(mu->bound(), rho->bound());
    DHrectBound<2> nu_sigma_ave;
    nu_sigma_ave.AverageBoxesInit(nu->bound(), sigma->bound());
    //ot::Print(mu_nu_ave);
    //ot::Print(rho_sigma_ave);
    
    double mu_rho_four_way_min_dist = mu_rho_ave.MinDistanceSq(nu_sigma_ave);
    double mu_rho_four_way_max_dist = mu_rho_ave.MaxDistanceSq(nu_sigma_ave);
    //double four_way_mid_dist = mu_nu_ave.MidDistanceSq(rho_sigma_ave);
    //printf("four_way_min: %g\n", four_way_min_dist);
    
    double up_bound = ComputeSingleIntegral_(mu_rho_min_dist, 
                                             nu_sigma_min_dist, 
                                             mu_rho_four_way_min_dist);
    double low_bound = ComputeSingleIntegral_(mu_rho_max_dist, 
                                              nu_sigma_max_dist, 
                                              mu_rho_four_way_max_dist);
             
                                                    
                                                                                                                                  
        
    Vector mu_center;
    mu->bound().CalculateMidpoint(&mu_center);
    Vector nu_center;
    nu->bound().CalculateMidpoint(&nu_center);
    Vector rho_center;
    rho->bound().CalculateMidpoint(&rho_center);
    Vector sigma_center;
    sigma->bound().CalculateMidpoint(&sigma_center);
    
    
    double approx_val = ComputeSingleIntegral_(mu_center, rho_center, 
                                               nu_center, sigma_center);
                        
    DEBUG_ASSERT(up_bound >= approx_val);
    DEBUG_ASSERT(approx_val >= low_bound);
    
    
    if (rho != sigma) {
      
      double mu_sigma_min_dist = mu->bound().MinDistanceSq(sigma->bound());
      double mu_sigma_max_dist = mu->bound().MaxDistanceSq(sigma->bound());
      
      double nu_rho_min_dist = nu->bound().MinDistanceSq(rho->bound());
      double nu_rho_max_dist = nu->bound().MaxDistanceSq(rho->bound());
      
      DHrectBound<2> mu_sigma_ave;
      mu_sigma_ave.AverageBoxesInit(mu->bound(), sigma->bound());
      DHrectBound<2> nu_rho_ave;
      nu_rho_ave.AverageBoxesInit(nu->bound(), rho->bound());
      
      double mu_sigma_four_way_min_dist = mu_sigma_ave.MinDistanceSq(nu_rho_ave);
      double mu_sigma_four_way_max_dist = mu_sigma_ave.MaxDistanceSq(nu_rho_ave);
      
      up_bound = up_bound + ComputeSingleIntegral_(mu_sigma_min_dist, 
                                                   nu_rho_min_dist, 
                                                   mu_sigma_four_way_min_dist);
      
      low_bound = low_bound + ComputeSingleIntegral_(mu_sigma_max_dist, 
                                                     nu_rho_max_dist, 
                                                     mu_sigma_four_way_max_dist);
      
      approx_val = approx_val + ComputeSingleIntegral_(mu_center, sigma_center, 
                                                       nu_center, rho_center);
      
      DEBUG_ASSERT(up_bound >= approx_val);
      DEBUG_ASSERT(approx_val >= low_bound);
      
    }
    
    // Because the exchange integrals have an extra 1/2 factor
    up_bound = up_bound * 0.5;
    low_bound = low_bound * 0.5;
    approx_val = approx_val * 0.5;
    
    
    
    up_bound = up_bound * rho->count() * sigma->count();
    low_bound = low_bound * rho->count() * sigma->count();
    approx_val = approx_val * rho->count() * sigma->count();
    
    DEBUG_ASSERT(up_bound >= approx_val);
    DEBUG_ASSERT(approx_val >= low_bound);
    
    double old_up_bound = up_bound;
    
    if (rho_sigma->stat().density_upper_bound() >= 0.0) {
      up_bound = up_bound * rho_sigma->stat().density_upper_bound();
    }
    else {
      up_bound = low_bound * rho_sigma->stat().density_upper_bound();
    }
    if (rho_sigma->stat().density_lower_bound() >= 0.0) {
      low_bound = low_bound * rho_sigma->stat().density_lower_bound();
    }
    else {
      low_bound = old_up_bound * rho_sigma->stat().density_lower_bound();
    }
    
    
    
    DEBUG_ASSERT(up_bound >= low_bound);
    
    
    // I'm not sure this is right, but it will work for now
    approx_val = approx_val * 0.5 * (rho_sigma->stat().density_upper_bound() + 
                                     rho_sigma->stat().density_lower_bound());
    
    
    // This is absolute error, I'll need to get the lower bound out of the 
    // mu_nu square tree to do relative
    double my_allowed_error = epsilon_exchange_ * rho->count() * sigma->count()
      / (number_of_basis_functions_ * number_of_basis_functions_);
    
    // The total error I'm incurring is the max error for one integral times 
    // the number of approximations I'm making
    DEBUG_ASSERT(up_bound >= approx_val);
    DEBUG_ASSERT(approx_val >= low_bound);
    double my_max_error = max((up_bound - approx_val), 
                              (approx_val - low_bound));
    
    // if using absolute error, then existing code is fine
    
    // For hybrid error 
    // assuming epsilon coulomb is the relative error tolerance
    if ((fabs(mu_nu->stat().entry_upper_bound()) <= hybrid_cutoff_) && 
        (fabs(mu_nu->stat().entry_lower_bound()) <= hybrid_cutoff_)) {
      
      DEBUG_ASSERT(mu_nu->stat().entry_upper_bound() >= 
                   mu_nu->stat().entry_lower_bound());
      
      // set my allowed_error to be the absolute bound
      my_allowed_error = my_allowed_error * epsilon_exchange_absolute_ / 
      epsilon_exchange_;
      
    }
    else {
      my_allowed_error = my_allowed_error * mu_nu->stat().entry_lower_bound();
    }
    
    // For relative error
    /*my_allowed_error = my_allowed_error * mu_nu->stat().entry_lower_bound();*/
    
    DEBUG_ONLY(*approximate_value = BIG_BAD_NUMBER);
    if (my_max_error < my_allowed_error) {
      
      can_prune = true;
      
      *approximate_value = approx_val;
      //printf("approx_val = %g\n", *approximate_value);
      
    }
    
    return can_prune;
    
  } // CanApproximateExchange_
  
  
  
  double ComputeSingleIntegral_(double mu_nu_dist, double rho_sigma_dist, 
                                double four_way_dist);
  
                                                              
  double ComputeSingleIntegral_(const Vector& mu_center, 
                                const Vector& nu_center, 
                                const Vector& rho_center, 
                                const Vector& sigma_center);
  
  
  bool RectangleOnDiagonal_(IntegralTree* mu, IntegralTree* nu);

    
   
  /**
   * Exhaustively computes the total integrals among the four nodes 
   *
   * TODO: account for symmetry here, particularly the mu+nu/rho+sigma kind
   *
   * Bug: Does it overcount when the nodes have more than one point?  I think 
   * it does in the case where rho and sigma are the same node.  There is a 
   * similar problem when mu and nu are the same.
   *
   * This is automatically rectangle safe, since all leaves should automatically 
   * be square when they're on the diagonal
   */
  void ComputeCoulombBaseCase_(SquareIntegralTree* mu_nu, 
                               SquareIntegralTree* rho_sigma) {
    
    IntegralTree* mu = mu_nu->query1();
    IntegralTree* nu = mu_nu->query2();
    IntegralTree* rho = rho_sigma->query1();
    IntegralTree* sigma = rho_sigma->query2();
    
    double max_entry = -DBL_INF;
    double min_entry = DBL_INF;
    
    DEBUG_ASSERT(mu->end() > nu->begin());
    DEBUG_ASSERT(rho->end() > sigma->begin());
    
    for (index_t mu_index = mu->begin(); mu_index < mu->end(); mu_index++) {
     
      for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
       
        double integral_value = coulomb_matrix_.ref(mu_index, nu_index);
        
        for (index_t rho_index = rho->begin(); rho_index < rho->end();
             rho_index++) {
         
          for (index_t sigma_index = sigma->begin(); sigma_index < sigma->end(); 
               sigma_index++) {
            
            Vector mu_vec;
            centers_.MakeColumnVector(mu_index, &mu_vec);
            Vector nu_vec;
            centers_.MakeColumnVector(nu_index, &nu_vec);
            Vector rho_vec;
            centers_.MakeColumnVector(rho_index, &rho_vec);
            Vector sigma_vec;
            centers_.MakeColumnVector(sigma_index, &sigma_vec);
            
            // Multiply by normalization to the fourth, since it appears 
            // once in each of the four integrals
            
            
            double this_integral = density_matrix_.get(rho_index, sigma_index) * 
                ComputeSingleIntegral_(mu_vec, nu_vec, rho_vec, sigma_vec);
              
            // this line gets it right
            // double this_integral = ComputeSingleIntegral_(mu_vec, nu_vec, rho_vec, sigma_vec);
            if (rho != sigma) {
              this_integral = this_integral * 2;
            }
            
            integral_value = integral_value + this_integral;
            
            } // sigma
          
        } // rho
        
        // Set both to account for mu/nu symmetry
        coulomb_matrix_.set(mu_index, nu_index, integral_value);
        // Necessary to fill in the lower triangle
        if (mu != nu) {
          coulomb_matrix_.set(nu_index, mu_index, integral_value);
        }
        
        if (integral_value > max_entry) {
          max_entry = integral_value;
        }
        if (integral_value < min_entry) {
          min_entry = integral_value;
        }
        
      } // nu
      
    } // mu
    
    mu_nu->stat().set_entry_upper_bound(max_entry);
    mu_nu->stat().set_entry_lower_bound(min_entry);
    
    index_t new_refs = rho->count() * sigma->count();
    
   if (rho != sigma) {
      new_refs = 2 * new_refs;
      DEBUG_ASSERT(!((rho->begin() < sigma->begin()) && 
                     (rho->end() > sigma->end())));
      DEBUG_ASSERT(!((rho->begin() > sigma->begin()) && 
                     (rho->end() < sigma->end())));
    }
    
    mu_nu->stat().set_remaining_references(mu_nu->stat().remaining_references() 
                                           - new_refs);
    
  } // ComputeCoulombBaseCase_
  
  /**
    * Exhaustively computes the total integrals among the four nodes 
   *
   * TODO: account for symmetry here, particularly the mu+nu/rho+sigma kind
   *
   * Bug: Does it overcount when the nodes have more than one point?  I think 
   * it does in the case where rho and sigma are the same node.  There is a 
   * similar problem when mu and nu are the same.
   *
   * Also rectangle safe, see above.
   */
  void ComputeExchangeBaseCase_(SquareIntegralTree* mu_nu, 
                                SquareIntegralTree* rho_sigma) {
    
    IntegralTree* mu = mu_nu->query1();
    IntegralTree* nu = mu_nu->query2();
    IntegralTree* rho = rho_sigma->query1();
    IntegralTree* sigma = rho_sigma->query2();
    
    double max_entry = -DBL_INF;
    double min_entry = DBL_INF;
    
    DEBUG_ASSERT(mu->end() > nu->begin());
    DEBUG_ASSERT(rho->end() > sigma->begin());
    
    for (index_t mu_index = mu->begin(); mu_index < mu->end(); mu_index++) {
      
      for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
        
        double integral_value = exchange_matrix_.ref(mu_index, nu_index);
        
        for (index_t rho_index = rho->begin(); rho_index < rho->end();
             rho_index++) {
          
          for (index_t sigma_index = sigma->begin(); sigma_index < sigma->end(); 
               sigma_index++) {
            
            
            Vector mu_vec;
            centers_.MakeColumnVector(mu_index, &mu_vec);
            Vector nu_vec;
            centers_.MakeColumnVector(nu_index, &nu_vec);
            Vector rho_vec;
            centers_.MakeColumnVector(rho_index, &rho_vec);
            Vector sigma_vec;
            centers_.MakeColumnVector(sigma_index, &sigma_vec);
            
            
            // Multiply by normalization to the fourth, since it appears 
            // once in each of the four integrals
            
            double kl_integral = density_matrix_.ref(rho_index, sigma_index) * 
              ComputeSingleIntegral_(mu_vec, rho_vec, nu_vec, sigma_vec) * 0.5;
              
            integral_value = integral_value + kl_integral;
              
            // Account for the rho-sigma partial symmetry
            // Need to make this rectangle safe
            if (rho != sigma) {  
            
              double lk_integral = density_matrix_.ref(sigma_index, rho_index) * 
                ComputeSingleIntegral_(mu_vec, sigma_vec, nu_vec, rho_vec) * 
                0.5;
            
              integral_value = integral_value + lk_integral;
              
            }
          } // sigma
          
        } // rho
        
        // Set both to account for mu/nu symmetry
        exchange_matrix_.set(mu_index, nu_index, integral_value);
        // Necessary to fill in the lower triangle
        if (mu != nu) {
          exchange_matrix_.set(nu_index, mu_index, integral_value);
        }
        
        if (integral_value > max_entry) {
          max_entry = integral_value;
        }
        if (integral_value < min_entry) {
          min_entry = integral_value;
        }        
        
      } // nu
      
    } // mu
    
    mu_nu->stat().set_entry_upper_bound(max_entry);
    mu_nu->stat().set_entry_lower_bound(min_entry);
    
    index_t new_refs = rho->count() * sigma->count();
    if (rho != sigma) {
      new_refs = 2 * new_refs;
    }
    mu_nu->stat().set_remaining_references(mu_nu->stat().remaining_references() 
                                           - new_refs);
    
  } // ComputeExchangeBaseCase_
  
  
  /**
   * Once an integral is approximated, this function fills in the approximate
   * value in the Fock matrix.
   * 
   * TODO: think about a smarter way to do this than loops (they won't save me 
   * much time.  Submatrices? - submatrices don't exist, this might be the best
   */
  void FillApproximationCoulomb_(SquareIntegralTree* mu_nu, 
                                 SquareIntegralTree* rho_sigma,
                                 double integral_approximation) {
  
    IntegralTree* mu = mu_nu->query1();
    IntegralTree* nu = mu_nu->query2();
    IntegralTree* rho = rho_sigma->query1();
    IntegralTree* sigma = rho_sigma->query2();
  
  // if mu and nu overlap, recursively call this on children
  // else if rho and sigma overlap, recursively call on children
  // Make sure to update the integral approximation (for both) 
  // and the bounds if splitting mu_nu
  // else do the below
  
    if (RectangleOnDiagonal_(mu, nu)) {
    
      PropagateBoundsDown_(mu_nu);
    
    // this doesn't hold because I haven't propagated them down this far
      DEBUG_ASSERT(mu_nu->stat().remaining_references() ==
                   mu_nu->left()->stat().remaining_references());
      
      FillApproximationCoulomb_(mu_nu->left(), rho_sigma, 
                                integral_approximation);
      FillApproximationCoulomb_(mu_nu->right(), rho_sigma, 
                                integral_approximation);
      
      PropagateBoundsUp_(mu_nu);
      
      /*
      // Should be fine if the above statement holds
      mu_nu->stat().set_remaining_references(
           mu_nu->left()->stat().remaining_references());
      
      // Because it's already been updated for the children
      mu_nu->stat().set_approximation_val(0.0);
      
      // It's just based on the children, right?
      // This is the same code as the propagate bounds function
      mu_nu->stat().set_entry_upper_bound(
          max(mu_nu->left()->stat().entry_upper_bound(), 
              mu_nu->right()->stat().entry_upper_bound()));
      
      mu_nu->stat().set_entry_lower_bound(
           min(mu_nu->left()->stat().entry_lower_bound(), 
               mu_nu->right()->stat().entry_lower_bound()));
      
      */
      
    }
    else if (RectangleOnDiagonal_(rho, sigma)) {
    
      FillApproximationCoulomb_(mu_nu, rho_sigma->left(), 
                                integral_approximation);
      FillApproximationCoulomb_(mu_nu, rho_sigma->right(), 
                                integral_approximation);
                                
      // Because the approximation has been counted twice
      mu_nu->stat().set_entry_upper_bound(mu_nu->stat().entry_upper_bound() - 
                                          integral_approximation);
      mu_nu->stat().set_entry_lower_bound(mu_nu->stat().entry_lower_bound() - 
                                          integral_approximation);
    
    }
    else {
  
    
      for (index_t mu_index = mu->begin(); mu_index < mu->end(); mu_index++) {
        
        for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
          
          double new_value = coulomb_matrix_.ref(mu_index, nu_index) + 
              integral_approximation;
          // Set both to account for mu/nu symmetry
          // This uses the same logic as the base case
          coulomb_matrix_.set(mu_index, nu_index, new_value);
          if (mu != nu) {
            coulomb_matrix_.set(nu_index, mu_index, new_value);
          }
          
        } // nu
        
      } // mu
      
      mu_nu->stat().set_entry_upper_bound(mu_nu->stat().entry_upper_bound() + 
                                          integral_approximation);

      mu_nu->stat().set_entry_lower_bound(mu_nu->stat().entry_lower_bound() + 
                                          integral_approximation);
                                          
      index_t new_refs = rho->count() * sigma->count();
      
      if (rho != sigma) {
        DEBUG_ASSERT(!((rho->begin() <= sigma->begin()) && 
                       (rho->end() >= sigma->end())));
        DEBUG_ASSERT(!((rho->begin() >= sigma->begin()) && 
                       (rho->end() <= sigma->end())));
        new_refs = 2 * new_refs;
      }
      
      mu_nu->stat().set_remaining_references(
          mu_nu->stat().remaining_references() - new_refs);
                                             
      mu_nu->stat().set_approximation_val(integral_approximation);
      
    }
  
  } // FillApproximationCoulomb_()
  
  /**
    * Once an integral is approximated, this function fills in the approximate
   * value in the Fock matrix.
   * 
   * TODO: think about a smarter way to do this than loops (they won't save me 
   * much time.  Submatrices? - submatrices don't exist, this might be the best
   */
  void FillApproximationExchange_(SquareIntegralTree* mu_nu, 
                                 SquareIntegralTree* rho_sigma,
                                 double integral_approximation) {
    
    IntegralTree* mu = mu_nu->query1();
    IntegralTree* nu = mu_nu->query2();
    IntegralTree* rho = rho_sigma->query1();
    IntegralTree* sigma = rho_sigma->query2();
    
    if (RectangleOnDiagonal_(mu, nu)) {
      
      PropagateBoundsDown_(mu_nu);
      
      // this doesn't hold because I haven't propagated them down this far
      DEBUG_ASSERT(mu_nu->stat().remaining_references() ==
                   mu_nu->left()->stat().remaining_references());
      
      FillApproximationExchange_(mu_nu->left(), rho_sigma, 
                                integral_approximation);
      FillApproximationExchange_(mu_nu->right(), rho_sigma, 
                                integral_approximation);
      
      PropagateBoundsUp_(mu_nu);
      
    }
    else if (RectangleOnDiagonal_(rho, sigma)) {
      
      FillApproximationExchange_(mu_nu, rho_sigma->left(), 
                                integral_approximation);
      FillApproximationExchange_(mu_nu, rho_sigma->right(), 
                                integral_approximation);
      
      // Because the approximation has been counted twice
      mu_nu->stat().set_entry_upper_bound(mu_nu->stat().entry_upper_bound() - 
                                          integral_approximation);
      mu_nu->stat().set_entry_lower_bound(mu_nu->stat().entry_lower_bound() - 
                                          integral_approximation);
      
    }
    else {
    
      for (index_t mu_index = mu->begin(); mu_index < mu->end(); mu_index++) {
        
        for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
          
          double new_value = exchange_matrix_.ref(mu_index, nu_index) + 
          integral_approximation;
          // Set both to account for mu/nu symmetry
          // This uses the same logic as the base case
          exchange_matrix_.set(mu_index, nu_index, new_value);
          // I could probably check for this on the outside and make this function
          // faster
          if (mu != nu) {
            exchange_matrix_.set(nu_index, mu_index, new_value);
          }
          
        } // nu
        
      } // mu
      
      mu_nu->stat().set_entry_upper_bound(mu_nu->stat().entry_upper_bound() + 
                                          integral_approximation);
      
      mu_nu->stat().set_entry_lower_bound(mu_nu->stat().entry_lower_bound() + 
                                          integral_approximation);
      
      index_t new_refs = rho->count() * sigma->count();
      if (rho != sigma) {
        new_refs = 2 * new_refs;
      }
      mu_nu->stat().set_remaining_references(mu_nu->stat().remaining_references() 
                                             - new_refs);
      
      mu_nu->stat().set_approximation_val(integral_approximation);
    
    }
    
  } // FillApproximationExchange_()
  
  
  /**
   * Send bound information down the tree before recursive call.
   */
  void PropagateBoundsDown_(SquareIntegralTree* query) {
  
    query->left()->stat().set_remaining_references(
        query->stat().remaining_references());
    query->right()->stat().set_remaining_references(
        query->stat().remaining_references());
    
    if (query->stat().approximation_val() != 0.0) {
      
      query->left()->stat().set_entry_upper_bound(
          query->left()->stat().entry_upper_bound() + 
          query->stat().approximation_val());
      query->right()->stat().set_entry_upper_bound(
          query->right()->stat().entry_upper_bound() + 
          query->stat().approximation_val());
      
      query->left()->stat().set_entry_lower_bound(
          query->left()->stat().entry_lower_bound() + 
          query->stat().approximation_val());
      query->right()->stat().set_entry_lower_bound(
          query->right()->stat().entry_lower_bound() + 
          query->stat().approximation_val());
        
      query->stat().set_approximation_val(0.0);
      
    }
  
  } // PropagateBoundsDown_()
  
  /** 
   * Send bound information up the tree after a recursive call
   */
  void PropagateBoundsUp_(SquareIntegralTree* query) {
  
    double min_entry = query->left()->stat().entry_lower_bound();
    double max_entry = query->left()->stat().entry_upper_bound();
    
    min_entry = min(min_entry, 
                    query->right()->stat().entry_lower_bound());
    max_entry = max(max_entry, 
                    query->right()->stat().entry_upper_bound());
                    
    query->stat().set_entry_upper_bound(max_entry);
    query->stat().set_entry_lower_bound(min_entry);
   
    query->stat().set_remaining_references(
        query->left()->stat().remaining_references());
    
    DEBUG_ASSERT(query->stat().remaining_references() == 
                 query->right()->stat().remaining_references());
  
  } // PropagateBoundsUp_()
  
  
  /**  
   * Handles the recursive calls for the Coulomb part of the computation
   */
  void ComputeCoulombRecursion_(SquareIntegralTree* query, 
                                SquareIntegralTree* reference) {
  
    DEBUG_ASSERT(query->query1()->end() > query->query2()->begin());
    DEBUG_ASSERT(reference->query1()->end() > reference->query2()->begin());
                                           
    double integral_approximation;
         
    if (query->is_leaf() && reference->is_leaf()) {
    
      coulomb_base_cases_++;
      ComputeCoulombBaseCase_(query, reference);
      
    }
    else if(CanApproximateCoulomb_(query, reference, &integral_approximation)) {
      
      DEBUG_ASSERT(integral_approximation != BIG_BAD_NUMBER);
      
      coulomb_approximations_++;
      
      FillApproximationCoulomb_(query, reference, integral_approximation);
      
    }        
    else if (query->is_leaf()) {
    // Don't need to propagate stats
    
      ComputeCoulombRecursion_(query, reference->left());
      ComputeCoulombRecursion_(query, reference->right());
    
    }
    else if (reference->is_leaf()) {
    // Need to propagate some stats
      
      PropagateBoundsDown_(query);
      
      ComputeCoulombRecursion_(query->left(), reference);
      ComputeCoulombRecursion_(query->right(), reference);
      
      PropagateBoundsUp_(query);
    
    }
    else {
      
      PropagateBoundsDown_(query);
      
      ComputeCoulombRecursion_(query->left(), reference->left());
      ComputeCoulombRecursion_(query->left(), reference->right());
      
      ComputeCoulombRecursion_(query->right(), reference->left());
      ComputeCoulombRecursion_(query->right(), reference->right());

      PropagateBoundsUp_(query);
      
    }   
              
  } // ComputeCoulombRecursion_()
  

  /**  
  * Handles the recursive calls for the Exchange part of the computation
  */
  void ComputeExchangeRecursion_(SquareIntegralTree* query, 
                                SquareIntegralTree* reference) {
    
    double integral_approximation;
    
    if (query->is_leaf() && reference->is_leaf()) {
      
      exchange_base_cases_++;
      ComputeExchangeBaseCase_(query, reference);
      
    }
    else if (CanApproximateExchange_(query, reference, &integral_approximation)) {
      
      DEBUG_ASSERT(integral_approximation != BIG_BAD_NUMBER);
      
      exchange_approximations_++;
      
      FillApproximationExchange_(query, reference, integral_approximation);
      
    }        
    else if (query->is_leaf()) {
      // Don't need to propagate stats
      
      ComputeExchangeRecursion_(query, reference->left());
      ComputeExchangeRecursion_(query, reference->right());
      
    }
    else if (reference->is_leaf()) {
      // Need to propagate some stats
      
      PropagateBoundsDown_(query);
      
      ComputeExchangeRecursion_(query->left(), reference);
      ComputeExchangeRecursion_(query->right(), reference);
      
      PropagateBoundsUp_(query);
      
    }
    else {
      
      PropagateBoundsDown_(query);
      
      ComputeExchangeRecursion_(query->left(), reference->left());
      ComputeExchangeRecursion_(query->left(), reference->right());
      
      ComputeExchangeRecursion_(query->right(), reference->left());
      ComputeExchangeRecursion_(query->right(), reference->right());
      
      PropagateBoundsUp_(query);
      
    }
       
  } // ComputeExchangeRecursion_()


  /**
   * Set the upper and lower bounds on a result in the root node of the square 
   * tree.  These bounds are then propagated up and down the tree during the 
   * recursive calls.  
   *
   * TODO: Needs unit tests
   *
   * It is possible to tighten these for children, but only in the case where 
   * the global density bounds have the same sign.  
   */
  void SetEntryBounds_() {
  
    double density_upper = square_tree_->stat().density_upper_bound();
    double density_lower = square_tree_->stat().density_lower_bound();
    
    DEBUG_ASSERT(density_upper >= density_lower);
    
    double entry_upper;
    double entry_lower;
    
    double max_dist = square_tree_->query1()->bound().MaxDistanceSq(
        square_tree_->query2()->bound());
    
    if (density_upper > 0) {
      // then, the largest value is when all the distances are 0
      
      entry_upper = density_upper * number_of_basis_functions_ * 
                    number_of_basis_functions_;
    
    }
    else {
      // then, the largest value is when all the distances are max
    
      entry_upper = density_upper * 
                    ComputeSingleIntegral_(max_dist, max_dist, max_dist) * 
                    number_of_basis_functions_ * number_of_basis_functions_;
    
    }
    
    if (density_lower > 0) {
      //then, the smallest value is when all the distances are max
    
      entry_lower = density_lower * 
                    ComputeSingleIntegral_(max_dist, max_dist, max_dist) * 
                    number_of_basis_functions_ * number_of_basis_functions_;
    
    }
    else {
    
      entry_lower = density_lower * number_of_basis_functions_ * 
                    number_of_basis_functions_;
    
    }
    
    DEBUG_ASSERT(entry_upper >= entry_lower);
    
    square_tree_->stat().set_entry_upper_bound(entry_upper);
    square_tree_->stat().set_entry_lower_bound(entry_lower);
  
  } // SetEntryBounds_
  
  void ResetTreeForExchange_(SquareIntegralTree* root) {
  
    if (root != NULL) {
      
      root->stat().set_remaining_references(number_of_basis_functions_ * 
                                            number_of_basis_functions_);
      
      root->stat().set_approximation_val(0.0);
    
      ResetTreeForExchange_(root->left());
      ResetTreeForExchange_(root->right());  
      
    }
  
  } // ResetTreeForExchange_()
  
  /** 
   * Resets the tree after the density matrix changes.
   *
   * This is also used between Coulomb and Exchange computations, which is 
   * probably wrong.  
   */
  void ResetTree_(SquareIntegralTree* root) {
    
    double max_density;
    double min_density;
    
    if (root->is_leaf()) {
      
      max_density = -DBL_INF;
      min_density = DBL_INF;
      
      for (index_t i = root->query1()->begin(); i < root->query1()->end(); 
           i++) {
      
        for (index_t j = root->query2()->begin(); j < root->query2()->end(); 
             j++) {
        
          double this_density = density_matrix_.ref(i, j);
          if (this_density > max_density) {
            max_density = this_density;
          }
          if (this_density < min_density) {
            min_density = this_density;
          }
        
        } // j
      
      } // i
      
    } // leaf
    else {
    
      ResetTree_(root->left());
      ResetTree_(root->right());
      
      max_density = max(root->left()->stat().density_upper_bound(), 
                        root->right()->stat().density_upper_bound());
      min_density = min(root->left()->stat().density_lower_bound(), 
                        root->right()->stat().density_lower_bound());
                               
    } // non-leaf
    
    root->stat().set_density_upper_bound(max_density);
    root->stat().set_density_lower_bound(min_density);
    
    root->stat().set_remaining_references(number_of_basis_functions_ * 
                                          number_of_basis_functions_);
    
    root->stat().set_approximation_val(0.0);
    
  } // ResetTree_()
  
      
public:
  
  double ErfLikeFunction(double z);
  
  
  /**
   * Initialize the class with the centers of the data points, the fx module,
   * bandwidth
   */
  void Init(const Matrix& centers_in, struct datanode* mod, double band) {
  
    // Needs to be copied because it will be permuted 
    centers_.Copy(centers_in);
    
    module_ = mod;
    
    bandwidth_ = band;
    
    epsilon_ = fx_param_double(module_, "epsilon", 0.01);
    // Later, I can make the factor a tuning parameter
    epsilon_coulomb_ = 0.5 * epsilon_;
    epsilon_exchange_ = 0.5 * epsilon_;
    
    epsilon_absolute_ = 
        fx_param_double(module_, "epsilon_absolute", 1.0);
        
    epsilon_coulomb_absolute_ = 0.5 * epsilon_absolute_;
    epsilon_exchange_absolute_ = 0.5 * epsilon_absolute_;
    
    hybrid_cutoff_ = fx_param_double(module_, "hybrid_cutoff", 0.1);
    
    
    coulomb_approximations_ = 0;
    exchange_approximations_ = 0;
    coulomb_base_cases_ = 0;
    exchange_base_cases_ = 0;
    num_absolute_prunes_ = 0;
    num_relative_prunes_ = 0;
    
    number_of_basis_functions_ = centers_.n_cols();
    fx_format_result(module_, "N", "%d", number_of_basis_functions_);
    
    // A hack to make the pruning always absolute, since I can't input infinity
    // with fx
    if (hybrid_cutoff_ > number_of_basis_functions_) {
      hybrid_cutoff_ = DBL_INF;
    }
    
    // The common normalization constant of all the Gaussians
    normalization_constant_fourth_ = pow((2 * bandwidth_ / math::PI), 3);
    
    coulomb_matrix_.Init(number_of_basis_functions_, 
                         number_of_basis_functions_);
    coulomb_matrix_.SetZero();
    
    exchange_matrix_.Init(number_of_basis_functions_, 
                          number_of_basis_functions_);
    exchange_matrix_.SetZero();
    
    fock_matrix_.Init(number_of_basis_functions_, number_of_basis_functions_);
    fock_matrix_.SetZero();
        
    leaf_size_ = fx_param_int(module_, "leaf_size", 20);
    
    tree_ = tree::MakeKdTreeMidpoint<IntegralTree>(centers_, leaf_size_, 
                &old_from_new_centers_, NULL);
    
    // Set up the indices of the nodes for symmetry pruning
    traversal_index_ = 0;
    //PreOrderTraversal_(tree_);
    //tree_->Print();
    
    square_tree_ = new SquareIntegralTree();
    square_tree_->Init(tree_, tree_, number_of_basis_functions_);
    //square_tree_->Print();
    //SetEntryBounds_();
    
  } // Init
  
  /** 
   * Returns the permutation of the basis centers
   */
  void GetPermutation(ArrayList<index_t>* perm) {
  
    perm->Copy(old_from_new_centers_);
  
  } // GetPermutation()
  
  /**
   * Call this after the density matrix is permuted in the SCF solver
   */
  void SetDensity(const Matrix& updated_density) {
  
    density_matrix_.Copy(updated_density);
    
    ResetTree_(square_tree_);
    
    
    SetEntryBounds_();
  
  }
  
  /**
   * Updates the density matrix and clears the Fock matrix between iterations
   * Should also reset the tree for the next iteration.
   */
  void UpdateMatrices(const Matrix& new_density) {
  
    //This isn't necessary since it's already an alias
    density_matrix_.CopyValues(new_density);
    
    // Reset tree density bounds
    ResetTree_(square_tree_);
    
    SetEntryBounds_();
    
    coulomb_matrix_.SetZero();
    exchange_matrix_.SetZero();
    fock_matrix_.SetZero();
  
  } // UpdateMatrices()
  
  /**
   * Drives the computation, assuming that all the parameters are correct
   */
  void ComputeFockMatrix() {
    
    //tree_->Print();
    //square_tree_->Print();
    
    fx_timer_start(module_, "coulomb_recursion");
    ComputeCoulombRecursion_(square_tree_, square_tree_);  
    fx_timer_stop(module_, "coulomb_recursion");
    // Will need to be followed by clearing the tree and computing the exchange 
    // matrix
    // I think this is the only resetting the tree will need
    SetEntryBounds_();
    ResetTreeForExchange_(square_tree_);
    fx_timer_start(module_, "exchange_recursion");
    ComputeExchangeRecursion_(square_tree_, square_tree_);
    fx_timer_stop(module_, "exchange_recursion");
    
    // Then by adding both into the Fock matrix
    la::AddTo(coulomb_matrix_, &fock_matrix_);
    la::Scale(-1, &exchange_matrix_);
    la::AddTo(exchange_matrix_, &fock_matrix_);
    
  } // ComputeTwoElectronIntegrals
  
  const Matrix& FockMatrix() const {
    return fock_matrix_;
  }
  
  /**
   * Returns the computed Fock matrix.  For now, it just prints it, but should 
   * eventually return it in a useable form for the SCF procedure.
   */
  void OutputFockMatrix(Matrix* fock_out, Matrix* coulomb_out, 
                        Matrix* exchange_out, 
                        ArrayList<index_t>* old_from_new) {
  
    //printf("number_of_approximations_ = %d\n", number_of_approximations_);
    //printf("number_of_base_cases_ = %d\n\n", number_of_base_cases_);
    fx_format_result(module_, "epsilon_coulomb", "%g", epsilon_coulomb_);
    fx_format_result(module_, "epsilon_exchange", "%g", epsilon_exchange_);
    fx_format_result(module_, "coulomb_approximations", "%d", 
                     coulomb_approximations_);
    fx_format_result(module_, "exchange_approximations", "%d", 
                     exchange_approximations_);
    fx_format_result(module_, "coulomb_base_cases", "%d", 
                     coulomb_base_cases_);
    fx_format_result(module_, "exchange_base_cases", "%d", 
                     exchange_base_cases_);
    fx_format_result(module_, "normalization", "%g", 
                     normalization_constant_fourth_);
    fx_format_result(module_, "abs_prunes", "%d", num_absolute_prunes_);
    fx_format_result(module_, "rel_prunes", "%d", num_relative_prunes_);
                     
   /* printf("Multi-tree Coulomb:\n");
    coulomb_matrix_.PrintDebug();
    
    printf("Multi-tree Exchange:\n");
    exchange_matrix_.PrintDebug();
    */
    if (fock_out) {
      fock_out->Copy(fock_matrix_);
    }
    if (coulomb_out) {
      coulomb_out->Copy(coulomb_matrix_);
    }
    if (exchange_out) {
      exchange_out->Copy(exchange_matrix_);
    }
    
    if (old_from_new) {
      old_from_new->Copy(old_from_new_centers_);
    }
    
    // Need to output the Fock matrix
    // should I unpermute here?
    // Maybe keep it permuted in the other code, and unpermute at the end?
    
    // For now, unpermute it here
    
    
  } // OutputFockMatrix()
  
 
}; // class DualTreeIntegrals



#endif 