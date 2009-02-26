#ifndef MULTI_TREE_FOCK_H
#define MULTI_TREE_FOCK_H

#include "fastlib/fastlib.h"
#include "square_fock_tree.h"

const fx_entry_doc multi_tree_fock_entries[] = {
  {"epsilon", FX_PARAM, FX_DOUBLE, NULL, 
   "The relative error cutoff.  Default:0.01\n"},
  {"epsilon_absolute", FX_PARAM, FX_DOUBLE, NULL, 
    "The absolute error cutoff.  Default: 1.0\n"},
  {"hybrid_cutoff", FX_PARAM, FX_DOUBLE, NULL, 
  "The cutoff for hybrid error control.  If the value can be proven to be \n"
  "below the cutoff, then absolute error pruning is used.  For only relative \n"
  "error, set to 0; for only absolute, set to any value larger than the number\n"
  "of basis functions.  Default: 0.1\n"},
  {"N", FX_RESULT, FX_INT, NULL, 
  "The total number of basis functions, as in the dimension of the Fock matrix.\n"},
  {"leaf_size", FX_PARAM, FX_INT, NULL, 
    "The size of the leaves in the tree.  Default: 10\n"},
  {"epsilon_split", FX_PARAM, FX_DOUBLE, NULL, 
    "Controls the allocation of error between the Coulomb and exchange \n"
    "computations.  A setting of 1 allocates all the error to the Coulomb side.\n"
    "Only values in the interval (0,1) are permitted.  Default: 0.5\n"},
  FX_ENTRY_DOC_DONE
};


class MultiTreeFock {

public:


private:

  class SingleNodeStat {
  
   private:
  
    // The node's index in a pre-order depth-first traversal of the tree
    index_t node_index_;
    
    double min_bandwidth_;
    double max_bandwidth_;
    
    index_t height_;
    
    // I don't think these matter in single nodes
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
              const SingleNodeStat& left, const SingleNodeStat& right) {
      
      Init();
      height_ = max(left.height(), right.height()) + 1;
      
      min_bandwidth_ = min(left.min_bandwidth(), right.min_bandwidth());
      max_bandwidth_ = max(left.max_bandwidth(), right.max_bandwidth());
      
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
      DEBUG_ASSERT(new_min < max_bandwidth_);
      DEBUG_ASSERT(new_min > 0);
      min_bandwidth_ = new_min;
    } //set_min_bandwidth
    
    void set_max_bandwidth(double new_max) {
      DEBUG_ASSERT(new_max > min_bandwidth_);
      DEBUG_ASSERT(new_max > 0);
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
  
  }; // class SingleNodeStat
  
 public:
  // This assumes identical bandwidth small Gaussians
  // Otherwise, I'll need something other than a Matrix
  // I should also consider something better than bounding boxes
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, SingleNodeStat> FockTree; 
  
  typedef SquareFockTree<FockTree> SquareTree;
  
 private:
    
  // The tree 
  FockTree* tree_;
  
  // the square tree
  SquareTree* square_tree_;
  
  // Centers of the basis functions
  // assuming one entry per function
  Matrix centers_;
  
  // The fx module
  fx_module* module_;
  
  // The vector of bandwidths
  Vector exponents_;
  
  // The vector of momenta
  Vector momenta_;
  
  // The number of times an approximation is invoked
  int coulomb_approximations_;
  int exchange_approximations_;
  
  index_t num_absolute_prunes_;
  index_t num_relative_prunes_;
  
  
  // The number of times the base case is called
  int coulomb_base_cases_;
  int exchange_base_cases_;
  
  // Controls the allocation of error between Coulomb and exchange computations
  // A value of 1 allocates all of the error to the Coulomb computation
  double epsilon_split_;
  
  // The value eps governing relative error
  double epsilon_;
  
  // the amount of error allocated to the coulomb matrix
  double epsilon_coulomb_;
  
  // the amount of error allocated to the exchange matrix
  double epsilon_exchange_;
  
  // The absolute analogs of the above 
  double epsilon_absolute_;
  double epsilon_coulomb_absolute_;
  double epsilon_exchange_absolute_;
  
  // Values that can be guaranteed to be less than this value are pruned by the 
  // absolute criterion
  double hybrid_cutoff_;
  
  // The return values are stored here
  // fock_matrix__.ref(i, j) is the fock matrix entry i, j
  Matrix fock_matrix_;
  
  // The exchange contribution
  Matrix exchange_matrix_;
  
  // The coulomb contribution
  Matrix coulomb_matrix_;
  
  // The density matrix
  Matrix density_matrix_;
  
  // The total number of basis functions
  // this is the dimensionality of the density matrix
  index_t number_of_basis_functions_;
  
  // what is this for?
  index_t traversal_index_;
  
  // Stores the permutation used in tree-building
  ArrayList<index_t> old_from_new_centers_;
  
  // Size of leaves in the tree
  int leaf_size_;
  
  
  //////////////// Functions /////////////////////////////
  
  /**
   * Determines if the Coulomb interaction between the given square nodes can 
   * currently be approximated.  If so, then *approx_val holds the estimate
   */
  bool CanApproximateCoulomb_(SquareTree* mu_nu, SquareTree* rho_sigma, 
                              double* approx_val);
  
  /**
   * Determines if the Exchange interaction can be approximated, and if so, 
   * fills in *approx_val with the estimate.
   */
  bool CanApproximateExchange_(SquareFockTree* mu_nu, SquareFockTree* rho_sigma, 
                               double* approx_val);
             
 /**
  * Base cases
  */                  
  void ComputeCoulombBaseCase_(SquareFockTree* mu_nu, 
                               SquareFockTree* rho_sigma);

  void ComputeExchangeBaseCase_(SquareFockTree* mu_nu, 
                                SquareFockTree* rho_sigma);


  /**
   * Recursive calls
   */  
  void ComputeCoulombRecursion_(SquareIntegralTree* query, 
                                SquareIntegralTree* ref);
  
  
  void ComputeExchangeRecursion_(SquareIntegralTree* query, 
                                 SquareIntegralTree* ref);
  
  
 public:
 
  void Init(const Matrix& centers_in, const Matrix& exp_in
            const Matrix& momenta_in, const Matrix& density_in, 
            fx_module* mod) {
  
    // Needs to be copied because it will be permuted 
    centers_.Copy(centers_in);
    
    module_ = mod;
    
    exponents_.Copy(exp_in.ptr(), centers_.n_cols());
    momenta_.Copy(mom_in.ptr(), centers_.n_cols());
    
    epsilon_ = fx_param_double(module_, "epsilon", 0.01);
    
    epsilon_split_ = fx_param_double(module_, "epsilon_split", 0.5);
    ASSERT(epsilon_split_ < 1.0 && epsilon_split_ > 0.0);
    
    epsilon_coulomb_ = epsilon_split_ * epsilon_;
    epsilon_exchange_ = (1 - epsilon_split_) * epsilon_;
    DEBUG_ASSERT(epsilon_coulomb_ + epsilon_exchange_ == epsilon_);
    
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
    
    
    number_of_basis_functions_ = centers_.n_cols() + 
        (index_t)2*la::Dot(momenta_, momenta_);
    fx_format_result(module_, "N", "%d", number_of_basis_functions_);
    
    // A hack to make the pruning always absolute, since I can't input infinity
    // with fx
    if (hybrid_cutoff_ > number_of_basis_functions_) {
      hybrid_cutoff_ = DBL_INF;
    }
    
    coulomb_matrix_.Init(number_of_basis_functions_, 
                         number_of_basis_functions_);
    coulomb_matrix_.SetZero();
    
    exchange_matrix_.Init(number_of_basis_functions_, 
                          number_of_basis_functions_);
    exchange_matrix_.SetZero();
    
    fock_matrix_.Init(number_of_basis_functions_, number_of_basis_functions_);
    fock_matrix_.SetZero();
    
    leaf_size_ = fx_param_int(module_, "leaf_size", 10);
    
    tree_ = tree::MakeKdTreeMidpoint<FockTree>(centers_, leaf_size_, 
                                               &old_from_new_centers_, NULL);
    
    // Do I use this for anything?
    // Set up the indices of the nodes for symmetry pruning
    traversal_index_ = 0;
    
    square_tree_ = new SquareFockTree();
    square_tree_->Init(tree_, tree_, number_of_basis_functions_);
    
  
  } // Init()
  
  // Should see how CFMM code unpermutes and use that
  void GetPermutation(ArrayList<index_t>* perm) {
    perm->Copy(old_from_new_centers_);
  } // GetPermutation()
  
  void ComputeFockMatrix();
    

}; // class MultiTreeFock


#endif 
