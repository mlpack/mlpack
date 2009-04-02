#ifndef MULTI_TREE_FOCK_H
#define MULTI_TREE_FOCK_H

#include "fastlib/fastlib.h"
#include "square_fock_tree.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"

const fx_entry_doc multi_tree_fock_entries[] = {
  {"epsilon", FX_PARAM, FX_DOUBLE, NULL, 
   "The relative error cutoff.  Default:0.01\n"},
  {"N", FX_RESULT, FX_INT, NULL, 
  "The total number of basis functions, as in the dimension of the Fock matrix.\n"},
  {"leaf_size", FX_PARAM, FX_INT, NULL, 
    "The size of the leaves in the tree.  Default: 10\n"},
  {"epsilon_split", FX_PARAM, FX_DOUBLE, NULL, 
    "Controls the allocation of error between the Coulomb and exchange \n"
    "computations.  A setting of 1 allocates all the error to the Coulomb side.\n"
    "Only values in the interval (0,1) are permitted.  Default: 0.5\n"},
  {"coulomb_recursion", FX_TIMER, FX_CUSTOM, NULL, 
   "Amount of time spent computing J.\n"},
  {"exchange_recursion", FX_TIMER, FX_CUSTOM, NULL, 
   "Amount of time spent computing K.\n"},
  {"epsilon_coulomb", FX_RESULT, FX_DOUBLE, NULL, 
   "Amount of error allocated to the coulomb recursion.\n"},
  {"epsilon_exchange", FX_RESULT, FX_DOUBLE, NULL, 
   "Amount of error allocated to the exchange recursion.\n"},
  {"coulomb_approximations", FX_RESULT, FX_INT, NULL, 
   "The number of prunes made in the coulomb recursion.\n"},
  {"exchange_approximations", FX_RESULT, FX_INT, NULL, 
  "The number of prunes made in the exchange recursion.\n"},
  {"coulomb_base_cases", FX_RESULT, FX_INT, NULL, 
   "The number of base cases computed in the coulomb recursion.\n"},
  {"exchange_base_cases", FX_RESULT, FX_INT, NULL, 
   "The number of base_cases_computed in the exchange recursion.\n"},
  {"absolute_error", FX_PARAM, FX_BOOL, NULL, 
   "Specify this parameter to use absolute error, defaults to relative.\n"},
  {"tree_building", FX_TIMER, FX_CUSTOM, NULL, 
   "Time spent to build the kd-tree.\n"},
  {"square_tree_building", FX_TIMER, FX_CUSTOM, NULL, 
    "Time spent to build the square tree.\n"},
  {"coulomb_recursion", FX_TIMER, FX_CUSTOM, NULL, 
    "Computing the Coulomb matrix.\n"},
  {"exchange_recursion", FX_TIMER, FX_CUSTOM, NULL, 
    "Computing the exchange matrix.\n"},
  {"multi_time", FX_TIMER, FX_CUSTOM, NULL, 
    "Total time spent to initialize the trees and compute F.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc multi_mod_doc = {
  multi_tree_fock_entries, NULL, 
  "Algorithm module for multi tree method.\n"
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
      
      // need to set min and max bandwidth here
      // can't, the entries are being permuted, so I can't reference the 
      // exponents_ vector
      
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
      DEBUG_ASSERT(new_min <= max_bandwidth_);
      DEBUG_ASSERT(new_min > 0);
      min_bandwidth_ = new_min;
    } //set_min_bandwidth
    
    void set_max_bandwidth(double new_max) {
      DEBUG_ASSERT(new_max >= min_bandwidth_);
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
  
  // The number of times the base case is called
  int coulomb_base_cases_;
  int exchange_base_cases_;
  
  // Controls the allocation of error between Coulomb and exchange computations
  // A value of 1 allocates all of the error to the Coulomb computation
  double epsilon_split_;
  
  // The value eps governing error
  double epsilon_;
  
  double epsilon_coulomb_;
  double epsilon_exchange_;
  
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
  
  // true if the error is relative, false if absolute
  bool relative_error_;
  
  // having trouble with bounds very close to zero
  double bounds_cutoff_;
  
  
  //////////////// Functions /////////////////////////////
  
  /**
   * Normalization constants
   */
  double NodeMaxNorm_(FockTree* mu);
  
  double NodeMinNorm_(FockTree* mu);
  
  double NodeAveNorm_(FockTree* mu);  
  
  /**
   * Returns the maximum integral between two square tree nodes
   */
  double NodesMaxIntegral_(FockTree* mu, FockTree* nu, FockTree* rho, 
                           FockTree* sigma);
  
  double NodesMinIntegral_(FockTree* mu, FockTree* nu, FockTree* rho,
                           FockTree* sigma);
  
  double NodesMidpointIntegral_(FockTree* mu, FockTree* nu, FockTree* rho, 
                                FockTree* sigma);
                                
                                
  
  /**
   * Determines if the pair of nodes represent a non square square node
   * on the diagonal.  This is important for counting the number of repeated 
   * reference pairs accurately.  
   */
  bool RectangleOnDiagonal_(FockTree* mu, FockTree* nu);
  
  /**
   * Counts the number of entries on the diagonal in the square node.  This 
   * is needed for accurately counting how many references are involved in 
   * an approximation.  
   */
  index_t CountOnDiagonal_(SquareTree* rho_sigma);
  
  
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
  bool CanApproximateExchange_(SquareTree* mu_nu, SquareTree* rho_sigma, 
                               double* approx_val);
             
 /**
  * Base cases
  */                  
  void ComputeCoulombBaseCase_(SquareTree* mu_nu, 
                               SquareTree* rho_sigma);

  void ComputeExchangeBaseCase_(SquareTree* mu_nu, 
                                SquareTree* rho_sigma);
                                
  /**
   * Fill in approximations after pruning
   *
   * NOTE: these are almost the same function, the only difference is which
   * matrix gets written to at the end
   * How can I make them the same?
   */
  void FillApproximationCoulomb_(SquareTree* mu_nu, 
                                 SquareTree* rho_sigma,
                                 double integral_approximation);
                                 
  void FillApproximationExchange_(SquareTree* mu_nu, 
                                  SquareTree* rho_sigma,
                                  double integral_approximation);
    


  /**
   * Keep track of bounds
   */
  void SetExponentBounds_(FockTree *tree);
   
  void PropagateBoundsDown_(SquareTree* query);
  
  void PropagateBoundsUp_(SquareTree* query);

  void SetEntryBounds_();

  void ResetTreeForExchange_(SquareTree* root);
  
  void ResetTree_(SquareTree* root);
  
  /**
   * PermuteMatrices and vectors
   */
  void ApplyPermutation(ArrayList<index_t>& old_from_new, Matrix* mat);

  void ApplyPermutation(ArrayList<index_t>& old_from_new, Vector* vec);
   
  void UnApplyPermutation(ArrayList<index_t>& old_from_new, Matrix* mat);

  void UnApplyPermutation(ArrayList<index_t>& old_from_new, Vector* vec);


  /**
   * Recursive calls
   */  
  void ComputeCoulombRecursion_(SquareTree* query, 
                                SquareTree* ref);
  
  
  void ComputeExchangeRecursion_(SquareTree* query, 
                                 SquareTree* ref);
                                
  
  
 public:
 
  void Init(const Matrix& centers_in, const Matrix& exp_in,
            const Matrix& momenta_in, const Matrix& density_in, 
            fx_module* mod) {
  
    // Needs to be copied because it will be permuted 
    centers_.Copy(centers_in);
    
    module_ = mod;
    
    exponents_.Copy(exp_in.ptr(), centers_.n_cols());
    momenta_.Copy(momenta_in.ptr(), centers_.n_cols());
    
    epsilon_ = fx_param_double(module_, "epsilon", 0.01);
    
    epsilon_split_ = fx_param_double(module_, "epsilon_split", 0.5);
    if ((epsilon_split_ > 1.0) || (epsilon_split_ < 0.0)) {
      FATAL("Epsilon split must be between 0 and 1.\n");
    }
    
    epsilon_coulomb_ = epsilon_split_ * epsilon_;
    epsilon_exchange_ = (1 - epsilon_split_) * epsilon_;
    DEBUG_ASSERT(epsilon_coulomb_ + epsilon_exchange_ <= epsilon_);
    
    coulomb_approximations_ = 0;
    exchange_approximations_ = 0;
    coulomb_base_cases_ = 0;
    exchange_base_cases_ = 0;
    
    number_of_basis_functions_ = centers_.n_cols() + 
        (index_t)2*(index_t)la::Dot(momenta_, momenta_);
    fx_result_int(module_, "N", number_of_basis_functions_);
    
    coulomb_matrix_.Init(number_of_basis_functions_, 
                         number_of_basis_functions_);
    coulomb_matrix_.SetZero();
    
    exchange_matrix_.Init(number_of_basis_functions_, 
                          number_of_basis_functions_);
    exchange_matrix_.SetZero();
    
    fock_matrix_.Init(number_of_basis_functions_, number_of_basis_functions_);
    fock_matrix_.SetZero();
    
    leaf_size_ = fx_param_int(module_, "leaf_size", 10);
    
    printf("====Building Tree====\n");
    
    fx_timer_start(module_, "multi_time");
    
    fx_timer_start(module_, "tree_building");
    tree_ = tree::MakeKdTreeMidpoint<FockTree>(centers_, leaf_size_, 
                                               &old_from_new_centers_, NULL);
                              
    // set up the min and max exponents                 
    SetExponentBounds_(tree_);

    fx_timer_stop(module_, "tree_building");

                                               
    // Do I use this for anything?
    // Set up the indices of the nodes for symmetry pruning
    traversal_index_ = 0;
    

    printf("====Square Tree Building====\n");
    
    fx_timer_start(module_, "square_tree_building");
    square_tree_ = new SquareTree();
    square_tree_->Init(tree_, tree_, number_of_basis_functions_);
    fx_timer_stop(module_, "square_tree_building");
    
    // IMPORTANT: permute the exponents, mommenta, and density
    density_matrix_.Init(number_of_basis_functions_, number_of_basis_functions_);
    UpdateMatrices(density_in);
    
    ApplyPermutation(old_from_new_centers_, &exponents_);
    ApplyPermutation(old_from_new_centers_, &momenta_);
    
    relative_error_ = !fx_param_exists(module_, "absolute_error");
    
    square_tree_->stat().set_remaining_epsilon(epsilon_split_ * epsilon_);
    
    PropagateBoundsDown_(square_tree_);
    
    fx_timer_stop(module_, "multi_time");

    
    bounds_cutoff_ = 10e-50;
    
  } // Init()
  
  // Should see how CFMM code unpermutes and use that
  void GetPermutation(ArrayList<index_t>* perm) {
    perm->InitCopy(old_from_new_centers_);
  } // GetPermutation()
  
  /**
   * For use in between iterations of SCF solver
   */
  void UpdateMatrices(const Matrix& new_density);
    
  /**
   * Algorithm driver
   */
  void ComputeFockMatrix();
  
  /**
   * Output matrices
   *
   * TODO: make this function unpermute them first
   */
  void OutputFockMatrix(Matrix* fock_out, Matrix* coulomb_out, 
                        Matrix* exchange_out, 
                        ArrayList<index_t>* old_from_new);
  
  
}; // class MultiTreeFock


#endif 
