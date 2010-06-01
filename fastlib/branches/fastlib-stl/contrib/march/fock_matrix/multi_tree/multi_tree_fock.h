#ifndef MULTI_TREE_FOCK_H
#define MULTI_TREE_FOCK_H

#include "fastlib/fastlib.h"
#include "matrix_tree_impl.h"
#include "eri_bounds.h"

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
  {"num_schwartz_prunes", FX_RESULT, FX_INT, NULL,
   "The number of times the Schwartz prescreening estimate allowed a prune.\n"},
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
  {"bounds_cutoff", FX_PARAM, FX_DOUBLE, NULL,
    "Bounds computed to be below this value are set to zero.  Default: 0.0\n"},
  {"schwartz_pruning", FX_PARAM, FX_BOOL, NULL,
   "Specify this parameter to activate pruning based on the Schwartz inequality.\n"},
  {"num_integrals_computed", FX_RESULT, FX_INT, NULL,
  "The total number of integral computations.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc multi_mod_doc = {
  multi_tree_fock_entries, NULL, 
  "Algorithm module for multi tree method.\n"
};


class MultiTreeFock {

 private:
    
  // The tree 
  BasisShellTree* tree_;
  
  // the square tree
  MatrixTree* matrix_tree_;
  
  // Centers of the basis functions
  Matrix centers_;
  
  // The fx module
  fx_module* module_;
  
  // The vector of bandwidths
  Vector exponents_;
  
  // The vector of momenta
  Vector momenta_;
  
  // The number of times an approximation is invoked
  int num_coulomb_approximations_;
  int num_exchange_approximations_;
  
  index_t num_integrals_computed_;
  
  // The number of times the base case is called
  int num_coulomb_base_cases_;
  int num_exchange_base_cases_;
  
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
  
  // Stores the permutation used in tree-building
  ArrayList<index_t> old_from_new_shells_;
  
  // Size of leaves in the tree
  int leaf_size_;
  
  // true if the error is relative, false if absolute
  bool relative_error_;
  
  // having trouble with bounds very close to zero
  double bounds_cutoff_;
  
  // the number of times the schwartz bound works for a prune
  index_t num_schwartz_prunes_;
  
  ArrayList<BasisShell> shell_list_;
  ArrayList<BasisShell*> shell_ptr_list_;
  
  // The available error
  double epsilon_;
  
  double prescreening_cutoff_;
  int num_coulomb_prescreening_prunes_;
  int num_exchange_prescreening_prunes_;
  
  //////////////// Functions /////////////////////////////
  
  void PassBoundsUp_(MatrixTree* query);
  
  void PassBoundsDown_(MatrixTree* query);
  
  ///////// Recursive Calls ////////////////////////
  
  void ComputeBaseCaseCoulomb(MatrixTree* query, MatrixTree* reference);
  void ComputeBaseCaseExchange(MatrixTree* query, MatrixTree* reference);
  
  /**
   * Heuristic to determine which node to split
   * Returns true if query should be split
   * Returns false if reference
   */
  bool SplitQuery(MatrixTree* query, MatrixTree* reference);
  
  void NodeBoundsCoulomb(MatrixTree* query, MatrixTree* reference,
                  double* max_coulomb, double* min_coulomb);

  void NodeBoundsExchange(MatrixTree* query, MatrixTree* reference,
                          double* max_exchange, double* min_exchange);
  
  bool CanPruneCoulomb(MatrixTree* query, MatrixTree* reference,
                       double* approx_val, double* lost_error);

  bool CanPruneExchange(MatrixTree* query, MatrixTree* reference,
                        double* approx_val, double* lost_error);
  
  void DepthFirstRecursionCoulomb(MatrixTree* query, MatrixTree* reference);
  void DepthFirstRecursionExchange(MatrixTree* query, MatrixTree* reference);
  
  
  
  /**
   * PermuteMatrices and vectors
   */
  /*
  void ApplyPermutation(ArrayList<index_t>& old_from_new, Matrix* mat);

  void ApplyPermutation(ArrayList<index_t>& old_from_new, Vector* vec);
   
  void UnApplyPermutation(ArrayList<index_t>& old_from_new, Matrix* mat);

  void UnApplyPermutation(ArrayList<index_t>& old_from_new, Vector* vec);
   */

  
 public:
 
  void Init(const Matrix& centers_in, const Matrix& exp_in,
            const Matrix& momenta_in, const Matrix& density_in, 
            fx_module* mod) {
  
    // Needs to be copied because it will be permuted 
    centers_.Copy(centers_in);
    
    module_ = mod;
    
    exponents_.Copy(exp_in.ptr(), centers_.n_cols());
    momenta_.Copy(momenta_in.ptr(), centers_.n_cols());
    density_matrix_.Copy(density_in);

    epsilon_ = fx_param_double(module_, "epsilon", 0.01);
    
    prescreening_cutoff_ = fx_param_double(module_, "prescreening_cutoff", 0.0);
    
    num_coulomb_approximations_ = 0;
    num_exchange_approximations_ = 0;
    num_integrals_computed_ = 0;
    num_coulomb_base_cases_ = 0;
    num_exchange_base_cases_ = 0;    
    num_coulomb_prescreening_prunes_ = 0;
    num_exchange_prescreening_prunes_ = 0;
    
    number_of_basis_functions_ = eri::CreateShells(centers_, exponents_, momenta_,
                                                   &shell_list_);
    fx_result_int(module_, "N", number_of_basis_functions_);
    
    shell_ptr_list_.Init(shell_list_.size());
    for (index_t i = 0; i < shell_list_.size(); i++) {
      shell_ptr_list_[i] = &(shell_list_[i]);
    }
    
    leaf_size_ = fx_param_int(module_, "leaf_size", 10);
    
    printf("====Building Tree====\n");
    
    fx_timer_start(module_, "multi_time");
    
    fx_timer_start(module_, "tree_building");
    tree_ = shell_tree_impl::CreateShellTree(shell_ptr_list_, leaf_size_, 
                                             &old_from_new_shells_, NULL);
                              
    fx_timer_stop(module_, "tree_building");

    printf("====Matrix Tree Building====\n");
    
    matrix_tree_ = matrix_tree_impl::CreateMatrixTree(tree_, shell_ptr_list_, 
                                                      density_matrix_);
    
    // set up the pruning values
    matrix_tree_->set_remaining_references(matrix_tree_->num_pairs());
    // half for coulomb, half for exchange
    matrix_tree_->set_remaining_epsilon(epsilon_ * 0.5);
    
    relative_error_ = !fx_param_exists(module_, "absolute_error");
    
    fx_timer_stop(module_, "multi_time");

    
    bounds_cutoff_ = fx_param_double(module_, "bounds_cutoff", 0.0);
    if (bounds_cutoff_ < 0.0) {
      bounds_cutoff_ = 0.0;
    }
    
    num_schwartz_prunes_ = 0;
    
  } // Init()
  
  void Destruct() {
    
    centers_.Destruct();
    centers_.Init(1,1);
    
    exponents_.Destruct();
    exponents_.Init(1);
    
    momenta_.Destruct();
    momenta_.Init(1);
    
    coulomb_matrix_.Destruct();
    coulomb_matrix_.Init(1,1);
    
    exchange_matrix_.Destruct();
    exchange_matrix_.Init(1,1);
    
    fock_matrix_.Destruct();
    fock_matrix_.Init(1,1);
    
    old_from_new_shells_.Clear();
    //old_from_new_centers_.Init(1);
    
    density_matrix_.Destruct();
    density_matrix_.Init(1,1);
    
  } // Destruct()
  
  // Should see how CFMM code unpermutes and use that
  void GetPermutation(ArrayList<index_t>* perm) {
    perm->InitCopy(old_from_new_shells_);
  } // GetPermutation()
  
  /**
   * For use in between iterations of SCF solver
   */
  void UpdateDensity(const Matrix& new_density);
    
  /**
   * Algorithm driver
   */
  void Compute();
  
  /**
   * Output matrices
   *
   * TODO: make this function unpermute them first
   */
  void OutputFockMatrix(Matrix* fock_out, Matrix* coulomb_out, 
                        Matrix* exchange_out, 
                        ArrayList<index_t>* old_from_new);
  
  void OutputCoulomb(Matrix* coulomb_out);
  
  void OutputExchange(Matrix* exchange_out);
  
  
}; // class MultiTreeFock


#endif 
