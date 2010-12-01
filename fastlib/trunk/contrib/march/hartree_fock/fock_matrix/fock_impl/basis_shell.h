#ifndef BASIS_SHELL_H
#define BASIS_SHELL_H

#include "fastlib/fastlib.h"
#include "eri.h"

/**
 * A shell is a set of integrals with the same center, total angular momentum, 
 * and exponents, but with different orientations of the momentum.  
 * For example, the px, py, pz orbitals on a single atom form a shell
 *
 * I could consider the s orbital in an sp set as a part of the shell as well,
 * but the benefits are probably small.
 */
class BasisShell {

 public:

  //OT_DEF_BASIC(BasisShell);

  /*BasisShell() {}
  
  ~BasisShell() {}
  */
  
  // the index here is the first index into the Fock matrix for this shell
  // the code that creates the list of BasisShells should keep up with
  // how many functions there are
  // Compute list of BasisShells - returns total number of functions
  // then this number is the dimensionality of the matrices that can be 
  // used to init them / check that this matches the size of the input density
  // matrix
  void Init(const Vector& cent, double exp, index_t mom, index_t ind, 
            index_t list_ind) {
  
    DEBUG_ASSERT(mom >= 0);
    DEBUG_ASSERT(ind >= 0);
    DEBUG_ASSERT(exp >= 0);
    
    total_momentum_ = mom;
    exponent_ = exp;
  
    // alias instead?  I shouldn't need to change it
    center_.Copy(cent);
  
    num_functions_ = eri::NumFunctions(total_momentum_);
    
    matrix_indices_.Init(num_functions_);
    for (index_t i = 0; i < num_functions_; i++) {
      matrix_indices_[i] = ind + i;
    }
    
    list_index_ = list_ind;
    
    ComputeShellNormalization();
    
  } // Init()
  
  void Copy(BasisShell& inshell) {
  
    total_momentum_ = inshell.total_momentum();
    exponent_ = inshell.exp();
    // does this need to be Copy?  alias instead?
    center_.Copy(inshell.center());
    num_functions_ = inshell.num_functions();
    normalization_constants_.InitCopy(inshell.normalization_constants());
    matrix_indices_.InitCopy(inshell.matrix_indices());
    
  }
  
  /**
   * Fills in the array of normalization constants in Libint order.
   */
  void ComputeShellNormalization() {
    
    normalization_constants_.Init(num_functions_);
    
    index_t fun_index = 0;
    
    for (index_t i = 0; i <= total_momentum_; i++) {
      
      int nx = total_momentum_ - i;
      
      for (index_t j = 0; j <= i; j++) {
        
        int ny = i - j;
        int nz = j;
        
        double this_norm = eri::ComputeNormalization(exponent_, nx, ny, nz);
        DEBUG_ASSERT(!isinf(this_norm));
        normalization_constants_[fun_index] = this_norm;
        fun_index++; 
        
      }
      
    }
    
  } // ComputeNormalization Basis Shell
  
  
  double exp() const {
    return exponent_;
  }
  
  const Vector& center() const {
    return center_;
  }
  
  index_t num_functions() const {
    return num_functions_;
  }
  
  index_t total_momentum() const {
    return total_momentum_;
  }
  
  const ArrayList<double>& normalization_constants() const {
    return normalization_constants_;
  }
  
  const ArrayList<index_t>& matrix_indices() const {
    return matrix_indices_; 
  }
  
  double normalization_constant(index_t i) const {
    
    DEBUG_ASSERT(i < num_functions_);
    DEBUG_ASSERT(i >= 0);
    
    return normalization_constants_[i];
    
  }
  
  index_t matrix_index(index_t i) const {

    DEBUG_ASSERT(i < num_functions_);
    DEBUG_ASSERT(i >= 0);
    
    return matrix_indices_[i];
    
  }
  
  
  // these functions are all for LinK
  
  index_t current_mu() {
    return current_mu_;
  }
  
  void set_current_mu(index_t mu_in) {
    current_mu_ = mu_in;
  }
  
  double max_schwartz_factor() {
    return max_schwartz_factor_;
  }
  
  void set_max_schwartz_factor(double fac) {
    max_schwartz_factor_ = fac;
  }
  
  double current_density_entry() {
    return current_density_entry_;
  } 
  
  void set_current_density_entry(double entry) {
    current_density_entry_ = entry;
  }
  
  index_t list_index() const {
    return list_index_;
  }
    

 private:

  index_t total_momentum_;

  // number of basis functions in the shell, depends on the momentum
  index_t num_functions_;
  
  // need to be able to index the density and Fock matrix for each function
  // these are accessed in the libint order
  // i.e. p_x, p_y, p_z
  // see the Libint programmer's manual for more info
  ArrayList<index_t> matrix_indices_;

  Vector center_;
  
  double exponent_;
  
  ArrayList<double> normalization_constants_;
  
  // this shell's location in the master list of shells
  // used in LinK
  index_t list_index_;
  
  // used in sorting for LinK
  // is it really necessary to have these?
  // could I handle LinK some other way?
  index_t current_mu_;
  double current_density_entry_;
  
  // (mu_max|mu_max)^1/2 from LinK paper
  double max_schwartz_factor_;
    

}; // class BasisShell


#endif
