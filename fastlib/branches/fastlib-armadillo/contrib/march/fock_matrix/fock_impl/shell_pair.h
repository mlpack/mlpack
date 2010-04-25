#ifndef SHELL_PAIR_H
#define SHELL_PAIR_H

#include "basis_shell.h"
#include "eri.h"

//class BasisShell;

class ShellPair {

 public:

  ShellPair() {}
  
  ~ShellPair() {}
  
  // add list index here
  void Init(index_t M_index, index_t N_index, BasisShell* M_shell, 
            BasisShell* N_shell, index_t list_ind, const Matrix& density) {
    
    DEBUG_ASSERT(M_index <= N_index);
    M_index_ = M_index;
    N_index_ = N_index;
    
    M_shell_ = M_shell;
    N_shell_ = N_shell;
    
    integral_upper_bound_ = DBL_MAX;
    integral_lower_bound_ = -DBL_MAX;
    
    exponent_ = eri::ComputeGPTCenter(M_shell_->center(), M_shell_->exp(), 
                                      N_shell_->center(), N_shell_->exp(), 
                                      &center_);
    
    list_index_ = list_ind;
    
    // this includes a factor of pi
    overlap_ = eri::ComputeShellOverlap(*M_shell_, *N_shell_);
    
    density_bound_ = -DBL_MAX;
    for (index_t i = 0; i < M_shell_->num_functions(); i++) {
      for (index_t j = 0; j < N_shell_->num_functions(); j++) {
        
        double density_val = density.get(M_shell_->matrix_index(i), 
                                         N_shell_->matrix_index(j));
        density_bound_ = max(fabs(density_val), 
                             density_bound_);
        
      }
    }
    
  } // Init()         
            
  index_t M_index() const {
    return M_index_;
  }
  index_t N_index() const {
    return N_index_;
  }
  
  BasisShell* M_Shell() {
    return M_shell_;
  }
  
  BasisShell* N_Shell() {
    return N_shell_;
  }
  
  double exponent() const {
    return exponent_;
  }
  
  const Vector& center() const {
    return center_;
  }
  
  
  void set_indices(index_t M_index, index_t N_index) {
  
    DEBUG_ASSERT(M_index <= N_index);
    M_index_ = M_index;
    N_index_ = N_index;
  
  } // set_indices()
  
  void set_integral_upper_bound(double bd) {
    integral_upper_bound_ = bd;
  } 
  
  double integral_upper_bound() const {
    return integral_upper_bound_;
  }
  
  double schwartz_factor() const {
    return schwartz_factor_;
  }
  
  void set_schwartz_factor(double fac) {
    schwartz_factor_ = fac;
  }
  
  void set_list_index(index_t ind) {
    list_index_ = ind;
  }
  
  index_t list_index() const {
    return list_index_;
  }
  
  double overlap() const {
    return overlap_;
  }
  
  double density_bound() const {
    return density_bound_;
  } 
  
  // need to make sure that I update this between SCF iterations
  void set_density_bound(double bound) {
    DEBUG_ASSERT(bound >= 0.0);
    density_bound_ = bound;
  }

 private:

  // density matrix bounds
  // do these get used anywhere?
  double density_mat_upper_;
  double density_mat_lower_;
  
  double density_bound_;
  

  double integral_upper_bound_;
  double integral_lower_bound_;
  
  // The indices of the two shells that make up the pair in the list of shells
  index_t M_index_;
  index_t N_index_;
  
  BasisShell* M_shell_;
  BasisShell* N_shell_;
  
  index_t num_prunes_;
  
  // this shell pair's position in the master list of shell_pairs
  index_t list_index_;
  
  // gamma
  double exponent_;
  
  // the average center weighted by the bandwidths
  Vector center_;
  
  // the Schwartz factor
  double schwartz_factor_;

  double overlap_;
  
}; // class ShellPair




#endif