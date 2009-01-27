#ifndef BASIS_SHELL_H
#define BASIS_SHELL_H

/**
 * A shell is a set of integrals with the same center, total angular momentum, 
 * and exponents, but with different orientations of the momentum.  
 * For example, the px, py, pz orbitals on a single atom form a shell
 *
 * I could consider the s orbital in an sp set as a part of the shell as well
 */
class BasisShell {

 public:

  BasisShell() {}
  
  ~BasisShell() {}
  
  void Init(const Vector& cent, double exp, index_t mom, index_t ind) {
  
    DEBUG_ASSERT(mom >= 0);
    DEBUG_ASSERT(ind >= 0);
    DEBUG_ASSERT(exp >= 0);
    
    total_momentum_ = mom;
    start_index_ = ind;
    exponent_ = exp;
  
    center_.Copy(cent);
  
    if (mom == 0) {
    
      num_functions_ = 1;
    
    }
    else if (mom == 1) {
    
      num_functions_ = 3;
    
    }
    else {
      FATAL("Higher momenta not supported.");
    }
  
  } // Init()
  
  void Copy(BasisShell& inshell) {
  
    total_momentum_ = inshell.total_momentum();
    start_index_ = inshell.start_index();
    exponent_ = inshell.exp();
    center_.Copy(inshell.center());
    num_functions_ = inshell.num_functions();
  
  }
  
  double exp() {
    return exponent_;
  }
  
  const Vector& center() {
    return center_;
  }
  
  index_t num_functions() {
    return num_functions_;
  }
  
  index_t total_momentum() {
    return total_momentum_;
  }
  
  index_t start_index() {
    return start_index_;
  }
  

 private:

  // right now, only 0 (s) and 1 (p) are supported
  index_t total_momentum_;

  // number of contracted basis functions in the shell
  index_t num_functions_;
  
  Vector center_;
  
  double exponent_;
  
  // The index of the center in the Matrix
  index_t start_index_;
  
  //ArrayList<index_t> functions_;
  
  // I'll need to define these somewhere
  char* atom_type_;
  

}; // class BasisShell


#endif