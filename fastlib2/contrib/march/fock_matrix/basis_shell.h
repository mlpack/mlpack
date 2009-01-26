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
  
  void Init() {
  
  } // Init()

 private:

  // right now, only 0 (s) and 1 (p) are supported
  index_t total_momentum_;

  // number of contracted basis functions in the shell
  index_t num_functions_;
  
  Vector center_;
  
  ArrayList<BasisFunction> functions_;
  
  // I'll need to define these somewhere
  char* atom_type_;
  

}; // class BasisShell


#endif