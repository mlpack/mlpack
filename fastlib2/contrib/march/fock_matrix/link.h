#ifndef LINK_H
#define LINK_H

#include "fastlib/fastlib.h"
#include "eri.h"
#include "basis_shell.h"
#include "shell_pair.h"



class Link {

 private:

  Matrix centers_;
  
  Matrix fock_matrix_;
  
  ArrayList<BasisShell> shell_list_;
  ArrayList<ShellPair> shell_pair_list_;

 public:

  void Init() {
  
    
  
  }
    
  void ComputeFockMatrix();

  void OutputFockMatrix();

}; // class Link






#endif