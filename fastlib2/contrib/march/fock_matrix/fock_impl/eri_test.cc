#include "eri.h"


int main(int argc, char* argv[]) {

  eri::ERIInit();

  
  Vector A_vec;
  A_vec.Init(3);
  A_vec[0] = 0.0;
  A_vec[1] = 0.0;
  A_vec[2] = 0.0;
  double A_exp = 1.0;
  int A_mom = 1;
  
  BasisShell A_Shell;
  A_Shell.Init(A_vec, A_exp, A_mom, 1);
  
  Vector B_vec;
  B_vec.Init(3);
  B_vec[0] = 1.0;
  B_vec[1] = 0.0;
  B_vec[2] = 0.0;
  double B_exp = 2.5;
  int B_mom = 1;
  
  BasisShell B_Shell;
  B_Shell.Init(B_vec, B_exp, B_mom, 0);
  
  /*
  Vector C_vec;
  C_vec.Init(3);
  C_vec[0] = 0.0;
  C_vec[1] = 1.0;
  C_vec[2] = 0.0;
  double C_exp = 1.5;
  int C_mom = 0;
  
  BasisShell C_Shell;
  C_Shell.Init(C_vec, C_exp, C_mom, 2);
  
  Vector D_vec;
  D_vec.Init(3);
  D_vec[0] = 0.0;
  D_vec[1] = 0.0;
  D_vec[2] = 1.0;
  double D_exp = 0.8;
  int D_mom = 1;
  
  BasisShell D_Shell;
  D_Shell.Init(D_vec, D_exp, D_mom, 3);
  */

  
  int num_integrals = eri::NumFunctions(A_mom) * eri::NumFunctions(B_mom);
  
  double* integrals = eri::ComputeOverlapIntegrals(A_Shell, B_Shell);
  
  for (index_t i = 0; i < num_integrals; i++) {
   
    printf("integrals[%d] = %g\n", i, integrals[i]);
    
  }
  
  free(integrals);
  
  eri::ERIFree();

} // main()