/*
Distribution
 |         \----------------
 |          \         \     \
Gaussian    HMM      LDS    (Multinomial?)


MMK operates on two Distribution objects of the same class



 */

#include "lds.h"
#include "mmk.h"

int main(int argc, char *argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);


  LDS lds_1;
  lds_1.Init(2, 1, true);

  LDS lds_2;
  lds_2.Init(2, 1, true);


  MeanMapKernel<LDS> mmk_lds;
  mmk_lds.Init(1, 10);

  printf("k(lds_1, lds_2) = %f\n", mmk_lds.Compute(lds_1, lds_2));

  
  printf("\n\n\n\n");

  fx_done(root);
  

  return SUCCESS_PASS;
}
