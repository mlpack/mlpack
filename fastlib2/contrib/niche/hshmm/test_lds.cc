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

  if(argc < 3) {
    printf("USAGE: testlds suffix1 suffix2\n");
    return SUCCESS_FAIL;
  }

  
  LDS lds_1;
  lds_1.Init(2, 1, true, argv[1]);

  LDS lds_2;
  lds_2.Init(2, 1, true, argv[2]);


  MeanMapKernel<LDS> mmk_lds;
  mmk_lds.Init(1, 10);

  double kernel_result = mmk_lds.Compute(lds_1, lds_2);
  printf("k(lds_1, lds_2) = %f\n", kernel_result);
  
  FILE *outfile = fopen("kernel_result.txt", "w");
  fprintf(outfile, "%.54f", kernel_result);
  fclose(outfile);

  
  printf("\n\n\n\n");
  
  fx_done(root);
  

  return SUCCESS_PASS;
}
