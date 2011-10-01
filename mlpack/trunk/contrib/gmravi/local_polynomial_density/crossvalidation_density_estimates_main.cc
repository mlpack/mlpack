#include "fastlib/fastlib_int.h"
#include "cross_validation.h"
#include "naive_kde.h"
#include "cross_validation.h"
#include "naive_kde_local_polynomial.h"
#include "naive_local_likelihood.h"

/* This piece of code takes in different bandwidth values and
   evaluates the cross validation score for these different values of
   h

*/

int main(int argc, char *argv[]){
  
  fx_init(argc,argv);
  //struct datanode * cvm=
  //fx_submodule(NULL,"cvm","cvm");
  
  
  //The reference file name
  const char *ref_file_name=fx_param_str_req(NULL,"data");
  
  
  //The bandwidth file
  
  const char *bandwidth_file_name=fx_param_str_req(NULL,"bandwidth");
  Matrix references;
  Matrix bandwidth;

  //READ ALL DATASETS/////////////////////

  //Load the reference file

   data::Load(ref_file_name,&references);
 
   //Load the bandwidth file
   data::Load(bandwidth_file_name, &bandwidth);

  //Lets do all calculations using a gaussian kernel

   
   printf("Loaded all files\n");
   CrossValidation<GaussianKernel> cross_validation;
   cross_validation.Init(references,bandwidth);
   cross_validation.Compute();

   printf("******************************\n");
   //cross_validation.Compute();
   return 1;
   fx_done();
   
}










