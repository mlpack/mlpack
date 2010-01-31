#include "fastlib/fastlib_int.h"
#include "regression2.h"
#define  MAXDOUBLE 32768.0

int main (int argc, char *argv[]){

  fx_init (argc, argv);

  char *rfname=(char*)malloc(40);
  char *qfname=(char*)malloc(40); 
  strcpy(rfname,fx_param_str_req (NULL, "data"));
  strcpy(qfname,fx_param_str_req (NULL,"query"));
  
  Regression2 <GaussianKernel> reg2;
  printf("going to initialization function...\n");
  Dataset ref_dataset ;
  Dataset q_dataset;
  
  Matrix query_dataset;
  Matrix reference_dataset;

  ref_dataset.InitFromFile(rfname);
  reference_dataset.Own(&(ref_dataset.matrix()));
  
  q_dataset.InitFromFile(qfname);
  query_dataset.Own(&(q_dataset.matrix()));

  reg2.Init(query_dataset,reference_dataset);
  printf("Initializations done..\n");

  //  reg2.Compute(fx_param_double (NULL, "tau", 0.1));
  reg2.Compute(0.1);
  ArrayList<Matrix> wfkde_results;
  wfkde_results.Copy(reg2.get_results());

  //NaiveRegression2 <GaussianKernel> naive_reg2;
  // naive_reg2.Init();
  //naive_reg2.Compute();
  //naive_reg2.ComputeMaximumRelativeError(wfkde_results);
  //printf("done..\n");
  fx_done();
}

