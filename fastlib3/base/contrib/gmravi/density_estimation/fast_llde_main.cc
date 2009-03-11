#include "fastlib/fastlib.h"
#include "fast_llde.h"

int main(int argc, char *argv[]){
  
  fx_init(argc,argv);
  
  //DEFINE THE SUBMODULE FROM WHERE THE DATA WILL BE PICKED UP
  
  struct datanode *fastlikelihood=
    fx_submodule(NULL,"fastlikelihood","fastlikelihood");
  
  //READ IN ALL PARAMETERS
  
  //READ IN REFERENCE DATASET
  
  const char *ref_file_name=
    fx_param_str_req(fastlikelihood,"ref");
  
  //READ IN QUERY DATASET
  
  const char *query_file_name=
    fx_param_str_req(fastlikelihood,"query"); 

  //Read in the bandwidth
  double bandwidth=fx_param_double_req(fastlikelihood,"bandwidth");
  
  Matrix ref; //to hold the reference dataset
  Matrix query; //to hold the query dataset
  
  //LOAD THE DATASETS INTO MEMORY
  data::Load(ref_file_name,&ref);
  data::Load(query_file_name,&query);

  FastLLDE fast_llde;
  fast_llde.Init(query,ref,bandwidth);
  fast_llde.Compute(); 
  fx_done();
}

