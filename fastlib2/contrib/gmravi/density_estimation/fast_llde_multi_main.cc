#include "fastlib/fastlib.h"
#include "fast_llde_multi.h"
#include "naive_kde.h"
int main(int argc, char *argv[]){
  
  fx_init(argc,argv);
  
  //DEFINE THE SUBMODULE FROM WHERE THE DATA WILL BE PICKED UP
  
  struct datanode *fastlikelihood_multi=
    fx_submodule(NULL,"fastlikelihood_multi","fastlikelihood_multi");
  
  //READ IN ALL PARAMETERS
  
  //READ IN REFERENCE DATASET
  
  const char *ref_file_name=
    fx_param_str_req(fastlikelihood_multi,"ref");
  
  //READ IN QUERY DATASET
  
  const char *query_file_name=
    fx_param_str_req(fastlikelihood_multi,"query"); 

  //Read in the bandwidth
  double bandwidth=fx_param_double_req(fastlikelihood_multi,"bandwidth");
  
  Matrix ref; //to hold the reference dataset
  Matrix query; //to hold the query dataset
  
  //LOAD THE DATASETS INTO MEMORY
  data::Load(ref_file_name,&ref);
  data::Load(query_file_name,&query);

  FastLLDEMulti fast_llde_multi;
  fast_llde_multi.Init(query,ref,bandwidth);
  fast_llde_multi.Compute(); 

  NaiveKde <EpanKernel> naive_kde;
  naive_kde.Init(query,ref,bandwidth);
  naive_kde.Compute();

  // Vector naive_kde_results;
  //naive_kde_results.Init(query.n_cols());
  //naive_kde.get_density_estimates(naive_kde_results);
  fx_done();
}

