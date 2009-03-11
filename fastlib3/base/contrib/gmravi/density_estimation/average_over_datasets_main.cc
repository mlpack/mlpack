#include "fastlib/fastlib.h"
//#include "fast_llde_multi.h"
#include "average_over_datasets.h"

int main(int argc, char *argv[]){
  
  fx_init(argc,argv);

  //Read in the query set
  const char *query_file_name=
    fx_param_str_req(NULL,"query"); 
  
  //Read in the bandwidth
  double bandwidth=fx_param_double_req(NULL,"bandwidth");

  //Read the number of reference datasets involved in the averaging
  //process

  index_t num_of_datasets=fx_param_int_req(NULL,"num_of_datasets");

  //Load the query set into memory  
  Matrix query;
  data::Load(query_file_name,&query);

  AverageOverDatasets avod;
  avod.Init(query,bandwidth,num_of_datasets);
  avod.Compute();
  avod.PrintDebug();

}
