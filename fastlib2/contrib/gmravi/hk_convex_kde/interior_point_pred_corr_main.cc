#include "interior_point_pred_corr.h"
#include "special_la.h"
#include "fastlib/fastlib_int.h"
#include "ichol.h"

int main(int argc, char *argv[]){
  
  //Initalize FastExec
  fx_init(argc,argv,NULL);
  
  struct datanode* ipc=fx_submodule(NULL,"ipc");
  
  //Train data file is a reuqirement
  const char *train_file=fx_param_str_req(ipc,"train");
  
  Matrix train_data;
  
  //Load the train datasets
  
  data::Load(train_file,&train_data);

  printf("Train dataset has been read....\n");
  
  //Having got all the initial data lets create an object of the class HkIse
  
  HKInteriorPointPredictorCorrector hk_ippc; 
  hk_ippc.Init(train_data,ipc);
  hk_ippc.ComputeOptimalSolution();
  
  //Initialize the object and call functions on it
  
  return 1;
}
