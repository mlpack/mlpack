#include "hk_convex_ise.h"
#include "fastlib/fastlib_int.h"
#include "engine.h"
int main(int argc, char *argv[]){
  
  //Initalize FastExec
  fx_init(argc,argv,NULL);
  
  struct datanode* ise_module=fx_submodule(NULL,"ise");
  
  //Train data file is a reuqirement
  const char *train_file=fx_param_str_req(ise_module,"train");
  
  Matrix train_data;
  
  //Load the train datasets
  
  data::Load(train_file,&train_data);
  
  //Having got all the initial data lets create an object of the class HkIse
  
  HkConvexIse hk_convex_ise;
  
  //Initialize the object and call functions on it
  
  hk_convex_ise.Init(train_data,ise_module);
  hk_convex_ise.ComputeOptimalKernel();
  //fx_done(NULL);
  //  fx_silence(NULL);
}
