
/** This piece of code does LSCV for variable bandwidth kde, with the
    bandiwdth of each reference point picked by using NN
    distances. Once the optimal parameters are found then finally the
    optimal parameters are used to calculate the density estimates at
    the different test points
*/

#include "fastlib/fastlib.h"
#include "variable_nn_based_kde.h"
int main(int argc, char *argv[]){

  //Initalize FastExec
  fx_init(argc,argv,NULL);

  printf("Will do variable bw kde...\n");
  
  struct datanode* variable=fx_submodule(NULL,"variable");
 
  VariableNNKDE vnkde;
  vnkde.Init(variable);
  vnkde.PerformLSCV();
  vnkde.ComputeTestDensities();
  double rmse=vnkde.get_rmse();

  //get test densities

  Vector result;
  vnkde.get_test_densities(&result);

  FILE *fp=fopen("variable_suicide_test_unscaled.txt","w");
  
  for(index_t i=0;i<result.length();i++){
    
    fprintf(fp,"%f\n",result[i]);
  }

  printf("RMSE is %f..\n",rmse);
}


