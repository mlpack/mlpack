#include "fastlib/fastlib_int.h"
#include "nn_kde.h"
#include "mlpack/kde/naive_kde.h"

int main(int argc, char *argv[]){
  
  //Initalize FastExec
  fx_init(argc,argv,NULL);
  
  struct datanode* nn_kde_module=fx_submodule(NULL,"nn_kde");

  //Create an object of class AdaptiveKDE
  NNKDE nn_kde;
  nn_kde.Init(nn_kde_module);
  nn_kde.ComputeTestDensities();


  Vector test_densities;
  nn_kde.get_test_densities(&test_densities);


  double rmse_nn_kde;
  rmse_nn_kde=nn_kde.get_rmse();

  printf("NN  kde test densities are...\n");
  test_densities.PrintDebug();

  FILE *fp=fopen("mog_new/mog_new_nn_kde_multi.txt","w");

  for(index_t i=0;i<test_densities.length();i++){
    if(i!=test_densities.length()-1){
      fprintf(fp,"%f\n",test_densities[i]);
    }
    else{
      fprintf(fp,"%f",test_densities[i]);
    }
  }
  
  printf("RMSE of NN KDE is %f...\n",rmse_nn_kde);
}
