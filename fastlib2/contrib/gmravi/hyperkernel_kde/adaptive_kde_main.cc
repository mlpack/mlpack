#include "fastlib/fastlib_int.h"
#include "adaptive_kde.h"
#include "mlpack/kde/naive_kde.h"

int main(int argc, char *argv[]){
  
  //Initalize FastExec
  fx_init(argc,argv,NULL);
  
  struct datanode* adaptive=fx_submodule(NULL,"adaptive");

  //Create an object of class AdaptiveKDE
  AdaptiveKDE adaptive_kde;
  adaptive_kde.Init(adaptive);
  printf("Initialization performed.....\n");
  adaptive_kde.ComputeDensities();


  Vector test_densities;
  adaptive_kde.get_test_densities(test_densities);


  printf("Got adaptive kde test densities...\n");
  double rmse_adaptive_kde;
  double hellinger_adaptive_kde;
  
  adaptive_kde.get_rmse_and_hellinger_distance(&rmse_adaptive_kde,
  			    &hellinger_adaptive_kde);
  

  printf("Adaptive kde test densities are...\n");
  test_densities.PrintDebug();

  
  FILE *fp=fopen("./mog3_multi/mog3_multi_rect_adaptive.txt","w");
  for(index_t i=0;i<test_densities.length();i++){
     
     fprintf(fp,"%f\n",test_densities[i]);
   }
  
  
  printf("RMSE of adaptive KDE is %f...\n",rmse_adaptive_kde);
  printf("Hellinger distance  of adaptive KDE is %f...\n",hellinger_adaptive_kde);
}
