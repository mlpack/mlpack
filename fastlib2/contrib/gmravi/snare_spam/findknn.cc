#include "fastlib/fastlib.h"
#include "mlpack/allknn/allknn.h"
#include "findknn.h"
int main(int argc, char *argv[]){
  fx_init(argc,argv,NULL);
  FindAllkNN find_all_knn;
  find_all_knn.Init();
  fx_done(NULL);
}
