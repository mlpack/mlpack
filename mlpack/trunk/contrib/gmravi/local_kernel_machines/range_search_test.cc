#include "range_search.h"

int main(int argc, char* argv[]){

  fx_module *root = fx_init(argc, argv, NULL);
 
  Matrix q_mat;

  Matrix r_mat;

  double bw=0.35;


  const char *fp1=
    "/net/hu17/gmravi/research/matlab_codes/local_svm/datasets/covtype/covtype_data_train.txt";
  

  const char *fp2=    
    "/net/hu17/gmravi/research/matlab_codes/local_svm/datasets/covtype/covtype_data_test.txt";
  
  data::Load(fp1,&q_mat);  
  data::Load(fp2,&r_mat);  
  
  RangeSearch rs;
  
  rs.Init(r_mat,q_mat,bw);
  rs.PerformRangeSearch();

  fx_done(root);

}
