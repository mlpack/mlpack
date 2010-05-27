#include "special_la.h"
#include "fastlib/fastlib_int.h"

int main(int argc, char *argv[]){

  fx_init(argc,argv,NULL);
  
  Vector diag_mat;
  
  Matrix lower_triang;

  Matrix usual_mat;

  
  diag_mat.Init(3);
  
  diag_mat[0]=1; diag_mat[1]=1; diag_mat[2]=3;
  
  lower_triang.Init(2,2);
  lower_triang.SetZero();
  
  lower_triang.set(0,0,1);
  lower_triang.set(1,0,2);;
  
  lower_triang.set(1,1,4);

  usual_mat.Init(3,3);

  usual_mat.set(0,0,1);
  usual_mat.set(1,0,2);
  usual_mat.set(2,0,3);

  usual_mat.set(0,1,3);
  usual_mat.set(1,1,4);
  usual_mat.set(2,1,5);

  usual_mat.set(0,2,4);
  usual_mat.set(1,2,5);
  usual_mat.set(2,2,6);



  Matrix upper_triang_mat;
  upper_triang_mat.Init(2,3);

  upper_triang_mat.set(0,0,1);
  upper_triang_mat.set(1,0,0);
  upper_triang_mat.set(0,1,2);
  upper_triang_mat.set(1,1,4);

  upper_triang_mat.set(0,2,3);
  upper_triang_mat.set(1,2,5);




  //printf("Upper triangular matrix is..\n");
  //upper_triang_mat.PrintDebug();


  printf("usual matrix is...\n");
  usual_mat.PrintDebug();

  
  //PostMultiply with lower triangular

  //special_la::PreMultiplyLowerTriangMatrixInit(lower_triang,usual_mat,&res_mat);

  //printf("Will call the function...\n");

  //  printf("The result is...\n");
  // res_mat.PrintDebug();


  //Post Multiply with upper triang matrix

  //special_la::PostMultiplyUpperTriangMatrixInit(usual_mat,upper_triang_mat,
  //					&res_mat);

  Matrix L_mat;
  
  L_mat.Init(3,1);

  L_mat.set(0,0,1);
  L_mat.set(1,0,2);
  L_mat.set(2,0,3);

  ArrayList <index_t> perm;
 

  Vector diag;
  diag.Init(3);
  diag[0]=2;
  diag[1]=2;
  diag[2]=2;

  Vector alpha;

  alpha.Init(3);
  alpha[0]=1;
  alpha[1]=2;
  alpha[2]=3;

  //printf("Everything initialized. Will do matrix inverse vector calculation...\n");
  
  //special_la::MatrixInverseTimesVectorInit(L_mat,diag,alpha,&res);
 

  Matrix chol_factor;
  chol_factor.Init(3,1);
  chol_factor.set(0,0,2);
  chol_factor.set(1,0,3);
  chol_factor.set(2,0,1);

  ArrayList <index_t > perm_mat;
  perm_mat.Init(3);
  perm_mat[0]=2;
  perm_mat[1]=0;
  perm_mat[2]=1;
   
  
  //special_la::PostMultiplyMatrixWithVectorGivenCholeskyInit(chol_factor,perm_mat,alpha,&res);
    
  Vector test;
  test.Init(3);
  test[0]=1;
  test[1]=2;
  test[2]=3;

  test.PrintDebug();

  usual_mat.PrintDebug();

  Matrix res;
  special_la::PreMultiplyMatrixWithPermutationMatrixInit(perm_mat,usual_mat,&res);

  printf("Result of pre multiplication with a permutation matrix is...\n");
  res.PrintDebug();
}
