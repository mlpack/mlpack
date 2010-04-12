#include "ocas_smo.h"
#include "ocas.h"
class OCASSMOTest{

public:

  OCASSMO ocas_smo;

  ArrayList <Vector> subgradients_mat;

  ArrayList <double> intercepts_vec;
  
  double lambda_reg_const;

  
  void SetUpVariables(Vector &alpha_vector,Vector &primal_solution){

    int num_subgradients;
    int num_intercepts;
    int num_dims=3;


    num_subgradients=num_intercepts=3;
    
    // The subgradients

    subgradients_mat.Init(num_subgradients);
    intercepts_vec.Init(num_intercepts);
    for(int i=0;i<num_subgradients;i++){
      
      subgradients_mat[i].Init(num_dims);
    }
    
    /*Matrix total_data;
    const char *fp=
      "/net/hu17/gmravi/research/matlab_codes/local_svm/total_data.txt";

    data::Load(fp,&total_data);
    Matrix total_data_trans;
    la::TransposeInit(total_data,&total_data_trans);

    //total_data.PrintDebug();
    
    // Each row of total_data_trans is a subgradient. Except the last 2
    // columns which represents the lambda value and the intercepts

    for(int i=0;i<num_subgradients;i++){

      for(int j=0;j<num_dims;j++){
	
	subgradients_mat[i][j]=total_data_trans.get(i,j); 
      }
      // Add the intercept
      intercepts_vec[i]=total_data_trans.get(i,num_dims);
      }*/
    
    // Finally add lambda
    //lambda_reg_const=total_data_trans.get(0,num_dims+1);
    
    
    lambda_reg_const=0.0001;
    subgradients_mat[0][0]=0.0;
    subgradients_mat[0][1]=0.0;
    subgradients_mat[0][2]=0.0;
    intercepts_vec[0]=0.0;
    

    subgradients_mat[1][0]=-0.064385;
    subgradients_mat[1][1]=0.178969;
    subgradients_mat[1][2]= 0.255574;
    intercepts_vec[1]=0.364661;
    
    subgradients_mat[2][0]=0.035163;
    subgradients_mat[2][1]=-0.003051;
    subgradients_mat[2][2]=-0.054543;
    intercepts_vec[2]=-0.054543;
    

    int subgradients_mat_n_cols=subgradients_mat.size();
    int intercepts_vec_length=intercepts_vec.size();
    
    if(num_subgradients!=subgradients_mat_n_cols|| 
       num_subgradients!=intercepts_vec_length){

      
      printf("There seems to be a mistake in assigning subgradients and intercepts...\n");
      exit(0);
    }

    
    ocas_smo.Init(subgradients_mat,intercepts_vec,lambda_reg_const,
		  alpha_vector,primal_solution);

    /*printf("Initialized OCAS....\n");

    printf("Number of subgradients=%d..\n",
	   ocas_smo.num_subgradients_available_);

    printf("The subgradients are...\n");
    for(index_t i=0;i<ocas_smo.num_subgradients_available_;i++){

      ocas_smo.subgradients_mat_[i].PrintDebug();
      printf("intercept:%f....\n",ocas_smo.intercepts_vec_[i]); 
      }*/
    
    printf("The regularization constant is %f...\n",
	   ocas_smo.lambda_reg_const_);

    /*printf("The alpha vector is ..\n");
  
    printf("I0 indices are ....\n");
    for(index_t i=0;i<ocas_smo.I0_indices_.size();i++){
     
      printf("%d,",ocas_smo.I0_indices_[i]);
    }
    printf("I1 indices are....\n");
    
    for(index_t i=0;i<ocas_smo.I1_indices_.size();i++){
     
      printf("%d,",ocas_smo.I1_indices_[i]);
    }

    printf("There are %d I2 indices ...\n",ocas_smo.I2_indices_.size());

    for(index_t i=0;i<ocas_smo.I2_indices_.size();i++){
      
      printf("%d,",ocas_smo.I2_indices_[i]);
      }*/
  }



  void Test1(){
    
    ocas_smo.SolveOCASSMOProblem_();
  }
};

int main(){

  OCASSMOTest ocas_smo_test;

  Vector alpha_vector;
  alpha_vector.Init(3);
  alpha_vector.SetZero();
  alpha_vector[0]=0.999641;
  alpha_vector[1]=0.000359;
  alpha_vector[2]=0.0;
  
  
  Vector primal_solution;
  primal_solution.Init(3);
  primal_solution.SetZero();
  
  ocas_smo_test.SetUpVariables(alpha_vector,primal_solution);

  ocas_smo_test.Test1();
}
