#include "fastlib/fastlib.h"
#include "ocas.h"

#ifndef OCAS_SMO_H_
#define OCAS_SMO_H_
#define SMALL pow(10,-6)
class OCASSMO{

 public:
  // The current number of subgradients/intercepts available
  int num_subgradients_available_;

  // The regularization constant

  double lambda_reg_const_;

  // The smoothing kernel bandwidth

  double smoothing_kernel_bandwidth_;
    
  // The alpha_vec that is used in the SMO step of the OCAS algorithm.
  Vector  alpha_vec_;

  // The primal solution

  Vector primal_solution_;
  
  // The subgradients of the non-differentiable part og the objective
  // R(w)=\frac{1}{n}\sum_{i=1}^n max{(1-y-i \langle w,x_i\rangle)K(x_i,x
  // Because one subgradient is added at each point of time, hence we need a 
  // dynamic data structure.
  
  ArrayList<Vector> subgradients_mat_;
  ArrayList <double> intercepts_vec_;
  
  //Cache of F_i values for points in $I_0$
  // F_i=\frac{1}{\lambda} \sum_{j=1}^{t+1} \langle a_i,a_j\rangle \alpha_j- 
  // beta_up and beta_low 
  
  double beta_up_;
  double beta_low_;
  
  int i_up_;
  int i_low_;

  // These variables are the working set variables
  int index_working_set_variables_1_;
  int index_working_set_variables_2_;

  // The F value for the working set variables

  double F_working_set_variables_1_;
  double F_working_set_variables_2_;
  
  // F_i values for elements in $I_0$
  
  ArrayList <double> Fi_for_I0_;
  
  
  // Data structures for I_0, I_1,I_2
  
  // Variables involved in the SMO optimization
  
  
  ArrayList <int> I0_indices_;
  ArrayList <int> I1_indices_;
  ArrayList <int> I2_indices_;

  // Finally the tolerance

  double tau_;

  // Handy variables. They tell you the index of a dual variable in
  // I0,I1,I2. We particularly want these values for the working set
  // variables
  int position_of_i_in_I0_;
  int position_of_i_in_I1_;
  int position_of_i_in_I2_;
  
  int position_of_j_in_I0_;
  int position_of_j_in_I1_;
  int position_of_j_in_I2_;

  // flags to check if one of the sets is destructed.
  
  int flag_I0_indices_;
  int flag_I1_indices_;
  int flag_I2_indices_;


  int num_dims_;

  double eps_;

  void SMOMainRoutine_();
  
  int ExamineSubgradient_(int);
  
  void UpdateBetaUpAndBetaLow_(double, int);

  void UpdateBetaUpAndBetaLowUsingI0_(double,double);
  
  int CheckIfInI0_(int);
  
  int CheckIfInI1_(int);
  
  int CheckIfInI2_(int);
  
  double ComputeFiValue_(int);
  
  int CheckForOptimality_(int,double);

  void ResetPositions_();

  void UpdateSets_(double,int,int,int,int,int);

  void DeleteFromI0_(int,int);

  void DeleteFromI1_(int,int);
  
  void DeleteFromI2_(int,int);
  
  void AddToI0_(int);
  
  void AddToI1_(int);

  void AddToI2_(int);

  void AddToFiForI0_(double);

  void DeleteFromFiForI0_(int);

  void UpdateFiForI0_(double,double);

  void UpdateFiValuesOfWorkingSetVariables_(double,double);

  int TakeStep_();
  
  void Init( ArrayList<Vector>& ,
	     ArrayList <double>&,double, 
	     Vector&,double);
  
  void SolveOCASSMOProblem_();

  void get_primal_solution(Vector &);

  void get_dual_solution(Vector &);
  void InitializeUsingWarmStart_(Vector &);
  void InitializeIndexSets_();
  void InitializeFiForI0_();
  void CalculatePrimalSolution_();
  void PrintSubgradientsAndIntercepts_();

  
}; 
     
#endif    
  

 
