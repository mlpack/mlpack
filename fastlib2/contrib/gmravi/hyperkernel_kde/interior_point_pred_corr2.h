#ifndef INTERIOR_POINT_PRED_CORR2_H
#define INTERIOR_POINT_PRED_CORR2_H
#include "ichol_dynamic.h"
#include "fastlib/fastlib.h"
#include "dte.h"
#include "special_la.h"
#define CALCULATED 1
#define NOT_CALCULATED 0
//#define UPDATE_FRACTION 0.05
#define EPSILON 0.01
#define NUM_ITERATIONS 15
#define SPARSITY_THRESHOLD 0.1
#define SMALL_STEP pow(10,-7)
#define MAX_NUM_ITERATIONS 150

class HKInteriorPointPredictorCorrector{
  
 private:

  //The essentials

  struct datanode *module_;

  //The train set

  Matrix train_set_;
 
  Matrix test_set_;

  //The train and test classes if they are available

  Matrix train_set_classes_;
  Matrix test_set_classes_;

  //Matrix  true_test_densities_;

  //We shall be using the gaussian hyperkernel and the derived
  //gaussian hyperkernel through our code. These two kernels require
  //two parameters namely sigma and sigma_h

  //This is like the bw of the gaussian hyperkernel
  double sigma_hk_;

  //This measures the interaction between 2 pais of points. If set to
  //infinity then this interaction is considered to be constant
  //irrespective of the distance between the pairs of points

  double sigma_h_hk_;

  //The regularization factor
  
  double lambda_;
  
  //The number of train points
  
  index_t num_train_points_;
  
  //The dimensionality of dataset
  index_t num_dims_;
  
  //Variables involved in the optimization process(both primal and dual)

  //The vector involved in the primal optimization 
  Vector v_vector_;
  
  Vector a_vector_;
  
  //This is the constant involved in the constraint of the primal
  //optimization problem
  
  double primal_constant_;
 
  //The primal and the dual variables
  Vector beta_vector_;
  Vector delta_beta_pred_vector_;

  Vector gamma_vector_;
  Vector delta_gamma_pred_vector_;

  Vector D_vector_; //D<-\frac{\beta_i}{\gamma_i}

  double psi_;
  double delta_psi_pred_;

  //Cholesky factor of the matrix 2M

  Matrix chol_factor_;

  //The permutation matrix stored as an array of integers
  ArrayList<index_t> perm_mat_;

  //The permuted Cholesky factor. This is PL

  Matrix permuted_chol_factor_;


  //Set of variables that are required for the predictor and the
  //corrector steps

  Vector v1_vector_;
  Vector v2_vector_;
  double s1_;

  //This is just the dot product between \beta and \gamma. At
  //optimality this should be 0
  double mu_;

  //The sigma value. This value along with $\mu$ helps calculate the
  //path following constant $\tau=\mu \times \sigma$
  double sigma_pred_corr_;
  
  //Step lengths for primal and dual variable

  double primal_step_length_;
  double dual_step_length_;

  //Keep a tab on the number of iterations

  index_t num_iterations_;

  //We shall use the square of the number of train points at lots of
  //places. hence it is better to evaluate it for once and all

  index_t sqd_num_train_points_;

  //Number of test points
  index_t num_test_points_;

  //Vector of computed_test_densities
  Vector computed_test_densities_;

  //Penalized ISE of train set 

  double penalized_ise_train_set_;

  //Maximum gap ratio that you want before the predictor-corrector iteration stop

  double max_gap_ratio_;

  //Maximum gap

  double max_gap_;

  //The feasibility gap for primal

  double epsilon_primal_;

  //The feasibility gap for dual

  double epsilon_dual_;

  //The surrogate duality gap

  double epsilon_duality_gap_;

  //A boolean flag to know if we are doing classification or not

  bool classification_task_flag_;
  


 public:

  //getter and setter functions
  
  
  double get_penalized_ise_train_set(){
    
    Vector A_trans_beta;
    la::MulInit(beta_vector_,permuted_chol_factor_,&A_trans_beta); //$A^T\beta$
    
    double quad_value;
    
    
    //Note since the cholesky factor that we have right now is the
    //cholesky factorization of 2M, we finally need to divide the
    //quad_value by 2 to get the quadratic associated with \beta^TM\beta
        
    quad_value=la::Dot(A_trans_beta,A_trans_beta)/2; //quad value is \beta^TAA^T\beta    
    double linear_value=la::Dot(beta_vector_,a_vector_);
    penalized_ise_train_set_=quad_value-linear_value;
    //printf("The penalized ise is of the train set is %f...\n",penalized_ise_train_set_);
    return penalized_ise_train_set_;
  }

  double get_ise_train_set(){
    
    
    double ise_train_set;
    ICholDynamic ichol_dynamic;
    Matrix chol_factor_K_prime_matrix;
    ArrayList <index_t> perm_mat_K_prime_matrix;
    
    ichol_dynamic.Init(train_set_,sigma_h_hk_,sigma_hk_,0); //lambda =0
    perm_mat_K_prime_matrix.Init(sqd_num_train_points_);
    ichol_dynamic.Compute(chol_factor_K_prime_matrix,perm_mat_K_prime_matrix);
    
    Matrix permuted_chol_factor_K_prime_matrix;
    special_la::
      PreMultiplyMatrixWithPermutationMatrixInit(perm_mat_K_prime_matrix,
						 chol_factor_K_prime_matrix,
						 &permuted_chol_factor_K_prime_matrix);
    
    Vector A_trans_beta;
    la::MulInit(beta_vector_,permuted_chol_factor_K_prime_matrix,&A_trans_beta); //$A^T\beta$
    
    double quad_value;
    
    quad_value=la::Dot(A_trans_beta,A_trans_beta); //quad value is \beta^TAA^T\beta
    
    //printf("a vector is...\n");
    //a_vector_.PrintDebug();
    
    double linear_value=la::Dot(beta_vector_,a_vector_);
    
    ise_train_set=quad_value-linear_value;
    
    printf("The ise of the train set is %f...\n",ise_train_set);
    //printf("quad_val=%f\n",quad_value);
    //printf("linear value=%f..\n",linear_value);
    return ise_train_set;
  }

  
  double get_negative_log_likelihood_test(){
    
    double likelihood=1;
    for(index_t i=0;i<num_test_points_;i++){
      
      likelihood*=computed_test_densities_[i];      
    }
    
    return -log(likelihood);
  }
  
  
  void get_test_densities(Vector *densities_res){

    ComputeTestDensities_();
    densities_res->Copy(computed_test_densities_);    
  }
  
  HKInteriorPointPredictorCorrector(){
    
    
  }
  
  ~HKInteriorPointPredictorCorrector(){
    
    
  }
  
 private:
  
  void GetLinearPartOfObjective_(){
    
    GaussianHyperKernel ghk;
    ghk.Init(sigma_hk_,sigma_h_hk_,num_dims_);
    double norm_const=ghk.CalcNormConstant();
    
    double constant=2.0/(num_train_points_*(num_train_points_-1));
    //printf("Constant is ....%f\n",constant);
    
    //THIS HAS TO CHANGE WHEN I USE A MULTIPLICATIVE KERComputeNEL
    GaussianKernel gpq; //This is the gaussian kernel between x_p,x_q
    gpq.Init(sigma_hk_*sqrt(2),num_dims_);
    
    //Iterations begin from here...................
    for(index_t p=0;p<num_train_points_;p++){
      
      double *x_p=train_set_.GetColumnPtr(p);
      Vector vec_x_p;
      vec_x_p.Alias(x_p,num_dims_);
	
      for(index_t q=p;q<num_train_points_;q++){
	
	double sum_of_row=0;
	double  *x_q=train_set_.GetColumnPtr(q);
	Vector vec_x_q;
	vec_x_q.Alias(x_q,num_dims_);
	
	double sqd_dist=la::DistanceSqEuclidean(vec_x_p,vec_x_q);
	double gpq_unnorm_val=gpq.EvalUnnormOnSq(sqd_dist);
	
	for(index_t r=0;r<num_train_points_;r++){
	  
	  double *x_r=train_set_.GetColumnPtr(r);
	  Vector vec_x_r;
	  vec_x_r.Alias(x_r,num_dims_);
	  
	  for(index_t s=r+1;s<num_train_points_;s++){
	    
	    double *x_s=train_set_.GetColumnPtr(s);
	    Vector vec_x_s;
	    vec_x_s.Alias(x_s,num_dims_);
	    sum_of_row+=
	      ghk.EvalUnnormPartial1(vec_x_p,vec_x_q,vec_x_r,vec_x_s);
	  }
	}
	
	//With this we have summed up over the row p,q. The sum needs
	//to be doubled up, since we considered only the strict upper
	//triangle of the row, and because the matrix is symmetric,
	//the elements of the lower triangle will be the same, and
	//hence the required sum is twice the original sum. Note
	//diagonal elements are not involved
	
	double total_sum_of_elem_row=2*sum_of_row*gpq_unnorm_val/norm_const;
	
	a_vector_[p*num_train_points_+q]=
	  constant*total_sum_of_elem_row;
	
	
	//Since a_{p,q}=a_{q,p}
	a_vector_[q*num_train_points_+p]=
	  constant*total_sum_of_elem_row;
      }
    } 
  }
  

  //Linear Constraints and the corresponding helper functions
  
  void DualTreeEvaluationOfLinearConstraint_(index_t p,Vector &row_p){
    
    //bw=sqrt(6\sigma^2+4\sigma_h^2)
    
    double bandwidth=sqrt(6*sigma_hk_*sigma_hk_+4*sigma_h_hk_*sigma_h_hk_); 
    double tau=0.004;
    
    //Having fixed p, let q vary from q=p onwards
    
    double *db_x_p=train_set_.GetColumnPtr(p);
    Vector x_p;
    x_p.Alias(db_x_p,num_dims_);

    
    //Extract the qset first q starts from index p(filling the upper
    //traingle, by symmetry lower traingle can be obtained)

    Matrix qset_temp;
    train_set_.MakeColumnSlice(p,num_train_points_-p,&qset_temp); 

    
    //Extract the rset. The rset is simply 2x_i-x_p
    
    Matrix rset_temp;
    la::ScaleInit(2,train_set_,&rset_temp); //temp <-2x_i

    for(index_t i=0;i<rset_temp.n_cols();i++){

      Vector vec;      
      double *col_ptr=rset_temp.GetColumnPtr(i); //gets a column of the vector

      vec.Alias(col_ptr,num_dims_);

      la::SubFrom(x_p,&vec); //temp <- temp-x_p (temp=2*x_i) 
    }

    DTreeEvaluation<GaussianKernel> dte;
    dte.Init(qset_temp,rset_temp,bandwidth,tau);
    dte.Compute();
    dte.GetEstimatesInitialized(p,row_p);
    
    GaussianKernel gk;
    gk.Init(sigma_hk_*sqrt(2));
    double norm_const=gk.CalcNormConstant(num_dims_);

    //Having got the estimates multiply with <x_p,x_q>_{2\sigm,a^2}
    for(index_t q=p;q<num_train_points_;q++){
      
      double *x_q=train_set_.GetColumnPtr(q);
      Vector vec_x_q;
      vec_x_q.Alias(x_q,num_dims_);
      double sqd_dist=la::DistanceSqEuclidean(x_p,vec_x_q); //x_p-x_q
      double unnorm_val=gk.EvalUnnormOnSq(sqd_dist);
      double mult_factor=unnorm_val/norm_const;
      row_p[q-p]*=mult_factor;
    }
  }
  
  void GetLinearConstraint_(){
    
    
    //The linear constraint requires us to evaluate a vector of the
    //form $v_{p,q}=\sum_{i=1}^m
    //e^-\frac{\left(-2x_i+x_p+x_q\right)^2}{12\sigma^2+8\sigma_h^2}
    
    //Fix p and let q vary from q>=p
    
    for(index_t p=0;p<num_train_points_;p++){

      //Lets evaluate the pth row using dual tree computations
      
      Vector row_p;
      row_p.Init(num_train_points_-p);
      DualTreeEvaluationOfLinearConstraint_(p,row_p);  

      //Having got the row_p evaluations store them
      
      for(index_t q=p;q<num_train_points_;q++){
	
	index_t row_num1=p*num_train_points_+q;
	v_vector_[row_num1]=row_p[q-p];
	
	//By symmetry 
	index_t row_num2=q*num_train_points_+p;
	v_vector_[row_num2]=row_p[q-p];
      }
    }
  }

  //////////////////////////THE POSTPROCESSING STAGE/////////////////



  //This is the function that makes use of the beta vector to
  //calculate densities

  void ComputeTestDensities_(){

    //Number miscalssified
    index_t num_misclassified=0;


    //printf("beta vector is...\n");
    //beta_vector_.PrintDebug();
    
    GaussianHyperKernel ghk;
    ghk.Init(sigma_hk_,sigma_h_hk_,num_dims_);
    
    GaussianKernel gk;
    gk.Init(sigma_hk_*sqrt(2));
    
    //$ \hat{f}\left(x\right)=\frac{1}{m}\sum_{i=1}^m \langle
    //x,x_i\rangle \sum_{p=1}^m\sum_{q=1}^m \beta_{p,q}\langle
    //x_p,x_q\rangle_\sigma^2 \langle
    //(x+x_i)/2,(x_p+x_q)/2_{\sigma^2+\sigma_h^2}
    
    //A bunch of normalization constants
    
    double norm_const_hyperkernel=ghk.CalcNormConstant();
    
    printf("Normalization constant is %f...\n",norm_const_hyperkernel);

    //printf("Number of test points are %d..\n",num_test_points_);

    for(index_t test_pt=0;test_pt<num_test_points_;test_pt++){
      
      //Compute density for each test point
      
      //get the test point first
      
      double *x=test_set_.GetColumnPtr(test_pt);

      double total_contrib=0;
      double total_signed_contrib=0;

      for(index_t i=0;i<num_train_points_;i++){
	
	double *x_i=train_set_.GetColumnPtr(i);
	double sqd_dist_x_x_i=la::DistanceSqEuclidean (num_dims_,x,x_i);
	double gaussian_due_to_x_x_i=gk.EvalUnnormOnSq(sqd_dist_x_x_i);
	double contrib_due_to_x_i=0;           
	//Contribution due to x_i

	for(index_t p=0;p<num_train_points_;p++){

	  double *x_p=train_set_.GetColumnPtr(p);
	  
	  for(index_t q=0;q<num_train_points_;q++){
	  
	    index_t counter=p*num_train_points_+q;
	    
	    double *x_q=train_set_.GetColumnPtr(q);
	    
	    double partial_hyperkernel_val=
	      ghk.EvalUnnormPartial1(num_dims_,x,x_i,x_p,x_q);
	    
	    contrib_due_to_x_i+=
	      beta_vector_[counter]*partial_hyperkernel_val;
	  }
	}//////////

	//Contrib due to x_i calculated

	contrib_due_to_x_i*=(gaussian_due_to_x_x_i/norm_const_hyperkernel);

	if(classification_task_flag_==1){

	  double signed_contrib_due_to_x_i=
	    (contrib_due_to_x_i*train_set_classes_.get(i,0));
	  total_signed_contrib+=signed_contrib_due_to_x_i;
	}
	
	total_contrib+=contrib_due_to_x_i;
	//printf("Kernel contribution of i=%d is %f\n",i+1,contrib_due_to_x_i);
      }
      //total contrib calculated.....
      computed_test_densities_[test_pt]=
	total_contrib/num_train_points_;
      
      if(classification_task_flag_==1){
	index_t predicted_class=signum_(total_signed_contrib);
	if(predicted_class!=(int)test_set_classes_.get(test_pt,0)){
	  num_misclassified++;      
	}
      }
    }
   //printf("calculated test densities are...\n");
    //computed_test_densities_.PrintDebug();

    printf("computed TEST DESNTIIES^^^^^^^^^\n");
    if(classification_task_flag_==1){
      printf("NUmber of points misclassified are *****************%d....\n",num_misclassified);
      printf("Percentage misclassified are !!!!!!!!!!!!!!!%f...\n",(double)num_misclassified/num_test_points_);

    }
  }

  int signum_(double a){

    if(a<0)
      return -1;
    else
      return 1;
  }

  //To calculate ISE we need to find the cholesky factor of
  //\underline{K}



  int CheckIfVectorIsCloseToZero_(Vector &a){

    index_t len=a.length();
    for(index_t i=0;i<len;i++){
      if(fabs(a[i])>EPSILON){
	printf("This particular component is %f..\n",fabs(a[i]));
	return -1;
      } 
    }
    return 1;
  }
  
  void FormScalars1PredictorStep_(){

    //s1<-c-\beta^Tv
    //c is the primal constant

    double beta_trans_v=la::Dot(beta_vector_,v_vector_);

    s1_=primal_constant_-beta_trans_v;

    //printf("The scalar s1_ is...%f\n",s1_);
  }

  void FormVectorv2PredictorStep_(){
    
    //For the predictor step v_2=-\beta

    Vector neg_beta_vector;

    la::ScaleInit(-1,beta_vector_,&neg_beta_vector);

    v2_vector_.CopyValues(neg_beta_vector);
  }

  void FormVectorv1PredictorStep_(){

    //v1=a+\gamma-\psi v-2M\beta

    Vector a_plus_gamma;
    la::AddInit(a_vector_,gamma_vector_,&a_plus_gamma);

    //Get psi_v
    Vector psi_v;
    la::ScaleInit(psi_,v_vector_,&psi_v);

    Vector a_plus_gamma_minus_psi_v;
    la::SubInit(psi_v,a_plus_gamma,
		&a_plus_gamma_minus_psi_v);

    Vector two_M_beta;    
    special_la::
      PostMultiplyMatrixWithVectorGivenCholeskyInit(chol_factor_,
						    perm_mat_,
						    beta_vector_,
						    &two_M_beta);
    
    //v1 <-a+\gamma-\psiv-2M\beta
    la::SubOverwrite(two_M_beta,a_plus_gamma_minus_psi_v,
		     &v1_vector_);
  }


  //Do the same thing for corrector steps too

  void FormScalars1CorrectorStep_(){
    
    s1_=0;
    
  }
  
  void FormVectorv1CorrectorStep_(){

    v1_vector_.SetZero();
  }


  void FormVectorv2CorrectorStep_(){
    
    for(index_t i=0;i<sqd_num_train_points_;i++){
      
      double val1=(sigma_pred_corr_*mu_-
		   ((delta_beta_pred_vector_[i]*delta_gamma_pred_vector_[i])));
      
      v2_vector_[i]=val1/gamma_vector_[i];
    }
  }
  
  

  
  //The first step of the primal-dual path following method. This
  //solves a linear system of equations, having neglected the delta
  //terms

  double EvaluateDeltaPsi_(double *denominator_delta_psi,index_t flag){

    double delta_psi;
    
    //Evaluate v1-2Mv2;

    Vector two_M_v2;
    special_la::
      PostMultiplyMatrixWithVectorGivenCholeskyInit(chol_factor_,
						    perm_mat_,
						    v2_vector_,
						    &two_M_v2);

    Vector v1_minus_2M_v2;
    la::SubInit(two_M_v2,v1_vector_,&v1_minus_2M_v2);


    //We need to do (2M+D')^-1 (v1-2MV2). Lets call this results temp1

    //D' it the diagonal matrix with elements
    //\frac{\gamma_i}{\beta_i}. However since SMW update requires only
    //D=D'^-1, we never need to calculate D'

    Vector temp1;
    special_la:: MatrixInverseTimesVectorInit(permuted_chol_factor_,
					      D_vector_,v1_minus_2M_v2,
					      &temp1);
    
    double temp2=la::Dot(v_vector_,temp1);
    
    //Check to see if denominator_delta_psi=0. if it is then recompute it

    double temp4=*denominator_delta_psi;
   
    if(flag!=CALCULATED){

      //This will calculate the actual value for the denominator
      Vector temp3;
      //We also need (2M+D')^-1v. Lets call it temp3;
      
      special_la:: MatrixInverseTimesVectorInit(permuted_chol_factor_,
						D_vector_,v_vector_,
						&temp3);
      temp4=la::Dot(v_vector_,temp3);
      *denominator_delta_psi=temp4;
    }
    
    double v_trans_v2=la::Dot(v_vector_,v2_vector_);

    //printf("Numerator of my calculations are %f..\n",v_trans_v2-s1_+temp2);
    delta_psi=(v_trans_v2-s1_+temp2)/temp4;
    return delta_psi;
  }


  //delta \beta=(2M+D')^-1 D'(v2+D(v1-\delta psi v))
  void EvaluateDeltaBeta_(double delta_psi,Vector &delta_beta_vector){
    
    //    printf("In evaluate delta beta...\n");
    
    Vector delta_psi_v;
    la::ScaleInit(delta_psi,v_vector_,&delta_psi_v);
    
    Vector v1_minus_delta_psi_v;
    la::SubInit(delta_psi_v,v1_vector_,&v1_minus_delta_psi_v); //v1-\delta psi v

    //prod1 <-D(v1-\delta psi v)
    Vector prod1;
    special_la::PreMultiplyVectorWithDiagonalMatrixInit(v1_minus_delta_psi_v,
							D_vector_,&prod1);

   
    Vector sum_of_v2_prod1;
    la::AddInit(prod1,v2_vector_,&sum_of_v2_prod1); 
    //With this we have v_2+D\left(v_1-\delta_psi\right)v


    // Form A^TDA. 

    //First do DA

    Matrix DA;

    special_la::PreMultiplyMatrixWithDiagonalMatrixInit(D_vector_,permuted_chol_factor_,&DA);

    Matrix A_tDA;

    la::MulTransAInit(permuted_chol_factor_,DA,&A_tDA);//We now have A^TDA

    
    //Add identity matrix to A^TDA
    for(index_t i=0;i<A_tDA.n_rows();i++){

      double val=A_tDA.get(i,i);
      A_tDA.set(i,i,val+1);
    }

    //We invert this matrix. Remember this is an in-place inversion 

    index_t flag=
      la::Inverse(&A_tDA);

    if(flag==SUCCESS_PASS){

      //printf("Matrix A_tDA has been successfully inverted..\n");
   
      
      //printf("Inversion done successfully...\n");
      
      //For the other cases we shall print out warning messages
    }
    else{
      if(flag==SUCCESS_WARN){
	
	printf("Warning issued in evaluation of delta_beta");
      }
      else{
	printf("Inversion failed, but will do pseudo inverse");
	special_la::InvertSquareMatrixUsingSVD(A_tDA);
      }
    }

    //Get A^T (v2+D(v1-\delta psi v)) . Remember A =permuted chol factor


    Vector permuted_chol_factor_trans_sum_of_v2_prod1;

    //permuted_chol_factor_trans_sum_of_v2_prod1.Init(permuted_chol_factor_.n_rows());

    //This operation does the transpose and multiplies with a vector

    //Remember we are doing a matrix transpose times vector
    //la::MulInit (const Vector &x, const Matrix &A, Vector *y)	 //y<-A'x

    //printf("About to do matrix transpose times vector..\n");						

    // printf("permuted chol factor is....\n");
    //permuted_chol_factor_.PrintDebug();

    la::MulInit(sum_of_v2_prod1,permuted_chol_factor_,
		&permuted_chol_factor_trans_sum_of_v2_prod1); 

    //printf("Sum of v2_prod1 is..\n");
    //sum_of_v2_prod1.PrintDebug();

    // printf(" permuted Chol factor is..\n");
    //permuted_chol_factor_.PrintDebug();
    

    //printf("Matrix transpose times vector multiplication comes out to..\n");
    //permuted_chol_factor_trans_sum_of_v2_prod1.PrintDebug();

    //printf("permuted_chol_factor_trans_sum_of_v2_prod formed....\n");

    //Small inverse times A^T(v2+D(v1-delta psi v))

    Vector temp1;

    la::MulInit(A_tDA,permuted_chol_factor_trans_sum_of_v2_prod1,&temp1);

    Vector A_temp1;

    la::MulInit(permuted_chol_factor_,temp1,&A_temp1);

    Vector D_A_temp1;

    special_la::PreMultiplyVectorWithDiagonalMatrixInit(D_vector_,A_temp1,
							&D_A_temp1);

     //delta beta is sum_of_v2_prod1-D_A_temp1

    la::SubOverwrite(D_A_temp1,sum_of_v2_prod1,&delta_beta_vector);
  }

  //delta gamma =D'(v2-delta \beta)
  
  //Evaluate delta beta
  
   void EvaluateDeltaGamma_(double delta_psi,Vector &delta_beta_vector,Vector &delta_gamma_vector){


     //\delta gamma= 2M\delta \beta +\delta \psi v-v_1

     Vector two_M_delta_beta;

     special_la::PostMultiplyMatrixWithVectorGivenCholesky(chol_factor_,
							   perm_mat_,
							   delta_beta_vector,
							   &two_M_delta_beta);
	
     //Lets find 2M\delta \beta +\delta psi v

     Vector delta_psi_v;

     la::ScaleInit(delta_psi,v_vector_,&delta_psi_v);

     Vector two_M_delta_beta_plus_delta_psi_v;

     la::AddInit(two_M_delta_beta,
		 delta_psi_v,
		 &two_M_delta_beta_plus_delta_psi_v);

    
     //Finally subtract v1
     la::SubOverwrite(v1_vector_,
		      two_M_delta_beta_plus_delta_psi_v,
		      &delta_gamma_vector);
   }

   void GetTheStartPoint_(){
     //The starting point is basically a set of values for
     //beta_vector_,gamma_vector_,psi_.
     
     /*  Since we are doing infeasible method our choice of
	 beta,\gamma,\psi will be very simple but they will be strictly
	 positive */
     
     beta_vector_.SetAll(1); 
     gamma_vector_.SetAll(1); 
     psi_=1; 
   }

   void GetTheStartPointAdvanced_(){ 

     printf("GOt the start point using advanced tech..\n");

      //The starting point is basically a set of values for
      //beta_vector_,gamma_vector_,psi_.
     
     //Since we are doing infeasible method our choice of 
     // beta,\gamma,\psi will be very simple but they will be 
     //strictly positive. Now ind order to avoid very infeasible 
     // points we shall solve a regularized KKT  
       
     //This is solved by setting v_1=a,v_2=0,s_1=primal_constant_ 
     
     //Temporary Primal Dual Variables
     Vector delta_beta_vector; 
     Vector delta_gamma_vector; 
     double delta_psi; 
     
     delta_beta_vector.Init(sqd_num_train_points_); 
     delta_gamma_vector.Init(sqd_num_train_points_);
     
     
     //Set v1_vector ,v2_vector and s_1 
     
     v1_vector_.CopyValues(a_vector_);
     s1_=primal_constant_; 
     v2_vector_.SetZero(); 
     D_vector_.SetAll(3); 
     
     index_t flag=NOT_CALCULATED;
     
     double denominator_delta_psi=0;//I dont care abt this value as 
                                    //far as obtaining a starting 
 				    //point is concerned 
     
     delta_psi=EvaluateDeltaPsi_(&denominator_delta_psi,flag);
     EvaluateDeltaBeta_(delta_psi,delta_beta_vector); 
     printf("Evaluated delta beta....\n"); 
     
     //delta_gamma evaluation is easy for this case. It is simply 
     //-D^(-1)\delta_beta 
     
     la::ScaleOverwrite (-1.0/3, delta_beta_vector,&delta_gamma_vector); 
     
     //To enforce strict positivity we shall set \beta=\max(\beta,1), \gamma=\max(\gamma,1) 
      
     for(index_t i=0;i<sqd_num_train_points_;i++){ 
       
       beta_vector_[i]=max(delta_beta_vector[i],1.0); 
       gamma_vector_[i]=max(delta_gamma_vector[i],1.0); 
      } 
     psi_=delta_psi; 
     
     /* printf("Found the start vectors by using an advanced method...\n"); */
/*      printf("The initial values are...\n"); */
/*      beta_vector_.PrintDebug(); */
/*      gamma_vector_.PrintDebug(); */
/*      printf("psi is %f..\n",psi_); */
   }
   
   //The D vector is simply a vector of ratio of beta to gamma components
   
   void GetDVector_(){
     
     double epsilon;
     epsilon=10^-4;
     
     //D_vector is \frac{\beta_i}{\gamma_i}
     
     index_t len=D_vector_.length();
     
     for(index_t i=0;i<len;i++){
       
       double val=
	 beta_vector_[i]/gamma_vector_[i];
       D_vector_[i]=val; 
     } 
   }
   
   void TakePredictorStep_(double *denominator_delta_psi,index_t flag){


     //Temporary Primal Dual Variables
     Vector delta_beta_vector;
     Vector delta_gamma_vector;
     double delta_psi;

     delta_beta_vector.Init(sqd_num_train_points_);
     delta_gamma_vector.Init(sqd_num_train_points_);
     
     delta_psi=EvaluateDeltaPsi_(denominator_delta_psi,flag);
     
     //printf("Evaluated delta psi..\n");
     EvaluateDeltaBeta_(delta_psi,delta_beta_vector);
     printf("Evaluated delta beta....\n");
     
     //printf("Evlauated delta beta...\n");
     EvaluateDeltaGamma_(delta_psi,delta_beta_vector,delta_gamma_vector);
     printf("Evaluated delta gamma..\n");
     
     //This routine could have been called while perfomring the
     //predictor step or corrector step.

     if(flag==NOT_CALCULATED){
       
       //This means this step has been called from the predictor step
       delta_beta_pred_vector_.CopyValues(delta_beta_vector);
       delta_gamma_pred_vector_.CopyValues(delta_gamma_vector);
       delta_psi_pred_=delta_psi;
   
     }
     else{
       //this step was called while performing the corrector
       //step. Since the predictor step values explicitlyu are not
       //required anymore, lets resue these vectors

       la::AddTo(delta_beta_vector,&delta_beta_pred_vector_);
       la::AddTo(delta_gamma_vector,&delta_gamma_pred_vector_);
       delta_psi_pred_+=delta_psi;
       
     }
   }
   
   void TakeCorrectorStep_(double *denominator_delta_psi,index_t flag){
     
     
     //Set vectors v1,v2, and scalar s1
      
     FormVectorv1CorrectorStep_();
     FormVectorv2CorrectorStep_();
     FormScalars1CorrectorStep_();
     

     printf("Formed vectors v1,v2,s1 for corrector step..\n");
     //Resolve the equations. For this we shall recall the Predictor
     //step with $v_1,v_2,s_1$ changed
     
     printf("Calling predictor step as a part of corrector step...\n ");
     TakePredictorStep_(denominator_delta_psi,flag);
     printf("Done with corrector step...\n");
   }
   
   //This step finds the appropriate Step size to update the primal 
   //and dual variables
   
   void CalculatePrimalAndDualStepSizes_(){

     //first lets calculate the primal step length
     
     index_t len=beta_vector_.length();
    
     primal_step_length_=dual_step_length_=1.0;

     for(index_t i=0;i<len;i++){
       
       //Positive values for \delta\beta and \delta\gamma admit any
       //amount of step length. Hence it is enough to iterate over
       //negative values for \delta\beta and \delta\gamma

       if(delta_beta_pred_vector_[i]<0){

	 double temp=min(-beta_vector_[i]/delta_beta_pred_vector_[i],1.0);
	 if(temp<primal_step_length_){
	   primal_step_length_=temp;
	 }
       }
       else{
	 //No calculation is required

       }

       //Do a similar thing for \gamma
       if(delta_gamma_pred_vector_[i]<0){
	 
	 double temp=min(-gamma_vector_[i]/delta_gamma_pred_vector_[i],1.0);
	 if(temp<dual_step_length_){

	   dual_step_length_=temp;
	 }
       }
       else{
	 //No calculation is required
	 
       }
     }
   }

  void UpdatePrimalAndDualVariables_(){

    la::Scale(primal_step_length_,&delta_beta_pred_vector_);
    
    //\beta <- \beta+delta_beta
    
    la::AddTo(delta_beta_pred_vector_,&beta_vector_);

    la::Scale(dual_step_length_,&delta_gamma_pred_vector_);

    //\gamma <-\gamma+delta \gamma

    la::AddTo(delta_gamma_pred_vector_,&gamma_vector_);

    psi_=psi_+dual_step_length_*delta_psi_pred_;
  }

  void CalculateMu_(){

    
    //mu is simply the dot product between $\beta$ and $\gamma$
    mu_=la::Dot(beta_vector_,gamma_vector_);
    mu_=mu_/sqd_num_train_points_;
  }

  void CalculateSigma_(){

    double mu_aff;
    
    //beta_copy=beta_vector_+(primal_step_length*delta_beta_vector_)
    Vector beta_copy;
    beta_copy.Copy(beta_vector_);
    la::AddExpert(primal_step_length_,delta_beta_pred_vector_, &beta_copy);
    
    //Do the same thing with the $\gamma$ vector too
  
    Vector gamma_copy;
    gamma_copy.Copy(gamma_vector_);
    la::AddExpert(dual_step_length_,delta_gamma_pred_vector_, &gamma_copy);
    
    mu_aff=(la::Dot(gamma_copy,beta_copy))/sqd_num_train_points_;

    double ratio=mu_aff/mu_;
    sigma_pred_corr_=ratio*ratio*ratio;
  }

  
  //This function simply flushes the delta_beta,delta_gamm and
  //delta_psi values to 0

  void FlushUpdates_(){
    
    delta_beta_pred_vector_.SetZero();
    delta_gamma_pred_vector_.SetZero();
    delta_psi_pred_=0;
  }

  int CalculateGapNew_(double *gap,double *gap_ratio){


    //First check for primal feasibility

    //This can be done by calculating the value of \\beta^Tv-c
    
    double beta_trans_v;
    beta_trans_v=la::Dot(beta_vector_,v_vector_);
    
    double length_beta=la::LengthEuclidean(beta_vector_);
    double length_gamma=la::LengthEuclidean(gamma_vector_);

    double primal_feasibility_gap=beta_trans_v-primal_constant_;
    if(primal_feasibility_gap>epsilon_primal_*(1+length_beta)){

      //Primal feasibnility is violated

      printf("epsilon_primal =%f,primal feasibility gap is %f..\n",epsilon_primal_,primal_feasibility_gap);
      return -1;

    }

    //Check for dual feasibility

    // Calculate the value of $a+\gamma-\psi v-2M\beta$
    
    Vector a_plus_gamma;
    la::AddInit(a_vector_,gamma_vector_,&a_plus_gamma);
    
    //Get psi_v
    Vector psi_v;
    la::ScaleInit(psi_,v_vector_,&psi_v);

    Vector a_plus_gamma_minus_psi_v;
    la::SubInit(psi_v,a_plus_gamma,
		&a_plus_gamma_minus_psi_v);

    Vector two_M_beta;    
    special_la::
      PostMultiplyMatrixWithVectorGivenCholeskyInit(chol_factor_,
						    perm_mat_,
						    beta_vector_,
						    &two_M_beta);
    
    Vector temp;
    //temp <-a+\gamma-\psiv-2M\beta
    la::SubInit(two_M_beta,a_plus_gamma_minus_psi_v,
		&temp);

    double dual_feasibility_gap=
      la::LengthEuclidean(temp);

    if(dual_feasibility_gap>epsilon_dual_*(1+length_gamma)){

      //Dual feasibility is violated
   
      return -1;
    }

    //Calculate the surrogate duality gap as showed in boyd
    //This requires the calculation of \beta^T\gamma

    double surrogate_duality_gap=
      la::Dot(beta_vector_,gamma_vector_);

    double remaining_expr=psi_*(primal_feasibility_gap); //\psi(\beta^Tv-c)

    if(fabs(surrogate_duality_gap-remaining_expr)>epsilon_duality_gap_){
          
      return -1;
    }

    //Erverything looks all rite
    
    printf("primal feasibility gap is %f..\n",primal_feasibility_gap);
    printf("dual feasibility gap is %f..\n",dual_feasibility_gap);
    printf("surrogate dual gap is %f..\n",surrogate_duality_gap);
    return 1;
}


  int CalculateGap_(double *gap,double *gap_ratio){


    //First check for primal feasibility

    //This can be done by calculating the value of \\beta^Tv-c
    
    double beta_trans_v;
    beta_trans_v=la::Dot(beta_vector_,v_vector_);
    
    double primal_feasibility_gap=beta_trans_v-primal_constant_;
    if(primal_feasibility_gap>epsilon_primal_){

      //Primal feasibnility is violated

      printf("epsilon_primal =%f,primal feasibility gap is %f..\n",epsilon_primal_,primal_feasibility_gap);
      return -1;

    }

    //Check for dual feasibility

    // Calculate the value of $a+\gamma-\psi v-2M\beta$
    
    Vector a_plus_gamma;
    la::AddInit(a_vector_,gamma_vector_,&a_plus_gamma);
    
    //Get psi_v
    Vector psi_v;
    la::ScaleInit(psi_,v_vector_,&psi_v);

    Vector a_plus_gamma_minus_psi_v;
    la::SubInit(psi_v,a_plus_gamma,
		&a_plus_gamma_minus_psi_v);

    Vector two_M_beta;    
    special_la::
      PostMultiplyMatrixWithVectorGivenCholeskyInit(chol_factor_,
						    perm_mat_,
						    beta_vector_,
						    &two_M_beta);
    
    Vector temp;
    //temp <-a+\gamma-\psiv-2M\beta
    la::SubInit(two_M_beta,a_plus_gamma_minus_psi_v,
		&temp);

    double dual_feasibility_gap=
      la::LengthEuclidean(temp);

    if(dual_feasibility_gap>epsilon_dual_){

      //Dual feasibility is violated
      printf("epsilon_dual=%f,dual feasibility gap is %f..\n",epsilon_dual_,dual_feasibility_gap);
      return -1;
    }

    //Calculate the surrogate duality gap as showed in boyd
    //This requires the calculation of \beta^T\gamma

    double surrogate_duality_gap=
      la::Dot(beta_vector_,gamma_vector_);

    if(surrogate_duality_gap>epsilon_duality_gap_){
      printf("surrogate dual gap is %f..\n",surrogate_duality_gap);
      
      return -1;
    }

    //Erverything looks all rite
    
    printf("primal feasibility gap is %f..\n",primal_feasibility_gap);
    printf("dual feasibility gap is %f..\n",dual_feasibility_gap);
    printf("surrogate dual gap is %f..\n",surrogate_duality_gap);
    return 1;


    //*************Old piece of code***********************/

    //gap =-\beta^T \gamma+\psi(\beta^Tv-c);

   /*  double beta_trans_gamma; */

/*     beta_trans_gamma=la::Dot(beta_vector_,gamma_vector_); */

/*     double beta_trans_v; */
/*     beta_trans_v=la::Dot(beta_vector_,v_vector_); */

/*     *gap=-1*(beta_trans_gamma-psi_*(beta_trans_v-primal_constant_)); */

    
/*     Vector A_trans_beta; */
/*     la::MulInit(beta_vector_,permuted_chol_factor_,&A_trans_beta); //$A^T\beta$ */

/*     //printf("A_trans beta is...\n"); */
/*     //A_trans_beta.PrintDebug(); */


/*     double quad_value; */

/*     quad_value=la::Dot(A_trans_beta,A_trans_beta); //quad value is \beta^TAA^T\beta */

/*     //Since A is the permuted cholesky factor of 2M and not M, the */
/*     //quad _value obtained above is twice the actual value */

/*     quad_value/=2; */

/*     double beta_trans_a; */

/*     beta_trans_a=la::Dot(beta_vector_,a_vector_); */

/*     double function_val=quad_value-beta_trans_a; */

/*     double denominator=fabs(function_val+(*gap)); */

/*     *gap_ratio=*gap/denominator; */
    
/*     //printf("Quad value is %f..\n",quad_value); */
    
/*     //printf("Linear value is %f..\n",beta_trans_a); */
    
/*     printf("Gap is %f..\n",*gap); */

/*     printf("Ratio is %f..\n",*gap/denominator); */

    //printf("Beta gamma product is %f..\n",beta_trans_gamma);

    //printf("The objective function value is %f..\n",function_val);
    //printf("..................................\n");

  }

  int CheckIfPositive_(){

    index_t beta_counter=0;
    index_t gamma_counter=0;
    for(index_t i=0;i<sqd_num_train_points_;i++){
      
      if(beta_vector_[i]<0){
	
	beta_counter++;
	
	printf("This beta vector beciame negative %f...\n",beta_vector_[i]);
      }
      
      if(gamma_vector_[i]<0){
	
	gamma_counter++;	
	printf("This gamma vector became negative %f...\n",gamma_vector_[i]);
      }
    }
    if(beta_counter>=1 ||gamma_counter>=1){
      printf("Optimization wasnt successful because solution is not positive....\n");
      return -1;
    }
    printf("beta and gamma vectors are positive...\n");
    return 1;
  }

  int CheckIfPrimalFeasible_(){
    //Check if \beta^Tv=c
    double dot_prod=la::Dot(beta_vector_,v_vector_);
    if(fabs(dot_prod-primal_constant_)<EPSILON){

      //succesful
      return 1;
    }
    //not so succesful
    printf("The gap in primal feasibility is %f..\n",fabs(dot_prod-primal_constant_));
    return -1;
   
  }

  int CheckForOptimality_(){
    
    //For optimality it is enough to check if $a+\gamma-\psi
    //v-2M\beta=0$.  This is equivalent to checking if v1=0. Hence we
    //shall calculate v1 and see how close it is to 0.

    FormVectorv1PredictorStep_();
   
    //It is enought to check if this vector is close to 0

    return CheckIfVectorIsCloseToZero_(v1_vector_);
  }


  int CheckIfKKTIsSatisfied_(){


   
    int flag1=CheckIfPositive_();
    if(flag1!=1){
      printf("Otimization was not succesful becasuse the final solution is not positive");
      return -1;
    }
   

    // ALL FURTHER CHECKS ARE NOT REQUIRED SINCE THEY HAVE ALREADY BEEN CHECKED

   /*  int flag2=CheckIfPrimalFeasible_(); */

/*     if(flag2!=1){ */
/*       printf("Otimization was not succesful becasuse the final solution is not primal feasible"); */
/*       return -1; */
/*     } */

/*     //CheckDualityGap_(); */
/*     int flag3=CheckForOptimality_(); */
/*     if(flag3!=1){ */

/*        printf("Otimization was not succesful becasuse the final solution is not optimal"); */
/*       return -1; */
/*     } */

    //Everything looks fine */
     return 1;
  }

  index_t PredictorCorrectorSteps_(){

    
    //Get the starting values for \beta, \gamma and \psi
    
    GetTheStartPoint_();
    
    printf("Got the start point..\n");
    index_t flag;
    
    double gap;
    double gap_ratio=DBL_MAX;
    index_t status=-1;
    while(status==-1&&num_iterations_<MAX_NUM_ITERATIONS){ 

      //With the current values for $\beta,\gamma$ calculate $\mu

      CalculateMu_();
           
      //We compute vectors v1,v2, s1, before taking the predictor
      //corrector steps.We also need the D matrix which is a diagonal matrix

      GetDVector_();      
      FormVectorv1PredictorStep_();
      FormVectorv2PredictorStep_();
      FormScalars1PredictorStep_();

      // printf("Got vectors v1,v2,s1 for predictor step...\n");

     
      // This forces the program to calculate the quantity
      // denominator_delta_psi
      
      double denominator_delta_psi=0;
      
      flag=NOT_CALCULATED; 
     
      //Calculate $\deta\beta,\delta\gamma,\delta\psi$ 
      TakePredictorStep_(&denominator_delta_psi,flag);

      // printf("Finished predictor step...\n");
      
      //Having taken the predictor step, lets measure the primal and
      //the dual step length's that can be taken

      CalculatePrimalAndDualStepSizes_();

      // printf("Found primal dual step sizes for predictor step to be %f,%f...\n",primal_step_length_,dual_step_length_);

      //Having calculated step sizes now calculate the value of
      //$\sigma$.$\sigma$ is simply the ratio of $\mu_aff$ to $\mu$.

      CalculateSigma_();

      //We are finished with the predictor step. Now we need to take
      //the corrector step.

      flag=CALCULATED;

      //   printf("Will take corrector steps...\n");
      
      TakeCorrectorStep_(&denominator_delta_psi,flag);      
     
      CalculatePrimalAndDualStepSizes_(); 

      //  printf("Found primal dual step sizes for corrector step to be %f,%f...\n",primal_step_length_,dual_step_length_);

      //Lets take much conservative steps.
      
      primal_step_length_=min(0.95*primal_step_length_,1.0);
      dual_step_length_=min(0.95*dual_step_length_,1.0);

      // printf("Final primal and dual step lengths are %f,%f..\n",primal_step_length_,dual_step_length_);
     
      //Having got the step size update the primal and the dual
      //variables

      UpdatePrimalAndDualVariables_();
      status=CalculateGapNew_(&gap,&gap_ratio);
      FlushUpdates_();
      
     
      
      printf("Finished ITERATION=%d...\n\n",num_iterations_);
      
      num_iterations_++;
      
    }

    //NOT REQUIRED SINCE THEY HAVE ALREADY BEEN DONE
   /*  if(fabs(gap_ratio)>max_gap_ratio_){ */
      
/*       printf("Ended with gap larger than max gap. Hence returning ERROR..\n"); */
/*       return -1; */
    
/*     } */
    printf("Number of iterations made are %d..\n",num_iterations_);
   
    return CheckIfKKTIsSatisfied_(); 
    
  }
  
  void GetLinearPartOfObjectiveAndLinearConstraintVectors_(){
    
    
    //We shall call the vector involved in the linear part of
    //objective as a_vector_
    
    GetLinearPartOfObjective_();

    // printf("Got linear part of the objective.....\n");
    
    //We shall call the vector involved in the linear part of
    //constraint as the v_vector_
    
    GetLinearConstraint_();
    
    /*  printf("V vector is...\n"); */
/*     v_vector_.PrintDebug(); */
    
    
/*     printf("a vector is..\n"); */
/*     a_vector_.PrintDebug(); */
  }
  
  
  void GetCholeskyFactorizationOfMMatrix_(){

    ICholDynamic ichol_dynamic;
    ichol_dynamic.Init(train_set_,sigma_h_hk_,sigma_hk_,lambda_);
    ichol_dynamic.Compute(chol_factor_,perm_mat_);  
    
    //We need the cholesky factorization of 2M everywhere.  hence we
    //shall scale the cholesky factor of M with sqrt(2).  It is
    //imperative to do the scaling now itself, so that the scaling
    //gets reflected in the permuted cholesky factor also
    

    la::Scale(sqrt(2),&chol_factor_);
    //chol_factor_.PrintDebug();
    
    //Also we use the permuted chol factor which is A=PL
    
    special_la::
      PreMultiplyMatrixWithPermutationMatrixInit(perm_mat_,
						 chol_factor_,
						 &permuted_chol_factor_);  

  }
  

 public:
  
  //This function drives the whole algorithm
  index_t ComputeOptimalSolution(){
    
    
    //The algorithm solves a QP which involves a linar term in the
    //objective and a linear constraint. Lets first get both these vectors

    //printf("Initial train set is...\n");
    //train_set_.PrintDebug();
    
    fx_timer_start(NULL,"preprocess");
    GetLinearPartOfObjectiveAndLinearConstraintVectors_();

    //Get the Cholesky factiorization of the matrix M
    
    GetCholeskyFactorizationOfMMatrix_();
     
    fx_timer_stop(NULL,"preprocess");
    printf("Preprocessing steps are all done..\n");

    //having obtained the vectors the algorithm works by taking a
    //predictor step and followed by a corrector step
    
    fx_timer_start(NULL,"opt");
    index_t success=PredictorCorrectorSteps_();
    fx_timer_stop(NULL,"opt");

    if(success==-1){

      return -1;
    }

    printf("Postprocessing step over...\n");
    
    return 1;
  }

  //We shall initialize all variables and set up for the primal dual
  //iteration

  void Init(Matrix &train_set,Matrix &test_set,struct datanode *module_in){
    
    //set the module to the incoming module
    module_=module_in;
    train_set_.Alias(train_set);

    num_train_points_=train_set_.n_cols();    
    
    test_set_.Alias(test_set);
    num_test_points_=test_set_.n_cols();

    //Lastly if the train and test classes are available read that up

    
    if(fx_param_int(module_in,"classification",0)==1){

      printf("WE ARE DOING CLASSIFICATion..\n");

      //Classification needs to be done on this dataset
      //Read both the train and test classes
      
      const char *train_classes_file=
	fx_param_str_req(module_,"train_class");

      const char *test_classes_file=
	fx_param_str_req(module_,"true_test_class");
      data::Load(train_classes_file,&train_set_classes_);
      data::Load(test_classes_file,&test_set_classes_);
      classification_task_flag_=1;
    }
    else{
      //Dummy initialization to avoid segfault
      train_set_classes_.Init(1,1);
      test_set_classes_.Init(1,1);
      classification_task_flag_=0;
    }
    
    //    if(fx_param_exists(module_,"true")){

    //const char *true_density_file=fx_param_str_req(module_,"true");
    //data::Load(true_density_file,&true_test_densities);
    //}
   
    //This ends the necessaary file reading routines. Lets read the
    //parameters of the gaussian kernel and the regularization
    //constant \lambda from the user
    
    sigma_hk_=fx_param_double_req(module_,"sigma");
    sigma_h_hk_=fx_param_double_req(module_,"sigma_h");
    lambda_=fx_param_double_req(module_,"lambda");


    printf("In the init function we have...\n");
    printf("sigma_h_hk=%f,sigma_hk_=%f,lambda=%f..\n",sigma_h_hk_,sigma_hk_,lambda_);
    max_gap_ratio_=fx_param_double(module_,"max_gap_ratio",0.0001);
    
    
    max_gap_=fx_param_double(module_,"max_gap",0.0001);
    


    // The feasibility gap for primal and dual
    epsilon_primal_=fx_param_double(module_,"epsilon_primal",0.001);
    epsilon_dual_=fx_param_double(module_,"epsilon_dual",0.001);
    
    //The surrogate duality gap
    
    epsilon_duality_gap_=fx_param_double(module_,"epsilon_duality_gap",0.001);
    
    
    //The dimensionality of the dataset
    num_dims_=train_set_.n_rows();

    printf("Number of dimensions are %d..\n",num_dims_);

    sqd_num_train_points_=num_train_points_*num_train_points_;

  
    //Initialize vectors that will be used

    v1_vector_.Init(sqd_num_train_points_);
    v2_vector_.Init(sqd_num_train_points_);
    v_vector_.Init(sqd_num_train_points_);
    a_vector_.Init(sqd_num_train_points_);
    beta_vector_.Init(sqd_num_train_points_);
    gamma_vector_.Init(sqd_num_train_points_);
    delta_beta_pred_vector_.Init(sqd_num_train_points_);
    delta_gamma_pred_vector_.Init(sqd_num_train_points_);

    printf("Number of test points are  %d...\n",num_test_points_);
    computed_test_densities_.Init(num_test_points_);
    
    //The permutation matrix. This will be used along with the chol
    //factor in calculations

    perm_mat_.Init(sqd_num_train_points_);
    D_vector_.Init(sqd_num_train_points_);

    //chol factor will be initialized in the routine GetCholFactor

    //Set delta_beta, delta_gamma and delta_psi to 0
    delta_beta_pred_vector_.SetZero();
    delta_gamma_pred_vector_.SetZero();
    delta_psi_pred_=0;


    double sigma_4=sigma_hk_*sigma_hk_*sigma_hk_*sigma_hk_;
    double sigma_2=sigma_hk_*sigma_hk_;
    double sigma_h_2=sigma_h_hk_*sigma_h_hk_;
    double sigma_h_4=sigma_h_hk_*sigma_h_hk_*sigma_h_hk_*sigma_h_hk_;


    //The primal constant
    double three_simga_sqd_plus_two_sigma_h_sqd;

    three_simga_sqd_plus_two_sigma_h_sqd=
      3*sigma_2+2*sigma_h_2;

    primal_constant_=num_train_points_*
      pow((sqrt(math::PI)*sqrt(sigma_2+sigma_h_2)*(3*sigma_2+2*sigma_h_2)/
	   sqrt(3*sigma_4+5*sigma_2*sigma_h_2+2*sigma_h_4)),num_dims_);
    printf("Primal constant is %f...\n",primal_constant_);
    num_iterations_=0;

  }
};
#endif

