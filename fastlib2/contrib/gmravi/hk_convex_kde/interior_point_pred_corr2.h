#ifndef INTERIOR_POINT_PRED_CORR2_H
#define INTERIOR_POINT_PRED_CORR2_H
#include "ichol_dynamic.h"
#include "fastlib/fastlib.h"
#include "dte.h"
#include "special_la.h"
#define CALCULATED 1
#define NOT_CALCULATED 0
#define UPDATE_FRACTION 0.05
#define EPSILON 0.0001
#define NUM_ITERATIONS 15
#define SPARSITY_THRESHOLD 0.1
#define SMALL_STEP pow(10,-7)
#define MAX_NUM_ITERATIONS 50
#include <map>

class HKInteriorPointPredictorCorrector{
  
 private:

  //The essentials

  struct datanode *module_;

  //The train set

  Matrix train_set_;

  Matrix test_set_;

  //Matrix  true_test_densities_;

  //We shall be using the gaussian hyperkernel and the derived
  //gaussian hyperkernel through our code. These two kernels require
  //two parameters namely sigma and sigma_h

  //This is like the bw of the gaussian hyperkernel
  double sigma_;

  //This measures the interaction between 2 pais of points. If set to
  //infinity then this interaction is considered to be constant
  //irrespective of the distance between the pairs of points

  double sigma_h_;

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
  Vector delta_beta_vector_;

  Vector gamma_vector_;
  Vector delta_gamma_vector_;

  Vector D_vector_; //D<-\frac{\beta_i}{\gamma_i}

  double psi_;
  double delta_psi_;

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


  //This is the path following constant. This will converge to 0
  double mu_;

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
    printf("The penalized ise is of the train set is %f...\n",penalized_ise_train_set_);
    return penalized_ise_train_set_;
    }
    double get_ise_train_set(){
    
      
      double ise_train_set;
      ICholDynamic ichol_dynamic;
      Matrix chol_factor_K_prime_matrix;
      ArrayList <index_t> perm_mat_K_prime_matrix;
      
      ichol_dynamic.Init(train_set_,sigma_h_,sigma_,0); //lambda =0
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
      
      printf("The ise is of the train set is %f...\n",ise_train_set);
      printf("quad_val=%f\n",quad_value);
      printf("linear value=%f..\n",linear_value);
      return ise_train_set;
    }


    double get_negative_log_likelihood_test(){
      
      double likelihood=1;
      for(index_t i=0;i<num_test_points_;i++){
	
	likelihood*=computed_test_densities_[i];      
      }
      
      return -log(likelihood);
    }
    
    
    void get_test_densities(Vector &densities_res){
      
      densities_res.Alias(computed_test_densities_);
    }
    
    HKInteriorPointPredictorCorrector(){
      
      
    }
    
    ~HKInteriorPointPredictorCorrector(){
      
      
    }
    
 private:
    
  
    void GetLinearPartOfObjective_(){
      
      GaussianHyperKernel ghk;
      ghk.Init(sigma_,sigma_h_,num_dims_);
      double norm_const=ghk.CalcNormConstant();
      
      double constant=2.0/(num_train_points_*(num_train_points_-1));
      printf("Constant is ....%f\n",constant);
      
      //THIS HAS TO CHANGE WHEN I USE A MULTIPLICATIVE KERComputeNEL
      GaussianKernel gpq; //This is the gaussian kernel between x_p,x_q
      gpq.Init(sigma_*sqrt(2),num_dims_);
      
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
    
    double bandwidth=sqrt(6*sigma_*sigma_+4*sigma_h_*sigma_h_); 
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
    gk.Init(sigma_*sqrt(2));
    double norm_const=gk.CalcNormConstant(num_dims_);

    //Having go the estimates multiply with <x_p,x_q>_{2\sigm,a^2}
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


    //printf("beta vector is...\n");
    //beta_vector_.PrintDebug();

    printf("Before calculating densities..\n");
    printf("Train set is..\n");
    train_set_.PrintDebug();

    printf("Test set is...\n");
    test_set_.PrintDebug();

    GaussianHyperKernel ghk;
    ghk.Init(sigma_,sigma_h_,num_dims_);
    
    GaussianKernel gk;
    gk.Init(sigma_*sqrt(2));
    
    //$ \hat{f}\left(x\right)=\frac{1}{m}\sum_{i=1}^m \langle
    //x,x_i\rangle \sum_{p=1}^m\sum_{q=1}^m \beta_{p,q}\langle
    //x_p,x_q\rangle_\sigma^2 \langle
    //(x+x_i)/2,(x_p+x_q)/2_{\sigma^2+\sigma_h^2}
    
    //A bunch of normalization constants
    
    double norm_const_hyperkernel=ghk.CalcNormConstant();
    
    //printf("Normalization constant is %f...\n",norm_const_hyperkernel);

    //printf("Number of test points are %d..\n",num_test_points_);

    for(index_t test_pt=0;test_pt<num_test_points_;test_pt++){
      
      //Compute density for each test point
      
      //get the test point first
      
      double *x=test_set_.GetColumnPtr(test_pt);

      //printf("Test point is %d\n",test_pt+1);

      double total_contrib=0;

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
	
	total_contrib+=contrib_due_to_x_i;
	//printf("Kernel contribution of i=%d is %f\n",i+1,contrib_due_to_x_i);
      }
      //total contrib calculated.....
      computed_test_densities_[test_pt]=
	total_contrib/num_train_points_;
    }
    printf("calculated test densities are...\n");
    computed_test_densities_.PrintDebug();
  }

  //To calculate ISE we need to find the cholesky factor of
  //\underline{K}
  
  void FormScalars1_(){

    //s1<-c-\beta^Tv
    //c is the primal constant

    double beta_trans_v=la::Dot(beta_vector_,v_vector_);

    s1_=primal_constant_-beta_trans_v;

    //printf("The scalar s1_ is...%f\n",s1_);
  }

  void FormVectorv2_(){

    double product=0.0;
    double val=1.0;

    for(index_t i=0;i<sqd_num_train_points_;i++){

      //the product of the delta terms.  This should be 0 for the
      //predictor step. However it becomes non-zero for the corrector
      //step
     
      val=gamma_vector_[i];

      product=
	delta_beta_vector_[i]*delta_gamma_vector_[i];

      double numerator=mu_-(beta_vector_[i]*gamma_vector_[i])-product;
	
      v2_vector_[i]=numerator/val;
    }

    /*printf("Formed vector v2 to get..\n");
      v2_vector_.PrintDebug();*/
  }

  void FormVectorv1_(){

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
  
  //The first step of the primal-dual path following method. This
  //solves a linear system of equations, having neglected the delta
  //terms

  void EvaluateDeltaPsi_(double *denominator_delta_psi,index_t flag){
    
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
    delta_psi_=(v_trans_v2-s1_+temp2)/temp4;
  }


  //delta \beta=(2M+D')^-1 D'(v2+D(v1-\delta psi v))
  void EvaluateDeltaBeta_(){
    
    printf("In evaluate delta beta...\n");
    
    Vector delta_psi_v;
    la::ScaleInit(delta_psi_,v_vector_,&delta_psi_v);
    
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

    la::SubOverwrite(D_A_temp1,sum_of_v2_prod1,&delta_beta_vector_);
  }

  //delta gamma =D'(v2-delta \beta)
  
  //Evaluate delta beta
  
   void EvaluateDeltaGamma_(){


     //\delta gamma= 2M\delta \beta +\delta \psi v-v_1

     Vector two_M_delta_beta;

     special_la::PostMultiplyMatrixWithVectorGivenCholesky(chol_factor_,
							   perm_mat_,
							   delta_beta_vector_,
							   &two_M_delta_beta);
	
     //Lets find 2M\delta \beta +\delta psi v

     Vector delta_psi_v;

     la::ScaleInit(delta_psi_,v_vector_,&delta_psi_v);

     Vector two_M_delta_beta_plus_delta_psi_v;

     la::AddInit(two_M_delta_beta,
		 delta_psi_v,
		 &two_M_delta_beta_plus_delta_psi_v);

    
     //Finally subtract v1
     la::SubOverwrite(v1_vector_,
		      two_M_delta_beta_plus_delta_psi_v,
		      &delta_gamma_vector_);
   }

   void GetTheStartPoint_(){
     //The starting point is basically a set of values for
     //beta_vector_,gamma_vector_,psi_.  Though it can be chosen in a
     //more intelligent way, I shall start with a very naive choice of
     //\beta,\gamma \psi
     
     beta_vector_.SetAll(1);
     gamma_vector_.SetAll(1);
     psi_=5;
     mu_=1;

     //An intelligent choice

     /* double temp=primal_constant_/sqd_num_train_points_; */

/*      double total_sum=0; */
     
/*      for(index_t i=0;i<sqd_num_train_points_-1;i++){ */
       
/*        if(v_vector_[i]>temp){ */

/* 	 beta_vector_[i]=0; */
/*        } */
/*        else{ */

/* 	 beta_vector_[i]=1; */
/* 	 total_sum+=v_vector_[i]; */
/*        } */
/*      } */
/*      beta_vector_[sqd_num_train_points_-1]=(primal_constant_-total_sum)/v_vector_[sqd_num_train_points_-1]; */

/*      //printf("beta vector is...\n"); */
/*      //beta_vector_.PrintDebug(); */
     
/*      //printf("gamma vector is...\n"); */
     //gamma_vector_.PrintDebug();

     
     
     // We shall obtain the starting point by solving a regularized KKT
     /*v1_vector_.CopyValues(a_vector_);
       s1_=primal_constant_;
       v2_vector_.SetZero();
       
       double val;
       val=0;
       
       index_t flag=NOT_CALCULATED;
       
       EvaluateDeltaPsi_(&val,&flag);
       EvaluateDeltaBeta_();
       EvaluateDeltaGamma_();
       
       //In order to get positivity constraints
       
       for(index_t i=0;i<sqd_num_train_points_;i++){
       
       beta_vector_[i]=max(delta_beta_vector_[i],1.0);
       gamma_vector_[i]=max(delta_gamma_vector_[i],1.0);
       }
       
       psi_=delta_psi_;*/ 
   }
   
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
     
     EvaluateDeltaPsi_(denominator_delta_psi,flag);

     //printf("Evaluated delta psi..\n");
     EvaluateDeltaBeta_();
     
     //printf("Evlauated delta beta...\n");
     EvaluateDeltaGamma_();
     
     //printf("Evaluated delta gamma...\n");
   }
   
   void TakeCorrectorStep_(double *denominator_delta_psi,index_t flag){
     
     //The corrector step involves updating v2 and resolving equations
     //Hence calculate v2
     
     FormVectorv2_(); 
       
     //Resolve the equations
     
     TakePredictorStep_(denominator_delta_psi,flag);
   }
   
   //This step finds the appropriate Step size to update the primal 
   //and dual variables
   
   double FindStepSize_(){
     
     /*  double min_lambda=1.0;
	 double lambda;
	 //Run an iterator over beta values and find min lambda
	 for(index_t i=0;i<sqd_num_train_points_;i++){
	 
	 if(delta_beta_vector_[i]<0){
	 lambda=
	 beta_vector_[i]*(UPDATE_FRACTION-1)/delta_beta_vector_[i];
	 }
	 if(lambda<min_lambda){
	 
	 min_lambda=lambda;
	 }
	 }
	 
	 for(index_t i=0;i<sqd_num_train_points_;i++){
	 
	 if(delta_gamma_vector_[i]<0){
	 lambda=
	 gamma_vector_[i]*(UPDATE_FRACTION-1)/delta_gamma_vector_[i];
	 }
	 
	 if(lambda<min_lambda){
	 
	 min_lambda=lambda;
	 
	 }
	 }
	 return min_lambda;*/
     
     
     //Iterator over beta
     double min_over_beta=DBL_MAX;
     double min_over_gamma=DBL_MAX;
     
     for(index_t i=0;i<sqd_num_train_points_;i++){
       
       double frac_beta=delta_beta_vector_[i]/beta_vector_[i];

       if(frac_beta < min_over_beta){
	 
	 min_over_beta=frac_beta;
       }
       
       double frac_gamma=delta_gamma_vector_[i]/gamma_vector_[i];

       if(frac_gamma < min_over_gamma){
	 
	 min_over_gamma=frac_gamma;
       }
     }

     double temp=1.0/(UPDATE_FRACTION-1);

     //printf("min_over_beta=%f..\n",min_over_beta);
     //printf("min_over_gamma=%f..\n",min_over_gamma);
     
     double lambda_inv=max(max(1.0,temp*min_over_beta),temp*min_over_gamma);
     return 1.0/lambda_inv;
   }

  void UpdatePrimalAndDualVariables_(double min_lambda){

    la::Scale(min_lambda,&delta_beta_vector_);
    
    //\beta <- \beta+delta_beta
    
    la::AddTo(delta_beta_vector_,&beta_vector_);

    la::Scale(min_lambda,&delta_gamma_vector_);

    //\gamma <-\gamma+delta \gamma

    la::AddTo(delta_gamma_vector_,&gamma_vector_);

    psi_=psi_+min_lambda*delta_psi_;
  }

  void UpdateMu_(double min_lambda){


    double small_number=0.00001;

   
    //$mu=
    //\frac{\beta^T\gamma}{m^2} \left(\frac{1-\lambda+epsilon}{10+\lambda}\right)$

    double dot_product_between_beta_and_gamma;

    dot_product_between_beta_and_gamma=la::Dot(beta_vector_,gamma_vector_);

    double frac=(1-min_lambda+small_number)/(10+min_lambda);
 
    mu_=(dot_product_between_beta_and_gamma/sqd_num_train_points_)*frac*frac;

  }
  
  //This function simply flushes the delta_beta,delta_gamm and
  //delta_psi values to 0

  void FlushUpdates_(){
    
    delta_beta_vector_.SetZero();
    delta_gamma_vector_.SetZero();
    delta_psi_=0;
  }


  void CalculateGap_(double *gap,double *gap_ratio){

    //gap =-\beta^T \gamma+\psi(\beta^Tv-c);

    double beta_trans_gamma;

    beta_trans_gamma=la::Dot(beta_vector_,gamma_vector_);

    double beta_trans_v;
    beta_trans_v=la::Dot(beta_vector_,v_vector_);

   *gap=beta_trans_gamma-psi_*(beta_trans_v-primal_constant_);

    
    Vector A_trans_beta;
    la::MulInit(beta_vector_,permuted_chol_factor_,&A_trans_beta); //$A^T\beta$

    //printf("A_trans beta is...\n");
    //A_trans_beta.PrintDebug();


    double quad_value;

    quad_value=la::Dot(A_trans_beta,A_trans_beta); //quad value is \beta^TAA^T\beta

    //Since A is the permuted cholesky factor of 2M and not M, the
    //quad _value obtained above is twice the actual value

    quad_value/=2;

    double beta_trans_a;

    beta_trans_a=la::Dot(beta_vector_,a_vector_);

    double function_val=quad_value-beta_trans_a;

    double denominator=fabs(2*function_val+*gap);

    *gap_ratio=*gap/denominator;
    
    //printf("Quad value is %f..\n",quad_value);
    
    //printf("Linear value is %f..\n",beta_trans_a);
    
    printf("Gap is %f..\n",*gap);

    printf("Ratio is %f..\n",*gap/denominator);

    printf("Beta gamma product is %f..\n",beta_trans_gamma);

    printf("The objective function value is %f..\n",function_val);
    printf("..................................\n");

  }

  void CheckIfSolutionIsPositive_(){
    
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
    printf("beta counter=%d and gamma counter=%d..\n",beta_counter,gamma_counter);
    if(beta_counter>=1 ||gamma_counter>=1){
      printf("OPtimization wasnt successful....\n");
      exit(0);

    }
       
  }

  index_t PredictorCorrectorSteps_(){

    
    //Get the starting values for beta and gamma and \psi
    
    GetTheStartPoint_();
    printf("psi=%f mu_=%f..\n",psi_,mu_);
    
    
    index_t flag;
    
    double gap;
    double gap_ratio=DBL_MAX;
    while(fabs(gap_ratio)>max_gap_ratio_&&num_iterations_<MAX_NUM_ITERATIONS){ 
      
      
      //We compute vectors v1,v2, s1, before taking the predictor
      //corrector steps. While v1,s1 are same for both the predictor
      //and the correctors steps, v2 changes for the corrector
      //step. Also we need the D_vector


      GetDVector_();
      
      FormVectorv1_();
      FormVectorv2_();
      FormScalars1_();

      double denominator_delta_psi=0;
      
      // This forces the program to calculate the quantity
      // denominator_delta_psi
      
      flag=NOT_CALCULATED; 
      
      TakePredictorStep_(&denominator_delta_psi,flag);

      //Since predictor step has calculated the quantity denominator_delta_psi 
      //set the flag
      
      flag=CALCULATED;
      
      TakeCorrectorStep_(&denominator_delta_psi,flag);
      
      double min_lambda=FindStepSize_();

      if(min_lambda<SMALL_STEP){
	
	printf("min_lambda is %f..\n",min_lambda);
	
	return -1;
      }
      
     
      //Having got the step size update the primal and the dual
      //variables
      
      UpdatePrimalAndDualVariables_(min_lambda);
      
      //Update mu
      
      UpdateMu_(min_lambda);
      
      FlushUpdates_();
      
      CalculateGap_(&gap,&gap_ratio);
      
      printf("Finished ITERATION=%d...\n\n",num_iterations_);

      num_iterations_++;
    }

    if(gap_ratio>max_gap_ratio_){

      return -1;
    }

    CheckIfSolutionIsPositive_();
    return 1;
  }
    
  void GetLinearPartOfObjectiveAndLinearConstraintVectors_(){
    
    
    //We shall call the vector involved in the linear part of
    //objective as a_vector_
    
    GetLinearPartOfObjective_();

    printf("Got linear part of the objective.....\n");
    
    //We shall call the vector involved in the linear part of
    //constraint as the v_vector_
    
    GetLinearConstraint_();
    
    // printf("V vector is...\n");
    //v_vector_.PrintDebug();
    
    
    //printf("a vector is..\n");
    //a_vector_.PrintDebug();
  }
  
  
  void GetCholeskyFactorizationOfMMatrix_(){

    ICholDynamic ichol_dynamic;
    ichol_dynamic.Init(train_set_,sigma_h_,sigma_,lambda_);
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

    ComputeTestDensities_();
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

  

    printf("The test set is...\n");
    test_set_.PrintDebug();

   
    //This ends the necessaary file reading routines. Lets read the
    //parameters of the gaussian kernel and the regularization
    //constant \lambda from the user
    
    sigma_=fx_param_double_req(module_,"sigma");
    sigma_h_=fx_param_double_req(module_,"sigma_h");
    lambda_=fx_param_double_req(module_,"lambda");

    printf("In the init function we have...\n");
    printf("sigma_h=%f,sigma=%f,lambda=%f..\n",sigma_h_,sigma_,lambda_);
    max_gap_ratio_=fx_param_double(module_,"max_gap_ratio",0.0001);
    
    

    //The dimensionality of the dataset
    num_dims_=train_set_.n_rows();

    printf("Number of dimensions are %d..\n",num_dims_);

    sqd_num_train_points_=num_train_points_*num_train_points_;

    printf("Squared number of train points are %d..\n",sqd_num_train_points_);

    //THIS IS A TEMPORARY ARRANGEMENT..........

    //............................................
    //Initialize vectors that will be used

    v1_vector_.Init(sqd_num_train_points_);
    v2_vector_.Init(sqd_num_train_points_);
    v_vector_.Init(sqd_num_train_points_);
    a_vector_.Init(sqd_num_train_points_);
    beta_vector_.Init(sqd_num_train_points_);
    gamma_vector_.Init(sqd_num_train_points_);
    delta_beta_vector_.Init(sqd_num_train_points_);
    delta_gamma_vector_.Init(sqd_num_train_points_);

    computed_test_densities_.Init(num_test_points_);
    
    //The permutation matrix. This will be used along with the chol
    //factor in calculations

    perm_mat_.Init(sqd_num_train_points_);
    D_vector_.Init(sqd_num_train_points_);

    //chol factor will be initialized in the routine GetCholFactor

    //Set delta_beta, delta_gamma and delta_psi to 0
    delta_beta_vector_.SetZero();
    delta_gamma_vector_.SetZero();
    delta_psi_=0;


    double sigma_4=sigma_*sigma_*sigma_*sigma_;
    double sigma_2=sigma_*sigma_;
    double sigma_h_2=sigma_h_*sigma_h_;
    double sigma_h_4=sigma_h_*sigma_h_*sigma_h_*sigma_h_;


    //The primal constant
    double three_simga_sqd_plus_two_sigma_h_sqd;

    three_simga_sqd_plus_two_sigma_h_sqd=
      3*sigma_2+2*sigma_h_2;


    primal_constant_=
      sqrt(sqd_num_train_points_*math::PI)* 
      three_simga_sqd_plus_two_sigma_h_sqd*
      sqrt(sigma_2+sigma_h_2);

    primal_constant_/=
      sqrt(3*sigma_4+5*sigma_2*sigma_h_2+
	   2*sigma_h_4);

    printf("Primal constant is %f...\n",primal_constant_);
    num_iterations_=0;

  }
};
#endif

