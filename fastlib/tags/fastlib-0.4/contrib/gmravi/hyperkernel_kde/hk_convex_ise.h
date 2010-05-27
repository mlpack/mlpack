#ifndef HK_CONVEX_ISE_H
#define HK_CONVEX_ISE_H
#include "fastlib/fastlib.h"
#include "hyperkernels.h"
#include "ichol.h"
#include "dte.h"
#include "special_la.h"
#define FAILURE -1
#define EPSILON 0.00000001
class HkConvexIse{
  FORBID_ACCIDENTAL_COPIES(HkConvexIse);
 private:
  
  //The module. This will be initialized with the incoming module
  struct datanode *module_;
  
  //The reference set
  Matrix train_set_;

  //The query set(test)
  Matrix test_set_;

  //The true densities(test set). This will be available only for
  //synthetic datasets. Since files read are stored as matrices we
  //shall define this variable as a matrix
  
  Matrix true_test_densities_;
  
  //The Computed densities of the test set

  Vector computed_test_densities_;

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

  //The linear constraint

  Vector At_linear_constraint_;

  //Linear part of the objective

  Vector At_linear_objective_;

  //The cholesky factor of the matrix M

  Matrix chol_factor_;

  //The eigen values and eigen vectors of the M matrix, used in our
  //optimization problem

  Matrix eigen_vectors_;

  Vector eigen_values_;

 public:

  //Constructor and Destructor
  HkConvexIse(){


  }
  ~HkConvexIse(){

  }

  
  /**In this function we shall form the quadratic part of the
     objective. This weill require me to do an online incomplete
     cholesky of the matrix involved in the qaudratic part. The matrix
     is \frac{K'}{m^2}+\lambda K. We shall use ~\cite{shai fine}
  **/
  void FormQuadraticPartOfObjective_(){
    
    IChol ichol;
    ichol.Init(train_set_,sigma_h_,sigma_,lambda_);

    //lower triangular incomplete cholesky factor
    ichol.Compute(chol_factor_);

    
    //Having got the chol_factor lets obtain the eigen decomposition
    //of the original matrix using the cholesky factors. This can be done by 
    // an SVD of the cholesky factor L (M=LL')
   
   
    Matrix temp_eigen_vectors,temp;
    
    la::SVDInit(chol_factor_,&eigen_values_,&temp_eigen_vectors,&temp);

    //to form the quadratic constraints we need the vectors
    //\sqrt{\lambda_i}p_i where p_i is the eigen vector of the
    //original matrix M lambda_i is the eigen value of the original
    //matrix. However lambda_i=square of the signaular valuess of the
    //svd of L. hence \sqrt{\lambda_i}=signular value of L. Hence
    //\sqrt{\lambda_i}p_i amounts to scaling the column of the matrix
    //with the corresponding eigen value. Remember the eigen vectors
    //will be stored in a row-majore format
    
    //We shall do this by cleverly using the lapack utility ScaleRows


    la::TransposeInit(temp_eigen_vectors,&eigen_vectors_);
 
    la::ScaleRows(eigen_values_,&eigen_vectors_);

    printf("Number of eigen values are %d...\n",eigen_vectors_.n_cols());
    
    //eigen_vectors_now hold the scaled the eigen vectors_ which we
    //shall use to build the matrix required for the optimization
    //procedure. 

  }

  //This function fors the linear part of the QP. The linear part is
  //-\beta^Ta where a_{p,q}= \frac{2}{m(m-1)}\sum_{i=1}^m\sum_{j=1}^m
  //\underline{K}\left(\left(x_i,x_j\right),\left(x_p,x_q\right)\right)
  

  void FormLinearPartOfObjective_(){

    GaussianHyperKernel ghk;
    ghk.Init(sigma_,sigma_h_,num_dims_);
    double norm_const=ghk.CalcNormConstant();

    //THIS HAS TO CHANGE WHEN I USE A MULTIPLICATIVE KERNEL
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
	//hence the required sum is twice the original sum+the sum of
	//diagonal element

	double total_sum_of_elem_row=2*sum_of_row*gpq_unnorm_val/norm_const;

	At_linear_objective_[p*num_train_points_+q]=
	  -total_sum_of_elem_row;
	

	//Since a_{p,q}=a_{q,p}
	At_linear_objective_[q*num_train_points_+p]=
	  -total_sum_of_elem_row;
      }
    }
    index_t sqd_num_train_points=num_train_points_*num_train_points_;

    printf("Sqd number of train points  are...%d\n",sqd_num_train_points);

    At_linear_objective_[sqd_num_train_points]=0;
    At_linear_objective_[sqd_num_train_points+1]=-1;

  }

  //Linear Constraints and the corresponding helper functions
  
  void DualTreeEvaluationOfLinearConstraint_(index_t p,Vector &row_p){

    //bw=sqrt(6\sigma^2+4\sigma_h^2)

    double bandwidth=sqrt(6*sigma_*sigma_+4*sigma_h_*sigma_h_); 
    double tau=0.0004;

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

  
  
  void FormLinearConstraint_(){
 

    //The linear constraint requires us to evaluate a vector of the
    //form $v_{p,q}=\sum_{i=1}^m
    //e^-\frac{\left(-2x_i+x_p+x_q\right)^2}{12\sigma^2+8\sigma_h^2}

    //Fix p and let q vary from q>=p
    
    for(index_t p=0;p<num_train_points_;p++){ ///THIS HAS TO CHANGE........................
      
      //Lets evaluate the pth row using dual tree computations

      Vector row_p;
      row_p.Init(num_train_points_-p);
      DualTreeEvaluationOfLinearConstraint_(p,row_p);  

      //Having got the row_p evaluations store them
     
      for(index_t q=p;q<num_train_points_;q++){

	index_t row_num1=p*num_train_points_+q;
	At_linear_constraint_[row_num1]=row_p[q-p];

	//By symmetry 
	index_t row_num2=q*num_train_points_+p;
	At_linear_constraint_[row_num2]=row_p[q-p];
      }
    }

    //the last 2 elements are 0's

    At_linear_constraint_[num_train_points_*num_train_points_]=0;
    At_linear_constraint_[num_train_points_*num_train_points_+1]=0;
  }

  
  //Form linear constraint naively
  /*void FormLinearConstraintNaively_(){
    
  //THIS HAS TO CHANGE FOR A MULTIDIMENSIONAL SETTING..........
    
  GaussianKernel gk;
  double bandwidth=sqrt(6*sigma_*sigma_+4*sigma_h_*sigma_h_);
   
  Matrix temp;
  temp.Init(num_train_points_,num_train_points_);

  for(index_t p=0;p<train_set_.n_cols();p++){

  double *x_p=train_set_.GetColumnPtr(p);
  Vector vec_x_p;
  vec_x_p.Alias(x_p,num_dims_);

  for(index_t q=0;q<train_set_.n_cols();q++){
	
  double *x_q=train_set_.GetColumnPtr(q);
  Vector vec_x_q;
  vec_x_q.Alias(x_q,num_dims_);

  //Initialize it once again because it was
  //initialized with sigma*sqrt(2)

  gk.Init(bandwidth); 
  double kernel_contrib=0;

  for(index_t i=0;i<train_set_.n_cols();i++){

  double *x_i=train_set_.GetColumnPtr(i);
  Vector vec_x_i;
  vec_x_i.Alias(x_i,num_dims_);
	  
  Vector x_p_minus_x_i;
  la::SubInit (vec_x_i,vec_x_p,&x_p_minus_x_i); //x_p-x_i

  Vector x_i_minus_x_q;
  la::SubInit (vec_x_q,vec_x_i,&x_i_minus_x_q); //x_i-x_q
	  
  double sqd_dist=
  la::DistanceSqEuclidean(x_p_minus_x_i,x_i_minus_x_q);

  kernel_contrib+=gk.EvalUnnormOnSq(sqd_dist);
  }

  //I need a multiplying factor of sigma*sqrt(2)
  gk.Init(sigma_*sqrt(2));

  double sqd_dist=la::DistanceSqEuclidean(vec_x_p,vec_x_q); //x_p-x_q
  double mult_factor=gk.EvalUnnormOnSq(sqd_dist);
  double norm_const=gk.CalcNormConstant(num_dims_);

  //mult_factor=norm_const=1;
  kernel_contrib*=mult_factor/norm_const;
  temp.set(p,q,kernel_contrib);
  }
  }
  printf("NUmber of train points are %d.....\n",num_train_points_);
    

  //  printf("The linear constraint matrix is....\n");
  temp.PrintDebug();

  //printf("comparing the errors by comparing the naive estimates...\n");
  double max_diff=DBL_MIN;
  double diff=0;
  for(index_t p=0;p<num_train_points_;p++){

  for(index_t q=0;q<num_train_points_;q++){

  index_t row_num=p*num_train_points_+q;
  
  if(fabs(At_linear_constraint_[row_num])>EPSILON && 
     fabs(temp.get(p,q))>EPSILON){
    diff=
      fabs(At_linear_constraint_[row_num]-temp.get(p,q))/temp.get(p,q);

    printf("Fast estimate and naive estimates and diff are  %f,%f,%f..\n",
	   At_linear_constraint_[row_num],temp.get(p,q),diff);
  }
  else{
    diff=0;
  }
  
  if(diff>max_diff){
    
    max_diff=diff;
  }
  //printf("Max difference is %f\n",max_diff);
  }
  }
  if(max_diff<0.0004){
    printf("Safe,Safe,Safe...\n");
  }
  else{
    
    printf("I FAIL FAIL FAIL...\n");

    printf("Max difference is..%f\n",max_diff);
  }
  
  }*/
  
  //This function is rite now printing onto a file the At_matrix.

  //However once we can get the code to link with matlab engine we
  //should be able to avoid the printing procedure
  void PerformSeDuMiOptimization_(){

   
    //This routine will stack the At constraints on top of each other.

    //At_linear objective is ready

    //At_linear constraint is also ready

    //eigen values needs to be appened with a matrix of 0's. The size
    //of this matrix will be (num_eigen_valuesXsqd_num_train_points+2)
    
    Matrix appended_eigen_vectors;
    special_la::AppendMatrixWithZerosInit(eigen_vectors_,2,&appended_eigen_vectors);

    //Having got the appended matrix now stack this matrix on top of
    //each other

    //[At]=[At_linear_objective;At_linear_constaint;-At_linear_constraint;appended_eigen_vectors];
    Matrix temp1;
    if(special_la::StackVectorVectorInit(At_linear_objective_,At_linear_constraint_,&temp1)==FAILURE){

      printf("Vectors that u were trying to stack are of unequal columns lengths...\n");
      exit(0);
    }


    Matrix temp2;

    la::Scale(-1,&At_linear_constraint_);
    if(special_la::StackMatrixVectorInit(temp1,At_linear_constraint_,&temp2)==FAILURE){

      printf("Matrix and vector to be stacked have different number of columns...\n");
      exit(0);
    }
    
    Matrix temp3;
    if(special_la::StackMatrixMatrixInit(temp2,appended_eigen_vectors,&temp3)==
       FAILURE){

      printf("Matrices to be stacked have different number of columns...\n");
      exit(0);
    }
    
    //having stacked all the matrices I shall now print it onto a file

    //The first line of the file will specify the parameters of optimization, 
    //followed by  the stacked matrix

    Vector temp4;
    temp4.Init(num_train_points_*num_train_points_+2);

    Matrix temp5_print;

    //sigma_h,sigam,lambda,num_train_points
    temp4.SetZero();
    temp4[0]=sigma_h_;
    temp4[1]=sigma_;
    temp4[2]=lambda_;
    temp4[3]=num_train_points_;

    special_la::StackVectorMatrixInit(temp4,temp3,&temp5_print);

    printf("The size of the matrix to be dumped is %d %d..\n",
	   temp5_print.n_rows(),temp5_print.n_cols());
    printf("All Computations performed. Will dump the At matrix onto a file....\n");
    FILE *fp;
    fp=fopen("At_matrix.txt","w");
    temp5_print.PrintDebug(NULL,fp);
  }
  
  //This function computes the optimal kernel from the hyper-kernel
  void ComputeOptimalKernel(){
    
    //To Calculate the optimal kernel we solve an optimization
    //problem, which involves forming the matrix out of the objective 
    //(the At matrix).
    
    //The objective has two parts namely the quadratic and the linear
    //part 

    //The constraints are positivity constraints and the normalization
    //constraints
    
    //We shall begin with forming the linear part of the
    //objective
    

    fx_timer_start(NULL,"linear_part_obj");
    FormLinearPartOfObjective_();
    fx_timer_stop(NULL,"linear_part_obj");
    printf("Linear part of objective formed...\n");


    // Forming the linear constraints now.........
    
    //We shall do this by dual tree calculations.
    
    fx_timer_start(NULL,"linear_const");
    FormLinearConstraint_();
    // FormLinearConstraintNaively_();
    fx_timer_stop(NULL,"linear_const");
    printf("Linear constraints have been formed...\n");   



    //Next formulate the quadratic part of the objective

    fx_timer_start(NULL,"quad_part_obj");
    FormQuadraticPartOfObjective_();
    fx_timer_stop(NULL,"quad_part_obj");

    printf("Quadratic part of objective formeed...\n");
          
    //PerformSeDuMiOptimization_();
      
    
  }

  void Init(Matrix &train_set, struct datanode *module_in){
    
    //set the module to the incoming module
    module_=module_in;
    train_set_.Alias(train_set);


    num_train_points_=train_set_.n_cols();

    //Check for the existence of a query set and accordingly read data
    if(fx_param_exists(module_,"query")){

      //Load the dataset
      const char *test_file=fx_param_str_req(module_,"query");
      data::Load(test_file,&test_set_);

      //Since there is a  test file hence we shall compute the test densities
      computed_test_densities_.Init(test_set_.n_cols());
    }
    else{
      
      test_set_.Init(0,0); //This avoids segmentation fault
      computed_test_densities_.Init(0);
    }

    if(fx_param_exists(module_,"true")){
      
      const char *true_density_file=fx_param_str_req(module_,"true");

      //Since the true test densities are given, hence
      data::Load(true_density_file,&true_test_densities_);
    }
    else{
      true_test_densities_.Init(0,0);
    }

    //This ends the necessaary file reading routines. Lets read the
    //parameters of the gaussian kernel and the regularization
    //constant \lambda from the user

    sigma_=fx_param_double_req(module_,"sigma");
    sigma_h_=fx_param_double_req(module_,"sigma_h");
    lambda_=fx_param_double_req(module_,"lambda");
    num_dims_=train_set_.n_rows();
    
    At_linear_constraint_.Init(num_train_points_*num_train_points_+2);

    index_t len_linear=num_train_points_*num_train_points_+2;

    At_linear_objective_.Init(len_linear);
  }
};
#endif
