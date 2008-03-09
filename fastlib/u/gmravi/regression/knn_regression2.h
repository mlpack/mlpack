#ifndef KNN_REGRESSION2_H
#define KNN_REGRESSION2_H
#define LEAF_SIZE 50

#include <fastlib/fastlib.h>
#include "allknn.h"
#include "pseudo_inverse.h"
#include "./dense_lpr/naive_lpr.h"
#include "./dense_lpr/matrix_util.h"

#define CALCULATE_FOR_REFERENCE_POINTS_ 0
#define CALCULATE_FOR_QUERY_POINTS_ 1


template<typename TKernel> class KNNRegression{

 private:


  //The number of k-nearest neighbours
  index_t k_;

  //The number of dimensions of the dataset
  index_t number_of_dimensions_;

  double root_mean_square_deviation_;

  //To store the mean of the residuals
  //This  is required during cross validation stage
  double sigma_hat_;

  //The cross validation score of this model
  double cross_validation_score_;

  //The first and the second degrees of freedom
  double df1_,df2_;
  
  double max_relative_error_regression_estimates_query_;

  double max_relative_error_confidence_interval_upper_query_;

  double max_relative_error_confidence_interval_lower_query_;

  double average_relative_error_confidence_interval_lower_query_;

  double average_relative_error_confidence_interval_upper_query_;


  double max_relative_error_regression_estimates_reference_;

  double max_relative_error_confidence_interval_upper_reference_;

  double max_relative_error_confidence_interval_lower_reference_;

  double average_relative_error_confidence_interval_lower_reference_;

  double average_relative_error_confidence_interval_upper_reference_;


  //Data structure to hole the knn points and their distances. We
  //shall use the same arrays to hold the k-nn of the reference points
  //and the query points

  ArrayList<index_t> nn_neighbours_;
  ArrayList<double> nn_distances_;

  //The regression estimates calculated using the knn based local
  //fitting model for the query points
  
  Vector regression_estimates_;

  //This holds the distance of the kth nearest point
  Vector  kth_nn_distances_;

  //The global smoothing parameter to be used. Note the knn-based
  //method is an adaptive bandwidth method. hence we use a global
  //smoothing parameter
  
  double global_smoothing_;

  //The kind of kernel to be used

  TKernel kernel_;

  //The query and the reference sets

  Matrix qset_;
  Matrix rset_;

  //The reference values

  Matrix rset_weights_;

  //Matrices required for local linear calculations

  Matrix b_twb_;
  Matrix b_twy_;

  Matrix b_twb_inv_;

  //Matrix for Confidence Interval estimates

  Matrix b_tw2b_;

  //A vector to store the confidence intervals of all the query points

  Vector confidence_interval_query_points_;

  //A vector to store the confidence intervals of all the query points
  
  Vector confidence_interval_reference_points_;
  

  //A vector to store the ||l(x)||^2 values. They
  //will be used for C.I calculations of the query and the reference points

  Vector sqdlength_of_weight_diagram_;

  //This is a generic function and can be used to calculate the
  //C.I. of the query point either by using local linear fitting or
  //NWR regression


  //regression estimates of the referernce values

  // Vector regression_estimates_reference_;

  void CalculateConfidenceInterval_(){

    for(index_t q=0;q<qset_.n_cols();q++){

      //Once we have the length of the hat vector we should be able to
      //calculate the upper and lower bounds of the C.I for the query
      //point

      //printf("In query point calculations..\n");
      //printf("sigma_hat_ is %f\n",sigma_hat_);
      double lower_bound=
	regression_estimates_[q]-
	1.96*sigma_hat_*sqrt(1+sqdlength_of_weight_diagram_[q]);

      confidence_interval_query_points_[q*2]=lower_bound;

      double upper_bound=
	regression_estimates_[q]+
	1.96*sigma_hat_*sqrt(1+sqdlength_of_weight_diagram_[q]);

      confidence_interval_query_points_[q*2+1]=upper_bound;

      //printf("sqd length is %f\n",sqdlength_of_weight_diagram_[q]);
      //printf("regression estimate of %d-query point is %f\n",q,regression_estimates_[q]);
      //printf("Upper bound is %f\n",upper_bound);
      //printf("lower bound id %f\n",lower_bound);
    }
  }

  void CalculateConfidenceIntervalOfReferencePoints_(){
    
    //printf("for reference points sigma_hat is %f\n",sigma_hat_);
    for(index_t q=0;q<rset_.n_cols();q++){
      
      //Once we have the length of the hat vector we should be able to
      //calculate the upper and lower bounds of the C.I for the query
      //point
      
      
      double lower_bound=
	regression_estimates_[q]-
	1.96*sigma_hat_*sqrt(sqdlength_of_weight_diagram_[q]);
      
      confidence_interval_reference_points_[q*2]=lower_bound;
      
      double upper_bound=
	regression_estimates_[q]+
	1.96*sigma_hat_*sqrt(sqdlength_of_weight_diagram_[q]);
      
      confidence_interval_reference_points_[q*2+1]=upper_bound;

     
      // printf("sqd length is %f\n",sqdlength_of_weight_diagram_[q]);
      //printf("regression estimate of %d reference point is %f\n",q,regression_estimates_[q]);
      //printf("Upper bound is %f\n",upper_bound);
      //printf("lower bound id %f\n",lower_bound);
    }

  }




  //This function calculates the squared length of a vector
  double SquaredLengthOfVector_(Vector vec,index_t len){

    double sqdlength=0;
    for(index_t i=0;i<len;i++){
      sqdlength+=vec[i]*vec[i];
    }
    return sqdlength;
  }


  //This returns the influence of q_point on its own estimate. This
  //shall be used to calculate the df1

  double CalculateInfluenceForLocalLinear_(double *q_point,index_t q){

    //influence is norm_const*[1,q_point] (B^TWB)^-1 [1,q_point]^T

    Matrix point;
    point.Init(1,number_of_dimensions_+1);
    point.set(0,0,1);

    for(index_t j=1;j<number_of_dimensions_+1;j++){

      point.set(0,j,q_point[j-1]);
    }

    //Now lets multiply [1,q_point] with (B^TWB)^-1 and stores it in
    //temp. 

    Matrix temp;
    la::MulInit(point,b_twb_inv_,&temp);


    //NOw multiply temp with [1,q_point]^T. So lets first compute the
    //tranpose of [1,q_point]

    Matrix point_transpose;
    la::TransposeInit(point,&point_transpose);

    Matrix influence_matrix;
    la::MulInit(temp,point_transpose,&influence_matrix);

    //Lets calculate the normalization constant
    double bw=global_smoothing_*sqrt(nn_distances_[q*k_+k_-1]); 

    printf("BW used for initialization is %f\n",bw);
    kernel_.Init(bw);
    double norm_constant=1/kernel_.CalcNormConstant(number_of_dimensions_);
    return norm_constant*influence_matrix.get(0,0);
  }

  //This function will calculate ||l(x)||^2 for the point provided as
  //argument. This function when called over all the reference points
  //gives the df2 and when called over the query points can be used in
  //confidence interval estimation

  double CalculateSqdWeightDiagramLocalLinear_(double *q_point){

    // ([1,q_point] (B^TWB)^-1) (B^T W^2 B) ((B^TWB)^-1[1,q_point]T)

    Matrix point;
    point.Init(1,qset_.n_rows()+1);
    point.set(0,0,1);

    for(index_t j=1;j<qset_.n_rows()+1;j++){

      point.set(0,j,q_point[j-1]);
    }  

    Matrix left; 
    //Note b_twb_ holds the  inverse of B^TWB matrix
    la::MulInit(point,b_twb_inv_,&left); 
   
    Matrix right;
    la::TransposeInit(left,&right);

    //Note:Middle matrix has already been calculated during Compute
    //Matrix calculations. all that remains left to be done is
    //multiply these three matrices

    Matrix temp;
    la::MulInit(b_tw2b_,right,&temp);

    Matrix sqd_length; //this is the squared length of the l(x) vector
    la::MulInit(left,temp,&sqd_length);
    return sqd_length.get(0,0);

  }

  void ComputeMatrices_(double *q_point,index_t q){

    //printf("came to compute matries...\n");

    //for each reference point which are the k-nearest neighbours of
    //the query point ''q''
    // printf("For a new point ...\n");
    
    for (index_t r = 0; r < k_; r++){ 
   
      //Set up the bandwidth as per the reference point
         
      //The index of the rth nearest neighbour
      index_t knn_point=nn_neighbours_[q*k_+r];
      //printf("knn point is %d\n",knn_point);
	
      //the distance of the kth nearest neighbour of knn_point
      double dist=kth_nn_distances_[knn_point];
      
      double bw=global_smoothing_*sqrt(dist);
      printf("bw is %f\n",bw);
      kernel_.Init(bw);
      //printf("number of ref points are %d\n",rset_.n_cols());
	
      //Get reference point
      const double *r_col = rset_.GetColumnPtr(knn_point);
      
      // pairwise distance and kernel value
      double dsqd =
	la::DistanceSqEuclidean (number_of_dimensions_, q_point, r_col);
     
      dsqd=nn_distances_[q*k_+r];

      printf("distance of this neighbour is %f\n",dsqd);
      printf("radius of this kernel is %f\n",dist);
      
      double ker_value = kernel_.EvalUnnormOnSq (dsqd)/
	kernel_.CalcNormConstant(number_of_dimensions_);
      printf("Kernel value is %f\n",ker_value);

      
      
      for(index_t col = 0; col < number_of_dimensions_+ 1; col++){	
	//along each direction
	
	for(index_t row=0; row< number_of_dimensions_+ 1; row++){
	  
	  
	  //Lets gradually fill up all the elements of the matrices 
	  //b_twy_ and b_twb_
	  
	  //Fill in b_twy_ first
	  
	  if(col==0){
	    
	    //For this column 
	    if (row != 0){
	     
	      // printf("In column 0 and row not equal to 0\n");
	      //Fill B^TWY naive
	      
	      double val=b_twy_.get(row,col)+ 
		ker_value * rset_weights_.get(knn_point,0) * 
		  rset_.get (row-1,knn_point);
	      
	      b_twy_.set(row,col,val);
	      
	      //Now fill B^TWB naive
	      
	      double val1=b_twb_.get(row,col) + 
		ker_value * rset_.get(row-1,knn_point);
	      b_twb_.set(row,col,val1);
	      
	     
	      //Also fill in B^T W^2 B matrix
	      
	      double val2=b_tw2b_.get(row,col) + 
		ker_value * ker_value*
		rset_.get(row-1,knn_point);
	      b_tw2b_.set(row,col,val2);   
	    }
	    
	    else{
	      
	      //Fill B^TWY naive 
	      
	      double val= b_twy_.get(row,col) + 
		ker_value * rset_weights_.get(knn_point,0);
	      
	      b_twy_.set(row,col,val);
	      
	      //Now fill B^TWB
	      
	      double val1=b_twb_.get(row,col) 
		+ ker_value ;
	      
	      b_twb_.set(row,col,val1);
	      
	      //Also fill B^T W^2 B matrix
	      double val2=b_tw2b_.get(row,col) 
		+ ker_value *ker_value;
	      
	      b_tw2b_.set(row,col,val2);
	      
	      
	    }
	  }//end of col 0...............
	  
	  //Column!=0
	  
	  else{
	    
	    if(row!=0){
	      //printf("In column not 0 and row not not equal to 0\n");
	      
	      //Only B^TWB naive estimates get filled up
	      //printf("lets calculate row val\n");
	      //printf("row val is %f\n",rset_.get(row-1,knn_point));

	      double val1=b_twb_.get(row,col) 
		+ ker_value* rset_.get(row-1,knn_point)* 
		  rset_.get(col-1,knn_point);
	      //printf("val1 calc is %f\n",val1);
	      b_twb_.set(row,col,val1);
	      //printf("val1 is %f\n",val1);
	      
	      double val2=b_tw2b_.get(row,col) 
		+ ker_value*ker_value* 
		rset_.get(row-1,knn_point)* 
		rset_.get(col-1,knn_point);
	      
	      b_tw2b_.set(row,col,val2);
	      //printf("val2 is %f\n",val2);
	      
	    }
	    
	    else{
	      
	      //printf("In column 0 and row  equal to 0\n");
	      double val1=b_twb_.get(row,col) 
		+ ker_value* rset_.get(col-1,knn_point);
	      b_twb_.set(row,col,val1);
	      
	      double val2=b_tw2b_.get(row,col) 
		+ ker_value* ker_value*
		rset_.get(col-1,knn_point);
	      b_tw2b_.set(row,col,val2);
	      
	    }
	  }
	}
      }
    }
  }
  




  void PerformKNNLocalLinearRegression_(double *q_point,
					index_t q, 
					index_t flag,
					double &influence, 
					double &sqdlength,
					double &estimate){
    
    //This involves computing 2 matrices namely B^TWB and B^TWY nad
    //the matrix B^T W^2B for C.I calculations
    
    ComputeMatrices_(q_point,q);
    //printf("Matrices computed...\n");

    //Let q_matrix hold [q,q_point]^T

    Matrix q_matrix;
    q_matrix.Init(1,number_of_dimensions_+1);
    q_matrix.set(0,0,1);

    for(index_t col=1;col<number_of_dimensions_+1;col++){
      
      q_matrix.set(0,col,q_point[col-1]);
    }

    //printf("Q matrix formed,...\n");

    if(flag==CALCULATE_FOR_REFERENCE_POINTS_){

      //Lets also define matrices namely b_twb_cross_validation and
      //b_twy_cross_validation. These are termporary matrices

      Matrix b_twb_cross_validation,b_twy_cross_validation;
      Matrix b_twb_cross_validation_inv;

      b_twb_cross_validation_inv.Init(b_twb_.n_rows(),b_twb_.n_cols());

      //We shall modify these matrices to get cross validation score...
      //We shall subtract from B^TWB 1/NormConstat * [1,q_point]^T [1.q_point]
      
      //q_matrix holds [1,q_point]
      //q_matrix_transpose will hold [1,q_point]^T
      
      Matrix q_matrix_transpose,temp_product;
      la::TransposeInit(q_matrix,&q_matrix_transpose);
      la::MulInit(q_matrix_transpose,q_matrix,&temp_product);
      //temp_product will have [1,q_point]^T [1,q_point]
      
      //this is the distance to the k nearest tneighbour of the
      //reference point. That is the radius of the kernel associated
      //with the query point itself

      double bw=global_smoothing_*nn_distances_[q*k_+k_-1]; 
      kernel_.Init(bw);
      double norm_constant=1/kernel_.CalcNormConstant(number_of_dimensions_);
      la::Scale(norm_constant,&temp_product);
       
  

      //With this b_twb now holds b_twb
      //without considering the query
      //points self contribution. that is
      //the cross validated value

      la::SubInit(temp_product,b_twb_,&b_twb_cross_validation); 
      
      //Similarily to get b^TWY value without considering its own
      //contribution b_twy_cross_validation <- b_twy_ -
      //rset_weight*normconstant [1,q_point]^T

      //Scale B^TWY by rset_weight*nomalization constant. 


      //Note: q_matrix_tranpose no longer contains [1,q_point]^T

      la::Scale(rset_weights_.get(q,0)*norm_constant,&q_matrix_transpose); 
      la::SubInit(q_matrix_transpose,b_twy_,&b_twy_cross_validation);

      //printf("cross validated btwy[%d] is\n",q);
      //b_twy_cross_validation.PrintDebug();

      //Now cross validated regression estimate is obtained by
      //multiplying BTWB^-1 with B^TWY

      
      Matrix temp1,temp2;
     
      MatrixUtil::PseudoInverse(b_twb_cross_validation,&b_twb_cross_validation_inv);  
      la::MulInit (b_twb_cross_validation_inv,b_twy_cross_validation, &temp1);
      //printf("first prioduct found..\n");
      
      //Now lets multiply the resulting q_matrix with temp1 to get the
      //regression estimate
      
      la::MulInit(q_matrix,temp1,&temp2);  
     
      double regression_estimate_cross_validation=temp2.get(0,0);
      double diff=regression_estimate_cross_validation-rset_weights_.get(q,0);
    
      cross_validation_score_+=diff*diff;
    }
      
    //printf("will get the actual estimates now...\n");
    //So we now have B^TWB and B^TWY matrices.
    Matrix temp;
    MatrixUtil::PseudoInverse(b_twb_,&b_twb_inv_);  
   
    la::MulInit (b_twb_inv_,b_twy_, &temp);

     
    //Now lets multiply the resulting q_matrix with temp to get the
    //regression estimate
    
    Matrix temp2;
    la::MulInit(q_matrix,temp,&temp2);  
    estimate=temp2.get(0,0);
    // printf("regression estimate found...\n");
   
    //To calculate the second deg of freedom in case it is a reference
    //set calculations and for C.I estimate in case it is a query side
    //calculation, we need to calculate the sqd length of weight diagram

    sqdlength=CalculateSqdWeightDiagramLocalLinear_(q_point);
    //printf("Calculated sqd length also..\n");

    if(flag==CALCULATE_FOR_REFERENCE_POINTS_){
      //Also if we are doing regression on reference points then we
      //need to cacluclate the influence
     
      influence=CalculateInfluenceForLocalLinear_(q_point,q);
    }
    
  }

  //This will perform KNN based NWR Regression on the q_matrix using
  //r_matrix. also note that if we are caclualting the regression
  //estiamtes on the reference set for calculating statistics of the
  //reference set then we also calculate the first and second degerees
  //of freedom
 
  void PerformKNNNWRegression_(Matrix &q_matrix,index_t flag){

    //So we have the knn of all the points in q_matrix and the kth
    //nearest neighbours of all the points in q_matrix

    index_t number_of_points=q_matrix.n_cols();

    Vector weight_diagram;
    weight_diagram.Init(k_);
    for(index_t q=0;q<number_of_points;q++){

      double numerator=0;
      double denominator=0;
      weight_diagram.SetAll(0);
      
      for(index_t l=0;l<k_;l++){
	//for each of the knn of the point q we find out the kth nn
	//distance
	
	//The index of the lth nearest neighbour
	index_t knn_point=nn_neighbours_[q*k_+l];
	
	//printf("Distance of this nearest neighbour %f\n",nn_distances_[q*k_+l]);

	//the distance of the kth nearest neighbour of knn_point. This
	//is the radius of the kernel
	double dist=kth_nn_distances_[knn_point];
	
	double bw=global_smoothing_*sqrt(dist);
	kernel_.Init(bw);

	weight_diagram[l]=kernel_.EvalUnnormOnSq(nn_distances_[q*k_+l])/
	  kernel_.CalcNormConstant(number_of_dimensions_);

	double new_numerator=rset_weights_.get(knn_point,0)
	  *kernel_.EvalUnnormOnSq(nn_distances_[q*k_+l])/
	  kernel_.CalcNormConstant(number_of_dimensions_);

	numerator+=new_numerator;

	double new_denominator=kernel_.EvalUnnormOnSq(nn_distances_[q*k_+l])/
	  kernel_.CalcNormConstant(number_of_dimensions_);

	denominator+=new_denominator;
	//printf("bw is %f\n",bw);
	//printf("New numerator is %lf\n",new_numerator);
	//printf("New denominator is %lf\n",new_denominator);
	//printf("Unnoralized value is %lf\n",kernel_.EvalUnnormOnSq(nn_distances_[q*k_+l]));
	//printf("normalization const is %lf\n", 1/kernel_.CalcNormConstant(number_of_dimensions_));
      }
      
      regression_estimates_[q]=numerator/denominator;
      //printf("the point being considered is %d\n",q);
      //printf("numerator was %f\n",numerator);
      printf("denominator is %f\n",denominator);
      printf("regression estimates are %f\n",regression_estimates_[q]);
      if(isnan(regression_estimates_[q])||isinf(regression_estimates_[q])){

	printf("regression estimate is inf!!!!");
	exit(0);

      }
      //printf("\n");
      
      for(index_t i=0;i<k_;i++){
	
	weight_diagram[i]/=denominator;
      }
    
      double sqdlength=SquaredLengthOfVector_(weight_diagram,k_); 


      //Calculate cross validation score. for cross validation we need
      //to discard the contribution of its own observation in the
      //final regression value

      if(flag==CALCULATE_FOR_REFERENCE_POINTS_){


	/*******************Set df1 and df2 here ****************************************/



	//printf("weight diagram caclulated as %f...\n",weight_diagram[0]);
	df1_+=weight_diagram[0];
	
	if(weight_diagram[0]>1.0){
	  
	  printf("the influence was too heavy...\n");
	  exit(0);
	}
	
	//printf("df1 has become %f\n",df1_);
	if(isnan(weight_diagram[0])){
	  printf("weight diagram is nan\n");
	  exit(0);
	}

	df2_+=sqdlength;

	/*********************************************************************/
	
	double bw=global_smoothing_*sqrt(nn_distances_[q*k_+k_-1]);
	kernel_.Init(bw);
	double norm_constant=(1/kernel_.CalcNormConstant(number_of_dimensions_));
	double denominator_cross_validation=denominator-norm_constant;

	double numerator_cross_validation=numerator-
	  (rset_weights_.get(q,0)*norm_constant);


	printf("numerator cross validation is %g\n",numerator_cross_validation);
	printf("will subtract %g\n",rset_weights_.get(q,0)*norm_constant);

       
	printf("denominator cross validation is %f\n",denominator_cross_validation);

	double regression_estimate_cross_validation;



	//If the numerator of cross validation and denominator are both 0
	if(numerator_cross_validation==0){
	  printf("Yes numerator is small..\n");
	}


	if(abs(numerator_cross_validation)<=pow(10,-10) && abs(denominator_cross_validation)<=pow(10,-10)){

	  printf("regression estimate for cross validation has been made 0\n");
	  regression_estimate_cross_validation=0;
	}

	else{
	  regression_estimate_cross_validation=
	    numerator_cross_validation/denominator_cross_validation;
	}


	double diff=regression_estimate_cross_validation-rset_weights_.get(q,0);
       
	cross_validation_score_+=diff*diff;

	printf("Cross validation score has become %f\n",cross_validation_score_);
      }

      //We shall calculate the C.I. for both the refererence set as
      //well as the query set of points
      //printf("Storing things in weight diagram sqd length..\n");

      sqdlength_of_weight_diagram_[q]=sqdlength;      
      
    }

    //Performed NWR on all the reference points........................
    if(flag==CALCULATE_FOR_REFERENCE_POINTS_){
      cross_validation_score_/=rset_.n_cols();
      cross_validation_score_=sqrt(cross_validation_score_);
    }
  }


  void CalculateSigmaHat_(const char *method){

    //For this we need to calculate the regression estimates of the
    //reference set first using local fitting methods

    //printf("Method is %s\n",method);
    if(!strcmp(method,"nwr")){

      //Performing regression on the reference set. Please note this
      //function also finds out the first and second degrees of
      //freedom

      //printf("Came to compute sigma hat of nwr calc...\n");

      //Lets perform KNN Based NWR Regression However before we do
      //that lets initialize the vector regression_estimates

      index_t flag=CALCULATE_FOR_REFERENCE_POINTS_;
      PerformKNNNWRegression_(rset_,flag);
      //printf("KNN Regression performed.......\n");

      //So we now have the NWR fits on the reference points.We need to
      //calcualte the sqd residual, by finding out the sqd difference
      //between the regression estiamtes found by NWR and the
      //observation values given

      double sqd_residual_error=0;
      for(index_t r=0;r<rset_.n_cols();r++){

	double diff=regression_estimates_[r]-rset_weights_.get(r,0);
	sqd_residual_error+=diff*diff;
      }

      printf("sqd residual error is %f\n",sqd_residual_error);
      printf("denominator is %f\n",rset_.n_cols()-2*df1_+df2_);

      if(abs(rset_.n_cols()-2*df1_+df2_)<=pow(10,-10)){
	printf("Too few neighbours chosen,,,exiting....\n");
	exit(0);
      }
      sigma_hat_=sqrt(sqd_residual_error/(rset_.n_cols()-2*df1_+df2_));     

      printf("degrees of freedom1 are %f\n",df1_);
      printf("degrees of freedom2 are %f\n",df2_);
      printf("sigma_hat is %f\n",sigma_hat_);
      printf("cross validation score is %f\n",cross_validation_score_);
      
    }

    //Do local linear fitting...............................................
    else{

      //printf("came to sigm ahat of local linear calc..\n");
      
      //We have found out the k-nearest neighbours of all the
      //reference points 
      
      double sqd_residual_error=0;

      //Initialize the regression estimates vector
      //regression_estimates_.Init(rset_.n_cols());

      for(index_t r=0;r<rset_.n_cols();r++){

	double influence=0;
	double sqdlength=0;
	double estimate=0;

	index_t flag=CALCULATE_FOR_REFERENCE_POINTS_;
	//We first perform local linear regression on
	//the reference set
	PerformKNNLocalLinearRegression_(rset_.GetColumnPtr(r),r,flag,influence,sqdlength,estimate);

	regression_estimates_[r]=estimate;
	df1_+=influence;

	df2_+=sqdlength;

	//printf("sqdlength was %f\n",sqdlength);
	//printf("df2 becomes %f\n",df2_);
	sqdlength_of_weight_diagram_[r]=sqdlength; 

	//Flush b_twb and b_twy and b_tw2b_ matrices for new computations
	b_twb_.SetZero();
	b_twy_.SetZero();
	b_tw2b_.SetZero();
      
	//With the above function call we have the regression estimates
	//of the different reference points
	
	//sigma_hat is nothing but the mean squared residuals
	
	double diff=estimate-rset_weights_.get(r,0);
	sqd_residual_error+=diff*diff;
	//printf("sqd residual error is %f\n",sqd_residual_error);
      }

      printf("Squared residual errror is %f\n",sqd_residual_error);

      if(abs(rset_.n_cols()-2*df1_+df2_)){

	printf("too few neighbours chose....exiting....\n");
	exit(0);

      }
      sigma_hat_=sqd_residual_error/(rset_.n_cols()-2*df1_+df2_);  

      if(sigma_hat_<0){

	sigma_hat_=0;
      }  	
      else{
	sigma_hat_=sqrt(sigma_hat_);
      }
      printf("degrees of freedom1 are %f\n",df1_);
      printf("degrees of freedom2 are %f\n",df2_);
      printf("sigma_hat is %f\n",sigma_hat_);

      cross_validation_score_/=rset_.n_cols();
      cross_validation_score_=sqrt(cross_validation_score_);
    }
  }


  //Here we get the statistics from the reference dataset. This will
  //involve calculation of the following
  //1) sigma_hat and the degrees of freedom
  //2) knn of all the reference points
  

    void GetStatisticsOfReferenceSet_(const char *method){

    //First lets find the nearest neighbours of all the reference
    //points

      AllkNN *all_knn;
      all_knn=new AllkNN();
      
      //printf("allknn object initialized...\n");
      //Initialize the object and call compute function

      fx_timer_start(NULL,"reference_nbs");
      all_knn->Init(rset_,rset_,LEAF_SIZE,k_);
      all_knn->ComputeNeighbors(&nn_neighbours_,&nn_distances_);
      fx_timer_stop(NULL,"reference_nbs");


      //printf("Nearset neighbouurs computed..\n");

      //Note for variable bandwidth we need the kth nearest neighbour
      //for each reference point. We have calculated the k-nearest
      //neighbours of all the points in the above steps. We shall store
      //them in the vector kth_nn_distances_reference_points
      
      kth_nn_distances_.Init(rset_.n_cols());
      for(index_t l=0;l<rset_.n_cols();l++){
	
	kth_nn_distances_[l]=nn_distances_[(l+1)*k_-1];
      }

      //printf("Kth neareast neighbours all found..for ref points\n");

      fx_timer_start(NULL,"calculate_sigma_hat");
      CalculateSigmaHat_(method);

      CalculateConfidenceIntervalOfReferencePoints_();
      //With this we have completed our reference side
      //calculations. we need to have these matrices uninitialized for
      //query side calculations. hence lets destruct them
      nn_neighbours_.Destruct();
      nn_distances_.Destruct();
      sqdlength_of_weight_diagram_.Destruct();
      regression_estimates_.Destruct();
      delete(all_knn);

      fx_timer_stop(NULL,"calculate_sigma_hat");

    }


    void CompareWithNaive_(const char *method){

      //We shall make 3 comparisons
      //1) Compute the max relative error of knn regression w.r.t naive
      //2) Calculate RMSE of knn regression w.r.t naive
      //3) Compute the diff in confidence band estimates of knn w.r.t naive

      //lets first declare and initialize a naive lpr object


      printf("came to compare function...\n");
      NaiveLpr<TKernel> naive_lpr;

      printf("Method is %s\n",method);
      
      struct datanode* lpr_module =
	fx_submodule(NULL, "lpr", "lpr_module");
      
   

      Matrix rset_weights_column;
      la::TransposeInit(rset_weights_,&rset_weights_column);
      if(!strcmp(method,"nwr")){
	index_t lpr_order=0;
	naive_lpr.Init(rset_,rset_weights_column,lpr_module,lpr_order,k_);
      }
      else{
	index_t lpr_order=1;
	naive_lpr.Init(rset_,rset_weights_column,lpr_module,lpr_order,k_);
      }
      //printf("Naive lpr initialized..\n");

     
      //Lets call the compute function
      Vector regression_estimates_query_naive;
      ArrayList<DRange> query_confidence_bands_naive;
      Vector query_magnitude_weight_diagrams_naive; 
      naive_lpr.Compute(qset_, &regression_estimates_query_naive,
			&query_confidence_bands_naive,
			&query_magnitude_weight_diagrams_naive);
	
	//With this naive lpr caclulations are all over. We shall now
	//perform the comparisons
	
	//get the maximum relative difference in our regression
	//estimates
      
      max_relative_error_regression_estimates_query_=
	MatrixUtil::
	MaxRelativeDifference(regression_estimates_query_naive,
			      regression_estimates_);
     
      //printf("query regression estimates as per my methods are\n");
      //regression_estimates_.PrintDebug();

      //printf("query regression estimates as per naive method are..\n");
      //regression_estimates_query_naive.PrintDebug();

      printf("Reference regression estimates as per naive method are..\n");
      Vector regression_estimates_reference_naive;
      naive_lpr.get_regression_estimates(&regression_estimates_reference_naive);      
      //regression_estimates_reference_naive.PrintDebug();


          
      printf("Max relative error in regression estimates of query is %f\n",
	     max_relative_error_regression_estimates_query_);

      //We next get the max relative diff of knn based regression w.rt. naive
      

	/****************************Query side calculations****************************/
      Vector upper_bounds_knn_query;
      upper_bounds_knn_query.Init(qset_.n_cols());
      
      Vector upper_bounds_naive_query;
      upper_bounds_naive_query.Init(qset_.n_cols());

      Vector lower_bounds_knn_query;
      lower_bounds_knn_query.Init(qset_.n_cols());

      Vector lower_bounds_naive_query;
      lower_bounds_naive_query.Init(qset_.n_cols());

      for(index_t l=0;l<qset_.n_cols();l++){
	lower_bounds_knn_query[l]=confidence_interval_query_points_[2*l];
	upper_bounds_knn_query[l]=confidence_interval_query_points_[2*l+1];

	//printf("C.I %d is %f\n",l,lower_bounds_knn[l]);
	//printf("C.I.%d is %f\n",l,lower_bounds_knn[l]);

      }
      for(index_t l=0;l<qset_.n_cols();l++){
	lower_bounds_naive_query[l]=query_confidence_bands_naive[l].lo;
	upper_bounds_naive_query[l]=query_confidence_bands_naive[l].hi;
	//printf("lower_bounds_naive_query is %f\n",lower_bounds_naive_query[l]);
        //printf("Upper bounds naive is %f\n",upper_bounds_naive_query[l]);
	
      }


      //Having separated the lower and upper bounds lets compare

    

       max_relative_error_confidence_interval_lower_query_=
	MatrixUtil::MaxRelativeDifference(lower_bounds_naive_query,lower_bounds_knn_query);

       max_relative_error_confidence_interval_upper_query_=
	MatrixUtil::MaxRelativeDifference(upper_bounds_naive_query,upper_bounds_knn_query);

       average_relative_error_confidence_interval_lower_query_=
	MatrixUtil::AverageRelativeDifference(lower_bounds_naive_query,lower_bounds_knn_query);
      
       average_relative_error_confidence_interval_upper_query_=
	MatrixUtil::AverageRelativeDifference(upper_bounds_naive_query,upper_bounds_knn_query);

       printf("On comparison with naive i have..\n");
       printf("Max relative error in regression estimates is %f\n",
	      max_relative_error_regression_estimates_query_);
       printf("Max rel err in lower bound of CI of query is %f\n",
	      max_relative_error_confidence_interval_lower_query_);
       printf("Max rel err in upper bound of CI for query is %f\n",
	      max_relative_error_confidence_interval_upper_query_);
       printf("average relative error lower for query is %f \n",
	      average_relative_error_confidence_interval_lower_query_);
       printf("average relative error upper for query is %f\n",
	      average_relative_error_confidence_interval_upper_query_);
       /***************************************************************************/
       
       //**********************REFERENCE SIDE CALCULATIONS************************/

       // max_relative_error_regression_estimates_reference_=
       //MatrixUtil::MaxRelativeDifference(regression_estimates_reference_naive,
       //			   regression_estimates_reference_);

       //printf("Maximum relative error of regression estimates on the reference set is %f\n",
       //     max_relative_error_regression_estimates_reference_);

       //C.I for the reference points by knn based methods

       Vector lower_bounds_knn_reference;
       Vector upper_bounds_knn_reference;

       lower_bounds_knn_reference.Init(rset_.n_cols());
       upper_bounds_knn_reference.Init(rset_.n_cols());


      
       for(index_t l=0;l<rset_.n_cols();l++){
	 lower_bounds_knn_reference[l]=confidence_interval_reference_points_[2*l];
	 upper_bounds_knn_reference[l]=confidence_interval_reference_points_[2*l+1];
      }

      
       //C.I for the reference points by naive based methods

       ArrayList<DRange> confidence_interval_reference_points_naive;
       naive_lpr.get_confidence_bands(&confidence_interval_reference_points_naive);
       printf("Length is %d\n",confidence_interval_reference_points_naive.size());

    
       Vector lower_bounds_naive_reference;
       Vector upper_bounds_naive_reference;

       lower_bounds_naive_reference.Init(rset_.n_cols());
       upper_bounds_naive_reference.Init(rset_.n_cols());
	 
       for(index_t l=0;l<rset_.n_cols();l++){
	 lower_bounds_naive_reference[l]=confidence_interval_reference_points_naive[l].lo;
	 upper_bounds_naive_reference[l]=confidence_interval_reference_points_naive[l].hi;
       }

       printf("all initializations done. Will do comparisons now..\n");

       //Now lets compare them

      max_relative_error_confidence_interval_upper_reference_=
	MatrixUtil::MaxRelativeDifference(upper_bounds_naive_reference,upper_bounds_knn_reference);

      max_relative_error_confidence_interval_lower_reference_=
	MatrixUtil::MaxRelativeDifference(lower_bounds_naive_reference,lower_bounds_knn_reference);
      
      printf("Maximum relative error upper of ref is %f\n",max_relative_error_confidence_interval_upper_reference_);

      printf("Maximum relative error lower of ref is %f\n",max_relative_error_confidence_interval_lower_reference_);

      average_relative_error_confidence_interval_lower_reference_=
	MatrixUtil::AverageRelativeDifference(lower_bounds_naive_reference,lower_bounds_knn_reference);
      
      average_relative_error_confidence_interval_upper_reference_=
	MatrixUtil::AverageRelativeDifference(upper_bounds_naive_reference,upper_bounds_knn_reference);

      printf("average relative error of upper bounds for reference points is %f\n",
	     average_relative_error_confidence_interval_upper_reference_);

      printf("average relative error of lower bounds for reference points is %f\n",
	     average_relative_error_confidence_interval_lower_reference_);  
    }


    void PrintDebug_(){

      FILE *fp;
      fp=fopen("knn_regression_results.txt","w+");
      //Lets Print the regression estimates of all the query points
      for(index_t q=0;q<qset_.n_cols();q++){

	fprintf(fp,"%f,[%f, %f]",regression_estimates_[q],
		confidence_interval_query_points_[2*q],confidence_interval_query_points_[2*q+1]);

	fprintf(fp,"\n");

      }
       
       fprintf(fp,"Max relative error in regression estimates is %f\n",
	      max_relative_error_regression_estimates_query_);

       fprintf(fp,"Max rel err in lower bound of CI of query is %f\n",
	      max_relative_error_confidence_interval_lower_query_);

       fprintf(fp,"Max rel err in upper bound of CI for query is %f\n",
	      max_relative_error_confidence_interval_upper_query_);

       fprintf(fp,"average relative error lower for query is %f \n",
	      average_relative_error_confidence_interval_lower_query_);

       fprintf(fp,"average relative error upper for query is %f\n",
	      average_relative_error_confidence_interval_upper_query_);

       fprintf(fp,"Maximum relative error upper of ref is %f\n",max_relative_error_confidence_interval_upper_reference_);
 
      fprintf(fp,"Maximum relative error lower of ref is %f\n",max_relative_error_confidence_interval_lower_reference_);

      fprintf(fp,"average relative error of upper bounds for reference points is %f\n",
	     average_relative_error_confidence_interval_upper_reference_);
      
      fprintf(fp,"average relative error of lower bounds for reference points is %f\n",
	     average_relative_error_confidence_interval_lower_reference_);  

      fprintf(fp,"cross_validation_score is %f\n",cross_validation_score_);


      fprintf(fp,"degree of freedom1 is %f\n",df1_);

      fprintf(fp,"degree of freedom2 is %f\n",df2_);

      fprintf(fp,"sigma hat is %f\n",sigma_hat_);

      fclose(fp);
      
    }
    
 public:
    
    void Compute(const char *method){
      
      printf("came to compute...\n");
      printf("method is %s\n",method);
      if(!strcmp(method,"nwr")){
	
	//Before we do the actual computations, lets calculate statistics
	//of the reference data.
	printf("Method was nwr...\n");

	//This method has been timed
	GetStatisticsOfReferenceSet_(method);
	printf("Got reference statistics..\n");

	//Now lets perform query side caclulations Before we do that
	//we need to set up the nearest neighbours

	fx_timer_start(NULL,"query_side_nbs");

	AllkNN *all_knn;
	all_knn=new AllkNN();
	
	printf("allknn object initialized...\n");
	//Initialize the object and call compute function
	all_knn->Init(qset_,rset_,LEAF_SIZE,k_);
	all_knn->ComputeNeighbors(&nn_neighbours_,&nn_distances_);
	fx_timer_stop(NULL,"query_side_nbs");



	/**********************Initialize destructed quantities***************************/
	//Also note the vector sqdlength_of_weight_diagram_ which was
	//destructed after reference side calculations will now have
	//to be initialized
	
	sqdlength_of_weight_diagram_.Init(2*qset_.n_cols());

	//Also the vector regression_estimates has beeen destructed. So lets initialize it now
	regression_estimates_.Init(qset_.n_cols());

	/**************************Initializes the destructed quantitites************************/

	index_t flag=CALCULATE_FOR_QUERY_POINTS_;

	printf("df1 is %f\n",df1_);
	printf("df2 is %f\n",df2_);

	printf("WILL PERFORM REGRESSION FOR QSET..\n");

	fx_timer_start(NULL,"knn_nwr");
	PerformKNNNWRegression_(qset_,flag);
	printf("df1 is %f\n",df1_);
	printf("df2 is %f\n",df2_);

	printf("Performed regression for qset too..\n");


	//With this we have the regression estiamtes at all the query
	//points and the values of ||l(x)||^2 for each and qvery query
	//point. We now calculate the confidence interval for weach
	//query point

	CalculateConfidenceInterval_();
	delete(all_knn);
	fx_timer_stop(NULL,"knn_nwr");
      }
      else{

	//Method is local linear 

	GetStatisticsOfReferenceSet_(method);



      
	//Now let us perform query side calculations. before we do
	//that we need to find the nearest neighbours of the query
	//points


	AllkNN *all_knn;
	all_knn=new AllkNN();

	fx_timer_start(NULL,"query_side_nbs");
	all_knn->Init(qset_,rset_,LEAF_SIZE,k_);
	all_knn->ComputeNeighbors(&nn_neighbours_,&nn_distances_);
	fx_timer_stop(NULL,"query_side_nbs");
	
	fx_timer_start(NULL,"knn_loc_linear");
	/**********************Initialize destructed quantities***************************/
	//Also note the vector sqdlength_of_weight_diagram_ which was
	//destructed after reference side calculations will now have
	//to be initialized
	
	sqdlength_of_weight_diagram_.Init(2*qset_.n_cols());

	//Also the vector regression_estimates has beeen destructed. So lets initialize it now
	regression_estimates_.Init(qset_.n_cols());


	

	/**************************Initializes the destructed quantitites************************/


	//Also note the vector sqdlength_of_weight_diagram_ which was
	//destructed after reference side calculations will now have
	//to be initialized

	sqdlength_of_weight_diagram_.Init(2*qset_.n_cols());
	for(index_t q=0;q<qset_.n_cols();q++){

	double influence=0;
	double sqdlength=0;
	double estimate=0;

	//Flush b_twb and b_twy and b_tw2b_ matrices for new computations
	b_twb_.SetZero();
	b_twy_.SetZero();
	b_tw2b_.SetZero();
	
	
	index_t flag=CALCULATE_FOR_QUERY_POINTS_;
	//the reference set
	
	//printf("Will Perform  KNN local linear regression on the query set..\n");
	//printf("Number of query points %d\n",qset_.n_cols());
	
	//This method is timed.............


	PerformKNNLocalLinearRegression_(qset_.GetColumnPtr(q),q,flag,
					 influence,sqdlength,estimate);

       

	//sqd length is useful for C.I estimates, and estimate is the
	//regression estimate at the query point
	
	
	//influence is no longer useful hence we are not considering it any further
	
	sqdlength_of_weight_diagram_[q]=sqdlength;
	
	
	//regression estimates of the query points
	regression_estimates_[q]=estimate;
	     
	}
	CalculateConfidenceInterval_();
	fx_timer_stop(NULL,"knn_loc_linear");
	delete(all_knn);

      }

    

      //So we have finisehd all our caclulations. Lets compare our results

      printf("Cross validation results are %f\n",cross_validation_score_);
      printf("Comparing with naive..\n");
      printf("degree of freeedom1 is %f\n",df1_);
      printf("degree of freedom2 is %f\n",df2_);
      CompareWithNaive_(method);

      /**************************PRINT RESULTS TO A FILE *******************************************/
      //printf("Priniting results to a file...\n");
      PrintDebug_();


    }

    void Init(double alpha, Matrix q_matrix, Matrix r_matrix,  
	      Matrix rset_weights, double global_smoothing){
      
    
      
    
      //Copy the qeuery and the reference matrices
      qset_.Copy(q_matrix);
      rset_.Copy(r_matrix);
      
      number_of_dimensions_=rset_.n_rows();

      //Set up the number of k-nearest neighbours
      k_=(int)(alpha*rset_.n_cols())+2;

      printf("K is %d\n",k_);
      
      //Copy the weights of the reference points. By weights we mean the
      //observed regression values at the reference points
      
      
      rset_weights_.Copy(rset_weights);

      global_smoothing_=1.0;
      
      //initialize the kernel with the global smoothing parameter
      kernel_.Init(global_smoothing_);
      
      
      //Lets now initialize all the data structures
      
      b_twb_.Init(number_of_dimensions_+1,number_of_dimensions_+1);
      b_tw2b_.Init(number_of_dimensions_+1,number_of_dimensions_+1);
      b_twy_.Init(number_of_dimensions_+1,1);
      b_twb_inv_.Init(number_of_dimensions_+1,number_of_dimensions_+1);

      regression_estimates_.Init(rset_.n_cols());

      //Set thses matrices all to 0

      b_twb_.SetZero();
      b_twy_.SetZero();
      b_tw2b_.SetZero();
      b_twb_inv_.SetZero();
      //Initialize the confidence interval. The confidence interval will
      //have the upper bound on the regression estimate and the lower
      //bound of the regression estimate for each query point. So it's
      //size will be twice the number of query points
      
      confidence_interval_query_points_.Init(2*qset_.n_cols());

      //Similarily for the reference points
      confidence_interval_reference_points_.Init(2*rset_.n_cols());

      sqdlength_of_weight_diagram_.Init(rset_.n_cols());
      printf("Everything nicely initialized..\n");
      cross_validation_score_=0;
      sigma_hat_=0;
      df1_=0;
      df2_=0;
       
      //Regression estimates of reference values

      //regression_estimates_reference_.Init(rset_.n_cols());
    }
};
#endif
