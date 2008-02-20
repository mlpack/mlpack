#ifndef KNN_REGRESSION_H
#define KNN_REGRESSION_H
#define LEAF_SIZE 10
#define CALCULATE_FOR_REFERENCE_POINTS_ 0
#define CALCULATE_FOR_QUERY_POINTS_ 1

#include <fastlib/fastlib.h>
#include "allknn.h"
#include "pseudo_inverse.h"


//This is a templatized class

template<typename TKernel> class KNNRegression{

  private:

  //An array to hold the index of neighbours and distances for all the query points

  ArrayList<index_t> nn_neighbours_;
  ArrayList<double> nn_distances_;


  //The confidence intervals calculation requires us to calculate the
  //regression estimates at all the reference points too.

  ArrayList<index_t> nn_neighbours_reference_points_;
                   
  ArrayList<double> nn_distances_reference_points_;



  //The bandwidth to be used
  double bandwidth_;


  //The kind of kernel to be used
  TKernel kernel_;


  //The regression estimates for the query points
  Vector regression_estimates_;

  //The regression estimates for the reference points. This is done by
  //local fitting methods but now the query points are the same as the
  //reference points

  Vector regression_estimates_reference_points_;

  //The reference dataset

  Matrix rset_;
  
  //The query dataset

  Matrix qset_;
  
  //The reference values
  
  Vector rset_weights_;
  

  //The number of neighbours u want

  index_t k_;

  //Matrices that are required for BTWB and BTWY calculations for
  //Local linear calculations

  ArrayList <Matrix> b_twb_;
  ArrayList <Matrix> b_twy_;


  //They store the upper and lower bounds of the confidence intervals
  //for each query point
  Vector confidence_interval_knn_nwr_;
  Vector confidence_interval_knn_local_linear_;


  double sigma_hat_;


  void CalculateSigmaHatKNNWR_(){

   
    double sum_of_squared_residual_errors=0;
    double df1=0.0;
    double df2=0.0;

    for(index_t r=0;r<rset_.n_cols();r++){
      
      //For each reference point. Compute the kernel sum of its knn
      
      double sum=0;
      for(index_t l=0;l<k_;l++){
	
	sum+=kernel_.EvalUnnormOnSq(nn_distances_reference_points_[r*k_+l]);
      }

      //So we now have the sum of the kernel weights of all the
      //knn. We shall proceed to calculate the degrees of freedom usig
      //these values

      df1+=1/sum; //(1/sum) is the influence of the reference point on its own estimate
      df2+=1/(sum*sum);  

      //We also need to run over all the reference points and compute the
      //squared residual error.

       sum_of_squared_residual_errors+=
	(rset_weights_[r]-regression_estimates_reference_points_[r])*(rset_weights_[r]-regression_estimates_reference_points_[r]);
      
    }

    printf("degress of freedom1 are %f\n",df1);
    printf("degrees of freedom2 are %f\n",df2);

    sigma_hat_=sum_of_squared_residual_errors/(rset_.n_cols()-2*df1+df2);

  }
  
  double CalculateTheSqdLengthOfHatVectorKNNWR_(index_t q){

    double sum=0;

    for(index_t l=0;l<k_;l++){
      
      sum+=kernel_.EvalUnnormOnSq(nn_distances_[q*k_+l]);
    }

    double sqdlength=0;
    for(index_t l=0;l<k_;l++){
      
      double contribution_due_to_other_point=kernel_.EvalUnnormOnSq(nn_distances_[q*k_+l]);
      double ratio=contribution_due_to_other_point/sum;
      sqdlength+=ratio*ratio;
    }

    printf("The squared length of hat vector for the query point q=%d is %f\n",q,sqdlength);
    return sqdlength;
  }


  void ComputeConfidenceIntervalLocalLinear_(){

    //As usual we first fit regression values at the reference points

    index_t flag=CALCULATE_FOR_REFERENCE_POINTS_;

    GetTheKNNNegihbours_(flag);


    //perform local linear regression on the reference set of points
    KNNLocalLinearRegressionRegression_(flag);

    CalculateSigmaHatKNNLocalLinear_();

  }


  
  void ComputeConfidenceIntervalKNNNWR_(){
      
    //The C.I requires us to calculate the regression fits at all the
    //reference points. hence we first check if the query set and the
    //reference set are the same. If they are then we dont need to do
    //anything special. if they are not then we first calculate the
    //fits at all the refernece points and then go on tc calculate the
    //confidence intervals     
      
 
    //So now we need to do a local fitting at each and every reference
    //point

    
    printf("Came to calculated confidence intervals\n");
    index_t flag=CALCULATE_FOR_REFERENCE_POINTS_;
    GetTheKNNNeighbours_(flag);

    printf("Got the knn...\n");
    KNNNWRegression_(flag);
    printf("Did KNNNWR regression...\n");
    
    
    
    //We need to do two things to compute the C.I. of an point. First
    //we need to calculate the sigma_hat and then we need to calculate
    //the length of l(x) vector. Lets do them in 2 functions.Note that
    //the sigma_hat is not query dependent and is same for all the query points
    
    CalculateSigmaHatKNNNWR_();

    printf("sigma hat is %f\n",sigma_hat_);

    //The next important thing to do is to calculate the length of
    //||l(x)|| vector. Lets call this vector as the hat vector
    
    for(index_t q=0;q<qset_.n_cols();q++){
      
      double sqdlength= CalculateTheSqdLengthOfHatVectorKNNWR_(q);
      
      //Once we have the length of the hat vector we should be able to
      //calculate the upper and lower bounds of the C.I for the query
      //point
      double lower_bound=regression_estimates_[q]+1.96*sigma_hat_*(1+sqdlength);
      confidence_interval_knn_nwr_[q*2]=lower_bound;

      double upper_bound=regression_estimates_[q]+1.96*sigma_hat_*(1+sqdlength);
      confidence_interval_knn_nwr_[q*2+1]=upper_bound;

      printf("Uopper bound is %f\n",upper_bound);
      printf("lower bound id %f\n",lower_bound);
    }
  }



  void KNNNWRegression_(index_t flag){   
    
    //For each query point
    
    index_t number_of_points;
    if(flag==CALCULATE_FOR_QUERY_POINTS_){
      
      number_of_points = qset_.n_cols();
    }
    
    //we need to calculate the regression estimates at the reference
    //points too

    else{

      number_of_points = rset_.n_cols();
    }
    
    printf("Number of points are %d\n",number_of_points);
    
    printf("The weights are ...\n");
    rset_weights_.PrintDebug();

    for(index_t q=0; q<number_of_points; q++){
      printf("q is %d\n",q);
      
      float numerator=0;
      float denominator=0;

      //For each neighbour
      printf("abt to check flag...\n");
      if(flag==CALCULATE_FOR_QUERY_POINTS_){ 
	printf("Calculating for query points..\n");

	for(index_t l=0;l<k_;l++){
	  
	  numerator+=
	    rset_weights_[nn_neighbours_[q*k_+l]]*
	    kernel_.EvalUnnormOnSq (nn_distances_[q*k_+l]); 
	  
	  denominator+=kernel_.EvalUnnormOnSq(nn_distances_[q*k_+l]);
	}
	regression_estimates_[q]=numerator/denominator;
      }
      
      else{
	printf("calculating for reference points....\n");
	
	for(index_t l=0;l<k_;l++){

	  printf
	    ("Neighbours set..%d\n",nn_neighbours_reference_points_[q*k_+l]);  

	  numerator+=
	    rset_weights_[nn_neighbours_reference_points_[q*k_+l]]*
	    kernel_.EvalUnnormOnSq (nn_distances_reference_points_[q*k_+l]); 
	  
	  denominator+=
	    kernel_.EvalUnnormOnSq(nn_distances_reference_points_[q*k_+l]);
	}
	regression_estimates_reference_points_[q]=numerator/denominator;
	printf("regression estimate is %f\n",
	       regression_estimates_reference_points_[q]);
      }
    }
  }
  
  
  void ComputeMatrices(){
    
    for (index_t q = 0; q < qset_.n_cols (); q++){	//for each query point
      const double *q_col = qset_.GetColumnPtr (q);
      
      //for each reference point which are the k-nearest neighbours of
      //the query point
      
      for (index_t r = 0; r < k_; r++){ 
	
	//Get reference point
	const double *r_col = rset_.GetColumnPtr(nn_neighbours_[q*k_+r]);
	
	// pairwise distance and kernel value
	double dsqd =
	  la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	
	double ker_value = kernel_.EvalUnnormOnSq (dsqd);
	
	for(index_t col = 0; col < rset_.n_rows () + 1; col++){	//along each direction
	  
	  for(index_t row=0; row< rset_.n_rows () + 1; row++){
	    
	    
	    //Lets gradually fill up all the elements of the matrices 
	    //b_twy_ and b_twb_
	    
	    //Fill in b_twy_ first
	    
	    if(col==0){
	      
	      //For this column 
	      if (row != 0){
		
		//Fill B^TWY naive
		
		double val=b_twy_[q].get(row,col)+ 
		  ker_value * rset_weights_[nn_neighbours_[q*k_+r]] * 
		  rset_.get (row- 1,nn_neighbours_[q*k_+r]);
		
		b_twy_[q].set(row,col,val);
		
		//Now fill B^TWB naive
		
		double val1=b_twb_[q].get(row,col) + 
		  ker_value * rset_.get(row-1,nn_neighbours_[q*k_+r]);
		b_twb_[q].set(row,col,val1);
		
	      }
	      
	      else{
		
		//Fill B^TWY naive 
		
		double val= b_twy_[q].get(row,col) + 
		  ker_value * rset_weights_[nn_neighbours_[q*k_+r]];
		
		b_twy_[q].set(row,col,val);
		
		//Now fill B^TWB
		
		double val1=b_twb_[q].get(row,col) 
		  + ker_value ;
		
		b_twb_[q].set(row,col,val1);
		
		
	      }
	    }//end of col 0...............
	    
	    //Column!=0
	    
	    else{
	      if(row!=0){
		//Only B^TWB naive estimates get filled up
		double val1=b_twb_[q].get(row,col) 
		  + ker_value* 
		  rset_.get(row-1,nn_neighbours_[q*k_+r])* 
		  rset_.get(col-1,nn_neighbours_[q*k_+r]);
		
		b_twb_[q].set(row,col,val1);
	      }
	      
	      else{
		double val1=b_twb_[q].get(row,col) 
		  + ker_value* rset_.get(col-1,nn_neighbours_[q*k_+r]);
		b_twb_[q].set(row,col,val1);
		
	      }
	    }
	  }
	}
      }
    }
   }
  
  void KNNLocalLinearRegression(){
    
    //This will involve calculation of 2 matrices. One is BTWB and
    //another is BTWY.First lets compute BTWB

    ComputeMatrices();

    printf("Having computed these matrices i have...\n");

    for(index_t q=0;q<qset_.n_cols();q++){

      b_twb_[q].PrintDebug();
      b_twy_[q].PrintDebug();

    }
    //This returns the inverse of the matrix BTWB

      for(index_t q=0;q<qset_.n_cols();q++){
      PseudoInverse::FindPseudoInverse(qset_.n_cols(),b_twb_[q]);
      }

    //We now have both (B^TWB)-1 and (B^TWY). The Regression Estimates
    //will be obtained by simply multiplying these 2 matrices
    
    
    //So we now have for each query point the (B^TWB)-1 matri and B^TWY
    //matrix. In order to get the regression estimates we perform
    //y_hat= [1,q] (B^TWB)^-1 (B^TWY) where q are the coordinates of the query
    //point
    
    //lets perform these multiplications by using LaPack utilities
    
    //The vector temp will hold the multiplication of (B^TWB)^-1 and
    //(B^TWY)
    
    
    //Now lets form a vector q_vector using the coordinates of the query
    //point
    
    ArrayList<Matrix> temp;
    temp.Init(qset_.n_cols());
    
    for(index_t q=0;q<qset_.n_cols();q++){
      
      la::MulInit (b_twb_[q],b_twy_[q], &temp[q]);

      printf("The product is ..\n");
      temp[q].PrintDebug();

      Matrix q_matrix;
      
      //Initialize q matrix and set it up
      q_matrix.Init(1,qset_.n_rows()+1);
      q_matrix.set(0,0,1);
      
      for(index_t col=0;col<qset_.n_rows();col++){
	
	q_matrix.set(0,col+1,qset_.get(col,q));
      }
      
      
      //Now lets multiply the resulting q_matrix with temp to get the
      //regression estimate
      
      Matrix temp2;
      la::MulInit(q_matrix,temp[q],&temp2);  
      regression_estimates_[q]=temp2.get(0,0);
      
    }

    regression_estimates_.PrintDebug();
    }


  void GetTheKNNNeighbours_(index_t flag){
    

    //First lets get the knn of all the query points We shall
    //accomplish this by calling the functions related to the object
    
    AllkNN *all_knn;
    all_knn=new AllkNN();
    printf("The query set is ..\n");
    qset_.PrintDebug();

    if(flag==CALCULATE_FOR_QUERY_POINTS_){
      all_knn->Init(qset_,rset_,LEAF_SIZE,k_);
      //Now lets call the compute function.
      all_knn->ComputeNeighbors(&nn_neighbours_,&nn_distances_);
    }

    else{
      printf("Calculated the nearest neighbours for the reference points too...\n");

      all_knn->Init(rset_,rset_,LEAF_SIZE,k_);
      all_knn->ComputeNeighbors
	(&nn_neighbours_reference_points_,&nn_distances_reference_points_);

    }
  }



 public:
 
  
  void Compute(const char *method){
    

    //lets declare an object of type AllKNNDualTree to find out the
    //k-nearest neighbours by dual tree recurusion
    
    index_t flag=CALCULATE_FOR_QUERY_POINTS_;
    
    GetTheKNNNeighbours_(flag);
    
    
    if(!strcmp(method,"nwr")){
      
      //Do KNN based Nadaraya Watson regression
      KNNNWRegression_(flag);
      ComputeConfidenceIntervalKNNNWR_();
      
    }
    
    else{
      flag=CALCULATE_FOR_REFERENCE_POINTS_;
      
      //Do KNN based Local linear regression
      KNNLocalLinearRegression();
      ComputeConfidenceIntervalKNNLocalLinear_();

    }
    
  }

  void Init(index_t k, Matrix q_matrix, Matrix r_matrix, 
	    Vector rset_weights, double bandwidth){
    
    k_=k;
    bandwidth_=bandwidth;
   

    //Alias the query and reference datassets
    qset_.Copy(q_matrix);
    rset_.Copy(r_matrix);
    printf("The matrixes are...\n");
    qset_.PrintDebug();
    rset_.PrintDebug();

    //Alias the rset_weights. These are the regression values of the
    //reference points and are know to us

    rset_weights_.Copy(rset_weights);

    //Initialize the kerneL
    kernel_.Init(bandwidth);

  

    //Also lets initialize regression estimates
    regression_estimates_.Init(qset_.n_cols());


    //Lets initialize the regression estimates of the reference points
    //too

    regression_estimates_reference_points_.Init(rset_.n_cols());

    //Lets initialize the matrices for local linear estimates

    b_twb_.Init(qset_.n_cols());

    //Initialize each matrix now

    for(index_t q=0;q<qset_.n_cols();q++){

      b_twb_[q].Init(rset_.n_rows()+1,rset_.n_rows()+1);
      b_twb_[q].SetZero();

    }

    //similarily lets initialize BTWY too

    b_twy_.Init(qset_.n_cols());

    for(index_t q=0;q<qset_.n_cols();q++){
      
      b_twy_[q].Init(rset_.n_rows()+1,1);
      b_twy_[q].SetZero();
    }

    //Initialize the confidence intervals also. Their size is twice
    //the size of the query set so as to store both the upper and
    //lower bounds

    confidence_interval_knn_nwr_.Init(2*qset_.n_cols());
    confidence_interval_knn_local_linear_.Init(2*qset_.n_cols());


    //Set them to all 0's
    confidence_interval_knn_nwr_.SetZero();
    confidence_interval_knn_local_linear_.SetZero();

  }
};

#endif
