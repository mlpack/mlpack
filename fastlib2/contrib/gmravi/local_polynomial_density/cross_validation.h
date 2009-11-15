#ifndef CROSS_VALIDATION_H
#define CROSS_VALIDATION_H

#include "naive_kde_local_polynomial.h"
#include "naive_kde.h"
#include "naive_local_likelihood.h"

template <typename TKernel> class CrossValidation{


 private:

  //The matrix of referene points
  Matrix rset_;

  //The set of bandwidths in a  column major format
  Matrix bandwidth_;


  // I need a kernel for this code because a part of the naive kde
  // crossvalidation requires kernel summation
  TKernel kernel_; 
 
  NaiveLocalLikelihood <GaussianVectorKernel> naive_local_likelihood_;
  NaiveKde < GaussianKernel> naive_kde_;

  NaiveKdeLP <ErfDiffKernel> naive_kde_lp_;
  

  //Naive KDE cross validation score. This is an error function hence
  //we need that bandwidth where this error function takes the least
  //value. Similar is the case for local polynomial expansion

  double min_naive_kde_score_;
  double max_local_likelihood_cvs_;
  double min_naive_kde_lp_score_;

  index_t naive_kde_bandwidth_;
  index_t naive_kde_lp_bandwidth_;
  index_t likelihood_bandwidth_;

  //Store the cross validation scores in a vector so that we can plot
  //the graphs of them

  Vector cvs_naive_kde_lp_;
  Vector cvs_naive_kde_;
  Vector cvs_local_likelihood_;

  //Get the local likelihood cross validation score
  
  void GetLikelihoodCrossValidationScore_(index_t b){

    //The result of local likelihood density calculations
    Vector local_likelihood_density_results;
    local_likelihood_density_results.Init(rset_.n_cols());

    local_likelihood_density_results.SetZero();
    //We need \sum log \hat{f}_{-i}\left(X_i\right). Since local
    //likelihood density estimation already calculates this hence we
    //shall invoke the compute function on the local likelihood object

    //First set up the bandwidth
    
    naive_local_likelihood_.InitBandwidth(bandwidth_.GetColumnPtr(b));
    
    //Compute the densities at each and every point using local
    //likelihood methods
    
    naive_local_likelihood_.Compute();

    //Get the vector of densities using the objects getter method 
    naive_local_likelihood_.
      get_local_likelihood_densities(local_likelihood_density_results);

    //printf("Local likelihood density results are..\n");
    //  local_likelihood_density_results.PrintDebug();

    //Get the sum of log of these elements
    double likelihood_cvs=0;

    //printf("Local likelihood density results are...\n");
    //local_likelihood_density_results.PrintDebug();

    for(index_t i=0;i<local_likelihood_density_results.length();i++){
     
      likelihood_cvs+=log(local_likelihood_density_results[i]);
    }

    cvs_local_likelihood_[b]=likelihood_cvs;


    //If this score is greater than the present highest cvs then 
    if(likelihood_cvs > max_local_likelihood_cvs_){

      max_local_likelihood_cvs_=likelihood_cvs;
      likelihood_bandwidth_=b;
    }
   
    
    printf("Cross validation score for local likelihood is:%f\n",
	   likelihood_cvs);
  }

  //THis is the cross validation function for local least squares

  void GetCrossValidationScoreForLocalLeastSquares_(index_t b){

    //do this to get the local least squares density estimates
    naive_kde_lp_.InitBandwidth(bandwidth_.GetColumnPtr(b));
    naive_kde_lp_.Compute(); 

    Vector results;
    results.Init(rset_.n_cols());
    naive_kde_lp_.get_density_estimates(results);

    double int_f_f_hat=0; //this will store the total sum of densities
    
    double n=rset_.n_cols();
  
    //NOTE: WE ARE DEALING WITH A 1-D CASE ONLY. THE LOCAL LEAST
    //SQUARES EQUATIONS ARE NOT GENERAL

    double bw=bandwidth_.get(0,b);
    double total_value=0;

    for(index_t i=0;i<rset_.n_cols();i++){

      double *point_i=rset_.GetColumnPtr(i);
      
      //This is only for the 1-d case
      double x_i=point_i[0];
      
      for(index_t j=0;j<rset_.n_cols();j++){
	
	double *point_j=rset_.GetColumnPtr(j);
	
	//This is only for 1-D case
	
	double x_j=point_j[0];

	double diff_by_2h=(x_i-x_j)/(2*bw);

	double temp1=2*x_j*erf(diff_by_2h);
	double temp2=(bw/2.0-x_j)*erf(diff_by_2h+0.50)-
	  (x_j+bw/2)*erf(diff_by_2h-0.5);
	
	double temp3=2*x_i*erfc(diff_by_2h);
	double temp4=-(x_i+bw/2.0)*erfc(diff_by_2h+0.50)-
	  (x_i-bw/2.0)*erfc(diff_by_2h-0.50);
	
	double temp5=2*pow(math::E,-pow((diff_by_2h),2))-
	  pow(math::E,-pow(diff_by_2h+0.50,2))-
	  pow(math::E,-pow(diff_by_2h-0.50,2));
	
	double temp6=-(2*bw/sqrt(math::PI))*temp5;
	
	total_value+=temp1+temp2+temp3+temp4+temp6;
      }
      //Note that the code for local least squares expansion excludes
      //self contribution
      
      int_f_f_hat+=(2*results[i])/n;
    }
    
    double int_f_hat_sqd=total_value/(2*n*n*bw*bw);
    if(int_f_hat_sqd<0){

      printf("integral fhat sqd is %f\n",int_f_hat_sqd);
      exit(0);
    }
    else{
      // printf("Int f_hat_sqd is %f\n",int_f_hat_sqd);
    }
   
    double cross_validation_score_naive_kde_lp=
      int_f_hat_sqd-int_f_f_hat;
    
    printf("Cross validation score for naive kde lp is %f\n",
	   cross_validation_score_naive_kde_lp);

    cvs_naive_kde_lp_[b]=cross_validation_score_naive_kde_lp;
    
    if(cross_validation_score_naive_kde_lp<min_naive_kde_lp_score_){
      
      min_naive_kde_lp_score_=cross_validation_score_naive_kde_lp;
      naive_kde_lp_bandwidth_=b;
    } 
  }

  //This function does crossvalidation for KDE 

  void GetCrossValidationScoreForKDE_(index_t b){
    
    double bw=bandwidth_.get(0,b);

    //do this to get the kde estimates
    naive_kde_.InitBandwidth(bw);
    naive_kde_.Compute(); 

    Vector results;
    results.Init(rset_.n_cols());
    naive_kde_.get_density_estimates(results); 

    double int_f_hat=0;
    double int_f_hat_sqd=0;


    //THIS IS TRUE ONLY FOR A GAUSSIAN KERNEL
    kernel_.Init(2*bw);

    for(index_t i=0;i<rset_.n_cols();i++){

      for(index_t j=0;j<rset_.n_cols();j++){
	
	
	//Lets find the crossvalidation score for Naive KDE first....

	//This involves the convolution of the gaussian which is
	//another gaussian with twice the bandwidth
	

	
	double distance_sqd=la::DistanceSqEuclidean(rset_.n_rows(),
						rset_.GetColumnPtr(i),
						rset_.GetColumnPtr(j));

	int_f_hat_sqd+=kernel_.EvalUnnormOnSq(distance_sqd);
      }
      int_f_hat+=results[i];
    }

    index_t n=rset_.n_cols();
   
    //compute cross validation score
    double  cross_validation_score_naive_kde=bw*int_f_hat_sqd/(n*n)-2*int_f_hat/n;
    printf("Cross validation score for naive kde is %f\n",
	   cross_validation_score_naive_kde);
    
   
    cvs_naive_kde_[b]=cross_validation_score_naive_kde;
   
    if(cross_validation_score_naive_kde<min_naive_kde_score_){

      min_naive_kde_score_=cross_validation_score_naive_kde;
      naive_kde_bandwidth_=b;
    }

    //Set the bandwidth of the kernel back to h
    kernel_.Init(bw);
  }
  

  void PrintToFile_(){
    FILE *fp;
    fp=fopen("scores.txt","w+");
    for(index_t i=0;i<bandwidth_.n_cols();i++){
      
      //The cvs for local likelihood is being divided by 1000 so as to
      //maintain the scales
      fprintf(fp,"%f %f %f %f",
	      bandwidth_.get(0,i),cvs_naive_kde_[i],
	      cvs_naive_kde_lp_[i],
	      cvs_local_likelihood_[i]/1000);
      fprintf(fp,"\n");
    }
  }

 public:
  
  void Compute(){


    min_naive_kde_score_=DBL_MAX;
    min_naive_kde_lp_score_=DBL_MAX;
    max_local_likelihood_cvs_=-DBL_MAX;
    
    //We have all the density objects calculated. hence perform
    //calculations on these density objects

    printf("Number of points in reference set are %d\n",rset_.n_cols());
    
    //For each bandwidth
    for(index_t b=0;b<bandwidth_.n_cols();b++){

      printf("Bandwidth is:%f........\n",bandwidth_.get(0,b));
      
      //Calculate the cross validation score for naive kde

      GetCrossValidationScoreForKDE_(b);

      //Caluclate the cross validation score for local least squares 

      GetCrossValidationScoreForLocalLeastSquares_(b);

      GetLikelihoodCrossValidationScore_(b);
    }

    printf("At the end i have...\n");

    printf("KDE bandwidth:%f\n",bandwidth_.get(0,naive_kde_bandwidth_));
    printf("Local least squares bandwidth: %f\n",
       bandwidth_.get(0,naive_kde_lp_bandwidth_));
    printf("Local likelihood bandwidth:%f\n",
       bandwidth_.get(0,likelihood_bandwidth_));


    printf("The vectors are ...\n");

    printf("Naive kde lp...\n");
    cvs_naive_kde_lp_.PrintDebug();

    printf("Local likelihood cvs is...\n");
    cvs_local_likelihood_.PrintDebug();

    printf("Naive kde  is...\n");
    cvs_naive_kde_.PrintDebug();


    PrintToFile_();
  }

  //This function takes advantage of the fact that the true densities
  //are known to us. Hence it directly compares the estimate with the
  //true density
   

  void Init(Matrix &references, Matrix &bandwidth){


    //Copy the reference and the bandwidth 

    rset_.Init(references.n_rows(),references.n_cols());
    rset_.CopyValues(references);

    bandwidth_.Init(bandwidth.n_rows(),bandwidth.n_cols());   
    bandwidth_.CopyValues(bandwidth);

    printf("Bandwidths to be used is...\n");

    bandwidth_.PrintDebug();
    //Initialize naive_local_likelihood_ object
    
    //Naive local likelihood object accepts a vector of bandwidth
    //values. So lets provide the first column for local likelihood
    //calculations

    printf("First bandwidth for cv is %f...\n",bandwidth.GetColumnPtr(0)[0]);
  
    naive_local_likelihood_.Init(references,references,
				 bandwidth.GetColumnPtr(0),
				 bandwidth.n_rows());

    //Initialize the local least square object

    naive_kde_lp_.Init(references,references,
		       bandwidth.GetColumnPtr(0),
		       bandwidth.n_rows());

    //Initialize the naive kde object

    naive_kde_.Init(references,references,bandwidth.get(0,0));

    min_naive_kde_score_=DBL_MAX;
    min_naive_kde_lp_score_=DBL_MAX;
    max_local_likelihood_cvs_=-DBL_MAX;


    //Initialize the vectors

    cvs_naive_kde_lp_.Init(bandwidth_.n_cols());
    cvs_naive_kde_.Init(bandwidth_.n_cols());
    cvs_local_likelihood_.Init(bandwidth_.n_cols());

  
  }
  
};
#endif
