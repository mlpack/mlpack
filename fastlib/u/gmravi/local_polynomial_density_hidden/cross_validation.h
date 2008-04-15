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

  TKernel kernel_;

  NaiveLocalLikelihood <GaussianVectorKernel> naive_local_likelihood_;

  NaiveKdeLP <ErfDiffKernel> naive_kde_lp_;
  

  //Naive KDE cross validation score. This is an error function hence
  //we need that bandwidth where this error function takes the least
  //value. Similar is the case for local polynomial expansion

  double min_naive_kde_score;
  double max_local_likelihood_cvs;
  double min_naive_kde_lp_score;

  index_t naive_kde_bandwidth;
  index_t naive_kde_lp_bandwidth;
  index_t likelihood_bandwidth;

  //Get the local likelihood cross validation score
  
  void GetLikelihoodCrossValidationScore_(index_t b){

    //The result of local likelihood density calculations
    Vector local_likelihood_density_results;
    local_likelihood_density_results.Init(rset_.n_cols());

    //We need \sum log \hat{f}_{-i}\left(X_i\right). Since local
    //likelihood density estimation already calculates this hence we
    //shall invoke the compute function on the local likelihood object

    //First set up the bandwidth
    
    //    Matrix temp_matrix;
    //temp_matrix.Alias(bandwidth_.GetColumnPtr(b),1,bandwidth_.n_rows());

    naive_local_likelihood_.InitBandwidth(bandwidth_.GetColumnPtr(b));
    
    //Compute the densities at each and every point using local
    //likelihood methods

    naive_local_likelihood_.Compute();

    //Get the vector of densities using the objects getter method 
    naive_local_likelihood_.
      get_local_likelihood_densities(local_likelihood_density_results);
    
    //Get the sum of these elements
    double likelihood_cvs=0;

    for(index_t i=0;i<local_likelihood_density_results.length();i++){

      likelihood_cvs+=local_likelihood_density_results[i];
    }

    //If this score is greater than the present highest cvs then 
    if(likelihood_cvs > max_local_likelihood_cvs){
      max_local_likelihood_cvs=likelihood_cvs;
      likelihood_bandwidth=b;

    }

    printf("Cross validation score for local likelihood is:%f\n",
	   likelihood_cvs);
    printf("........\n");
  }


  //THis is the cross validation function for local least squares

  void GetCrossValidationScoreForLocalLeastSquares_(index_t b){

 
    naive_kde_lp_.InitBandwidth(bandwidth_.GetColumnPtr(b));
    naive_kde_lp_.Compute(); //do this to get the local least squares density estimates
    Vector results;
    results.Init(rset_.n_cols());
    naive_kde_lp_.get_density_estimates(results);

    double f_hat=0; //this will store the total sum of densities
    double total_diff_erf_values=0;

    double n=rset_.n_cols();
  
    double bw=bandwidth_.get(0,b);
    for(index_t i=0;i<rset_.n_cols();i++){

      for(index_t j=0;j<rset_.n_cols();j++){

	//Now lets get the cross validation score for local least squares
	
	double upper_erf_value_i=
	  erf((sqrt(3/8.0)/bw)*(rset_.get(0,i)+bw/2.0));
	
	double lower_erf_value_i=
	  erf((sqrt(3/8.0)/bw)*(rset_.get(0,i)-bw/2.0));
	
	double upper_erf_value_j=
	  erf((sqrt(3/8.0)/bw)*(rset_.get(0,j)+bw/2.0));
	double lower_erf_value_j=
	  erf((sqrt(3/8.0)/bw)*(rset_.get(0,j)-bw/2.0));
	
	
	double diff_erf_values_i=
	  upper_erf_value_i-lower_erf_value_i;
	
	double diff_erf_values_j=
	  upper_erf_value_j-lower_erf_value_j;
	
	total_diff_erf_values+=
	  diff_erf_values_i*diff_erf_values_j;      
      }

      f_hat+=results[i];
    }
 
    double cross_validation_score_naive_kde_lp=
      (total_diff_erf_values/(sqrt(3)*n*n*bw*bw))-(2*f_hat/n);
    
    printf("Cross validation score for naive kde lp is %f\n",
	   cross_validation_score_naive_kde_lp);
    
    if(cross_validation_score_naive_kde_lp<min_naive_kde_lp_score){
      
      min_naive_kde_lp_score=cross_validation_score_naive_kde_lp;
      naive_kde_lp_bandwidth=b;
    } 
  }

  //This function does crossvalidation for both KDE as well as local
  //least squares density estimation. However note that the cross
  //validation score for local least squares is true only for the 1-D
  //case

  void GetCrossValidationScoreForKDE_(index_t b){
    
    double total_kernel_value=0;
    double total_convolution_value=0;    
    double bw=bandwidth_.get(0,b);

    for(index_t i=0;i<rset_.n_cols();i++){

      
      for(index_t j=0;j<rset_.n_cols();j++){
	
	
	//Lets find the crossvalidation score for Naive KDE first....
	
	kernel_.Init(bw);
	
	double distance=la::DistanceSqEuclidean(rset_.n_rows(),
						rset_.GetColumnPtr(i),
						rset_.GetColumnPtr(j));
	
	double unnorm_value=kernel_.EvalUnnormOnSq(distance);
	double norm_value=unnorm_value/
	  kernel_.CalcNormConstant(rset_.n_rows());
	
	total_kernel_value+=norm_value;
	
	
	//Now lets find the convolution value. For a gaussian kernel it
	//is just the kernel value with twice the bandwidth
	kernel_.Init(2*bw);
	
	double unnorm_convolution=kernel_.EvalUnnormOnSq(distance);
	double norm_convolution=unnorm_convolution/
	  kernel_.CalcNormConstant(rset_.n_rows());

	total_convolution_value+=norm_convolution;
	
      }
    }

    index_t n=rset_.n_cols();
   
    //compute cross validation score
    double  cross_validation_score_naive_kde=
      (-2/(n*n-n))* 
      (total_kernel_value/bw)+
      (total_convolution_value/(n*n*bw))+2/((n-1)*bw);
    
    
    printf("bandwidth:%f\n",bw);
    printf("Cross validation score for naive kde is %f\n",
	   cross_validation_score_naive_kde);
    
   
   
    if(cross_validation_score_naive_kde<min_naive_kde_score){

      min_naive_kde_score=cross_validation_score_naive_kde;
      naive_kde_bandwidth=b;
    }
  }

 public:
  
  void Compute(){
    
    //We have all the density objects calculated. hence perform
    //calculations on these density objects

    
    //For each bandwidth
    for(index_t b=0;b<bandwidth_.n_cols();b++){
      

      //Calculate the cross validation score for naive kde
      GetCrossValidationScoreForKDE_(b);

      //Caluclate the cross validation score for local least squares 
      GetCrossValidationScoreForLocalLeastSquares_(b);

      GetLikelihoodCrossValidationScore_(b);
    }

    printf("At the end i have...\n");

    printf("KDE bandwidth:%f\n",bandwidth_.get(0,naive_kde_bandwidth));
    printf("Local least squares bandwidth: %f\n",
	   bandwidth_.get(0,naive_kde_lp_bandwidth));
    printf("Local likelihodd bandwidth:%f\n",
	   bandwidth_.get(0,likelihood_bandwidth));
  }

  void Init(Matrix &references, Matrix &bandwidth){


    //Copy the reference and the bandwidth 

    rset_.Init(references.n_rows(),references.n_cols());
    rset_.CopyValues(references);

    bandwidth_.Init(bandwidth.n_rows(),bandwidth.n_cols());   
    bandwidth_.CopyValues(bandwidth);

    //Initialize naive_local_likelihood_ object
    
    //Naive local likelihood obeject accepts a vector of bandwidth
    //values. So lets provide the first column for local likelihood
    //calculations

    naive_local_likelihood_.Init(references,references,
				 bandwidth.GetColumnPtr(0),
				 bandwidth.n_rows());

    //Initialize the local least square object

    naive_kde_lp_.Init(references,references,
		       bandwidth.GetColumnPtr(1),
		       bandwidth.n_rows());

    min_naive_kde_score=DBL_MAX;
    min_naive_kde_lp_score=DBL_MAX;
    max_local_likelihood_cvs=DBL_MIN;

  }
  
};
#endif
