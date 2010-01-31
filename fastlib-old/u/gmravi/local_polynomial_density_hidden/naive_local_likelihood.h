#ifndef NAIVE_LOCAL_LIKELIHOOD_H
#define NAIVE_LOCAL_LIKELIHOOD_H
#include "fastlib/fastlib.h"
#include "vector_kernel.h"
#define EPSILON pow(10,-10)
template <typename TKernel> class NaiveLocalLikelihood{


  FORBID_ACCIDENTAL_COPIES(NaiveLocalLikelihood);

 private:

  /*The query set */

  Matrix qset_;

  /* The reference set */

  Matrix rset_;

  /*Vector of bandwidths */
  Vector bandwidth_;
  
  /** The kernel for calculations */
  
  TKernel kernel_;
  
  /** The vector of densities */
  Vector densities_;
  
  // Get the vector difference between vec1 and vec2 and return the
  // result in the vector diff
  
  void GetVectorDifference_(double *vec1, double *vec2,index_t length,
			    Vector &diff){
    
    for(index_t l=0;l<length;l++){
      
      diff[l]=vec1[l]-vec2[l]; 
    }
    
  }
  
 public:
  
  
  //Constructors and Destructors
  
  
  NaiveLocalLikelihood(){
    
    
  }
  
  
  ~NaiveLocalLikelihood(){
    
    
  }
  
  
  //get density estimate
  
  
  void get_local_likelihood_densities(Vector &results){
    
    //Remember results is uninitialized.....
    results.CopyValues(densities_);    
  }

  bool IsSamePoint_(Vector &diff){

    for(index_t i=0;i<diff.length();i++){

      if(fabs(diff[i])>EPSILON){
	return false;
      }
    }
    return true;

  }

  //////////User level functions////////////
 public:


  void Compare(Matrix &foreign_density, Vector &boundary_points){

    //For each query point find out the R.M.S.E, and for the boundary
    //points find out the R.M.S.E

    double mean_sqd_error=0;
    double boundary_mean_sqd_error=0;
    index_t number_of_boundary_points=0;

    for(index_t q=0;q<qset_.n_cols();q++){

      mean_sqd_error+=pow(densities_[q]-foreign_density.get(0,q),2);

      //If dataset is one dimensional
      if(qset_.n_rows()==1){

	if(qset_.get(0,q)< boundary_points[1]||  
	   qset_.get(0,q)>boundary_points[2]){

	  boundary_mean_sqd_error+=pow(foreign_density.get(0,q)-
				       densities_[q],2);  
	  number_of_boundary_points++;
	}
      }
    }

    mean_sqd_error/=qset_.n_cols();
    boundary_mean_sqd_error/=number_of_boundary_points;

    double root_mean_sqd_error=sqrt(mean_sqd_error);
    double boundary_root_mean_sqd_error=sqrt(boundary_mean_sqd_error);

    printf("RESULTS FOR LOCAL LIKELIHOOD .........\n");
    
    printf("Total root mean squared error is %f\n",root_mean_sqd_error);

    printf("Boundary root mean squared error is %f\n",
	   boundary_root_mean_sqd_error);

    printf("Number of boundary points are %d\n",number_of_boundary_points);
  }


  void Compute(){

    //we shall use the vector kernels for local likelihood based
    //calculations

    //printf("Bandwidth being used is...\n");
    //bandwidth_.PrintDebug();

    for(index_t q=0;q<qset_.n_cols();q++){

      //The query point
      double *q_col=qset_.GetColumnPtr(q);
	
      double total_kernel_value=0.0;
      Vector exponential_term_along_each_dimension;
      exponential_term_along_each_dimension.Init(rset_.n_rows());
      exponential_term_along_each_dimension.SetZero();

      for(index_t r=0;r<rset_.n_cols();r++){

	//The reference point
	double *r_col=rset_.GetColumnPtr(r);
	
	//To hold the differene of 2 vectors
	Vector diff;
	diff.Init(rset_.n_rows());
	GetVectorDifference_(r_col,q_col,rset_.n_rows(),diff); 

       
	double kernel_value=kernel_.EvalUnnormOnVectorDifference(diff);

	if(!IsSamePoint_(diff)){
	  kernel_value/=kernel_.EvalNormConstant(rset_.n_rows());
	  total_kernel_value+=kernel_value;

	  for(index_t dim=0;dim<rset_.n_rows();dim++){
	    
	    exponential_term_along_each_dimension[dim]+=
	      kernel_value*diff[dim]/pow(bandwidth_[dim],2);
	  }
	}
      }

      //Having iterated over all the reference points. we now have
      //both the numerator and the denominator. 

      for(index_t i=0;i<rset_.n_rows();i++){

	exponential_term_along_each_dimension[i]/=total_kernel_value;
      }

      //Calculate the density at the query point
      for(index_t dim=0;dim<rset_.n_rows();dim++){
	total_kernel_value*=
	  pow(math::E,-0.50*pow(exponential_term_along_each_dimension[dim],2)*
	      pow(bandwidth_[dim],2));
      }
      //unnormalized density is nothing but this total_kernel_value 

      densities_[q]=total_kernel_value;
    }

    //Divide the unnomalized density with the normalization constant
    
    double norm_constant=kernel_.EvalNormConstant(rset_.n_rows());
    for(index_t q=0;q<qset_.n_cols();q++){

      densities_[q]/=(rset_.n_cols()*norm_constant);
    }
  }

  void Init(Matrix &query, Matrix &references, double *bwidth,index_t len){

    qset_.Init(query.n_rows(),query.n_cols());
    rset_.Init(references.n_rows(),references.n_cols());
    bandwidth_.Init(len);

    qset_.CopyValues(query);
    rset_.CopyValues(references);
    bandwidth_.CopyValues(bwidth);

    printf("BANDWIDTH BEING USED FOR LOCAL LIKELIHOOD CALC IS\n");
    bandwidth_.PrintDebug();

    kernel_.Init(bandwidth_);
    densities_.Init(qset_.n_cols());
    densities_.SetZero();
  }

  //This is most useful when u want to do cross validation and hence
  //are using the code with the same datasets but different
  //bandwidths, and the bandwidth matrix has been initialized before
  void InitBandwidth(double *bwidth){

    kernel_.InitInitialized(bwidth);

  }
};
#endif


