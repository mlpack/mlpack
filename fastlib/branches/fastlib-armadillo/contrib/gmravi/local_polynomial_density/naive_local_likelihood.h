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

    //printf("RESULTS FOR LOCAL LIKELIHOOD .........\n");
    
    //printf("Total root mean squared error is %f\n",root_mean_sqd_error);

    //    printf("Boundary root mean squared error is %f\n",
    //   boundary_root_mean_sqd_error);

    //printf("Number of boundary points are %d\n",number_of_boundary_points);
  }


  //Checks if the ref and the query point are the same or not by
  //checking the vector difference
  
  bool IsSame_(Vector &diff){

    for(index_t i=0;i<diff.length();i++){

      if(fabs(diff[i])>EPSILON){

	return false;
      }
    }
    return true;
  }


  void Compute(){
    
    //Along each direction

    double density;

    printf("In local likelihood calculations with bw=%f...\n",bandwidth_[0]);


    for(index_t q=0;q<qset_.n_cols();q++){//pick a query point

      double *q_col=qset_.GetColumnPtr(q);
      double exponential_term=0;

      Vector diff;
      diff.Init(qset_.n_rows());  

      double kernel_contrib=0;  

      index_t number_of_contributions=0;

      for(index_t k=0;k<bandwidth_.length();k++){

	number_of_contributions=0;
	
	double h_k=bandwidth_[k];
	double kernel_contrib_with_product=0.0;

	kernel_contrib=0;
	
	for(index_t i=0;i<rset_.n_cols();i++){ //For each reference point
    
	  double *r_col=rset_.GetColumnPtr(i); //Extract the ith reference point
	  
	  GetVectorDifference_(r_col,q_col,qset_.n_rows(),diff);
	  
	  if(!IsSame_(diff)){
	    
	    double kernel_val=kernel_.EvalUnnormOnVectorDifference(diff);
	    //printf("Kernel contribution to %d by %d is %f\n",q,i,kernel_val);
	    kernel_val/=kernel_.EvalNormConstant(rset_.n_rows());
	    double term_independent_of_j=(r_col[k]-q_col[k])/pow(h_k,2);
	    
	    kernel_contrib_with_product+=kernel_val*term_independent_of_j;
	    kernel_contrib+=kernel_val;
	    number_of_contributions++;
	  }
	}
	//printf("Simple kde density is %f\n",
	//kernel_contrib/number_of_contributions);
       
	
	if(fabs(kernel_contrib_with_product)>EPSILON && 
	   fabs(kernel_contrib)>EPSILON){
	  exponential_term+=
	    -0.5*pow((kernel_contrib_with_product/kernel_contrib)*h_k,2);
	}
	else{
	  if(fabs(kernel_contrib_with_product)<EPSILON && 
	     fabs(kernel_contrib)<EPSILON)
	    {

	      //if both quantites are small then the ratio of the 2
	      //quants is being considered as close to 0
	      exponential_term+=EPSILON; 
	      
	    }
	  else{
	    /*  printf("WRONG CALCULATIONS.........\n");

	    printf("kernel contrib with product is %f\n",
	    fabs(kernel_contrib_with_product));
	    
	    printf("kernel contrib is %f\n",
	    fabs(kernel_contrib));
	    if(fabs(kernel_contrib_with_product)>EPSILON){
	    
	    printf("kernel contrib with product is large\n");
	    }
	    
	    if(fabs(kernel_contrib)<EPSILON){
	    
	    printf("kernel contrib is small\n");
	    }*/

	    }
	}
     
      //printf("Number of contribution is %d\n",number_of_contributions);
      //printf("Kernel contrib is %f\n",kernel_contrib);
      //printf("Exponential term is %f\n",exponential_term);
     
      density=(kernel_contrib/number_of_contributions)*
	pow(math::E,exponential_term);

      //printf("the exponential shift is %f\n",pow(math::E,exponential_term));

      //printf("The density is %f\n",density);
      densities_[q]=density; 

      //reset number of contributions
      number_of_contributions=0;
      }
    }
  }

  void Init(Matrix &query, Matrix &references, double *bwidth,index_t len){

    qset_.Init(query.n_rows(),query.n_cols());
    rset_.Init(references.n_rows(),references.n_cols());
    bandwidth_.Init(len);

    qset_.CopyValues(query);
    rset_.CopyValues(references);
    bandwidth_.CopyValues(bwidth);


    printf("Bandwidth for local likelihood calculations is ...\n");
    bandwidth_.PrintDebug();

    //printf("BANDWIDTH BEING USED FOR LOCAL LIKELIHOOD CALC IS\n");
    //bandwidth_.PrintDebug();

    kernel_.Init(bandwidth_);
    densities_.Init(qset_.n_cols());
    densities_.SetZero();
  }

  //This is most useful when u want to do cross validation and hence
  //are using the code with the same datasets but different
  //bandwidths, and the bandwidth matrix has been initialized before

  void InitBandwidth(double *bwidth){
    //printf("BAndwidth being sent is %f......\n",bwidth[0]);
    kernel_.InitInitialized(bwidth);

    //reset the densities
    densities_.SetZero();

    bandwidth_.CopyValues(bwidth);
  }
};
#endif


