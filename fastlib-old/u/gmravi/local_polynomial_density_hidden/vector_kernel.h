#ifndef VECTOR_KERNEL_H
#define VECTOR_KERNEL_H
#include "fastlib/fastlib.h"
class ErfDiffKernel{

 private:
  Vector bwidth_;

 public:

 
  void Init(Vector &bwidth){

    //Set up the bandwidth. use this function when bwidth has not been
    //initialized

    bwidth_.Init(bwidth.length());
    bwidth_.CopyValues(bwidth);
  }

  /*Evaluate the kernel function for the given vector difference */

  double  EvalUnnormOnVectorDiff(Vector &diff){
    
    //Given the vector difference between the ref and the query points
    //we need to calculate the product of the difference of the erf
    //fuctions along each and every direction
    
    double product=1.0;
    for(index_t i=0;i<diff.length();i++){

      //Find erf(((x_id-x_d)/h sqrt(2)+1/2 sqrt(2)) - 
      //((x_id-x_d)/h sqrt(2)-1/2sqrt(2)))

      double constant=1.0/(2*sqrt(2.0));
  
      //printf("Bandwidth is %f\n",bwidth_.get(0,i));
      double upper_erf_value=erf((diff[i]/(bwidth_[i]*sqrt(2.0)))+
				  constant);
      double lower_erf_value=erf((diff[i]/(bwidth_[i]*sqrt(2.0)))-
				 constant);

      //printf("Upper erf value is %f\n",upper_erf_value);
      //printf("Lower erf value is %f\n",lower_erf_value);

      double diff_erf_value=upper_erf_value-lower_erf_value;
      product*=diff_erf_value;

      //printf("erf of large value is %f\n",erf(DBL_MAX));
    }
    return product;  
  }


  double EvalNormConstant(double dimensions,
			  index_t number_of_contributions){

    
    double norm_const=1.0;
    for(index_t i=0;i<dimensions;i++){
      
      norm_const*=bwidth_[i]*2;
    }
    return number_of_contributions*norm_const;
  }
};

class GaussianVectorKernel{

 private:
  Vector bwidth_;
  
 public:
 void InitInitialized(double *bwidth){

    //Use this function when bwidth_ has already been initialized
    bwidth_.CopyValues(bwidth);

  }

  void Init(Vector &bandwidth){
    
    bwidth_.Init(bandwidth.length());
    bwidth_.CopyValues(bandwidth);
  }
  
  double EvalUnnormOnVectorDifference(Vector &diff){
    
    GaussianKernel kernel;

    double kernel_value=1.0;

    for(index_t i=0;i<diff.length();i++){
      
      kernel.Init(bwidth_[i]);
      kernel_value*=kernel.EvalUnnormOnSq(diff[i]*diff[i]);
    }
    return kernel_value;
  }

 double EvalNormConstant(index_t number_of_dimensions){

   double norm_const=1.0;
   for(index_t dim=0;dim<number_of_dimensions;dim++){
     
     norm_const*=sqrt(2*math::PI)*bwidth_[dim];
   }
   return norm_const;
 }
};


#endif
