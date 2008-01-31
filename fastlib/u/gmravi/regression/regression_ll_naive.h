#ifndef REGRESSION_LL_NAIVE_H
#define REGRESSION_LL_NAIVE_H
#include "fastlib/fastlib_int.h"




/** This is the function which builds up the B^TWY 
 * vector by usimng matrix calculations. Note this can be 
 *  optimized by using fastlib's Lapack functions 
 */

template <typename TKernel>
void NaiveB_TWYCalculation<TKernel>::Compute (){
  
  // compute unnormalized sum
  for (index_t q = 0; q < qset_.n_cols (); q++){	//for each query point
    
    const double *q_col = qset_.GetColumnPtr (q);
    for(index_t row = 0; row < rset_.n_rows () + 1; row++){	//along each direction
      for(index_t col=0;col<1;col++){
	for (index_t r = 0; r < rset_.n_cols (); r++){	//for each reference point

	  //Get reference point
	  const double *r_col = rset_.GetColumnPtr (r);
	  
	  // pairwise distance and kernel value
	  double dsqd =
	    la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
	  double ker_value = kernel_.EvalUnnormOnSq (dsqd);
	 
	  if (row != 0){
	    
	    double val=b_twy_naive_estimate_[q].get(row,col)+ 
	      ker_value * rset_weights_[old_from_new_r_[r]] * rset_.get (row- 1, r);
	    
	    b_twy_naive_estimate_[q].set(row,col,val);
	    printf("value is %f\n",val);
	  }
	  else{
	    
	    double val= b_twy_naive_estimate_[q].get(row,col) + 
	      ker_value * rset_weights_[old_from_new_r_[r]];

	    b_twy_naive_estimate_[q].set(row,col,val);
	    //  printf("value is %f\n",val);
	    
	  }
	  
	}
      }
    }
  } 
}
template <typename TKernel>
void NaiveB_TWYCalculation<TKernel>::Init (Matrix & qset, Matrix & rset, 
	   ArrayList<index_t> &old_from_new_r,double bandwidth,Vector &rset_weights){
  
  // get datasets
  qset_.Alias (qset);
  rset_.Alias (rset);
  
  //get permutation
  old_from_new_r_.Copy(old_from_new_r);
  
  // get bandwidth
  kernel_.Init (bandwidth);
  
  //get the weights
  rset_weights_.Alias(rset_weights);   
  
  b_twy_naive_estimate_.Init (qset_.n_cols ());
  
  for (index_t i = 0; i < qset_.n_cols (); i++){
    b_twy_naive_estimate_[i].Init (rset_.n_rows () + 1,1);
    b_twy_naive_estimate_[i].SetZero ();
  } 
}

/** This is a temporary function that i am writing */
template <typename TKernel>
void NaiveB_TWYCalculation<TKernel>::print(){
  
  FILE *gp;
  gp=fopen("naive_b_twy.txt","w+");
  
  for(index_t q=0;q<qset_.n_cols();q++){
    
    
    // printf("The BTWY naive estimate for q is..\n");
    //b_twy_naive_estimate_[q].PrintDebug();
     b_twy_naive_estimate_[q].PrintDebug(NULL,gp);
  }
  fclose(gp);
}

#endif
