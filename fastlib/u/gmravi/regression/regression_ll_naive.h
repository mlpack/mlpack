#ifndef REGRESSION_LL_NAIVE_H
#define REGRESSION_LL_NAIVE_H
#include "fastlib/fastlib_int.h"
#include "pseudo_inverse.h"

template <typename TKernel>
double NaiveCalculation<TKernel>::SquaredFrobeniusNorm_(Matrix &a){

  //This function computes treats the natrix as a vector and computes
  //it's 1-norm
  double value=0;
  for(index_t col=0;col<a.n_cols();col++){
    for(index_t row=0;row<a.n_rows();row++){
      value+=a.get(row,col)*a.get(row,col);
    }
  }
  return value;
}



/** This is the function which builds up the B^TWY 
 * vector by usimng matrix calculations. Note this can be 
 *  optimized by using fastlib's Lapack functions 
 */


template <typename TKernel>
void NaiveCalculation<TKernel>::Compute (){
  fx_timer_start(NULL,"naive");
  
  // compute unnormalized sum
  for (index_t q = 0; q < qset_.n_cols (); q++){	//for each query point
    
    const double *q_col = qset_.GetColumnPtr (q);

    for (index_t r = 0; r < rset_.n_cols (); r++){ //for each reference point

      //Get reference point
      const double *r_col = rset_.GetColumnPtr (r);

      // pairwise distance and kernel value
      double dsqd =
	la::DistanceSqEuclidean (qset_.n_rows (), q_col, r_col);
      double ker_value = kernel_.EvalUnnormOnSq (dsqd);
      
      for(index_t col = 0; col < rset_.n_rows () + 1; col++){	//along each direction
	
	for(index_t row=0; row< rset_.n_rows () + 1; row++){


	  //Lets gradually fill up all the elements of the matrices 
	  //b_twy_naive_estimate and b_twb_naive_estimate	 

	  //Fill in b_twy_naive first
	
	  if(col==0){

	    //For this column 
	    if (row != 0){

	      //Fill B^TWY naive
	      
	      double val=b_twy_naive_estimate_[q].get(row,col)+ 
		ker_value * rset_weights_[old_from_new_r_[r]] * rset_.get (row- 1, r);
	      
	      b_twy_naive_estimate_[q].set(row,col,val);

	      //Now fill B^TWB naive

	      double val1=b_twb_naive_estimate_[q].get(row,col) + 
		ker_value * rset_.get(row-1,r);
	      b_twb_naive_estimate_[q].set(row,col,val1);
	      
	    }

	    else{

	      //Fill B^TWY naive 
	      
	      double val= b_twy_naive_estimate_[q].get(row,col) + 
		ker_value * rset_weights_[old_from_new_r_[r]];
	      
	      b_twy_naive_estimate_[q].set(row,col,val);

	      //Now fill B^TWB

	      double val1=b_twb_naive_estimate_[q].get(row,col) 
		+ ker_value ;

	      b_twb_naive_estimate_[q].set(row,col,val1);

	      
	    }
	  }//end of col 0...............

	  //Column!=0

	  else{
	    if(row!=0){
	      //Only B^TWB naive estimates get filled up
	      double val1=b_twb_naive_estimate_[q].get(row,col) 
		+ ker_value* rset_.get(row-1,r)* rset_.get(col-1,r);
	      
	      b_twb_naive_estimate_[q].set(row,col,val1);
	    }

	    else{
	      double val1=b_twb_naive_estimate_[q].get(row,col) 
		+ ker_value* rset_.get(col-1,r);
	      b_twb_naive_estimate_[q].set(row,col,val1);

	    }
	  }
	}
      }
    }
  }
  fx_timer_stop(NULL,"naive");
  Print_();
  //Having done this get regression estimates by calling the function
  //ObtainRegressionEstimates
  //ObtainRegressionEstimate_();
  //PrintRegressionEstimate_(fast_regression_estimate);
}

template <typename TKernel>
void NaiveCalculation<TKernel>:: ComputeMaximumRelativeError(ArrayList<Matrix> &fast_b_twy_estimate, ArrayList<Matrix> & fast_b_twb_estimate, char *pruning_criteria){

  if(!strcmp(pruning_criteria,"fnorm")){

    double max_frobenius_error_b_twb=0;
    double max_frobenius_error_b_twy=0;

    for(index_t q=0;q<qset_.n_cols();q++){

      //BTWY first...............
      Matrix temp1;
      la::SubInit (b_twy_naive_estimate_[q],fast_b_twy_estimate[q] , &temp1);
      double f_norm=SquaredFrobeniusNorm_(temp1);
      double f_norm_naive=SquaredFrobeniusNorm_(b_twy_naive_estimate_[q]);
      double rel_error_b_twy=fabs(f_norm)/f_norm_naive;
      printf("relative frobenius norm error for BTWY is %f\n",rel_error_b_twy);

      if(max_frobenius_error_b_twy<rel_error_b_twy){

	max_frobenius_error_b_twy=rel_error_b_twy;
      }      


      Matrix temp2;
      la::SubInit (b_twb_naive_estimate_[q],fast_b_twb_estimate[q] , &temp2);
      f_norm=SquaredFrobeniusNorm_(temp2);
      f_norm_naive=SquaredFrobeniusNorm_(b_twb_naive_estimate_[q]);
      double rel_error_b_twb=fabs(f_norm)/f_norm_naive;
      printf("relative frobenius norm error for BTWB is %f\n",rel_error_b_twb);

      if(max_frobenius_error_b_twb < rel_error_b_twb){

	max_frobenius_error_b_twb= rel_error_b_twb;
      }       
    }
    printf("The max frobenius error for BTWY is %f\n",max_frobenius_error_b_twy);
    printf("The max frobenius error for BTWB is %f\n",max_frobenius_error_b_twb);
  }
  //The pruning criteria is component wise......
  else{


    //Lets get the BTWB error first
    double max_error_on_the_whole=0;
   
    for(index_t q=0;q<qset_.n_cols();q++){
      
      double  max_error_for_this_point=0.0;

      for(index_t row=0;row<fast_b_twb_estimate[q].n_rows();row++){
	for(index_t col=0;col<fast_b_twb_estimate[q].n_cols();col++){
	  
	  double diff=
	    fabs(fast_b_twb_estimate[q].get(row,col)-b_twb_naive_estimate_[q].get(row,col));

	  double error=diff/b_twb_naive_estimate_[q].get(row,col);
	  if(error>max_error_for_this_point){

	    max_error_for_this_point=error;
	  }
	  
	}
      }//Calculations complete for this point

      printf("Maximum error for this point is %f\n",max_error_for_this_point);
      if(max_error_for_this_point<max_error_on_the_whole){
	max_error_on_the_whole=max_error_for_this_point;

      }
    }
    printf("Max error on the whole for BTWB is %f\n",	max_error_on_the_whole);
    //Now lets get BTWY error

    max_error_on_the_whole=0;
    
    for(index_t q=0;q<qset_.n_cols();q++){
      
      double max_error_for_this_point=0.0;
      
      for(index_t row=0;row<fast_b_twy_estimate[q].n_rows();row++){
	for(index_t col=0;col<fast_b_twy_estimate[q].n_cols();col++){
	  
	  double diff=
	    fabs(fast_b_twy_estimate[q].get(row,col)-b_twy_naive_estimate_[q].get(row,col));
	  
	  double error=diff/b_twb_naive_estimate_[q].get(row,col);
	  if(error>max_error_for_this_point){
	    
	    max_error_for_this_point=error;
	  }
	  
	}
      }  //Calculations complete for this point
      printf("Maximum error for this point is %f\n",max_error_for_this_point);
      
      if(max_error_for_this_point<max_error_on_the_whole){
	max_error_on_the_whole=max_error_for_this_point;
      }
    }
    printf("Max error on the whole for BTWY is %f\n",max_error_on_the_whole);
  }
}



template <typename TKernel>
void NaiveCalculation<TKernel>::Init (Matrix & qset, Matrix & rset, ArrayList<index_t> &old_from_new_r,double bandwidth,Vector &rset_weights){
  
  // get datasets
  qset_.Alias (qset);
  rset_.Alias (rset);
  
  //get permutation
  old_from_new_r_.Copy(old_from_new_r);
  
  // get bandwidth
  kernel_.Init (bandwidth);
  
  //get the weights
  rset_weights_.Alias(rset_weights);   


  //ArrayList of matrices to store results
  
  b_twy_naive_estimate_.Init (qset_.n_cols ());
  b_twb_naive_estimate_.Init(qset_.n_cols());

  
  for (index_t i = 0; i < qset_.n_cols (); i++){
    b_twy_naive_estimate_[i].Init (rset_.n_rows () + 1,1);
    b_twy_naive_estimate_[i].SetZero ();
  } 

  for (index_t i = 0; i < qset_.n_cols (); i++){
    b_twb_naive_estimate_[i].Init (rset_.n_rows () + 1,rset_.n_rows () + 1);
    b_twb_naive_estimate_[i].SetZero ();
  } 
 

}

/** This is a temporary function that i am writing */
template <typename TKernel>
void NaiveCalculation <TKernel>::Print_(){
  
  FILE *gp,*fp;
  gp=fopen("naive_b_twy.txt","w+");
  fp=fopen("naive_b_twb.txt","w+");
  
  for(index_t q=0;q<qset_.n_cols();q++){
    
    
    // printf("The BTWY naive estimate for q is..\n");
    //b_twy_naive_estimate_[q].PrintDebug();
     b_twy_naive_estimate_[q].PrintDebug(NULL,gp);
     b_twb_naive_estimate_[q].PrintDebug(NULL,fp);

  }
  fclose(gp);
  fclose(fp);
}






#endif
