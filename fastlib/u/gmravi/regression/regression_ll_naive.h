#ifndef REGRESSION_LL_NAIVE_H
#define REGRESSION_LL_NAIVE_H
#include "fastlib/fastlib_int.h"




/** This is the function which builds up the B^TWY 
 * vector by usimng matrix calculations. Note this can be 
 *  optimized by using fastlib's Lapack functions 
 */

template <typename TKernel>
void NaiveCalculation<TKernel>::Compute (){
  
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
  printf("The mapping is ..\n");
  for(index_t r=0;r<rset_.n_cols();r++){
    printf("old_from_new_r_[%d]=%d\n",r,old_from_new_r_[r]);
  }
}

template <typename TKernel>
void NaiveCalculation<TKernel>::Init (Matrix & qset, Matrix & rset, 
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
void NaiveCalculation <TKernel>::print(){
  
  FILE *gp;
  gp=fopen("naive_b_twy.txt","w+");
  
  for(index_t q=0;q<qset_.n_cols();q++){
    
    
    // printf("The BTWY naive estimate for q is..\n");
    //b_twy_naive_estimate_[q].PrintDebug();
     b_twy_naive_estimate_[q].PrintDebug(NULL,gp);
  }
  fclose(gp);
}


template <typename TKernel>

void  NaiveCalculation<TKernel>::ComputeMaximumRelativeError(ArrayList<Matrix> &fast_b_twy_estimate, ArrayList<Matrix> &fast_b_twb_estimate ){

  //here we measure the relative component wise error or the matrix BTWY for 
  //each query point

  double max_rel_error=DBL_MIN;

  for(index_t q=0;q<qset_.n_cols();q++){

    max_rel_error=DBL_MIN;
    for(index_t cols=0;cols<fast_b_twy_estimate[q].n_cols();cols++){
      for(index_t rows=0;rows<fast_b_twy_estimate[q].n_rows();rows++){

	//intf("The fast estimate is %f\n",
	//     fast_b_twy_estimate[q].get(rows,cols));

	//printf("The naive estimate is %f\n",
	//     b_twy_naive_estimate_[q].get(rows,cols));

	double rel_error=fabs(fast_b_twy_estimate[q].get(rows,cols)- 
			      b_twy_naive_estimate_[q].get(rows,cols))/
	  b_twy_naive_estimate_[q].get(rows,cols);

	if(rel_error>max_rel_error){
	  max_rel_error=rel_error;

	}
      }
    }
    printf("The maximum relative error for this point is %f\n",max_rel_error);
  }

  printf("Will print max rel errof ro BTWB..\n");
  max_rel_error=DBL_MIN;

  for(index_t q=0;q<qset_.n_cols();q++){

    max_rel_error=DBL_MIN;
    for(index_t cols=0;cols<fast_b_twb_estimate[q].n_cols();cols++){
      for(index_t rows=0;rows<fast_b_twb_estimate[q].n_rows();rows++){

	//intf("The fast estimate is %f\n",
	//     fast_b_twy_estimate[q].get(rows,cols));

	//printf("The naive estimate is %f\n",
	//     b_twy_naive_estimate_[q].get(rows,cols));

	double rel_error=fabs(fast_b_twb_estimate[q].get(rows,cols)- 
			      b_twb_naive_estimate_[q].get(rows,cols))/
	  b_twb_naive_estimate_[q].get(rows,cols);

	if(rel_error>max_rel_error){
	  max_rel_error=rel_error;

	}
      }
    }
    printf("The maximum relative error for this point is %f\n",max_rel_error);
  }

}



#endif
