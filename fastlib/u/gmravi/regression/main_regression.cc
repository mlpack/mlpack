
/** 
 * @file man_final.cc
 * 
 * This piece of code does local linear regression code by 
 * using a reference set which is entirely in the positive
 * quadrant and whose regression values are all positive.
 * Also we assume that the query points are all in the 
 * positive quadrant. By positive quadrant we mean that all 
 * the coordinates of all the reference point are positive.
 *
 * Local linear regression is achieved by evaluating the expression
 * [1,q](B^T W B) (B^T W Y) for a query point, by using all the 
 * reference points. Here we approximate B^TWB componentwise by a
 * certain factor and B^TWY by a certain factor. Approximation of 
 * B^T W Y is done in the file regression_vector.h and estimation
 * of B^T W B is done in the file regression_matrix.h 
 */


#include "fastlib/fastlib_int.h"
#include "regression_matrix.h"
#include "regression_vector.h"

int main (int argc, char *argv[]){

  fx_init (argc, argv);

  //const char *algorithm = fx_param_str_req (NULL, "method");
  //bool do_naive = fx_param_exists (NULL, "do_naive");

  
  FastVectorCalculation <GaussianKernel> fast_vector_calculation;

  //This will hold the results of B^TWY calculation
  ArrayList<Vector> fast_vector_calculation_results;

 
  Matrix query_dataset;
  Matrix reference_dataset;
 
  index_t num_query_points;
  index_t num_of_dimensions;

  Vector regression_estimates_naive;
  Vector regression_estimates;
  
  //This stores the order of permutations as a result of dual tree formation
  ArrayList <index_t> old_from_new_r;
  ArrayList <index_t> new_from_old_r;
  
  if (!strcmp (fx_param_str (NULL, "kernel", "gaussian"), "gaussian")){
    
    //First lets get B^TWY vector. We have declared an object 
    //type FastVectorCalculation  namely fast_vector_calculation.
    
    fast_vector_calculation.Init ();
    fast_vector_calculation.Compute (fx_param_double (NULL, "tau", 0.1));
   
    //Pull back the permuted and scaled(possibly) datasets from the module
    query_dataset.Alias(fast_vector_calculation.get_query_dataset());
    reference_dataset.Alias(fast_vector_calculation.get_reference_dataset()); 

    //Since we shall use the number of query points and 
    //the number of dimensions of the dataset regularly
    //lets store these values into 2 temporary variables

    num_query_points=query_dataset.n_cols();
    num_of_dimensions=query_dataset.n_rows();
    
    //Get the results of the vector calculations. Hence we now
    // have estimate of B^TWY for each query point

    //Firstly initialize the object fast_vector_calculation_results

    fast_vector_calculation_results.Init(num_query_points);  
    
    for(index_t i=0;i<num_query_points;i++)
      { 
	fast_vector_calculation_results[i].Init(num_of_dimensions+1);
      }
 
    //get density estimates from fast vector calculations 
    //and push it into the array fast_vector_calculation_results

      for(index_t q=0;q<num_query_points;q++)
	{
	  //for each query point
	  for(index_t d=0;d<num_of_dimensions+1;d++)
	    {
	      //along each dimension
	      fast_vector_calculation_results[q][d]=
		fast_vector_calculation.get_vector_estimates(q,d);
	    }
	}

      //get the permutation of the data, and use this permuted dataset
      //for all future calculations

      old_from_new_r.Copy(fast_vector_calculation.get_old_from_new_r());
      new_from_old_r.Copy(fast_vector_calculation.get_new_from_old_r());
    
      //Now lets get (B^TWB)^-1. 
      //This can be done by calling routines related to 
      //the object FastMatrixCalculation present in the file 
      //regression_matrix.h 

      

      FastMatrixCalculation <GaussianKernel> fast_matrix_calculation;   
      fast_matrix_calculation.Init(query_dataset,reference_dataset);
      fast_matrix_calculation.Compute(fx_param_double (NULL, "tau", 0.1));
     
   
      //This will hold the results of fast matrix calculations
      ArrayList<Matrix> fast_matrix_calculation_results;
      
      //This initializes fast_matrix_calculation_results
      fast_matrix_calculation_results.Copy(fast_matrix_calculation.get_results());
     
      
      
      //We now have to multiply the matrix
      // fast_matrix_calculation_results with fast_vector_calculation_results
      
      ArrayList<Vector> temp1;
      temp1.Init(num_query_points);  //This initializes temp1

     
      regression_estimates.Init(num_query_points);
      regression_estimates.SetZero();
     
      for(index_t q=0;q<num_query_points;q++){

	temp1[q].Init(num_of_dimensions+1);
	la::MulInit(fast_matrix_calculation_results[q],
		    fast_vector_calculation_results[q],&temp1[q]);
       
	for(index_t i=0;i<num_of_dimensions+1;i++){

	  if(i!=0){
	    regression_estimates[q]+=
	      temp1[q].get(i)*query_dataset.get(i-1,q);  
	  }	

	  else{
	    regression_estimates[q]+=temp1[q].get(i)*1;
	  }
	}
      }
      
      //With this we have calculated the regression 
      //estimates for the different query points
      
      //Lets do naive calculations too.......

 

      NaiveVectorCalculation <GaussianKernel> naive_vector_calculation;

      //This will hold the results of B^TWY
      ArrayList<Vector> naive_vector_calculation_results;
      
      if (!strcmp (fx_param_str (NULL, "kernel", "gaussian"), "gaussian")){
	
	
	//First lets get B^TWY by using naive methods. This can
	//be done by calling functions in regression_vector.h

	naive_vector_calculation.
	  Init (query_dataset,reference_dataset,old_from_new_r);
	naive_vector_calculation.Compute ();

	//initialize naive_vector_calculation_results
	num_query_points=query_dataset.n_cols();
	num_of_dimensions=query_dataset.n_rows();
    
	naive_vector_calculation_results.Init(num_query_points);  
	for(index_t i=0;i<num_query_points;i++)
	  { 
	    naive_vector_calculation_results[i].Init(num_of_dimensions+1); 
	  }
 
    
	
	for(index_t q=0;q<num_query_points;q++)
	  {
	    //for each query point
	    for(index_t d=0;d<num_of_dimensions+1;d++)
	      {
		//along each dimension
		naive_vector_calculation_results[q][d]=
		  naive_vector_calculation.get_vector_estimates(q,d);
	      }
	    //printf("Naive vector was\n");
	    //naive_vector_calculation_results[q].PrintDebug();
	  }
	
	
	//Now lets get (B^TWB)^-1.
	
	
	//The dataset retrieved in the previous steps is 
	//being used once again
	
	NaiveMatrixCalculation <GaussianKernel> naive_matrix_calculation;
	naive_matrix_calculation.Init(query_dataset,reference_dataset);  
	naive_matrix_calculation.Compute();
      
	//This will hold the results of 
	//Naive regression2 calculations

       ArrayList<Matrix> naive_matrix_calculation_results; 
       naive_matrix_calculation_results.Init(num_query_points);
       
       //This initializes fast_matrix_calculation_results

       for(index_t q=0;q<num_query_points;q++){
       naive_matrix_calculation_results[q].
	 Copy(naive_matrix_calculation.get_results(q)); 
       //printf("The naive matrix is ...\n");
       // naive_matrix_calculation_results[q].PrintDebug(); 
       }
       
       //We now have to multiply the matrix 
       //fast_matrix_calculation_results with 
       //fast_vector_calculation_results
       
       ArrayList<Vector> temp1;
       temp1.Init(num_query_points);  //This initializes temp1
       
      
       regression_estimates_naive.Init(num_query_points);
       regression_estimates_naive.SetZero();
       
       for(index_t q=0;q<num_query_points;q++){
	 
	 
	 temp1[q].Init(num_of_dimensions+1);
	
	 la::MulInit(naive_matrix_calculation_results[q],
		     naive_vector_calculation_results[q],&temp1[q]);
	 
	 //So from the previous step I have the vector
	 //(B^T W B)^-1 (B^T W Y). 
	 //We shall multiply this product with the vector
	 //[1,q]. Where  q are the coordinates of a query point

	 for(index_t i=0;i<num_of_dimensions+1;i++){
	   
	   if(i!=0){
	     
	     regression_estimates_naive[q]+=
	       temp1[q].get(i)*query_dataset.get(i-1,q);
	     
	   }	
	   
	   else{
	    
	     regression_estimates_naive[q]+=temp1[q].get(i)*1;
	   }
	 }	 
       }

       //Print both the naive estimates and the fast estimates on to a file
       //Note that the dataset has been scaled so the coordinates of the
       //pointed printed on the file are not the same as the original 
       //coordinates

       FILE *lp;
       lp=fopen("estimates_naive_fast.txt","w+");
       double relative_error=0;
       double error;
       double total_error=0;
       double mean_square_error;
       double max_relative_error;
       for(index_t q=0;q<num_query_points;q++){
	 
	 
	 error=(double)fabs(regression_estimates_naive[new_from_old_r[q]]-
			    regression_estimates[new_from_old_r[q]]);

	 relative_error=error/regression_estimates_naive[new_from_old_r[q]];
	 total_error+=pow(error,2);
	 
	 for(index_t d=0;d<num_of_dimensions;d++){
	   fprintf(lp,"%f, ",query_dataset.get(d,new_from_old_r[q]));
	 }

	 fprintf(lp,"naive: %2f, fast:%2f  diff:%2f\n",
		 regression_estimates_naive[new_from_old_r[q]],
		 regression_estimates[new_from_old_r[q]],relative_error);

	 if(relative_error>max_relative_error){
	   max_relative_error=relative_error;
	 }
       }
       //mean_square_error=error/num_query_points;
       fprintf(lp,"number of query points are %d\n",num_query_points);
       fprintf(lp,"total error=%f\n",total_error);
       fprintf(lp,"Max relative error=%f\n",max_relative_error);
       printf("Average relative error: %f\n",total_error/num_query_points);
       printf("Maximum relative error is %f\n",max_relative_error);
      }
  fx_done();
  }
}


