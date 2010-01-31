
// What does this piece of C++ do??

/** Local linear regression involves calculating for a query point
 * the product [1,q] (B^TWB)^-1 (B^TWY) where q are the coordinates of the
 * query point under consideration. Both B^TWB and B^TWY are query dependeny.
 * While B^TWB is a (D+1)X(D+1) matrix B^TWY is a (D+1) column vector. So we
 * divide the problem into 2 parts. We first evaluate B^TWY and then evaluate 
 * B^TWB for each query point.We do this by using dual tree methods.We include 
 * 2 header files namely regression_matrix.h and regression_vector.h which 
 * calculate B^TWB and B^TWY respectively
*/

//What are the arguments to this C++ code?

/** This code requires the following arguments from the user

1) kernel: type of kernel to be used. At the moment a piecce of code for 
   non-gaussian kernels is missing however one can replace the GaussianKernel 
   wherever it appears and replace it with appropriate kernel

2) data: The set of reference points. Note that we assume that all the reference
   points are in the positive half space. If not, one needs to transform so as to 
   move all the points to the positive quadrant

3) query: The set of query points

4) scaling: This can be set to "range" or "none". If the dataset is within [0,1] range 
   then there is no need for scaling. For larger datasetsi.e which are outside the 
   bound [0,1] we recommend scaling so as to achieve numerical stability

5) bandwidth: The bandwidth to be used for kernel calculations  

6) dwgts: The regression estimates for the reference points

For example one can compile the program with the following options
  --kernel=gaussian --data=refined_astroset.ds  --query=refined_astroset.ds  --scaling=range --bandwidth=0.3
*/





#include "fastlib/fastlib_int.h"
#include "regression_vector.h"
#include "regression_matrix.h"

int main (int argc, char *argv[]){

  fx_init (argc, argv);

  //const char *algorithm = fx_param_str_req (NULL, "method");
  //bool do_naive = fx_param_exists (NULL, "do_naive");

  FastVectorCalculation <GaussianKernel> fast_vector;

  //This will hold the results of B^TWY
  ArrayList<Vector> fast_vector_results;

 
  Matrix query_dataset;
  Matrix reference_dataset;
 
  index_t num_query_points;
  index_t num_of_dimensions;

  Vector regression_estimates_naive;
  Vector regression_estimates;
  ArrayList <index_t> old_from_new_r;
  ArrayList <index_t> new_from_old_r;

  if (!strcmp (fx_param_str (NULL, "kernel", "gaussian"), "gaussian")){
    
    //First lets get B^TWY
      fast_vector.Init ();

    fx_timer_start(NULL,"second_matrix");
    fast_vector.Compute (fx_param_double (NULL, "tau", 0.1));
    fx_timer_stop(NULL,"second_matrix");


    //Get the datasets. Note when we evaluated the 
    //vector B^TWY by building dual trees the order of 
    //query and reference points changed. Hence we are 
    //pulling back the data

    query_dataset.Alias(fast_vector.get_query_dataset());
    reference_dataset.Alias(fast_vector.get_reference_dataset());  

   
    num_query_points=query_dataset.n_cols();
    num_of_dimensions=query_dataset.n_rows();
    
    //initialize fast_vector_results
     fast_vector_results.Init(num_query_points);  
    
    for(index_t i=0;i<num_query_points;i++)
      { 
	fast_vector_results[i].Init(num_of_dimensions+1); 
     }
    
    //get vector estimates from fast vector calculations 
    //and push it into the array fast_vector_results

    for(index_t q=0;q<num_query_points;q++)
      {
	//for each query point
	for(index_t d=0;d<num_of_dimensions+1;d++)
	  {
	    //along each dimension
	    fast_vector_results[q][d]=fast_vector.get_vector_estimates(q,d);
	  }
      }

    //Pull back the results of fast vector calculations
    old_from_new_r.Copy(fast_vector.get_old_from_new_r());
    new_from_old_r.Copy(fast_vector.get_new_from_old_r());

    //Now lets get (B^TWB)^-1. 
    //This can be done by calling routines related the 
    //object Regression2 present in the file regression2.h 
    
    //The dataset retrieved in the previous steps is being used once again
    
    FastMatrixCalculation <GaussianKernel> fast_matrix_calc;
    fast_matrix_calc.Init(query_dataset,reference_dataset);

    fx_timer_start(NULL,"first_matrix");
    fast_matrix_calc.Compute(fx_param_double (NULL, "tau", 0.1));
    fx_timer_stop(NULL,"first_matrix");
    
    //This will hold the results of regression_matrix calculations
    ArrayList<Matrix> fast_matrix_results;
    
    //This initializes fast_matrix_results 
    fast_matrix_results.Copy(fast_matrix_calc.get_results());  
    
    
    //We now have to multiply the 
    //matrix fast_matrix__results with fast_vector_results
    
    ArrayList<Vector> temp1;
    temp1.Init(num_query_points);  //This initializes temp1
    
    regression_estimates.Init(num_query_points);
    regression_estimates.SetZero();
    
    for(index_t q=0;q<num_query_points;q++){
      
      
      temp1[q].Init(num_of_dimensions+1);
      la::MulInit(fast_matrix_results[q],fast_vector_results[q],&temp1[q]);
      
      for(index_t i=0;i<num_of_dimensions+1;i++){
	
	if(i!=0){
	  
	  regression_estimates[q]+=temp1[q].get(i)*query_dataset.get(i-1,q);
	}
	
	else{
	  
	  regression_estimates[q]+=temp1[q].get(i)*1;
	}
      }
    }
    
  }

  //Lets do naive calculations too.......

  NaiveVectorCalculation <GaussianKernel> naive_vector_calc;
 
  ArrayList<Vector> naive_vector_calc_results;

  if (!strcmp (fx_param_str (NULL, "kernel", "gaussian"), "gaussian")){
    

    //First lets get B^TWY

    naive_vector_calc.Init (query_dataset,reference_dataset,old_from_new_r);
    fx_timer_start(NULL,"second_matrix_naive");
    naive_vector_calc.Compute ();
    fx_timer_stop(NULL,"second_matrix_naive");

    //initialize naive_vector_calc_results
    num_query_points=query_dataset.n_cols();
    num_of_dimensions=query_dataset.n_rows();
    
    naive_vector_calc_results.Init(num_query_points); 

  
       for(index_t i=0;i<num_query_points;i++)
      { 
	naive_vector_calc_results[i].Init(num_of_dimensions+1);
      }
       
       //get density estimates from naive vector calculations 
    //and push it into the array naive_vector_calc_results
       
       for(index_t q=0;q<num_query_points;q++)
	 {
	   //for each query point
	   for(index_t d=0;d<num_of_dimensions+1;d++)
	     {
	       //along each dimension
	       naive_vector_calc_results[q][d]=
		 naive_vector_calc.get_vector_estimates(q,d);
	     }
	 }
       
      
      //Now lets get (B^TWB)^-1.


      //The dataset retrieved in the previous steps is being used once again
      
       NaiveMatrixCalculation <GaussianKernel> naive_matrix_calc;
       naive_matrix_calc.Init(query_dataset,reference_dataset);
       fx_timer_start(NULL,"first_matrix_naive");
       naive_matrix_calc.Compute();
       fx_timer_stop(NULL,"first_matrix_naive");


      //This will hold the results of Naive regression2 calculations
       ArrayList<Matrix> naive_matrix_calc_results;
       
       naive_matrix_calc_results.Copy(naive_matrix_calc.get_results());

       //We now have to multiply the matrix naive_matrix_results 
       //with naive_vector_results
       
       ArrayList<Vector> temp1;
       temp1.Init(num_query_points);  //This initializes temp1
       
      
       regression_estimates_naive.Init(num_query_points);
       regression_estimates_naive.SetZero();
       
       for(index_t q=0;q<num_query_points;q++){
	
	 temp1[q].Init(num_of_dimensions+1); 
	
	 la::MulInit(naive_matrix_calc_results[q],
		     naive_vector_calc_results[q],&temp1[q]);
	 
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

       FILE *lp;
       lp=fopen("estimates_fast_naive","w+");
       double relative_error=0;
       double error;
       double total_error=0;
       double mean_square_error;
       double max_relative_error;
       for(index_t q=0;q<num_query_points;q++){
	 
	 error=(double)fabs(regression_estimates_naive[new_from_old_r[q]]
			    -regression_estimates[new_from_old_r[q]]);

	 relative_error=error/regression_estimates_naive[new_from_old_r[q]];

	 total_error+=pow(error,2);
	 
	 for(index_t d=0;d<num_of_dimensions;d++){
	   fprintf(lp,"%f, ",query_dataset.get(d,new_from_old_r[q]));
	 }

	 fprintf(lp,"naive: %2f, fast:%2f  diff:%2f\n",regression_estimates_naive[new_from_old_r[q]],regression_estimates[new_from_old_r[q]],relative_error);

	 if(relative_error>max_relative_error){
	   max_relative_error=relative_error;
	 }
       }
       //mean_square_error=error/num_query_points;
       fprintf(lp,"number of query points are %d\n",num_query_points);
       fprintf(lp,"total error=%f\n",total_error);
       fprintf(lp,"Max relative error=%f\n",max_relative_error); 
       fclose(lp);       
  }
  fx_done();
}


