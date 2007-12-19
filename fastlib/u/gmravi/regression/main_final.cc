#include "fastlib/fastlib_int.h"
#include "regression.h"
#include "regression2.h"

int main (int argc, char *argv[]){

  fx_init (argc, argv);

  const char *algorithm = fx_param_str_req (NULL, "method");
  bool do_naive = fx_param_exists (NULL, "do_naive");

  FastKde <GaussianKernel> fast_kde;
  //This will hold the results of B^TWY
  ArrayList<Vector> fast_kde_results;

 
  Matrix query_dataset;
  Matrix reference_dataset;
 
  index_t num_query_points;
  index_t num_of_dimensions;

  Vector regression_estimates_naive;
  Vector regression_estimates;

  ArrayList <index_t> new_from_old_r;
  FILE *fp;
  FILE *gp;
  fp=fopen("fast_regression.txt","w+");
  gp=fopen("naive_regression.txt","w+");

  if (!strcmp (fx_param_str (NULL, "kernel", "gaussian"), "gaussian")){
    

    //First lets get B^TWY
    fx_timer_start(NULL,"second_matrix");
    fast_kde.Init ();
    fast_kde.Compute (fx_param_double (NULL, "tau", 0.1));
    fx_timer_stop(NULL,"second_matrix");

      //default value is 0.1
    if(fx_param_exists(NULL,"fast_kde_output"))
      {
	fast_kde.PrintDebug();
      }
	
    
    query_dataset.Alias(fast_kde.get_query_dataset());
    reference_dataset.Alias(fast_kde.get_reference_dataset());  // Hence with this wwe initialize both the datasets


    //printf("In main file the dataset received is ..\n");
    //query_dataset.PrintDebug();

    //initialize fast_kde_results
    num_query_points=query_dataset.n_cols();
    num_of_dimensions=query_dataset.n_rows();
    
    fast_kde_results.Init(num_query_points);  //this initializes fast_kde results
    
    for(index_t i=0;i<num_query_points;i++)
      { 
	fast_kde_results[i].Init(num_of_dimensions+1); //Each element is now initialized
      }
 
    //get density estimates from fast kde calculations and push it into the array fast_kde_results

   

      for(index_t q=0;q<num_query_points;q++)
	{
	  //for each query point
	  for(index_t d=0;d<num_of_dimensions+1;d++)
	    {
	      //along each dimension
	      fast_kde_results[q][d]=fast_kde.get_density_estimates(q,d);
	    }
	}
      new_from_old_r.Copy(fast_kde.get_new_from_old_r());

      printf("WILL NOW START REGRESSION2......................\n");

    //Now lets get (B^TWB)^-1. This can be done by calling routines related the object Regression2 present in the file regression2.h 


      //The dataset retrieved in the previous steps is being used once again
      
      Regression2 <GaussianKernel> reg2;
      printf("going to initialization function...\n");
      fx_timer_start(NULL,"first_matrix");
      reg2.Init(query_dataset,reference_dataset);
      reg2.Compute(fx_param_double (NULL, "tau", 0.1));
      fx_timer_stop(NULL,"first_matrix");
      printf("Initializations done..\n");

      //This will hold the results of regression2 calculations
      ArrayList<Matrix> wfkde_results;
      
      wfkde_results.Copy(reg2.get_results());  //This initializes wfkde_results
      
      
      //We now have to multiply the matrix wfkde_results with fast_kde_results
      
      ArrayList<Vector> temp1;
      temp1.Init(num_query_points);  //This initializes temp1

     
      regression_estimates.Init(num_query_points);
      regression_estimates.SetZero();
     
      for(index_t q=0;q<num_query_points;q++){


	temp1[q].Init(num_of_dimensions+1); //and this initializes each element of temp1
	//printf("fast_kde_results[%d] \n",q);
	//fast_kde_results[q].PrintDebug();
	la::MulInit(wfkde_results[q],fast_kde_results[q],&temp1[q]);
       
	for(index_t i=0;i<num_of_dimensions+1;i++){

	  if(i!=0){

	    //printf("temp1[%d][%d] is %f\n",q,i,temp1[q].get(i));
	    //printf("query_dataset[%d][%d] is %f\n",i-1,q,query_dataset.get(i-1,q));
	    regression_estimates[q]+=temp1[q].get(i)*query_dataset.get(i-1,q);

	  }	

	  else{
	    //printf("temp1[%d][%d] is %f\n",q,i,temp1[q].get(i));
	    regression_estimates[q]+=temp1[q].get(i)*1;
	  }
	}

	fprintf(fp,"Point:");

	for(index_t d=0;d<query_dataset.n_rows();d++){

	  fprintf(fp,"%f,",query_dataset.get(d,q));
	}
	  fprintf(fp,"  ",regression_estimates[q]);
	printf("Hence regression estimate is %f\n",regression_estimates[q]);
      }
      //printf("The regression estimates are ..\n");
      //regression_estimates.PrintDebug();
  }


  //Lets do naive calculations too.......



  NaiveKde <GaussianKernel> naive_kde;
  //This will hold the results of B^TWY
  ArrayList<Vector> naive_kde_results;

  if (!strcmp (fx_param_str (NULL, "kernel", "gaussian"), "gaussian")){
    

    //First lets get B^TWY

    //printf("In main function query dataset is..........\n");
    //query_dataset.PrintDebug();

    fx_timer_start(NULL,"second_matrix_naive");
    naive_kde.Init (query_dataset,reference_dataset,new_from_old_r);
    naive_kde.Compute ();
    fx_timer_stop(NULL,"second_matrix_naive");

    //initialize fast_kde_results
    num_query_points=query_dataset.n_cols();
    num_of_dimensions=query_dataset.n_rows();
    
    naive_kde_results.Init(num_query_points);  //this initializes fast_kde results
    
    for(index_t i=0;i<num_query_points;i++)
      { 
	naive_kde_results[i].Init(num_of_dimensions+1); //Each element is now initialized
      }
 
    //get density estimates from fast kde calculations and push it into the array fast_kde_results

   

      for(index_t q=0;q<num_query_points;q++)
	{
	  //for each query point
	  for(index_t d=0;d<num_of_dimensions+1;d++)
	    {
	      //along each dimension
	      naive_kde_results[q][d]=naive_kde.get_density_estimates(q,d);
	    }
	}

      printf("WILL NOW START REGRESSION2......................\n");

    //Now lets get (B^TWB)^-1. This can be done by calling routines related the object Regression2 present in the file regression2.h 


      //The dataset retrieved in the previous steps is being used once again
      
       NaiveRegression2 <GaussianKernel> naive_reg2;
       printf("going to initialization function...\n");

       fx_timer_start(NULL,"first_matrix_naive");
       naive_reg2.Init(query_dataset,reference_dataset);
       naive_reg2.Compute();
       fx_timer_stop(NULL,"first_matrix_naive");

       printf("Initializations done..\n");
       
       //This will hold the results of regression2 calculations
       ArrayList<Matrix> naive_reg2_results;
       
       naive_reg2_results.Copy(naive_reg2.get_results());  //This initializes wfkde_results
       
       
       //We now have to multiply the matrix wfkde_results with fast_kde_results
       
       ArrayList<Vector> temp1;
       temp1.Init(num_query_points);  //This initializes temp1
       
      
       regression_estimates_naive.Init(num_query_points);
       regression_estimates_naive.SetZero();
       
       for(index_t q=0;q<num_query_points;q++){
	 
	 
	 temp1[q].Init(num_of_dimensions+1); //and this initializes each element of temp1
	
	 la::MulInit(naive_reg2_results[q],naive_kde_results[q],&temp1[q]);
	 
	 for(index_t i=0;i<num_of_dimensions+1;i++){
	   
	   if(i!=0){
	     
	     // printf("temp1[%d][%d] is %f\n",q,i,temp1[q].get(i));
	     //printf("query_dataset[%d][%d] is %f\n",i-1,q,query_dataset.get(i-1,q));
	     regression_estimates_naive[q]+=temp1[q].get(i)*query_dataset.get(i-1,q);
	     
	   }	
	   
	   else{
	     //printf("temp1[%d][%d] is %f\n",q,i,temp1[q].get(i));
	     regression_estimates_naive[q]+=temp1[q].get(i)*1;
	   }
	 }
	 fprintf(gp,"Point:");

	 for(index_t d=0;d<query_dataset.n_rows();d++){

	   fprintf(gp,"%f,",query_dataset.get(d,q));
	 }
	 fprintf(gp,"  ",regression_estimates_naive[q]);
	 
       }
       // printf("The regression estimates are ..\n");
       //regression_estimates_naive.PrintDebug();

       FILE *lp;
       lp=fopen("estimates.txt","w+");
      double relative_error=0;
      double error;
      for(index_t q=0;q<num_query_points;q++){

	error=(double)fabs(regression_estimates_naive[q]-regression_estimates[q])/regression_estimates_naive[q];

	for(int d=0;d<num_of_dimensions;d++){
	  fprintf(lp,"%f,",query_dataset.get(d,q));
	}
	fprintf(lp,"ren: %3f, refast:%3f diff:%3f\n",regression_estimates_naive[q],regression_estimates[q],error);
	if(error>relative_error){
	  relative_error=error;
	}
      }
      printf("maximum relative error is %f\n",relative_error);
  }
  fx_done();
}


