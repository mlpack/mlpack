#include "fastlib/fastlib_int.h"
#include "regression.h"
#define  MAXDOUBLE 32768.0

int main (int argc, char *argv[]){
  fx_init (argc, argv);

  const char *algorithm = fx_param_str_req (NULL, "method");
  bool do_naive = fx_param_exists (NULL, "do_naive");

  Matrix query_dataset;
  Matrix reference_dataset;
  Vector rset_weights;

  ArrayList<Vector> fast_kde_results;
  FastKde <GaussianKernel> fast_kde;

  index_t num_query_points;
  index_t num_of_dimensions;

  if (!strcmp (algorithm, "fast")){
    if (!strcmp (fx_param_str (NULL, "kernel", "gaussian"), "gaussian")){
    
      printf ("O(p^D) expansion KDE\n");
      fast_kde.Init ();
      fast_kde.Compute (fx_param_double (NULL, "tau", 0.1));

      //default value is 0.1
      if(fx_param_exists(NULL,"fast_kde_output"))
	{
	  fast_kde.PrintDebug();
	}
	
      //initialize fast_kde_results
      num_query_points=fast_kde.get_query_dataset().n_cols();
      num_of_dimensions=fast_kde.get_query_dataset().n_rows();
    
      fast_kde_results.Init(num_query_points);

      for(index_t i=0;i<num_query_points;i++)
	{ 
	  fast_kde_results[i].Init(num_of_dimensions+1);
	} 


      //default value is 0.1
      printf("all computations done......");
      if (fx_param_exists (NULL, "fast_kde_output")) {
	fast_kde.PrintDebug ();
      }

      printf("Size of reference datset is %d\n",fast_kde.get_reference_dataset().n_cols());
      //get density estimates from fast kde calculations

      for(index_t q=0;q<num_query_points;q++)
	{
	  //for each query point
	  for(index_t d=0;d<num_of_dimensions+1;d++)
	    {
	      //along each dimension
	      fast_kde_results[q][d]=fast_kde.get_density_estimates(q,d);
	    }
	}
      
    }

    query_dataset.Alias(fast_kde.get_query_dataset());
    reference_dataset.Alias(fast_kde.get_reference_dataset());
    rset_weights.Alias(fast_kde.get_reference_weights());
    printf("all values copied for naive kde...\n");
  }

  if (do_naive){
    //Vector fast_kde_results;
	  
    NaiveKde <GaussianKernel> naive_kde;
    naive_kde.Init (query_dataset, reference_dataset);

    naive_kde.Compute ();
 
    if (fx_param_exists (NULL, "naive_kde_output")){
      naive_kde.PrintDebug ();
    }
     
     naive_kde.ComputeMaximumRelativeError(fast_kde_results);
  }
  printf("about to call the destructor of fast_kde...\n");
  fx_done ();
  return 0;
}

