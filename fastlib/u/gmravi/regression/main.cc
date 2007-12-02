#include "fastlib/fastlib_int.h"
#include "regression.h"
#define  MAXDOUBLE 32768.0

int
main (int argc, char *argv[])
{
  fx_init (argc, argv);

  const char *algorithm = fx_param_str_req (NULL, "method");
  bool do_naive = fx_param_exists (NULL, "do_naive");

  Matrix query_dataset;
  Matrix reference_dataset;
  Vector rset_weights;
  ArrayList<Vector> fast_kde_results;
  FastKde <GaussianKernel> fast_kde;
  if (!strcmp (algorithm, "fast"))
    {
      if (!strcmp (fx_param_str (NULL, "kernel", "gaussian"), "gaussian"))
	{
	  //  printf("will do  fast kde with gaussian kernel....\n");
	  ArrayList<Vector> fast_kde_results;
	
	  printf ("O(p^D) expansion KDE\n");
	 
	  fast_kde.Init ();
	  fast_kde.Compute (fx_param_double (NULL, "tau", 0.1));	
	  //default value is 0.1

	  printf("all computations done......");
	  if (fx_param_exists (NULL, "fast_kde_output")) {
	    fast_kde.PrintDebug ();
	  }
	  printf("Size of reference datset is %d\n",fast_kde.get_reference_dataset().n_cols());
	  fast_kde_results.Init(fast_kde.get_query_dataset().n_cols());

	  for(int i=0;i<fast_kde.get_query_dataset().n_cols();i++)
	    fast_kde_results[i].Alias(fast_kde.get_density_estimates(i));
	  
	  
	  query_dataset.Copy (fast_kde.get_query_dataset());
	  reference_dataset.Copy (fast_kde.get_reference_dataset());
	  rset_weights.Copy(fast_kde.get_reference_weights());
	  printf("all values copied for naive kde...\n");
	}


      if (do_naive)
	{
	  //Vector fast_kde_results;
	  
	  NaiveKde <GaussianKernel> naive_kde;
	  naive_kde.Init (query_dataset, reference_dataset);

	  naive_kde.Compute ();
 
	  if (fx_param_exists (NULL, "naive_kde_output"))
	    {
	      naive_kde.PrintDebug ();
	    }
	  printf("Naive KDE done...\n");
	  printf("Fast kde results length is %d\n",fast_kde_results.size());
	  naive_kde.ComputeMaximumRelativeError(fast_kde_results);
	}
    }
  //Do fastkde wih epanechnikov kernel
  else
    {
      /* if (!strcmp (fx_param_str (NULL, "kernel", "epan"), "epan"))
	 {
	 FastKde < EpanKernel > fast_kde;
	 fast_kde.Init ();
	 fast_kde.Compute (fx_param_double (NULL, "tau", 0.1));

	 if (fx_param_exists (NULL, "fast_kde_output"))
	 {
	 fast_kde.PrintDebug ();
	 }

	 Vector fast_kde_results;
	 fast_kde_results.Alias (fast_kde.get_density_estimates ());

	 if (do_naive)
	 {
	 NaiveKde < EpanKernel > naive_kde;
	 naive_kde.Init (fast_kde.get_query_dataset (),
	 fast_kde.get_reference_dataset ());
	 naive_kde.Compute ();

	 if (fx_param_exists (NULL, "naive_kde_output"))
	 {
	 naive_kde.PrintDebug ();
	 }
	 naive_kde.ComputeMaximumRelativeError (fast_kde_results);
	 }
	 }*/
      printf("do nothing..............\n");

      fx_done ();
      return 0;
    }
}
