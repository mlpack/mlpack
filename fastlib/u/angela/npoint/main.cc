/**
 * @author: Angela N Grigoroaia
 * @file: main.cc
 *
 * @description:
 * Main file that reads the arguments and decides what part of the code to call.
 *
 * Input data and a matcher must be specified by:
 * 	--data=[csv or arff file containing the input data]
 * 	--matcher=[csv file containing the matcher(s)]
 * The following arguments are optional:
 * 	--n=[size of n-tuple (default 2)]
 * 	--nweights=[number of weights (default 0)]
 * 	--metric=[csv file containing the desired metric]
 * etc. 	
 */


#include "fastlib/fastlib.h"
#include "globals.h"
#include "metrics.h"
#include "matcher.h"
#include "datapack.h"
#include "naive.h"



int main(int argc, char *argv[])
{
 fx_init(argc,argv);

 const char *data_file = fx_param_str_req(NULL, "data");
 const char *matcher_file = fx_param_str_req(NULL, "matcher", NULL);
 const char *metric_file = fx_param_str(NULL, "metric",NULL);
 const int n = fx_param_int(NULL,"n",2);
 const int nweights = fx_param_int(NULL,"nweights",0);

 Dataset data;
 Matcher matcher;
 Metric metric;
 index_t i, j;
 int dim, n_points;
 double count = 0;


 if (!PASSED(data.InitFromFile(data_file))) {
	 fprintf(stderr, "%s: Couldn't open file '%s'. No datapoints available.\n", argv[0], data_file);
	 exit(1);
	}
 dim = data.n_features();
 n_points = data.n_points();
 fprintf(stderr, "Successfully loaded %d points in %d dimensions from %s.\n", n_points, dim, data_file);

 fprintf(stderr,"Loading metric\n");
 metric.InitFromFile(dim, metric_file);
 fprintf(stderr,"The following metric was created:\n");
 for (i=0;i<dim;i++) {
	 for (j=0;j<dim;j++) {
		 fprintf(stderr,"%f  ", metric.M.get(i,j));
	 }
	 fprintf(stderr,"\n");
	}

 fprintf(stderr,"Loading matcher\n");
 matcher.InitFromFile(n, matcher_file);
 fprintf(stderr,"The following matcher was created:\n");
 for  (i=0;i<n;i++) {
         for (j=0;j<n;j++) {
					 fprintf(stderr,"%f  ", matcher.lo.get(i,j));
				 }
         fprintf(stderr,"\n");
        }
 for  (i=0;i<n;i++) {
	 for (j=0;j<n;j++) {
		 fprintf(stderr,"%f  ", matcher.hi.get(i,j));
	 }
	 fprintf(stderr,"\n");
	}

 fprintf(stderr,"Running naive n-point\n");
 count = naive_npoint(data.matrix(), matcher, metric);
 fprintf(stderr,"\nThere are %f distinct matching %d-tuples\n\n", count, n);

 fx_done();
}

#if 0
double naive_npoint(Matrix data, Matcher matcher, Metric metric)
{
 double count = 0;
 index_t i, top;
 int n_points = data.n_cols(), n = matcher.n;
 Vector index;
 index.Init(n);

 /* Initializing the n-tuple index */
 for (i = 0; i < n; i++) {
	 index[i] = i;
 }
 top = n-1;
 /* Running the actual naive loop */
 do {
	 if (PASSED(matcher.Matches(data, index, metric))) {
		 count = count + 1;
	 }

	 /* Generating a new valid n-tuple index */
	 index[top] = index[top] + 1;
	 while (index[top] > (n_points - n + top)) {
		 index[top] = top-1;
		 top--;
		 /* If we can't make a new index we're done. Yupiii!!! */
		 if (top < 0) {
			 return count;
		 }
		 index[top] = index[top] + 1;
	 }
	 for (i=1;i<n;i++) {
		 if (index[i] <= index[i-1]) {
			 index[i] = index[i-1] + 1;
		 }
	 }	 
	 top = n-1;
 } 
 while (1);
}

#endif
