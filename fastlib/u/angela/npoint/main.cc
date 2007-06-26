/**
 * @author: Angela N Grigoroaia
 * @file: main.cc
 *
 * @description:
 * Main file that reads the arguments and decides what part of the code to call.
 */

/**
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

/* Initiale any global variables here. */

FILE *output = stderr;

/* End initialization for global variables. */


int main(int argc, char *argv[])
{
 fx_init(argc,argv);

 const char *metric_file = fx_param_str(NULL,"metric",NULL);
 const char *data_file = fx_param_str_req(NULL,"data");
 const char *matcher_file = fx_param_str(NULL,"matcher",NULL);
 const int n = fx_param_int(NULL,"n",2);
 const int nweights = fx_param_int(NULL,"nweights",0);

 DataPack data;
 Matcher matcher;
 Metric metric;
 Vector count;
 int i;

 fprintf(output,"\nLoading the data.\n");
 if ( !PASSED(data.InitFromFile(data_file,nweights)) ) {
	 fprintf(output,"Unable to load data. Exiting program.\n");
	 exit(1);
 }
 fprintf(output, "Successfully loaded %d points in %d dimensions and %d weights from %s.\n",
		 data.npoints, data.dimension, nweights, data_file);
 if (data.dimension < 1) {
	 fprintf(output,"The file contains only weights! Aborting run!\n\n");
	 exit(1);
 }

 fprintf(output,"\nLoading the matcher.\n");
 matcher.InitFromFile(n, matcher_file);
 fprintf(output,"The following matcher was created:\n");
 matcher.Print2File(output);

 fprintf(output,"\nLoading the metric.\n");
 metric.InitFromFile(data.dimension, metric_file);
 fprintf(output,"The following metric was created:\n");
 metric.Print2File(output);

 fprintf(output,"\nRunning naive n-point\n");
 count.Copy(naive_npoint(data, matcher, metric));
 fprintf(output,"\n\nThere are %f distinct matching %d-tuples\n", count[0], n);
 for (i = 1; i <= nweights; i++) {
 	fprintf(output,"The %d-th weighted count is %f \n", i, count[i]);
 }
 fprintf(output,"\n\n");

 fx_done();
}

