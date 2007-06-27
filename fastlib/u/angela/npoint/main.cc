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

 const char *data_file = fx_param_str_req(NULL,"data");
 const char *matcher_file = fx_param_str_req(NULL,"matcher");
 const char *metric_file = fx_param_str(NULL,"metric","default");
 const char *output_file = fx_param_str(NULL,"output","stderr");
 const int n = fx_param_int(NULL,"n",2);
 const int nweights = fx_param_int(NULL,"nweights",0);

 DataPack data;
 Matcher matcher;
 Metric metric;
 Vector count;
 String tmp;
 int i;

 fprintf(output,"Trying to open '%s'.\n",output_file);
 tmp.Copy(output_file);
 if ( tmp.CompareTo("stderr") ) {
	 FILE *tmp_file = fopen(output_file,"w");
	 if (tmp_file != NULL) {
		 fprintf(output,"Successfully opened '%s'. Subsequent messages and results will be written here.\n\n",output_file);
		 output = tmp_file;
	 }
	 else {
		 fprintf(output,"Could not open '%s'. Subsequent messages and results will be written to 'stderr'.\n\n",output_file);
	 }
 }
 tmp.Destruct();

 fprintf(output,"\nLoading the data.\n");
 if ( !PASSED(data.InitFromFile(data_file,nweights)) ) {
	 fprintf(output,"Fatal error: Unable to load data.\n\n");
	 exit(1);
 }
 fprintf(output, "Successfully loaded %d points with %d dimensions and %d weights from %s.\n",
		 data.num_points(), data.num_dimensions(), nweights, data_file);
 if (data.num_dimensions() < 1) {
	 fprintf(output,"Fatal error: The file contains only weights! No matching is possible. \n\n");
	 exit(1);
 }

 fprintf(output,"\nLoading the matcher.\n");
 if ( !PASSED(matcher.InitFromFile(n, matcher_file)) ) {
	 fprintf(output,"Fatal error: Could not load matcher.\n\n");
	 exit(1);
 }
 fprintf(output,"The following matcher was created:\n");
 matcher.Print2File(output);

 tmp.Copy(metric_file);
 if ( tmp.CompareTo("default") ) {
	 fprintf(output,"\nLoading the metric.\n");
	 if ( !PASSED(metric.InitFromFile(data.num_dimensions(), metric_file)) ) {
		 fprintf(output,"Error: Could not load metric. Trying to use euclidean metric for the specified dimension.\n");
	 }
 }
 else {
	 fprintf(output,"\nCreating the metric.\n");
	 metric.Init(data.num_dimensions());
 }
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

