#include <fastlib/fastlib.h>
#include "approx_max_ip.h"

#include <string>
#include <armadillo>

using namespace mlpack;
using namespace std;

PROGRAM_INFO("Approx IP Tester", "This program "
 	     "tests the maximum inner product search for a "
 	     "given query over a set of points (references) with"
	     " varying values of k, epsilon and alpha.", 
	     "approx_maxip");

// PARAM_MODULE("maxip_main", "Parameters for the main "
// 	     "file to compute the maximum inner product "
// 	     "for a given query over a given set of "
// 	     "references.");

PARAM_STRING_REQ("r", "The reference set", "");
PARAM_STRING_REQ("q", "The set of queries", "");

PARAM_INT("max_k", "The max value of knns to be tried.", "", 1);

PARAM_STRING("k_values", "The comma-separated list of values of"
	     " 'k' to be tried.", "", "");

PARAM_STRING_REQ("epsilons", "The comma-separated list of epsilons", "");
// PARAM_STRING_REQ("alphas", "The comma-separated list of alphas", "");

PARAM_INT("reps", "The number of times the rank-approximate"
	  " algorithm is to be repeated for the same setting.",
	  "", 1);

PARAM_STRING("rank_file", "The file containing the ranks.",
	     "", "");

// PARAM_STRING("speedup_file", "The file in which to save the speedups"
// 	     " for the different values of k.", "", "speedups.txt");
// PARAM_FLAG("print_speedups", "The flag to trigger the printing of"
// 	   " speedups.", "");

// PARAM_FLAG("check_nn", "The flag to trigger the checking"
// 	   " of the results by doing the naive computation.", "");


/**
 * This function checks if the neighbors computed 
 * by two different methods is the same.
 */

void compute_error(arma::mat rdata, arma::mat qdata,
		   arma::Mat<size_t> indices, arma::mat values);

void compute_error(arma::Mat<size_t> indices, arma::mat values,
		   size_t rdata_size);


int main (int argc, char *argv[]) {

  CLI::ParseCommandLine(argc, argv);

  arma::mat rdata, qdata;
  string rfile = CLI::GetParam<string>("r");
  string qfile = CLI::GetParam<string>("q");

  Log::Warn << "Loading files..." << endl;
  if (!data::Load(rfile.c_str(), rdata))
    Log::Fatal << "Reference file "<< rfile << " not found." << endl;

  if (!data::Load(qfile.c_str(), qdata)) 
    Log::Fatal << "Query file " << qfile << " not found." << endl;

  Log::Warn << "File loaded..." << endl;
  
  Log::Warn << "R(" << rdata.n_rows << ", " << rdata.n_cols 
	   << "), Q(" << qdata.n_rows << ", " << qdata.n_cols 
	   << ")" << endl;


  arma::Col<size_t> ks;
  size_t max_k, number_of_ks;

  if (CLI::HasParam("max_k")) {

    max_k = CLI::GetParam<int>("max_k");
    number_of_ks = max_k;
    ks.set_size(number_of_ks);

    for (size_t i = 0; i < max_k; i++)
      ks(i) = i+1;
  }

  if (CLI::HasParam("k_values")) {

    number_of_ks = 0;
    string k_values = CLI::GetParam<string>("k_values");

    ks.set_size(10);

    char *temp = (char *) k_values.c_str();
    char *pch = strtok(temp, ",");
    while (pch != NULL) {
      ks(number_of_ks++) = atoi(pch);
      pch = strtok(NULL, ",");
    }

    free(temp);
    free(pch);

    max_k = ks(number_of_ks -1);
  }

  arma::vec eps(25);
  arma::vec als(10);
  size_t num_eps = 0, num_als = 0;

  string epsilons = CLI::GetParam<string>("epsilons");

  char *temp = (char *) epsilons.c_str();
  char *pch = strtok(temp, ",");
  while (pch != NULL) {
    eps(num_eps++) = atof(pch);
    pch = strtok(NULL, ",");
  }

  free(temp);
  free(pch);

  Log::Info << number_of_ks << " values for k," << endl
	    << num_eps << " values for epsilon." << endl;

  Log::Warn << "Starting loop for Fast Approx-Search." << endl;

  ApproxMaxIP fast_approx;
  vector< arma::Mat<size_t>* > all_solutions;
  vector<double> all_th_speedups;

  // vector<double> all_actual_times;

  fast_approx.Init(qdata, rdata);

  for (size_t i = 0; i < number_of_ks; i++) {

    printf("k = %zu", ks(i)); fflush(NULL);

    for (size_t j = 0; j < num_eps; j++) {

      printf(", eps = %lg", eps(j)); fflush(NULL);

      fast_approx.WarmInit(ks(i), eps(j));

      arma::Mat<size_t>* indices = new arma::Mat<size_t>();
      arma::mat values ;
      double approx_comp = fast_approx.ComputeApprox(indices, &values);
      double th_speedup = rdata.n_cols / approx_comp;

      all_solutions.push_back(indices);
      all_th_speedups.push_back(th_speedup);

      // find a way to compute the actual times
      // double actual_time;

    }

    printf("\n");
  }
  
  Log::Warn << "Search completed for all values of k...checking results now"
	    << endl;

  // computing the precision values and the median ranks from the results
  vector<double> avg_precisions;
  vector<size_t> median_ranks;

  if (CLI::HasParam("rank_file"))
    compute_error(all_solutions, rdata.n_cols,
		  &avg_precision, &median_ranks);
  else
    compute_error(all_solutions, &rdata, &qdata,
		  &avg_precisions, &median_ranks);

  // print the results here somehow

}  // end main


void compute_error(vector< arma::Mat<size_t>* > solutions,
		   size_t rdata_size,
		   vector<double>* precisions,
		   vector<size_t>* median_ranks) {

  vector< arma::Col<size_t>* > all_ranks_lists;
  vector<size_t> all_corrects;

  // START WORK FROM HERE


  arma::Col<size_t> all_ranks_list;
  size_t all_correct = 0;

  size_t k = indices->n_rows;

  all_ranks_list.set_size(indices->n_cols * indices->n_rows);


  double perc_done = 10.0;
  double done_sky = 1.0;
        
  FILE *rank_file = fopen(CLI::GetParam<string>("rank_file").c_str(), "r");

  // do it with a loop over the queries.
  for (size_t i = 0; i < indices->n_cols; i++) {

    // obtaining the rank list
    arma::Col<size_t> srt_ind;
    srt_ind.set_size(rdata_size);

    if (rank_file != NULL) {
      char *line = NULL;
      size_t len = 0;
      getline(&line, &len, rank_file);

      char *pch = strtok(line, ",\n");
      size_t rank_index = 0;

      while(pch != NULL) {
	srt_ind(rank_index++) = atoi(pch);
	pch = strtok(NULL, ",\n");
      }

      free(line);
      free(pch);
      assert(rank_index == rdata_size);
    }

    for (size_t j = 0; j < k; j++) {

      size_t rank = srt_ind(indices(j, i)) + 1;
      all_ranks_list( i * k + j ) = rank;
      if (rank < k + 1)
	all_correct++;

    }

    double pdone = i * 100 / qdata.n_cols;

    if (pdone >= done_sky * perc_done) {
      if (done_sky > 1) {
	printf("\b\b\b=%zu%%", (size_t) pdone); fflush(NULL); 
      } else {
	printf("=%zu%%", (size_t) pdone); fflush(NULL);
      }
      done_sky++;
    }

  } // query-loop


  fclose(rank_file);

  double pdone = 100;

  if (pdone >= done_sky * perc_done) {
    if (done_sky > 1) {
      printf("\b\b\b=%zu%%", (size_t) pdone); fflush(NULL); 
    } else {
      printf("=%zu%%", (size_t) pdone); fflush(NULL);
    }
    done_sky++;
  }
  printf("\n");fflush(NULL);

  Log::Info << "Errors Computed!" << endl;


  double avg_precision = (double) all_correct 
    / (double) (k * qdata->n_cols);

  size_t median_rank = arma::median(all_ranks_list);

}


void compute_error(vector< arma::Mat<size_t>* > solutions,
		   arma::mat rdata, arma::mat qdata,
		   vector<double>* precisions,
		   vector<size_t>* median_ranks) {
}
