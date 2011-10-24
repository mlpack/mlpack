#include <fastlib/fastlib.h>
#include "exact_max_ip.h"

#include <string>
#include <armadillo>

using namespace mlpack;
using namespace std;

PROGRAM_INFO("Maximum IP Tester", "This program "
 	     "tests the maximum inner product search for a "
 	     "given query over a set of points (references) with"
	     " varying values of k.", 
	     "maxip");

// PARAM_MODULE("maxip_main", "Parameters for the main "
// 	     "file to compute the maximum inner product "
// 	     "for a given query over a given set of "
// 	     "references.");

PARAM_STRING_REQ("r", "The reference set", "");
PARAM_STRING_REQ("q", "The set of queries", "");

PARAM_INT("max_k", "The max value of knns to be tried.", "", 1);
PARAM_STRING("k_values", "The comma-separated list of values of"
	     " 'k' to be tried.", "", "");



PARAM_STRING("speedup_file", "The file in which to save the speedups"
	     " for the different values of k.", "", "speedups.txt");
PARAM_FLAG("print_speedups", "The flag to trigger the printing of"
	   " speedups.", "");

PARAM_FLAG("check_nn", "The flag to trigger the checking"
	   " of the results by doing the naive computation.", "");


/**
 * This function checks if the neighbors computed 
 * by two different methods is the same.
 */
size_t count_mismatched_neighbors(arma::mat, size_t, 
				  arma::mat, size_t);


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



  if (CLI::HasParam("max_k")) {

    MaxIP naive, fast_exact;
    double naive_comp = (double) rdata.n_cols;
    arma::Mat<size_t> nac;
    arma::mat din;

    size_t max_k = CLI::GetParam<int>("max_k");
    arma::vec speedups(max_k);

    if (CLI::HasParam("check_nn")) { 
      Log::Warn << "Starting naive computation..." <<endl;
      naive.InitNaive(qdata, rdata);
      naive.WarmInit(max_k);
      naive_comp = naive.ComputeNaive(&nac, &din);
      Log::Warn << "Naive computation done..." << endl;
    }

    Log::Warn << "Starting loop for Fast Exact Search." << endl;

    fast_exact.Init(qdata, rdata);

    for (knns = 1; knns <= max_k; knns++) {

      printf("k = %zu", knns); fflush(NULL);
      arma::Mat<size_t> exc;
      arma::mat die;
      double fast_comp = fast_exact.ComputeNeighbors(&exc, &die);

      if (CLI::HasParam("check_nn")) {
	size_t errors = count_mismatched_neighbors(din, max_k, die, knns);

	if (errors > 0) {
	  Log::Warn << knns << "-NN error: " << errors << " / "
		    << exc.n_elem << endl;
	}
      }

      speedups(knns -1) = naive_comp / fast_comp;
      printf(": %lg\n", speedups(knns -1));

      fast_exact.WarmInit(knns+1);
    }

    printf("\n");
  }

  if (CLI::HasParam("k_values")) {

    MaxIP naive, fast_exact;
    double naive_comp = (double) rdata.n_cols;
    arma::Col<size_t> nac;
    arma::vec din;

    size_t number_of_ks = 0;
    string k_values = CLI::GetParam<string>("k_values");
    arma::Col<size_t> ks(10);

    char *temp = (char *) k_values.c_str();
    char *pch = strtok(temp, ",");
    while (pch != NULL) {
      ks(number_of_ks++) = atoi(pch);
      pch = strtok(NULL, ",");
    }

    arma::vec speedups(number_of_ks);

    if (CLI::HasParam("check_nn")) { 
      Log::Warn << "Starting naive computation..." <<endl;
      naive.InitNaive(qdata, rdata);
      naive.WarmInit(ks(number_of_ks - 1));
      naive_comp = naive.ComputeNaive(&nac, &din);
      Log::Warn << "Naive computation done..." << endl;
    }

    Log::Warn << "Starting loop for Fast Exact Search." << endl;

    fast_exact.Init(qdata, rdata);

    for (size_t knns = 0; knns < number_of_ks; knns++) {

      printf("k = %zu", ks(knns)); fflush(NULL);
      arma::Col<size_t> exc;
      arma::vec die;
      double fast_comp = fast_exact.ComputeNeighbors(&exc, &die);

      if (CLI::HasParam("check_nn")) {
	size_t errors = count_mismatched_neighbors(din, ks(number_of_ks -1),
						   die, ks(knns));

	if (errors > 0) {
	  Log::Warn << ks(knns) << "-NN error: " << errors << " / "
		    << exc.n_elem << endl;
	}
      }

      speedups(knns -1) = naive_comp / fast_comp;
      printf(": %lg\n", speedups(knns -1));

      if (knns < number_of_ks - 1)
	fast_exact.WarmInit(ks(knns+1));
    }

    printf("\n");
  }
  
  Log::Warn << "Search completed for all values of k...printing results now"
	   << endl;

//   if (CLI::HasParam("print_speedups")) {
//     string speedup_file = CLI::GetParam<string>("speedup_file");
//     speedups.save(speedup_file, arma::raw_ascii);
//   }
}  // end main

size_t count_mismatched_neighbors(arma::mat v1, size_t k1,
				  arma::mat v2, size_t k2) {

  assert(v1.n_cols == v2.n_cols);
  size_t count_mismatched = 0;

  size_t num_queries = v1.n_cols;
  for(size_t i = 0; i < num_queries;  i++) {

    for (size_t j = 0; j < std::min(k1, k2); j++)
      if (v1(j, i) != v2(j, i)) 
	++count_mismatched;
  }

  return count_mismatched;
}
