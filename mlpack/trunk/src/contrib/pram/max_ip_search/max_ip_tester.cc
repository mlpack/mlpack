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
PARAM_INT_REQ("max_k", "The max value of knns to be tried.", "");
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
size_t count_mismatched_neighbors(arma::vec, size_t, 
				  arma::vec, size_t);


int main (int argc, char *argv[]) {

  IO::ParseCommandLine(argc, argv);

  arma::mat rdata, qdata;
  string rfile = IO::GetParam<string>("r");
  string qfile = IO::GetParam<string>("q");

  Log::Warn << "Loading files..." << endl;
  if (!data::Load(rfile.c_str(), rdata))
    Log::Fatal << "Reference file "<< rfile << " not found." << endl;

  if (!data::Load(qfile.c_str(), qdata)) 
    Log::Fatal << "Query file " << qfile << " not found." << endl;

  Log::Warn << "File loaded..." << endl;
  
  Log::Warn << "R(" << rdata.n_rows << ", " << rdata.n_cols 
	   << "), Q(" << qdata.n_rows << ", " << qdata.n_cols 
	   << ")" << endl;



  size_t knns = IO::GetParam<int>("maxip/knns");
  size_t max_k = IO::GetParam<int>("max_k");
  arma::vec speedups(max_k);

  MaxIP naive, fast_exact;
  double naive_comp = (double) rdata.n_cols;
  arma::Col<size_t> nac;
  arma::vec din;

  if (IO::HasParam("check_nn")) { 
    Log::Warn << "Starting naive computation..." <<endl;
    naive.InitNaive(qdata, rdata);
    naive.WarmInit(max_k);
    naive_comp = naive.ComputeNaive(&nac, &din);
    Log::Warn << "Naive computation done..." << endl;
  }

  Log::Warn << "Starting loop for Fast Exact Search." << endl;

  fast_exact.Init(qdata, rdata);

  printf("k = "); fflush(NULL);
  for (knns = 1; knns <= max_k; knns++) {

    printf("%zu", knns); fflush(NULL);
    arma::Col<size_t> exc;
    arma::vec die;
    double fast_comp = fast_exact.ComputeNeighbors(&exc, &die);

    if (IO::HasParam("check_nn")) {
      size_t errors = count_mismatched_neighbors(din, max_k, die, knns);

      if (errors > 0) {
	Log::Warn << knns << "-NN error: " << errors << " / "
		 << exc.n_elem << endl;
      }
    }

    speedups(knns -1) = naive_comp / fast_comp;

    fast_exact.WarmInit(knns+1);
    for (size_t i = 0; i < ceil(log10(knns + 0.001)); i++)
      printf("\b"); fflush(NULL);
  }

  printf("\n");

  Log::Warn << "Search completed for all values of k...printing results now"
	   << endl;

  if (IO::HasParam("print_speedups")) {
    string speedup_file = IO::GetParam<string>("speedup_file");
    speedups.save(speedup_file, arma::raw_ascii);
  }
}  // end main

size_t count_mismatched_neighbors(arma::vec v1, size_t k1,
				  arma::vec v2, size_t k2) {

  assert(v1.n_elem / k1 == v2.n_elem / k2);
  size_t count_mismatched = 0;

  size_t num_queries = v1.n_elem / k1;
  for(size_t i = 0; i < num_queries;  i++) {
    size_t ind1 = i * k1;
    size_t ind2 = i * k2;

    for (size_t j = 0; j < std::min(k1, k2); j++)
      if (v1(ind1 + j) != v2(ind2 + j)) 
	++count_mismatched;
  }

  return count_mismatched;
}
