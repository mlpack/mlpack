#include <armadillo>
#include <string>
#include <vector>

#include <mlpack/core.h>
#include <mlpack/core/kernels/lmetric.hpp>

using namespace mlpack;
using namespace std;

// Add params for large scale or small scale computation
PROGRAM_INFO("Distance Computer", "This program computes the "
	     "complete distance list for the given queries and "
	     "references.", "");

PARAM_STRING_REQ("r", "The reference set", "");
PARAM_STRING_REQ("q", "The set of queries", "");
PARAM_STRING_REQ("dist_file", "The file where the rank "
		 "matrix would be written in.", "");
PARAM_INT_REQ("num_q", "The number of queries to be used for "
	      "this computation.", "");

int main (int argc, char *argv[]) {
  
  CLI::ParseCommandLine(argc, argv);

  arma::mat rdata, qdata;
  string rfile = CLI::GetParam<string>("r");
  string qfile = CLI::GetParam<string>("q");

  Log::Warn << "Loading files..." << endl;
  if (rdata.load(rfile.c_str()) == false)
    Log::Fatal << "Reference file "<< rfile << " not found." << endl;

  if (qdata.load(qfile.c_str()) == false) 
    Log::Fatal << "Query file " << qfile << " not found." << endl;

  rdata = arma::trans(rdata);
  qdata = arma::trans(qdata);

  Log::Warn << "File loaded..." << endl;
  
  Log::Warn << "R(" << rdata.n_rows << ", " << rdata.n_cols 
	    << "), Q(" << qdata.n_rows << ", " << qdata.n_cols 
	    << ")" << endl;

  string dist_file = CLI::GetParam<string>("dist_file");
  FILE *pfile = fopen(dist_file.c_str(), "w");

  mlpack::kernel::SquaredEuclideanDistance dist_kernel 
    = mlpack::kernel::SquaredEuclideanDistance();

  double perc_done = 10.0;
  double done_sky = 1.0;
	
  // do it with a loop over the queries.
  size_t num_q = CLI::GetParam<int>("num_q");

  for (size_t i = 0; i < num_q; i++) {

    arma::vec q = qdata.unsafe_col(i);

    for (size_t j = 0; j < rdata.n_cols; j++) {

      arma::vec r = rdata.unsafe_col(j);
      // obtaining the distances
      fprintf(pfile, "%lg\n",
	      sqrt(dist_kernel.Evaluate(q,r)));
    }

    double pdone = i * 100 / num_q;

    if (pdone >= done_sky * perc_done) {
      if (done_sky > 1) {
	printf("\b\b\b=%zu%%", (size_t) pdone); fflush(NULL); 
      } else {
	printf("=%zu%%", (size_t) pdone); fflush(NULL);
      }
      done_sky++;
    }
  } // query-loop

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

  fclose(pfile);
  Log::Info << "Distances computed!" << endl;
} // end main
