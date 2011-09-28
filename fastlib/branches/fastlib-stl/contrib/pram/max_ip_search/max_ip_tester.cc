#include <fastlib/fastlib.h>
#include "exact_max_ip.h"

#include <string>
#include <armadillo>

using namespace mlpack;
using namespace std;

PROGRAM_INFO("Maximum IP Tester", "This program "
 	     "returns the maximum inner product for a "
 	     "given query over a set of points (references).", 
	     "maxip");

// PARAM_MODULE("maxip_main", "Parameters for the main "
// 	     "file to compute the maximum inner product "
// 	     "for a given query over a given set of "
// 	     "references.");

PARAM_STRING_REQ("r", "The reference set", "");
PARAM_STRING_REQ("q", "The set of queries", "");
PARAM_INT_REQ("max_k", "The max value of knns to be tried.", "");

/**
 * This function checks if the neighbors computed 
 * by two different methods is the same.
 */
size_t count_mismatched_neighbors(arma::Col<size_t>, arma::vec, 
 				arma::Col<size_t>, arma::vec);


int main (int argc, char *argv[]) {

  IO::ParseCommandLine(argc, argv);

  arma::mat rdata, qdata;
  string rfile = IO::GetParam<string>("r");
  string qfile = IO::GetParam<string>("q");

  IO::Info << "Loading files..." << endl;
  if (!data::Load(rfile.c_str(), rdata))
    IO::Fatal << "Reference file "<< rfile << " not found." << endl;

  if (!data::Load(qfile.c_str(), qdata)) 
    IO::Fatal << "Query file " << qfile << " not found." << endl;

  IO::Info << "File loaded..." << endl;
  
  IO::Info << "R(" << rdata.n_rows << ", " << rdata.n_cols 
	   << "), Q(" << qdata.n_rows << ", " << qdata.n_cols 
	   << ")" << endl;



  size_t knns = IO::GetParam<int>("maxip/knns");
  size_t max_k = IO::GetParam<int>("max_k");

  MaxIP naive, fast_exact;

  naive.InitNaive(qdata, rdata);
  fast_exact.Init(qdata, rdata);

  for (knns = 1; knns <= max_k; knns++) {
    double naive_comp, fast_comp;
    arma::Col<size_t> nac, exc;
    arma::vec din, die;

    naive_comp = naive.ComputeNaive(&nac, &din);
    fast_comp = fast_exact.ComputeNeighbors(&exc, &die);


    size_t errors = count_mismatched_neighbors(nac, din, exc, die);

    if (errors > 0) {
      IO::Warn << knns << "-NN error: " << errors << " / "
	       << nac.n_elem << endl;
      IO::Warn << "Speed of fast-exact over naive: "
	       << naive_comp << " / " << (float) fast_comp << " = "
	       <<(float) (naive_comp / fast_comp) << endl;
    }

    naive.WarmInit(knns+1);
    fast_exact.WarmInit(knns+1);
  }
}

// void compare_neighbors(arma::Col<size_t> *a, 
//                        arma::vec *da,
//                        arma::Col<size_t> *b, 
//                        arma::vec *db) {
  
//   IO::Info << "Comparing results for %zud queries", a->size());
//   DEBUG_SAME_SIZE(a->size(), b->size());
//   size_t *x = a->begin();
//   size_t *y = a->end();
//   size_t *z = b->begin();

//   for(size_t i = 0; x != y; x++, z++, i++) {
//     DEBUG_WARN_MSG_IF(*x != *z || (*da)[i] != (*db)[i], 
//                       "point %zud brute: %zud:%lf fast: %zud:%lf",
//                       i, *z, (*db)[i], *x, (*da)[i]);
//   }
// }

size_t count_mismatched_neighbors(arma::Col<size_t> a, 
 				arma::vec da,
 				arma::Col<size_t> b, 
 				arma::vec db) {
  // IO::Warn << "Comparing results for " << a.n_elem << " queries." << endl;
  assert(a.n_elem == b.n_elem);
  size_t count_mismatched = 0;

//   IO::Warn << "Mismatches: " << endl;
  for(size_t i = 0; i < a.n_elem;  i++) {
    if (da(i) != db(i)) {
      ++count_mismatched;
      // IO::Warn << da(i) - db(i) << endl;
    }
  }

//   IO::Warn << count_mismatched << " / " << a.n_elem
// 	    << " errors." << endl;
  return count_mismatched;
}
