#include <mlpack/core.h>
#include "approx_max_ip.h"
#include "check_nn_utils.h"

#include <string>
#include <vector>
#include <armadillo>

using namespace mlpack;
using namespace std;

int main (int argc, char *argv[]) {

  srand(time(NULL));

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

  if (CLI::GetParam<int>("max_k") != 1) {

    max_k = CLI::GetParam<int>("max_k");
    number_of_ks = max_k;
    ks.set_size(number_of_ks);

    for (size_t i = 0; i < max_k; i++)
      ks(i) = i+1;
  }

  if (CLI::GetParam<string>("klist") != "") {

    number_of_ks = 0;
    string k_values = CLI::GetParam<string>("klist");

    ks.set_size(10);

    char *pch = strtok((char *) k_values.c_str(), ",");
    while (pch != NULL) {
      ks(number_of_ks++) = atoi(pch);
      pch = strtok(NULL, ",");
    }

    free(pch);
    max_k = ks(number_of_ks -1);
  }

  arma::vec eps(100);
  // arma::vec als(10);
  size_t num_eps = 0; //, num_als = 0;

  string epsilons = CLI::GetParam<string>("epslist");

  char *pch = strtok((char *) epsilons.c_str(), ",");
  while (pch != NULL) {
    eps(num_eps++) = atof(pch);
    pch = strtok(NULL, ",");
  }

  free(pch);

  Log::Warn << number_of_ks << " values for k," << endl
	    << num_eps << " values for epsilon." << endl;


  Log::Warn << "Starting loop for Fast Approx-Search." << endl;

  // If you did multiple repetitions, put the loop here 
  // have the following:
  // size_t total_reps = CLI::GetParam<int>("reps");
  // arma::mat res = arma::zeros<arma::mat>(number_of_ks * num_eps, 5);
  // for (size_t reps = 0; reps < total_reps; reps++) {



  ApproxMaxIP fast_approx;
  vector< arma::Mat<size_t>* > all_solutions;
  vector<double> all_th_speedups;

  // vector<double> all_actual_times;

  fast_approx.InitApprox(qdata, rdata);

  for (size_t i = 0; i < number_of_ks; i++) {

    printf("k = %zu", ks(i)); fflush(NULL);

    for (size_t j = 0; j < num_eps; j++) {

      printf(", eps = %lg", eps(j)); fflush(NULL);

      fast_approx.WarmInitApprox(ks(i), eps(j));

      arma::Mat<size_t>* indices = new arma::Mat<size_t>();
      arma::mat* values = new arma::mat();
      double approx_comp = fast_approx.ComputeApprox(indices, values);
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
  vector<double> avg_precisions; //, avg_precisions2;
  vector<size_t> median_ranks; //2, median_ranks2;

  if (CLI::GetParam<string>("rank_file") != "") {
    string rank_file = CLI::GetParam<string>("rank_file");
    check_nn_utils::compute_error(rank_file, rdata.n_cols, all_solutions,
				  &avg_precisions, &median_ranks);
  } else {

    check_nn_utils::compute_error(&rdata, &qdata, all_solutions,
				  &avg_precisions, &median_ranks);
  }


  for (size_t i = 0; i < all_solutions.size(); i++) {
     all_solutions[i]->reset();
     delete(all_solutions[i]);
  }

  // print the results here somehow
  assert(avg_precisions.size() == number_of_ks * num_eps);
  assert(median_ranks.size() == number_of_ks * num_eps);
  assert(avg_precisions.size() == all_th_speedups.size());



  // If we are performing reps, we have to make sure 
  // that the check_nn_utils::compute_error() is called
  // only once (especially for the Yahoo data set)


//   for (size_t i = 0; i < number_of_ks; i++) {
//     for (size_t j = 0; j < num_eps; j++) {

//       res(i * num_eps + j, 0) += (double)  ks(i);
//       res(i * num_eps + j, 1) += eps(j);
// 	 res(i * num_eps + j, 2) += avg_precisions[i * num_eps + j];
// 	 res(i * num_eps + j, 3) += (double) median_ranks[i * num_eps + j];
// 	 res(i * num_eps + j, 4) += all_th_speedups[i * num_eps + j]);
//     }
//   }


// } // reps-loop
//   if (CLI::GetParam<string>("res_file") != "") {
  
//     string res_file = GetParam<string>("res_file");
//     FILE *res_fp = fopen(res_file.c_str(), "w");
//     for (size_t i = 0; i < res.n_rows; i++) {
//       for (size_t j = 0; j < res.n_cols; j++) {
// 	    fprintf(res_fp, "%lg", res(i, j) / (double) total_reps);
//          if (j == res.n_cols -1)
//            fprintf(res_fp, "\n");
//          else
//            fprintf(res_fp, ",");
//       }
//     }
//   } else {

//     printf("k\te\tp\tmr\tth_sp\n");
//     for (size_t i = 0; i < res.n_rows; i++) {
//       for (size_t j = 0; j < res.n_cols; j++) {
// 	    printf("%lg", res(i, j) / (double) total_reps);
//          if (j == res.n_cols -1)
//            printf("\n");
//          else
//            printf(",");
//       }
//     }
//   }



  if (CLI::GetParam<string>("res_file") != "") {
  
    string res_file = CLI::GetParam<string>("res_file");
    FILE *res_fp = fopen(res_file.c_str(), "w");
    for (size_t i = 0; i < number_of_ks; i++)
      for (size_t j = 0; j < num_eps; j++)
	fprintf(res_fp, "%zu,%lg,%lg,%zu,%lg\n", ks(i), eps(j),
		avg_precisions[i * num_eps + j],
		median_ranks[i * num_eps + j],
		all_th_speedups[i * num_eps + j]);
  
  } else {

    printf("k\te\tp\tmr\tth_sp\n");
    for (size_t i = 0; i < number_of_ks; i++) {
      for (size_t j = 0; j < num_eps; j++) {

	printf("%zu\t%lg\t%lg\t%zu\t%lg\n", ks(i), eps(j),
	       avg_precisions[i * num_eps + j],
	       median_ranks[i * num_eps + j],
	       all_th_speedups[i * num_eps + j]); fflush(NULL);
      }
    }
  }
}  // end main



PROGRAM_INFO("Approx IP Tester", "This program "
 	     "tests the maximum inner product search for a "
 	     "given query over a set of points (references) with"
	     " varying values of k, epsilon and alpha.", 
	     "approx_maxip");

PARAM_STRING_REQ("r", "The reference set", "");
PARAM_STRING_REQ("q", "The set of queries", "");

PARAM_STRING("klist", "The comma-separated list of values of"
	     " k to be tried.", "", "");

PARAM_STRING_REQ("epslist", "The comma-separated list of epsilons",
	     "");
// PARAM_STRING("alphas", "The comma-separated list of alphas", "");

PARAM_INT("reps", "The number of times the rank-approximate"
	  " algorithm is to be repeated for the same setting.",
  	  "", 1);

PARAM_INT("max_k", "The max value of knns to be tried.", "", 1);

PARAM_STRING("rank_file", "The file containing the ranks.",
	     "", "");
PARAM_STRING("res_file", "The file where the results are to be written.",
	     "", "");
