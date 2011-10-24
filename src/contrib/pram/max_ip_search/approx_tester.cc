#include <mlpack/core.h>
#include "approx_max_ip.h"

#include <string>
#include <vector>
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

PARAM_STRING("klist", "The comma-separated list of values of"
	     " k to be tried.", "", "");

PARAM_STRING_REQ("epslist", "The comma-separated list of epsilons",
	     "");
// PARAM_STRING_REQ("alphas", "The comma-separated list of alphas", "");

// PARAM_INT("reps", "The number of times the rank-approximate"
// 	  " algorithm is to be repeated for the same setting.",
// 	  "", 1);

PARAM_INT("max_k", "The max value of knns to be tried.", "", 1);

PARAM_STRING("rank_file", "The file containing the ranks.",
	     "", "");

// PARAM_STRING("speedup_file", "The file in which to save the speedups"
// 	     " for the different values of k.", "", "speedups.txt");
// PARAM_FLAG("print_speedups", "The flag to trigger the printing of"
// 	   " speedups.", "");


/**
 * This function checks if the neighbors computed 
 * by two different methods is the same.
 */
void compute_error(vector< arma::Mat<size_t>* > solutions,
		   size_t rdata_size,
		   vector<double>* precisions,
		   vector<size_t>* median_ranks);

void compute_error(vector< arma::Mat<size_t>* > solutions,
		   arma::mat rdata, arma::mat qdata,
		   vector<double>* precisions,
		   vector<size_t>* median_ranks);



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

  if (CLI::HasParam("klist")) {

    number_of_ks = 0;
    string k_values = CLI::GetParam<string>("klist");

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
  // arma::vec als(10);
  size_t num_eps = 0; //, num_als = 0;

  string epsilons = CLI::GetParam<string>("epslist");

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

  fast_approx.InitApprox(qdata, rdata);

  for (size_t i = 0; i < number_of_ks; i++) {

    printf("k = %zu", ks(i)); fflush(NULL);

    for (size_t j = 0; j < num_eps; j++) {

      printf(", eps = %lg", eps(j)); fflush(NULL);

      fast_approx.WarmInitApprox(ks(i), eps(j));

      arma::Mat<size_t> indices; // = new arma::Mat<size_t>();
      arma::mat values ;
      double approx_comp = fast_approx.ComputeApprox(&indices, &values);
      double th_speedup = rdata.n_cols / approx_comp;

      all_solutions.push_back(&indices);
      all_th_speedups.push_back(th_speedup);

      // find a way to compute the actual times
      // double actual_time;

    }

    printf("\n");
  }
  
  Log::Warn << "Search completed for all values of k...checking results now"
	    << endl;

  // computing the precision values and the median ranks from the results
  vector<double> avg_precisions1, avg_precisions2;
  vector<size_t> median_ranks2, median_ranks2;

  //  if (CLI::HasParam("rank_file"))
  compute_error(all_solutions, rdata.n_cols,
		&avg_precisions1, &median_ranks1);
  //else
  compute_error(all_solutions, rdata, qdata,
		&avg_precisions2, &median_ranks2);

  // print the results here somehow
  assert(avg_precisions1.size() == number_of_ks * num_eps);
  assert(median_ranks1.size() == number_of_ks * num_eps);
  assert(avg_precisions1.size() == all_th_speedups.size());

  assert(avg_precisions2.size() == number_of_ks * num_eps);
  assert(median_ranks2.size() == number_of_ks * num_eps);
  assert(avg_precisions2.size() == all_th_speedups.size());


  printf("k, epsilon, precision, median rank, th_speedup\n");
  for (size_t i = 0; i < number_of_ks; i++) {
    for (size_t j = 0; j < num_eps; j++) {

      printf("%zu, %lg, %lg, %zu, %lg\n", ks(i), eps(j),
	     avg_precisions1[i * num_eps + j],
	     median_ranks1[i*num_eps + j],
	     all_th_speedups[i*num_eps + j]); fflush(NULL);
    }
  }

  printf("-------------------\n"
	 "k, epsilon, precision, median rank, th_speedup\n");
  for (size_t i = 0; i < number_of_ks; i++) {
    for (size_t j = 0; j < num_eps; j++) {

      printf("%zu, %lg, %lg, %zu, %lg\n", ks(i), eps(j),
	     avg_precisions2[i * num_eps + j],
	     median_ranks2[i*num_eps + j],
	     all_th_speedups[i*num_eps + j]); fflush(NULL);
    }
  }

}  // end main


void compute_error(vector< arma::Mat<size_t>* > solutions,
		   size_t rdata_size,
		   vector<double>* precisions,
		   vector<size_t>* median_ranks) {

  vector< arma::Col<size_t>* > all_ranks_lists;
  vector<double> all_corrects;

  // set up the list first
  for (size_t i = 0; i < solutions.size(); i++) {

    arma::Col<size_t> all_ranks_list; // = new arma::Col<size_t>();
    double all_correct = 0.0;

    all_ranks_list.set_size(solutions[i]->n_cols * solutions[i]->n_rows);
    all_ranks_lists.push_back(&all_ranks_list);

    all_corrects.push_back(all_correct);

  }


  // 1. Pick up the rank list
  // 2. Go through the solution for that query
  // 3. precision and ranks

  double perc_done = 10.0;
  double done_sky = 1.0;
        
  FILE *rank_file = fopen(CLI::GetParam<string>("rank_file").c_str(), "r");


  size_t num_queries = solutions[1]->n_cols;
  size_t num_solutions = solutions.size();

  // do it with a loop over the queries.
  for (size_t i = 0; i < num_queries; i++) {

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


    // Going through all the solutions
    for (size_t ind = 0; ind < num_solutions; ind++) {

      size_t k = solutions[ind]->n_rows;

      for (size_t j = 0; j < k; j++) {

	size_t rank = srt_ind((*solutions[ind])(j, i)) + 1;
	(*all_ranks_lists[ind])( i * k + j ) = rank;
	if (rank < k + 1)
	  all_corrects[ind] += (1.0 / (double) k);

      } // top k neighbors
    } // all solutions for this query

    double pdone = i * 100 / num_queries;

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

  for (size_t ind = 0; ind < num_solutions; ind++) {

    precisions->push_back(all_corrects[ind]
			  / (double) num_queries);
    median_ranks->push_back(arma::median(*all_ranks_lists[ind]));

  }

  return;
}


void compute_error(vector< arma::Mat<size_t>* > solutions,
		   arma::mat rdata, arma::mat qdata,
		   vector<double>* precisions,
		   vector<size_t>* median_ranks) {

  vector< arma::Col<size_t>* > all_ranks_lists;
  vector<double> all_corrects;

  // set up the list first
  for (size_t i = 0; i < solutions.size(); i++) {

    arma::Col<size_t> all_ranks_list; // = new arma::Col<size_t>();
    double all_correct = 0.0;

    all_ranks_list.set_size(solutions[i]->n_cols * solutions[i]->n_rows);
    all_ranks_lists.push_back(&all_ranks_list);

    all_corrects.push_back(all_correct);

  }


  // 1. Pick up the rank list
  // 2. Go through the solution for that query
  // 3. precision and ranks

  double perc_done = 10.0;
  double done_sky = 1.0;
        
  size_t num_queries = solutions[1]->n_cols;
  assert(num_queries == qdata.n_cols);
  size_t num_solutions = solutions.size();

  // do it with a loop over the queries.
  for (size_t i = 0; i < num_queries; i++) {

    // obtaining the rank list

    // obtaining the ips
    arma::vec ip_q = arma::trans(arma::trans(qdata.col(i)) 
                                 * rdata);
      
    assert(ip_q.n_elem == rdata.n_cols);

    // obtaining the ranks
    arma::uvec srt_ind = arma::sort_index(ip_q, 1);

    // Going through all the solutions
    for (size_t ind = 0; ind < num_solutions; ind++) {

      size_t k = solutions[ind]->n_rows;

      for (size_t j = 0; j < k; j++) {

	size_t rank = srt_ind((*solutions[ind])(j, i)) + 1;
	(*all_ranks_lists[ind])( i * k + j ) = rank;
	if (rank < k + 1)
	  all_corrects[ind] += (1.0 / (double) k);

      } // top k neighbors
    } // all solutions for this query

    double pdone = i * 100 / num_queries;

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

  Log::Info << "Errors Computed!" << endl;

  for (size_t ind = 0; ind < num_solutions; ind++) {

    precisions->push_back(all_corrects[ind]
			  / (double) num_queries);
    median_ranks->push_back(arma::median(*all_ranks_lists[ind]));

  }

  return;
}
