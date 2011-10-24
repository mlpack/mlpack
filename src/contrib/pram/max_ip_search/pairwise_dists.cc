#include <mlpack/core.h>
#include <armadillo>
#include <string>
#include <vector>

using namespace mlpack;
using namespace std;

// Add params for large scale or small scale computation
PROGRAM_INFO("Rank Computer", "This program computes the "
	     "complete rank matrix for the given queries and "
	     "references.", "");

PARAM_STRING_REQ("r", "The reference set", "");
PARAM_STRING_REQ("q", "The set of queries", "");
PARAM_STRING_REQ("rank_matrix", "The file where the rank "
		 "matrix would be written in.", "");
PARAM_STRING("ip_mat_file", "The file where the ip "
	     "matrix would be written in.", "", "");

PARAM_FLAG("large_scale", "The flag to trigger the "
	   "large scale computation where you go "
	   "through the individual queries linearly.", "");

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

  string rank_matrix_file = CLI::GetParam<string>("rank_matrix");


  FILE *pfile = fopen(CLI::GetParam<string>("rank_matrix").c_str(),
		      "w");

  if (CLI::HasParam("large_scale")) {

    double perc_done = 10.0;
    double done_sky = 1.0;
	
    // do it with a loop over the queries.
    for (size_t i = 0; i < qdata.n_cols; i++) {

      // arma::vec q = qdata.col(i);

      // obtaining the ips
      arma::vec ip_q = arma::trans(arma::trans(qdata.col(i)) 
				   * rdata);
      
      assert(ip_q.n_elem == rdata.n_cols);

      // obtaining the ranks
      vector<pair<double, size_t> > ips(rdata.n_cols);
      for (size_t j = 0; j < rdata.n_cols; j++)
	ips[j] = make_pair(ip_q(j), j);

      sort(ips.begin(), ips.end());
      reverse(ips.begin(), ips.end());

      for (size_t j = 0; j < rdata.n_cols; j++) {
	fprintf(pfile, "%zu", ips[j].second);
	if (j == rdata.n_cols -1)
	  fprintf(pfile, "\n");
	else
	  fprintf(pfile, ",");
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
    Log::Info << "RANKS COMPUTED!" << endl;

  } else { // small-scale

    // doing the computation when the rank-matrix is small
    arma::mat ip_mat = arma::trans(qdata) * rdata;


    Log::Info << "IPs COMPUTED!" << endl;
    if (CLI::HasParam("ip_mat_file")) {
      string ip_file = CLI::GetParam<string>("ip_mat_file");
      ip_mat.save(ip_file, arma::raw_ascii);
    }

    // do it with a loop over the queries.
    for (size_t i = 0; i < qdata.n_cols; i++) {

      // obtaining the ips
      arma::vec ip_q = arma::trans(ip_mat.row(i));
      assert(ip_q.n_elem == rdata.n_cols);


      // obtaining the ranks
      vector<pair<double, size_t> > ips(rdata.n_cols);
      for (size_t j = 0; j < rdata.n_cols; j++)
	ips[j] = make_pair(ip_q(j), j);

      sort(ips.begin(), ips.end());
      reverse(ips.begin(), ips.end());
	
      for (size_t j = 0; j < rdata.n_cols; j++) {
	fprintf(pfile, "%zu", ips[j].second);
	if (j == rdata.n_cols -1)
	  fprintf(pfile, "\n");
	else
	  fprintf(pfile, ",");
      }
    }
	
    fclose(pfile);
    Log::Info << "RANKS COMPUTED!" << endl;
  }
}
