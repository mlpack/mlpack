#include <string>
#include <vector>
#include <armadillo>
#include <mlpack/core.h>

#include "check_nn_utils.h"

using namespace mlpack;
using namespace std;

// Add params for large scale or small scale computation
PROGRAM_INFO("Rank Matrix Inverter", "This program inverts the given "
	     "order matrix to give the complete rank matrix ", "");

PARAM_STRING_REQ("in", "The input file", "");
PARAM_STRING_REQ("out", "The output file", "");
PARAM_INT_REQ("n_rows", "The number of rows in the input file.", "");
PARAM_INT_REQ("n_cols", "The number of columns in the input file.", "");


int main (int argc, char *argv[]) {

  CLI::ParseCommandLine(argc, argv);

  string infile = CLI::GetParam<string>("in");
  string outfile = CLI::GetParam<string>("out");


  FILE *in_fp = fopen(infile.c_str(), "r");
  FILE *out_fp = fopen(outfile.c_str(), "w");

  size_t num_rows = CLI::GetParam<int>("n_rows");
  size_t num_cols = CLI::GetParam<int>("n_cols");

  // do it with a loop over the queries.
  // obtaining the rank list

  for (size_t i = 0; i < num_rows; i++) {
    arma::uvec srt_ind;
    srt_ind.set_size(num_cols);

    if (in_fp != NULL) {
      char *line = NULL;
      size_t len = 0;
      getline(&line, &len, in_fp);

      char *pch = strtok(line, ",\n");
      size_t rank_index = 0;

      while(pch != NULL) {
	srt_ind(rank_index++) = atoi(pch);
	pch = strtok(NULL, ",\n");
      }

      free(line);
      free(pch);
      assert(rank_index == num_cols);
    }

    arma::uvec* rank_ind = new arma::uvec();
    check_nn_utils::invert_index(srt_ind, rank_ind);

    for (size_t j = 0; j < num_cols; j++) {
      fprintf(out_fp, "%zu",(size_t) (*rank_ind)[j]);
      if (j == num_cols -1)
	fprintf(out_fp, "\n");
      else
	fprintf(out_fp, ",");
    }

    srt_ind.reset();
    rank_ind->reset();
    delete(rank_ind);
  }

  fclose(in_fp);
  fclose(out_fp);

  return 1;
}




