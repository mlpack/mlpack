#include "multibody.h"

int main(int argc, char *argv[])
{
  const char *datafile_name;
  int leaflen;
  Dataset dataset;
  Matrix data;
  bool do_naive;

  fx_init(argc, argv);

  // PARSE INPUTS
  datafile_name = fx_param_str(NULL, "data", "small.arff");
  leaflen = fx_param_int(NULL, "leaflen", 20);
  do_naive = fx_param_exists(NULL, "do_naive");

  // READING DATA
  fx_timer_start(NULL, "read_d");
  // read the dataset and get the matrix
  if (!PASSED(dataset.InitFromFile(datafile_name))) {
    fprintf(stderr, "main: Couldn't open file '%s'.\n", datafile_name);
    return 1;
  }
  data.Alias(dataset.matrix());
  fx_timer_stop(NULL, "read_d");

  // Multibody computation
  fx_timer_start(NULL,"multibody");

  fx_timer_stop(NULL, "multibody");

  // NAIVE
  if (do_naive) {
  }

  fx_done();
}
