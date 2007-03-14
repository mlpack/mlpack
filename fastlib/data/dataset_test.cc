#include "dataset.h"

#include "xrun/xrun.h"
#include "math/discrete.h"

int main(int argc, char *argv[]) {
  xrun_init(argc, argv);
  const char *in = xrun_param_str("in");
  const char *out = xrun_param_str("out");
  String type;
  
  type.Copy(xrun_param_str("type"));
  
  Dataset dataset;
  
  if (!PASSED(dataset.InitFromFile(in))) return 1;
  
  success_t result;
  
  if (type.EqualsNoCase("arff")) {
    result = dataset.WriteArff(out);
  } else if (type.EqualsNoCase("csv")) {
    result = dataset.WriteCsv(out, false);
  } else if (type.EqualsNoCase("csvh")) {
    result = dataset.WriteCsv(out, true);
  } else {
    result = SUCCESS_FAIL;
  }
  
  if (!PASSED(result)) {
    fprintf(stderr, "Error!\n");
    return 1;
  }
  
  ArrayList<index_t> permutation;
  math::MakeRandomPermutation(dataset.n_points(), &permutation);
  
  for (int k = 5; k < 10; k++) {
    Dataset test;
    Dataset train;
    int i = k - 5;
    dataset.SplitTrainTest(k, i, permutation, &train, &test);
    assert(test.n_points() + train.n_points() == dataset.n_points());
  }

  return 0;
}
