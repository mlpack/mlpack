/**
 * @file main.cc
 * @author Chip Mappus
 *
 * main for using infomax ICA method.
 */

#include "infomax_ica.h"
#include "test_infomax_ica.h"
#include "fastlib/fastlib.h"
#include "fastlib/data/dataset.h"
int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  const char *data = fx_param_str_req(NULL, "data");
  double lambda = fx_param_double(NULL,"lambda",0.001);
  int B = fx_param_int(NULL,"B",5);
  double epsilon = fx_param_double(NULL,"epsilon",0.001);
  Dataset dataset;
  dataset.InitFromFile(data);

  InfomaxICA *ica = new InfomaxICA(lambda, B, epsilon);

  ica->applyICA(dataset);  
  Matrix west;
  ica->getUnmixing(west);
  //ica->displayMatrix(west);

  fx_done();
}
