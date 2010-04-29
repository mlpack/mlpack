/**
 * @file main.cc
 * @author Chip Mappus
 *
 * main for using infomax ICA method.
 */

#include "infomax_ica.h"
#include "test_infomax_ica.h"
#include <fastlib/fastlib.h>
#include <fastlib/data/dataset.h>

#include <armadillo>
#include <fastlib/base/arma_compat.h>

const fx_entry_doc infomax_ica_main_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   "  The name of the file containing mixture data.\n"},
  {"lambda", FX_PARAM, FX_DOUBLE, NULL,
   "  The learning rate.\n"},
  {"B", FX_PARAM, FX_INT, NULL,
   "  Infomax data window size.\n"},
  {"epsilon", FX_PARAM, FX_DOUBLE, NULL,
   "  Infomax algorithm stop threshold.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc infomax_ica_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc infomax_ica_main_doc = {
  infomax_ica_main_entries, infomax_ica_main_submodules,
  "This performs ICA decomposition on a given dataset using the Infomax method.\n"
};

int main(int argc, char *argv[]) {
  fx_module *root = fx_init(argc, argv, &infomax_ica_main_doc);

  const char *data_file_name = fx_param_str_req(root, "data");
  double lambda = fx_param_double(root,"lambda",0.001);
  int B = fx_param_int(root,"B",5);
  double epsilon = fx_param_double(root,"epsilon",0.001);
  Matrix dataset;
  arma::mat tmp_dataset;
  data::Load(data_file_name, tmp_dataset);
  arma_compat::armaToMatrix(tmp_dataset, dataset);
  InfomaxICA *ica = new InfomaxICA(lambda, B, epsilon);

  ica->applyICA(dataset);  
  Matrix west;
  ica->getUnmixing(west);
  //ica->displayMatrix(west);

  fx_done(NULL);
}
