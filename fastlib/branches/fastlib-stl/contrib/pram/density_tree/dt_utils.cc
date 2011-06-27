#include "fastlib/fastlib.h"
#include "dt_utils.h"

int main (int argc, char* argv[]) {

  fx_module *root = fx_init(argc, argv, &dt_utils_doc);

  struct datanode *dtree_module = fx_submodule(root, "dtree");

  // call the driver than builds the optimal decision tree
  // for density estimation and also perform some 
  // analysis operations
  DTree *dtree_opt = dt_utils::Driver(root, dtree_module);

  fx_done(fx_root);
}

