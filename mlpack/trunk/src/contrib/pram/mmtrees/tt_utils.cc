#include "fastlib/fastlib.h"
#include "tt_utils.h"

int main (int argc, char* argv[]) {

  fx_module *root = fx_init(argc, argv, &tt_utils_doc);

  struct datanode *ttree_module = fx_submodule(root, "ttree");

  // call the driver than builds the optimal decision tree
  // for density estimation and also perform some 
  // analysis operations
  TTree *ttree_opt = tt_utils::Driver(root, ttree_module);

  fx_done(fx_root);
}

