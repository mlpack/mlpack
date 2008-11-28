/**
 * @file cart_driver.c
 *
 * Main method for implementing classification and regression 
 * tree class.
 * @see cartree.h
 */

#include "fastlib/fastlib.h"
#include "cartree.h"

const fx_entry_doc root_entries[] = {
  {"target", FX_REQUIRED, FX_INT, NULL,
  "Target Variable \n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc cart_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc root_doc = {
  root_entries, cart_submodules,
  "CART Parameters \n"
};

int main(int argc, char *argv[]){
  fx_module *root = fx_init(argc, argv, &root_doc);

  const char* fp;
  fp = fx_param_str(NULL, "data_file", "cpu.arff");

  int target_variable;
  target_variable = fx_param_int_req(NULL, "target");
  double alpha;
  alpha = fx_param_double(0, "alpha", 10);
 
  TrainingSet data;
  Vector firsts;
  data.Init(fp, firsts);
  
  if (target_variable >= data.GetFeatures()){
    target_variable = data.GetFeatures() - 1;
  }
  if (target_variable < 0){
    target_variable = 0;
  }

  CARTree tree;
  tree.Init(&data, firsts, 0, data.GetPointSize(), target_variable, alpha);

  tree.Grow();
  printf(" Tree has %d nodes. \n", tree.GetNumNodes());

  tree.Prune(alpha);

  fx_done(root);

}
