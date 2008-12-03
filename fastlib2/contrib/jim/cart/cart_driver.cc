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
  {"target", FX_PARAM, FX_INT, NULL,
  "Target Variable, if in data file \n"},
  {"data", FX_REQUIRED, FX_STR,  NULL,
   "Test data file \n"},
  {"labels", FX_PARAM, FX_STR, NULL,
   "Labels for classification \n"},
  {"alpha", FX_PARAM, FX_DOUBLE, NULL,
   "Pruning criterion \n"},
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
  fp = fx_param_str_req(NULL, "data");
  const char* fl;
  fl = fx_param_str(NULL, "labels", " ");
 
  FILE *classifications;
  classifications = fopen("results.csv", "w+");
  int target_variable;
  target_variable = fx_param_int(NULL, "target", 0);
  double alpha;
  alpha = fx_param_double(0, "alpha", 1.0);
   Matrix data_mat;
  TrainingSet data;
  Vector firsts;  
  if (!strcmp(" ", fl)){   
    data_mat.Init(1,1);
    data.Init(fp, firsts);
    if (target_variable >= data.GetFeatures()){
      target_variable = data.GetFeatures() - 1;
    }
    if (target_variable < 0){
      target_variable = 0;
    }
  } else {      
    data.InitLabels(fp);  
    data_mat.Init(data.GetFeatures()+1, data.GetPointSize());
    data.InitLabels2(fl, firsts, &data_mat);
    printf("Target Variable: %d \n", target_variable);
    target_variable = data.GetFeatures()-1;    
  }

  CARTree tree;
  tree.Init(&data, firsts, 0, data.GetPointSize(), target_variable, alpha);

  tree.Grow();
  printf(" Tree has %d nodes. \n", 2*tree.GetNumNodes()-1);

  printf("Pruning...\n");
  tree.Prune(alpha);
  printf("Tree has %d nodes. \n", 2*tree.GetNumNodes()-1);


  int errors = 0;
  for (int i = 0; i < data.GetPointSize(); i++){
    int j;
    j = data.WhereNow(i);
    double prediction = tree.Test(&data, j);
    errors = errors + (int)data.Verify(target_variable, prediction, j);  
    fprintf(classifications, "%f \n", prediction); 
  }
  printf("\n Misclassified Cases: %d \n \n", errors);

  fclose(classifications);

  fx_done(root);

 

}
