/**
 * @file main.cc
 *
 */

#include <string>

#include "fastlib/fastlib.h"
#include "dtree.h"

const fx_entry_doc dtree_main_entries[] = {
  {"d", FX_REQUIRED, FX_STR, NULL,
   " Data file \n"},
  {"folds", FX_PARAM, FX_INT, NULL,
   " Number of folds for cross validation.\n"},
  {"tree_file", FX_PARAM, FX_STR, NULL,
   " The file in which the tree would be printed.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc dtree_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc dtree_main_doc = {
  dtree_main_entries, dtree_submodules,
  "DTree Parameters \n"
};

int main(int argc, char *argv[]){

  fx_module *root = fx_init(argc, argv, &dtree_main_doc);
  std::string data_file = fx_param_str_req(root, "d");
  Matrix dataset;
  NOTIFY("Loading data file...\n");
  data::Load(data_file.c_str(), &dataset);

  NOTIFY("%"LI"d points in %"LI"d dims.", dataset.n_cols(),
	 dataset.n_rows());
  // finding the max and min vals for the dataset
  ArrayList<double> max_vals, min_vals;
  max_vals.Init(dataset.n_rows());
  min_vals.Init(dataset.n_rows());

  Matrix temp_d;
  la::TransposeInit(dataset, &temp_d);

  for (index_t i = 0; i < temp_d.n_cols(); i++) {
    Vector dim_vals;
    temp_d.MakeColumnVector(i, &dim_vals);

//     dim_vals.PrintDebug("Dim");

    std::vector<double> dim_vals_vec(dim_vals.ptr(),
				     dim_vals.ptr() + temp_d.n_rows());

    sort(dim_vals_vec.begin(), dim_vals_vec.end());
    min_vals[i] = *(dim_vals_vec.begin());
    max_vals[i] = *(dim_vals_vec.end() -1);
  }

//   dataset.PrintDebug("data");
//   temp_d.PrintDebug("transposed");

  // Initializing the tree
  DTree *dtree = new DTree();
  dtree->Init(&max_vals, &min_vals, dataset.n_cols());
  // Getting ready to grow the tree
  ArrayList<index_t> old_from_new;
  old_from_new.Init(dataset.n_cols());
  for (index_t i = 0; i < old_from_new.size(); i++) {
    old_from_new[i] = i;
  }

  // Growing the tree
  double old_alpha = 0.0;
  double alpha = dtree->Grow(dataset, &old_from_new);

  NOTIFY("%"LI"d leaf nodes in this tree", dtree->subtree_leaves());
  printf("%lg, %lg %lg\n",old_alpha, alpha, DBL_MAX);

  // sequential pruning and saving the alpha vals and the
  // values of c_t^2*r_t
  std::vector<std::pair<double, double> >  pruned_sequence;
  while (dtree->subtree_leaves() > 1) {
    std::pair<double, double> tree_seq (old_alpha,
					-1.0 * dtree->subtree_leaves_error());
    pruned_sequence.push_back(tree_seq);
    old_alpha = alpha;
    alpha = dtree->PruneAndUpdate(old_alpha);
     NOTIFY("%"LI"d leaf nodes in this tree", dtree->subtree_leaves());
     printf("%lg, %lg\n",old_alpha, alpha);
     DEBUG_ASSERT(alpha > old_alpha);
  } // end while
  std::pair<double, double> tree_seq (old_alpha,
 				      -1.0 * dtree->subtree_leaves_error());
  pruned_sequence.push_back(tree_seq);

  printf("%"LI"d trees in the sequence.\n",(index_t) pruned_sequence.size());

//   for (std::vector<std::pair<double,double> >::iterator it
// 	 = pruned_sequence.begin(); it < pruned_sequence.end(); it++) {
//     printf("%lg ", it->first);
//   } // end for

//   printf("\n");

//   // cross-validation here which I will do later
//   index_t folds = fx_param_int(root, "folds", 10);


//   // outputting the optimal tree
//   std::string output_file
//     = fx_param_str(root, "treee_file", "output.txt");

  fx_param_bool(root, "fx/silent", 1);
  fx_done(root);
}
