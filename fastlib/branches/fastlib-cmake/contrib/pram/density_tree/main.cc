/**
 * @file main.cc
 * @ Parikshit Ram (pram@cc.gatech.edu)
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
  {"train_time", FX_TIMER, FX_CUSTOM, NULL,
   " Training time for obtaining the optimal tree.\n"},
  {"test_time", FX_TIMER, FX_CUSTOM, NULL,
   " Testing time for the optimal decision tree.\n"},
  {"print_tree", FX_PARAM, FX_BOOL, NULL,
   " Whether to print the tree or not.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc dtree_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc dtree_main_doc = {
  dtree_main_entries, dtree_submodules,
  "DTree Parameters \n"
};

void PermuteMatrix(const Matrix&, Matrix*);

int main(int argc, char *argv[]){

  srand( time(NULL));
  fx_module *root = fx_init(argc, argv, &dtree_main_doc);
  std::string data_file = fx_param_str_req(root, "d");
  Matrix dataset;
  NOTIFY("Loading data file...\n");
  data::Load(data_file.c_str(), &dataset);

  NOTIFY("%"LI"d points in %"LI"d dims.", dataset.n_cols(),
	 dataset.n_rows());

//   // getting information about the feature type
//   // REAL, INTEGER, NOMINAL. Most probably using
//   // enum ... don't know how to use it though.
//   ArrayList<enum> dim_type;
//   dim_type.Init(dataset.n_rows());
//   for (index_t i = 0; i < dim_type.size(); i++) {
//     // assign dim type somehow
//   } // end for

  // finding the max and min vals for the dataset
  ArrayList<double> max_vals, min_vals;
  max_vals.Init(dataset.n_rows());
  min_vals.Init(dataset.n_rows());

  Matrix temp_d;
  la::TransposeInit(dataset, &temp_d);

  for (index_t i = 0; i < temp_d.n_cols(); i++) {
    // if (dim_type[i] != NOMINAL) {
    Vector dim_vals;
    temp_d.MakeColumnVector(i, &dim_vals);
    std::vector<double> dim_vals_vec(dim_vals.ptr(),
				     dim_vals.ptr() + temp_d.n_rows());

    sort(dim_vals_vec.begin(), dim_vals_vec.end());
    min_vals[i] = *(dim_vals_vec.begin());
    max_vals[i] = *(dim_vals_vec.end() -1);
    // }
  }

  // Initializing the tree
  DTree *dtree = new DTree();
  dtree->Init(max_vals, min_vals, dataset.n_cols());
  // Getting ready to grow the tree
  ArrayList<index_t> old_from_new;
  old_from_new.Init(dataset.n_cols());
  for (index_t i = 0; i < old_from_new.size(); i++) {
    old_from_new[i] = i;
  }

  // Saving the dataset since it would be modified
  // while growing the tree
  Matrix new_dataset;
  new_dataset.Copy(dataset);

  // starting the training timer
  fx_timer_start(root, "train_time");
  // Growing the tree
  double old_alpha = 0.0;
  double alpha = dtree->Grow(new_dataset, &old_from_new);

  NOTIFY("%"LI"d leaf nodes in this tree", dtree->subtree_leaves());

//   // computing densities for the train points in the
//   // big tree
//   for (index_t i = 0; i < dataset.n_cols(); i++) {
//     Vector test_p;
//     dataset.MakeColumnVector(i, &test_p);
//     double f = dtree->ComputeValue(test_p);
//     printf("%lg ", f);
//   } // end for
//   printf("\n");

  // sequential pruning and saving the alpha vals and the
  // values of c_t^2*r_t
  std::vector<std::pair<double, double> >  pruned_sequence;
  while (dtree->subtree_leaves() > 1) {
    std::pair<double, double> tree_seq (old_alpha,
					-1.0 * dtree->subtree_leaves_error());
    pruned_sequence.push_back(tree_seq);
    old_alpha = alpha;
    alpha = dtree->PruneAndUpdate(old_alpha);
    DEBUG_ASSERT_MSG((alpha < DBL_MAX)||(dtree->subtree_leaves() == 1),
		     "old_alpha:%lg, alpha:%lg, tree size:%"LI"d",
		     old_alpha, alpha, dtree->subtree_leaves());
    DEBUG_ASSERT(alpha > old_alpha);
  } // end while
  std::pair<double, double> tree_seq (old_alpha,
 				      -1.0 * dtree->subtree_leaves_error());
  pruned_sequence.push_back(tree_seq);

  NOTIFY("%"LI"d trees in the sequence, max_alpha:%lg.\n",
	 (index_t) pruned_sequence.size(), old_alpha);


  // cross-validation here
  index_t folds = fx_param_int(root, "folds", 10);
  NOTIFY("Starting %"LI"d-fold Cross validation", folds);

  // Permute the dataset once just for keeps
  Matrix pdata;
  //  PermuteMatrix(dataset, &pdata);
  pdata.Copy(dataset);
  //  pdata.PrintDebug("Per");
  index_t test_size = dataset.n_cols() / folds;

  // Go through each fold
  for (index_t fold = 0; fold < folds; fold++) {
    // NOTIFY("Fold %"LI"d...", fold+1);

    // break up data into train and test set
    Matrix test;
    index_t start = fold * test_size,
      end = min ((fold + 1) * test_size,dataset.n_cols());
    pdata.MakeColumnSlice(start, end - start, &test);
    Matrix train;
    train.Init(pdata.n_rows(), pdata.n_cols() - (end - start));
    index_t k = 0;
    for (index_t j = 0; j < pdata.n_cols(); j++) {
      if (j < start || j >= end) {
	Vector temp_vec;
	pdata.MakeColumnVector(j, &temp_vec);
	train.CopyVectorToColumn(k++, temp_vec);
      } // end if
    } // end for
    DEBUG_ASSERT(k == train.n_cols());

    // go through the motions
    ArrayList<double> max_vals_cv, min_vals_cv;
    max_vals_cv.Init(train.n_rows());
    min_vals_cv.Init(train.n_rows());

    Matrix temp_t;
    la::TransposeInit(train, &temp_t);

    for (index_t i = 0; i < temp_t.n_cols(); i++) {
      Vector dim_vals;
      temp_t.MakeColumnVector(i, &dim_vals);

      std::vector<double> dim_vals_vec(dim_vals.ptr(),
				       dim_vals.ptr()
				       + temp_t.n_rows());
      sort(dim_vals_vec.begin(), dim_vals_vec.end());
      min_vals_cv[i] = *(dim_vals_vec.begin());
      max_vals_cv[i] = *(dim_vals_vec.end() -1);
    } // end for

    // Initializing the tree
    DTree *dtree_cv = new DTree();
    dtree_cv->Init(max_vals_cv, min_vals_cv, train.n_cols());
    // Getting ready to grow the tree
    ArrayList<index_t> old_from_new_cv;
    old_from_new_cv.Init(train.n_cols());
    for (index_t i = 0; i < old_from_new_cv.size(); i++) {
      old_from_new_cv[i] = i;
    }

    // Growing the tree
    old_alpha = 0.0;
    alpha = dtree_cv->Grow(train, &old_from_new_cv);

    // sequential pruning with all the values of available
    // alphas and adding values for test values
    std::vector<std::pair<double, double> >::iterator it;
    for (it = pruned_sequence.begin();
	 it < pruned_sequence.end() -2; ++it) {
      
      // compute test values for this state of the tree
      double val_cv = 0.0;
      for (index_t i = 0; i < test.n_cols(); i++) {
	Vector test_point;
	test.MakeColumnVector(i, &test_point);
	val_cv += dtree_cv->ComputeValue(test_point);
      }

      // update the cv error value
      it->second = it-> second - 2.0 * val_cv / (double) dataset.n_cols();

      // getting the new alpha value and pruning accordingly
      old_alpha = sqrt(((it+1)->first) * ((it+2)->first));
      alpha = dtree_cv->PruneAndUpdate(old_alpha);
    } // end for

    // compute test values for this state of the tree
    double val_cv = 0.0;
    for (index_t i = 0; i < test.n_cols(); i++) {
      Vector test_point;
      test.MakeColumnVector(i, &test_point);
      val_cv += dtree_cv->ComputeValue(test_point);
    }
    // update the cv error value
    it->second = it->second - 2.0 * val_cv / (double) dataset.n_cols();

  } // end for

  double optimal_alpha = -1.0, best_cv_error = DBL_MAX;
  std::vector<std::pair<double, double> >::iterator it;
  for (it = pruned_sequence.begin();
       it < pruned_sequence.end() -1; ++it) {
    if (it->second < best_cv_error) {
      best_cv_error = it->second;
      optimal_alpha = it->first;
    } // end if
  } // end for

  // stopping the training timer
  fx_timer_stop(root, "train_time");

  // Initializing the tree
  DTree *dtree_opt = new DTree();
  dtree_opt->Init(max_vals, min_vals, dataset.n_cols());
  // Getting ready to grow the tree
  for (index_t i = 0; i < old_from_new.size(); i++) {
    old_from_new[i] = i;
  }

  // Saving the dataset since it would be modified
  // while growing the tree
  new_dataset.Destruct();
  new_dataset.Copy(dataset);

  // Growing the tree
  old_alpha = 0.0;
  alpha = dtree_opt->Grow(new_dataset, &old_from_new);
  NOTIFY("%"LI"d leaf nodes in this tree\n opt_alpha:%lg",
	 dtree_opt->subtree_leaves(), optimal_alpha);

  while (old_alpha < optimal_alpha) {
    old_alpha = alpha;
    alpha = dtree_opt->PruneAndUpdate(old_alpha);
    DEBUG_ASSERT_MSG((alpha < DBL_MAX)||(dtree->subtree_leaves() == 1),
		     "old_alpha:%lg, alpha:%lg, tree size:%"LI"d",
		     old_alpha, alpha, dtree->subtree_leaves());
    DEBUG_ASSERT(alpha > old_alpha);
  } // end while

  // Pruning with optimal alpha
  NOTIFY("%"LI"d leaf nodes in this tree", dtree_opt->subtree_leaves());

  if (fx_param_bool(root, "print_tree", false)) {
    dtree_opt->WriteTree(0);
    printf("\n");fflush(NULL);
  }
//   // computing densities for the train points in the
//   // optimal tree

//   // starting the test timer
  fx_timer_start(root, "test_time");
//   for (index_t i = 0; i < dataset.n_cols(); i++) {
//     Vector test_p;
//     dataset.MakeColumnVector(i, &test_p);
//     double f = dtree_opt->ComputeValue(test_p);
//     printf("%lg\n", f);
//   } // end for
//   //  printf("\n");
//   fflush(NULL);
  fx_timer_stop(root, "test_time");

//   // outputting the optimal tree
//   std::string output_file
//     = fx_param_str(root, "treee_file", "output.txt");

  fx_param_bool(root, "fx/silent", 0);
  fx_done(root);
}

void PermuteMatrix(const Matrix& input, Matrix *output) {

  ArrayList<index_t> perm_array;
  index_t size = input.n_cols();
  Matrix perm_mat;
  perm_mat.Init(size, size);
  perm_mat.SetAll(0.0);
  srand( time(NULL));
  math::MakeRandomPermutation(size, &perm_array);
  for(index_t i = 0; i < size; i++) {
    perm_mat.set(perm_array[i], i, 1.0);
  }
  la::MulInit(input, perm_mat, output);
  return;
}
