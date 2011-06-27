/**
 * @file dt_utils.h
 * @ Parikshit Ram (pram@cc.gatech.edu)
 *
 */

#ifndef DT_UTILS_H
#define DT_UTILS_H

#include <string>

#include "fastlib/fastlib.h"
#include "dtree.h"

#define FACTOR 1.0e+300

const fx_entry_doc dt_utils_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   " Data file.\n"},
  {"labels", FX_PARAM, FX_STR, NULL,
   " The file containing the labels for the training points.\n"},
  {"num_classes", FX_PARAM, FX_INT, NULL,
   " Total number of classes in the training set.\n"},
  {"folds", FX_PARAM, FX_INT, NULL,
   " Number of folds for cross validation. For LOOCV "
   "enter 0.\n"},
  {"train_unpruned_output", FX_PARAM, FX_STR, NULL,
   " The file in which the estimated density values at"
   " the training points are output for the unpruned tree.\n"},
  {"train_output", FX_PARAM, FX_STR, NULL,
   " The file in which the estimated density values at"
   " the training points are output.\n"},
  {"compute_train_output", FX_PARAM, FX_BOOL, NULL,
   " This flag is provided to compute the density at"
   " the training points.\n"},
  {"test", FX_PARAM, FX_STR, NULL,
   "The points at which the density is to be computed"
   " using the tree.\n"},
  {"test_output", FX_PARAM, FX_STR, NULL,
   "File in which the density at the test points is "
   "to be output.\n"},
  {"train_time", FX_TIMER, FX_CUSTOM, NULL,
   " Training time for obtaining the optimal tree.\n"},
  {"test_time", FX_TIMER, FX_CUSTOM, NULL,
   " Testing time for the optimal decision tree.\n"},
  {"print_tree", FX_PARAM, FX_BOOL, NULL,
   " Whether to print the tree or not.\n"},
  {"tree_file", FX_PARAM, FX_STR, NULL,
   " The file in which the tree would be printed.\n"},
  {"print_vi", FX_PARAM, FX_BOOL, NULL,
   " Whether to print the variable importance of this"
   " tree.\n"},
  {"vi_file", FX_PARAM, FX_STR, NULL,
   " The file in which to write the variable"
   " importance.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc dt_utils_submodules[] = {
  {"dtree", &dtree_doc,
   " Contains the parameters for growing the tree.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc dt_utils_doc = {
  dt_utils_entries, dt_utils_submodules,
  "Parameters to train and test a decision tree"
  " for density estimation.\n"
};

namespace dt_utils {

  struct datanode *dt_util_module;

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

  void PrintLeafMembership(DTree *dtree,
			   const Matrix& data,
			   const Matrix& labels,
			   index_t num_classes) {

    index_t num_leaves = dtree->TagTree(0);
    dtree->WriteTree(0, stdout);printf("\n");fflush(NULL);
    Matrix table;
    table.Init(num_leaves, num_classes);
    table.SetZero();
    for (index_t i = 0; i < data.n_cols(); i++) {
      Vector test_p;
      data.MakeColumnVector(i, &test_p);
      index_t leaf_tag = dtree->FindBucket(test_p);
      index_t label = (index_t) labels.get(0, i);
      //      printf("%"LI"d,%"LI"d - %lg\n", leaf_tag, label, 
      // 	    table.get(leaf_tag, label));
      table.set(leaf_tag, label, 
		table.get(leaf_tag, label) + 1.0);
    } // end for

    table.PrintDebug("Classes in each leaf");
    return;
    // maybe print some more statistics if these work out well
  }

  void PrintVariableImportance(DTree *dtree,
			       index_t dims, FILE *fp) {
    ArrayList<long double> *imps = new ArrayList<long double>();
    imps->Init(dims);
    for (index_t i = 0; i < imps->size(); i++)
      (*imps)[i] = 0.0;
    
    dtree->ComputeVariableImportance(imps);
    long double max = 0.0;
    for (index_t i = 0; i < imps->size(); i++)
      if ((*imps)[i] > max)
	max = (*imps)[i];
    printf("Max: %Lg\n", max); fflush(NULL);

    for (index_t i = 0; i < imps->size() -1; i++)
      fprintf(fp, "%Lg,", (*imps)[i]);

    fprintf(fp, "%Lg\n", (*imps)[imps->size() -1]);

    return;
  }

  DTree *Trainer(const Matrix& dataset, 
		 index_t folds, 
		 datanode* dtree_mod) {


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
    dtree->Init(max_vals, min_vals,
		dataset.n_cols(), dtree_mod);
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

    // Growing the tree
    long double old_alpha = 0.0;
    long double alpha = dtree->Grow(new_dataset, &old_from_new);
    double new_f = dtree->st_estimate();
    double old_f = new_f;

    NOTIFY("%"LI"d leaf nodes in this tree, min_alpha: %Lg",
	   dtree->subtree_leaves(), alpha);

    // computing densities for the train points in the
    // full tree if asked for.
    if (fx_param_exists(dt_util_module, "train_unpruned_output")) {
      FILE *fp = 
	fopen(fx_param_str_req(dt_util_module,
			       "train_unpruned_output"), "w");
      if (fp != NULL) {
	for (index_t i = 0; i < dataset.n_cols(); i++) {
	  Vector test_p;
	  dataset.MakeColumnVector(i, &test_p);
	  double f = dtree->ComputeValue(test_p, false);
	  fprintf(fp, "%lg\n", f);
	} // end for
	fclose(fp);
      }
    }

    // exit(0);
    // sequential pruning and saving the alpha vals and the
    // values of c_t^2*r_t
    std::vector<std::pair<long double, long double> >  pruned_sequence;
    std::vector<double> change_in_estimate;
    while (dtree->subtree_leaves() > 1) {
      std::pair<long double, long double> tree_seq (old_alpha,
		    -1.0 * dtree->subtree_leaves_error());
      long double last = dtree->subtree_leaves_error();
      pruned_sequence.push_back(tree_seq);
      change_in_estimate.push_back(fabs(new_f - old_f));
      old_alpha = alpha;
      old_f = new_f;
      alpha = dtree->PruneAndUpdate(old_alpha);
      new_f = dtree->st_estimate();
      DEBUG_ASSERT_MSG((alpha < LDBL_MAX)||(dtree->subtree_leaves() == 1),
		       "old_alpha:%Lg, alpha:%Lg, tree size:%"LI"d",
		       old_alpha, alpha, dtree->subtree_leaves());
      DEBUG_ASSERT(alpha > old_alpha);
      DEBUG_ASSERT(dtree->subtree_leaves_error() >= -1.0 * tree_seq.second);
      // printf("%Lg\n", dtree->subtree_leaves_error() - last);
 
    } // end while
    std::pair<long double, long double> tree_seq (old_alpha,
		  -1.0 * dtree->subtree_leaves_error());
    pruned_sequence.push_back(tree_seq);
    change_in_estimate.push_back(fabs(new_f - old_f));

    NOTIFY("%"LI"d trees in the sequence, max_alpha:%Lg.\n",
	   (index_t) pruned_sequence.size(), old_alpha);

    // exit(0);

    ////////////// JT : REMOVE //////////////
//     std::vector<std::pair<long double, long double> >::iterator iter;
//     long double last = 0.0;
//     for (iter = pruned_sequence.begin();
// 	 iter < pruned_sequence.end(); ++iter) {
//       // printf("%Lg,%Lg\n", iter->first, iter->second - last);
//       last = iter->second;
//     } // end for
    ////////////////////////////
    // exit(0);


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

      // go through the motions - computing 
      // the maximum and minimum for each dimensions
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
      dtree_cv->Init(max_vals_cv, min_vals_cv,
		     train.n_cols(), dtree_mod);
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
      std::vector<std::pair<long double, long double> >::iterator it;
      for (it = pruned_sequence.begin();
	   it < pruned_sequence.end() -2; ++it) {
      
	// compute test values for this state of the tree
	long double val_cv = 0.0;
	for (index_t i = 0; i < test.n_cols(); i++) {
	  Vector test_point;
	  test.MakeColumnVector(i, &test_point);
	  val_cv += dtree_cv->ComputeValue(test_point, false);
	}

	// update the cv error value
	it->second -= 2.0 * val_cv / (long double) dataset.n_cols();

	// getting the new alpha value and pruning accordingly
	old_alpha = sqrt(((it+1)->first) * ((it+2)->first));
	alpha = dtree_cv->PruneAndUpdate(old_alpha);
      } // end for

      // compute test values for this state of the tree
      long double val_cv = 0.0;
      for (index_t i = 0; i < test.n_cols(); i++) {
	Vector test_point;
	test.MakeColumnVector(i, &test_point);
	val_cv += dtree_cv->ComputeValue(test_point, false);
      }
      // update the cv error value
      it->second -= 2.0 * val_cv / (long double) dataset.n_cols();

    } // end for

    long double optimal_alpha = -1.0, best_cv_error = LDBL_MAX;
    std::vector<std::pair<long double, long double> >::iterator it;
    std::vector<double>::iterator jt;
    for (it = pruned_sequence.begin(), jt = change_in_estimate.begin();
	 it < pruned_sequence.end() -1; ++it, ++jt) {
      // printf("%Lg,%Lg,%Lg\n", it->first, it->second, best_cv_error - it->second);
      if (0 < best_cv_error - it->second) {
	best_cv_error = it->second;
	optimal_alpha = it->first;
      } // end if
    } // end for

    //printf("OA: %Lg\n", optimal_alpha);
    //exit(0);

    // Initializing the tree
    DTree *dtree_opt = new DTree();
    dtree_opt->Init(max_vals, min_vals, dataset.n_cols(), dtree_mod);
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
    NOTIFY("%"LI"d leaf nodes in this tree\n opt_alpha:%Lg",
	   dtree_opt->subtree_leaves(), optimal_alpha);
    //   printf("%"LI"d leaf nodes in this tree\n opt_alpha:%Lg\n",
    // 	 dtree_opt->subtree_leaves(), optimal_alpha);

    // Pruning with optimal alpha
    while (old_alpha < optimal_alpha) {
      old_alpha = alpha;
      alpha = dtree_opt->PruneAndUpdate(old_alpha);
      DEBUG_ASSERT_MSG((alpha < DBL_MAX)||(dtree->subtree_leaves() == 1),
		       "old_alpha:%Lg, alpha:%Lg, tree size:%"LI"d",
		       old_alpha, alpha, dtree->subtree_leaves());
      DEBUG_ASSERT(alpha > old_alpha);
    } // end while

    NOTIFY("%"LI"d leaf nodes in this tree\n old_alpha:%Lg",
	   dtree_opt->subtree_leaves(), old_alpha);

    return dtree_opt;
  }

  DTree* Driver(datanode *dt_util_mod, datanode *dtree_mod) {

    srand( time(NULL));
    dt_util_module = dt_util_mod;
    std::string data_file
      = fx_param_str_req(dt_util_module, "data");
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

    // starting the training timer
    fx_timer_start(dt_util_module, "train_time");

    // cross-validation here
    index_t folds = fx_param_int(dt_util_module, "folds", 10);
    if (folds == 0) {
      folds = dataset.n_cols();
      NOTIFY("Starting Leave-One-Out Cross validation");
    } else 
      NOTIFY("Starting %"LI"d-fold Cross validation", folds);


 
    // obtaining the optimal tree
    DTree *dtree_opt = Trainer(dataset, folds, dtree_mod);
 
    // stopping the training timer
    fx_timer_stop(dt_util_module, "train_time");

    // starting the test timer
    fx_timer_start(dt_util_module, "test_time");


    // computing densities for the train points in the
    // optimal tree
    if (fx_param_bool(dt_util_module, "compute_train_output", false)) {
      FILE *fp;
      fp = NULL;

      // opening the file to write if provided
      if (fx_param_exists(dt_util_module, "train_output")) {
	std::string train_density_file
	  = fx_param_str_req(dt_util_module, "train_output");
	fp = fopen(train_density_file.c_str(), "w");
      }

      for (index_t i = 0; i < dataset.n_cols(); i++) {
	Vector test_p;
	dataset.MakeColumnVector(i, &test_p);
	double f = dtree_opt->ComputeValue(test_p, false);
	if (fp != NULL)
	  fprintf(fp, "%lg\n", f);
      } // end for

      if (fp != NULL)
	fclose(fp);
    }

    // computing the density at the provided test points
    // and outputting the density in the given file.
    if (fx_param_exists(dt_util_module, "test")) {
      std::string test_file = fx_param_str_req(dt_util_module, "test");
      Matrix test_set;
      NOTIFY("Loading test data...\n");
      data::Load(test_file.c_str(), &test_set);

      NOTIFY("%"LI"d points in %"LI"d dims.", test_set.n_cols(),
	     test_set.n_rows());

      FILE *fp;
      fp = NULL;

      if (fx_param_exists(dt_util_module, "test_output")) {
	std::string test_density_file
	  = fx_param_str_req(dt_util_module, "test_output");
	fp = fopen(test_density_file.c_str(), "w");
      }

      for (index_t i = 0; i < test_set.n_cols(); i++) {
	Vector test_p;
	test_set.MakeColumnVector(i, &test_p);
	double f = dtree_opt->ComputeValue(test_p, false);
	if (fp != NULL)
	  fprintf(fp, "%lg\n", f);
      } // end for

      if (fp != NULL) 
	fclose(fp);
    }

    fx_timer_stop(dt_util_module, "test_time");


    // printing the final tree
    if (fx_param_bool(dt_util_module, "print_tree", false)) {
      if (fx_param_exists(dt_util_module, "tree_file")) {
	std::string tree_file = fx_param_str(dt_util_module,
					     "tree_file",
					     "pruned_tree.txt");
	FILE *fp = fopen(tree_file.c_str(), "w");
	if (fp != NULL) {
	  dtree_opt->WriteTree(0, fp);
	  fclose(fp);
	}
      } else {
	dtree_opt->WriteTree(0, stdout);
	printf("\n");
      }
    }

    // print the leaf memberships for the optimal tree
    if (fx_param_exists(dt_util_module, "labels")) {
      std::string labels_file = fx_param_str_req(dt_util_module, "labels");
      Matrix labels;
      NOTIFY("loading labels.\n");
      data::Load(labels_file.c_str(), &labels);
      index_t num_classes = fx_param_int_req(dt_util_module, "num_classes");

      DEBUG_ASSERT(dataset.n_cols() == labels.n_cols());
      DEBUG_ASSERT(labels.n_rows() == 1);

      PrintLeafMembership(dtree_opt, dataset, labels, num_classes);
    }


    if(fx_param_bool(dt_util_module, "print_vi", false)) {
      if(fx_param_exists(dt_util_module, "vi_file")) {
	std::string vi_file = fx_param_str_req(dt_util_module, "vi_file");
	FILE *fp = fopen(vi_file.c_str(), "w");
	if (fp != NULL) {
	  PrintVariableImportance(dtree_opt, dataset.n_rows(), fp);
	  fclose(fp);
	}
      } else 
	PrintVariableImportance(dtree_opt, dataset.n_rows(), stdout);
    }

    return dtree_opt;
  } // end Driver()

}; // namespace dt_utils

#endif
