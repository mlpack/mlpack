/**
 * @file tt_utils.h
 * @ Parikshit Ram (pram@cc.gatech.edu)
 *
 */

#ifndef TT_UTILS_H
#define TT_UTILS_H

#include <string>

#include "fastlib/fastlib.h"
#include "ttree.h"

#define FACTOR 1.0e+300

const fx_entry_doc tt_utils_entries[] = {
  {"x", FX_REQUIRED, FX_STR, NULL,
   " Data file containing the x values "
   "(in row major format) .\n"},
  {"y", FX_REQUIRED, FX_STR, NULL,
   " The file containing the y values for the training points.\n"},
  {"folds", FX_PARAM, FX_INT, NULL,
   " Number of folds for cross validation. For LOOCV "
   "enter 0.\n"},
  {"train_unpruned_output", FX_PARAM, FX_STR, NULL,
   " The file in which the estimated tent values at"
   " the training points are output for the unpruned tree.\n"},
  {"train_output", FX_PARAM, FX_STR, NULL,
   " The file in which the estimated tent values at"
   " the training points are output.\n"},
  {"compute_train_output", FX_PARAM, FX_BOOL, NULL,
   " This flag is provided to compute the tent values at"
   " the training points.\n"},
  {"test", FX_PARAM, FX_STR, NULL,
   "The points at which the tent values are to be computed"
   " using the best CV tree.\n"},
  {"test_output", FX_PARAM, FX_STR, NULL,
   "File in which the tent values at the test points is "
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

const fx_submodule_doc tt_utils_submodules[] = {
  {"ttree", &ttree_doc,
   " Contains the parameters for growing the tree.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc tt_utils_doc = {
  tt_utils_entries, tt_utils_submodules,
  "Parameters to train and test a decision tree"
  " for density estimation.\n"
};

namespace tt_utils {

  struct datanode *tt_util_module;

  // FIX THIS FUNCTCLIN: To permute the y value as well
  void PermuteMatrix(const Matrix& input, const Vector& y,
		     Matrix *output, Vector * oy) {

    ArrayList<size_t> perm_array;
    size_t size = input.n_cols();
    Matrix perm_mat;
    perm_mat.Init(size, size);
    perm_mat.SetAll(0.0);
    srand( time(NULL));
    math::MakeRandomPermutation(size, &perm_array);

    oy->Init(size);

    for(size_t i = 0; i < size; i++) {
      perm_mat.set(perm_array[i], i, 1.0);
      (*oy)[i] = y[perm_array[i]];
    }
    la::MulInit(input, perm_mat, output);
    return;
  }

//   void PrintLeafMembership(TTree *ttree,
// 			   const Matrix& data,
// 			   const Matrix& labels,
// 			   size_t num_classes) {

//     size_t num_leaves = ttree->TagTree(0);
//     ttree->WriteTree(0, stdout);printf("\n");fflush(NULL);
//     Matrix table;
//     table.Init(num_leaves, num_classes);
//     table.SetZero();
//     for (size_t i = 0; i < data.n_cols(); i++) {
//       Vector test_p;
//       data.MakeColumnVector(i, &test_p);
//       size_t leaf_tag = ttree->FindBucket(test_p);
//       size_t label = (size_t) labels.get(0, i);
//       //      printf("%zu"d,%zu"d - %lg\n", leaf_tag, label, 
//       // 	    table.get(leaf_tag, label));
//       table.set(leaf_tag, label, 
// 		table.get(leaf_tag, label) + 1.0);
//     } // end for

//     table.PrintDebug("Classes in each leaf");
//     return;
//     // maybe print some more statistics if these work out well
//   }

  void PrintVariableImportance(TTree *ttree,
			       size_t dims, FILE *fp) {
    ArrayList<long double> *imps = new ArrayList<long double>();
    imps->Init(dims);
    for (size_t i = 0; i < imps->size(); i++)
      (*imps)[i] = 0.0;
    
    ttree->ComputeVariableImportance(imps);
    long double max = 0.0;
    for (size_t i = 0; i < imps->size(); i++)
      if ((*imps)[i] > max)
	max = (*imps)[i];
    printf("Max: %Lg\n", max); fflush(NULL);

    for (size_t i = 0; i < imps->size() -1; i++)
      fprintf(fp, "%Lg,", (*imps)[i]);

    fprintf(fp, "%Lg\n", (*imps)[imps->size() -1]);

    return;
  }

  TTree *Trainer(const Matrix& x, 
		 const Vector& y,
		 size_t folds, 
		 datanode* ttree_mod) {


    // finding the max and min vals for the x
    ArrayList<double> max_vals, min_vals;
    max_vals.Init(x.n_rows());
    min_vals.Init(x.n_rows());

    Matrix temp_d;
    la::TransposeInit(x, &temp_d);

    for (size_t i = 0; i < temp_d.n_cols(); i++) {
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


    NOTIFY("MAX - MIN Computed.");

    // Saving the x since it would be modified
    // while growing the tree
    Matrix new_x;
    new_x.Copy(x);

    Vector new_y;
    new_y.Copy(y);

    // Initializing the tree
    TTree *ttree = new TTree();
    ttree->Init(max_vals, min_vals,
		x.n_cols(), new_y, ttree_mod);
    // Getting ready to grow the tree
    ArrayList<size_t> old_from_new;
    old_from_new.Init(x.n_cols());
    for (size_t i = 0; i < old_from_new.size(); i++) {
      old_from_new[i] = i;
    }

    NOTIFY("Init done");

    // Growing the tree
    long double old_alpha = 0.0;
    long double alpha = ttree->Grow(new_x, new_y, &old_from_new);
    // double new_f = ttree->st_estimate();
    // double old_f = new_f;

    NOTIFY("%zu"d leaf nodes in this tree, min_alpha: %Lg",
	   ttree->subtree_leaves(), alpha);

    // computing densities for the train points in the
    // full tree if asked for.
    if (fx_param_exists(tt_util_module, "train_unpruned_output")) {
      FILE *fp = 
	fopen(fx_param_str_req(tt_util_module,
			       "train_unpruned_output"), "w");
      if (fp != NULL) {
	for (size_t i = 0; i < x.n_cols(); i++) {
	  Vector test_p;
	  x.MakeColumnVector(i, &test_p);
	  double f = ttree->ComputeValue(test_p, false);
	  fprintf(fp, "%lg\n", f);
	} // end for
	fclose(fp);
      }
    }

    // exit(0);
    // sequential pruning and saving the alpha vals
    // FIX: In TTree, you just need to save the alpha values,
    // during CV, you just consider the test errors 
    // on the validation set
    std::vector<std::pair<double, int> >  buff_error_vec;
    // std::vector<double> change_in_estimate;
    std::vector<double> pruned_sequence;
    while (ttree->subtree_leaves() > 1) {
      std::pair<double, int> tmp_pair (0.0, 0);
      buff_error_vec.push_back(tmp_pair);
      pruned_sequence.push_back(old_alpha);
//       change_in_estimate.push_back(fabs(new_f - old_f));
      old_alpha = alpha;
//       old_f = new_f;
      alpha = ttree->PruneAndUpdate(old_alpha);
//       new_f = ttree->st_estimate();
      DEBUG_ASSERT_MSG((alpha < LDBL_MAX)||(ttree->subtree_leaves() == 1),
		       "old_alpha:%Lg, alpha:%Lg, tree size:%zu"d",
		       old_alpha, alpha, ttree->subtree_leaves());
      DEBUG_ASSERT(alpha > old_alpha);
//       DEBUG_ASSERT(ttree->subtree_leaves_error() >= -1.0 * tree_seq.second);
      // printf("%Lg\n", ttree->subtree_leaves_error() - last);
 
    } // end while
//     std::pair<long double, long double> tree_seq (old_alpha,
// 		  -1.0 * ttree->subtree_leaves_error());
    pruned_sequence.push_back(old_alpha);
    // std::pair<double, int> tmp_pair (0.0, 0);
    // buff_error_vec.push_back(tmp_pair);
//     change_in_estimate.push_back(fabs(new_f - old_f));

    NOTIFY("%zu"d trees in the sequence, max_alpha:%Lg.\n",
	   (size_t) pruned_sequence.size(), old_alpha);

    
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


    // Permute the x once just for keeps
    Matrix pdata;
    Vector py;
    pdata.Copy(x);
    py.Copy(y);

    // PermuteMatrix(x, y, &pdata, &py);

    size_t test_size = x.n_cols() / folds;

    // Go through each fold
    for (size_t fold = 0; fold < folds; fold++) {
      // NOTIFY("Fold %zu"d...", fold+1);

      // break up data into train and test set
      Matrix test;
      Vector test_y;
      size_t start = fold * test_size,
	end = min ((fold + 1) * test_size,x.n_cols());
      pdata.MakeColumnSlice(start, end - start, &test);
      py.MakeSubvector(start, end - start, &test_y);
      Matrix train;
      Vector train_y;
      train.Init(pdata.n_rows(), pdata.n_cols() - (end - start));
      train_y.Init(pdata.n_cols() - (end - start));
      size_t k = 0;
      for (size_t j = 0; j < pdata.n_cols(); j++) {
	if (j < start || j >= end) {
	  train_y[k] = py[j];

	  Vector temp_vec;
	  pdata.MakeColumnVector(j, &temp_vec);
	  train.CopyVectorToColumn(k++, temp_vec);
	} // end if
      } // end for
      DEBUG_ASSERT(k == train.n_cols());
      DEBUG_ASSERT(k == train_y.length());

      // go through the motions - computing 
      // the maximum and minimum for each dimensions
      ArrayList<double> max_vals_cv, min_vals_cv;
      max_vals_cv.Init(train.n_rows());
      min_vals_cv.Init(train.n_rows());

      Matrix temp_t;
      la::TransposeInit(train, &temp_t);

      for (size_t i = 0; i < temp_t.n_cols(); i++) {
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
      TTree *ttree_cv = new TTree();
      ttree_cv->Init(max_vals_cv, min_vals_cv,
		     train.n_cols(), train_y, ttree_mod);
      // Getting ready to grow the tree
      ArrayList<size_t> old_from_new_cv;
      old_from_new_cv.Init(train.n_cols());
      for (size_t i = 0; i < old_from_new_cv.size(); i++) {
	old_from_new_cv[i] = i;
      }

      // Growing the tree
      old_alpha = 0.0;
      alpha = ttree_cv->Grow(train, train_y, &old_from_new_cv);

      // sequential pruning with all the values of available
      // alphas and adding values for test values
      std::vector<double>::iterator it;
      std::vector<std::pair<double, int> >::iterator jt
	= buff_error_vec.begin();
      for (it = pruned_sequence.begin();
	   it < pruned_sequence.end() -2; ++it, ++jt) {
      
	// compute test values for this state of the tree
	// long double val_cv = 0.0;
	double buff = 0.0;
	int error = 0;
	for (size_t i = 0; i < test.n_cols(); i++) {
	  Vector test_point;
	  test.MakeColumnVector(i, &test_point);
	  double val = ttree_cv->ComputeValue(test_point, false);
	  if (val < test_y[i]) {
	    error++;
	  } else {
	    buff += (val - test_y[i]);
	  }
	}

	jt->first += buff;
	jt->second += error;

	// getting the new alpha value and pruning accordingly
	old_alpha = sqrt((*(it+1)) * (*(it+2)));
	alpha = ttree_cv->PruneAndUpdate(old_alpha);
      } // end for

      // compute test values for this state of the tree
      double buff = 0.0;
      int error = 0;
      for (size_t i = 0; i < test.n_cols(); i++) {
	Vector test_point;
	test.MakeColumnVector(i, &test_point);
	double val = ttree_cv->ComputeValue(test_point, false);

	// printf("%lg, ", val);

	if (val < test_y[i]) {
	  error++;
	} else {
	  buff += (val - test_y[i]);
	}
      }

      // printf("\n");
      jt->first += buff;
      jt->second += error;

      DEBUG_ASSERT(pruned_sequence.size() == buff_error_vec.size()+1);

      // exit(0);
    } // end for


    // TAI: thing here to worry about is which one is to 
    // be considered the best CV error, one with the lowest
    // buff value or least error value - its a tradeoff
    // THIS HAS TO BE WORKED OUT

    // For now thinking of using the lowest alpha for which 
    // error == 0, since I am guessing, lower the alpha,
    // lower the buff value.
    double optimal_alpha = -1.0;
    int best_error = -1;
    std::vector<double>::iterator it;
    std::vector<std::pair<double, int> >::iterator jt
      = buff_error_vec.begin();
    size_t i = 0;

    for (it = pruned_sequence.begin();
	 it < pruned_sequence.end() -1; ++it, ++jt) {

      i++;
      printf("%zu"d: %lg,%lg,%d\n",i, (*it), jt->first, jt->second);

      if ((best_error == -1) || (jt->second < best_error)) {
	best_error = jt->second;
	optimal_alpha = (*it);
      } // end if
    } // end for

    printf("OA: %lg\n", optimal_alpha);
    exit(0);

    // Saving the x since it would be modified
    // while growing the tree
    new_x.Destruct();
    new_x.Copy(x);

    new_y.Destruct();
    new_y.Copy(y);

    // Initializing the tree
    TTree *ttree_opt = new TTree();
    ttree_opt->Init(max_vals, min_vals,
		    x.n_cols(),
		    new_y, ttree_mod);
    // Getting ready to grow the tree
    for (size_t i = 0; i < old_from_new.size(); i++) {
      old_from_new[i] = i;
    }

    // Growing the tree
    old_alpha = 0.0;
    alpha = ttree_opt->Grow(new_x, new_y, &old_from_new);
    NOTIFY("%zu"d leaf nodes in this tree\n opt_alpha:%lg",
	   ttree_opt->subtree_leaves(), optimal_alpha);
    //   printf("%zu"d leaf nodes in this tree\n opt_alpha:%Lg\n",
    // 	 ttree_opt->subtree_leaves(), optimal_alpha);

    // Pruning with optimal alpha
    while (old_alpha < optimal_alpha) {
      old_alpha = alpha;
      alpha = ttree_opt->PruneAndUpdate(old_alpha);
      DEBUG_ASSERT_MSG((alpha < DBL_MAX)||(ttree->subtree_leaves() == 1),
		       "old_alpha:%Lg, alpha:%Lg, tree size:%zu"d",
		       old_alpha, alpha, ttree->subtree_leaves());
      DEBUG_ASSERT(alpha > old_alpha);
    } // end while

    NOTIFY("%zu"d leaf nodes in this tree\n old_alpha:%Lg",
	   ttree_opt->subtree_leaves(), old_alpha);

    return ttree_opt;
  }

  TTree* Driver(datanode *tt_util_mod, datanode *ttree_mod) {

    srand( time(NULL));
    tt_util_module = tt_util_mod;
    std::string x_file
      = fx_param_str_req(tt_util_module, "x");
    Matrix x;
    NOTIFY("Loading data file...\n");
    data::Load(x_file.c_str(), &x);

    std::string y_file
      = fx_param_str_req(tt_util_module, "y");
    Matrix y_mat;
    NOTIFY("Loading data file...\n");
    data::Load(y_file.c_str(), &y_mat);

    DEBUG_ASSERT(y_mat.n_rows() == 1);
    Vector y;
    y.Init(y_mat.n_cols());

    for (size_t i = 0; i < y_mat.n_cols(); i++)
      y[i] = y_mat.get(0, i);


    DEBUG_ASSERT(x.n_cols() == y.length());



    NOTIFY("%zu"d points in %zu"d dims.",
	   x.n_cols(), x.n_rows());

    //   // getting information about the feature type
    //   // REAL, INTEGER, NOMINAL. Most probably using
    //   // enum ... don't know how to use it though.
    //   ArrayList<enum> dim_type;
    //   dim_type.Init(x.n_rows());
    //   for (size_t i = 0; i < dim_type.size(); i++) {
    //     // assign dim type somehow
    //   } // end for

    // starting the training timer
    fx_timer_start(tt_util_module, "train_time");

    // cross-validation here
    size_t folds = fx_param_int(tt_util_module, "folds", 10);
    if (folds == 0) {
      folds = x.n_cols();
      NOTIFY("Starting Leave-One-Out Cross validation");
    } else 
      NOTIFY("Starting %zu"d-fold Cross validation", folds);


    // stopping the training timer
    fx_timer_stop(tt_util_module, "train_time");

    // obtaining the optimal tree
    TTree *ttree_opt = Trainer(x, y, folds, ttree_mod);
 

    exit(0);


    // starting the test timer
    fx_timer_start(tt_util_module, "test_time");


    // computing densities for the train points in the
    // optimal tree
    if (fx_param_bool(tt_util_module, "compute_train_output", false)) {
      FILE *fp;
      fp = NULL;

      // opening the file to write if provided
      if (fx_param_exists(tt_util_module, "train_output")) {
	std::string train_density_file
	  = fx_param_str_req(tt_util_module, "train_output");
	fp = fopen(train_density_file.c_str(), "w");
      }

      for (size_t i = 0; i < x.n_cols(); i++) {
	Vector test_p;
	x.MakeColumnVector(i, &test_p);
	double f = ttree_opt->ComputeValue(test_p, false);
	if (fp != NULL)
	  fprintf(fp, "%lg\n", f);
      } // end for

      if (fp != NULL)
	fclose(fp);
    }

    // computing the density at the provided test points
    // and outputting the density in the given file.
    if (fx_param_exists(tt_util_module, "test")) {
      std::string test_file = fx_param_str_req(tt_util_module, "test");
      Matrix test_set;
      NOTIFY("Loading test data...\n");
      data::Load(test_file.c_str(), &test_set);

      NOTIFY("%zu"d points in %zu"d dims.", test_set.n_cols(),
	     test_set.n_rows());

      FILE *fp;
      fp = NULL;

      if (fx_param_exists(tt_util_module, "test_output")) {
	std::string test_density_file
	  = fx_param_str_req(tt_util_module, "test_output");
	fp = fopen(test_density_file.c_str(), "w");
      }

      for (size_t i = 0; i < test_set.n_cols(); i++) {
	Vector test_p;
	test_set.MakeColumnVector(i, &test_p);
	double f = ttree_opt->ComputeValue(test_p, false);
	if (fp != NULL)
	  fprintf(fp, "%lg\n", f);
      } // end for

      if (fp != NULL) 
	fclose(fp);
    }

    fx_timer_stop(tt_util_module, "test_time");


    // printing the final tree
    if (fx_param_bool(tt_util_module, "print_tree", false)) {
      if (fx_param_exists(tt_util_module, "tree_file")) {
	std::string tree_file = fx_param_str(tt_util_module,
					     "tree_file",
					     "pruned_tree.txt");
	FILE *fp = fopen(tree_file.c_str(), "w");
	if (fp != NULL) {
	  ttree_opt->WriteTree(0, fp);
	  fclose(fp);
	}
      } else {
	ttree_opt->WriteTree(0, stdout);
	printf("\n");
      }
    }

    // print the leaf memberships for the optimal tree
//     if (fx_param_exists(tt_util_module, "labels")) {
//       std::string labels_file = fx_param_str_req(tt_util_module, "labels");
//       Matrix labels;
//       NOTIFY("loading labels.\n");
//       data::Load(labels_file.c_str(), &labels);
//       size_t num_classes = fx_param_int_req(tt_util_module, "num_classes");

//       DEBUG_ASSERT(x.n_cols() == labels.n_cols());
//       DEBUG_ASSERT(labels.n_rows() == 1);

//       PrintLeafMembership(ttree_opt, x, labels, num_classes);
//     }


    if(fx_param_bool(tt_util_module, "print_vi", false)) {
      if(fx_param_exists(tt_util_module, "vi_file")) {
	std::string vi_file = fx_param_str_req(tt_util_module, "vi_file");
	FILE *fp = fopen(vi_file.c_str(), "w");
	if (fp != NULL) {
	  PrintVariableImportance(ttree_opt, x.n_rows(), fp);
	  fclose(fp);
	}
      } else 
	PrintVariableImportance(ttree_opt, x.n_rows(), stdout);
    }

    return ttree_opt;
  } // end Driver()

}; // namespace tt_utils

#endif
