/**
 * @file dt_utils.hpp
 * @ Parikshit Ram (pram@cc.gatech.edu)
 *
 * This file implements functions to perform
 * different tasks with the Density Tree class.
 */

#ifndef __MLPACK_METHODS_DET_DT_UTILS_HPP
#define __MLPACK_METHODS_DET_DT_UTILS_HPP

#include <string>

#include <mlpack/core.hpp>
#include "dtree.hpp"

using namespace std;

namespace mlpack {
namespace det {

  template<typename eT>
  void PrintLeafMembership(DTree<eT> *dtree,
			   const arma::Mat<eT>& data,
			   const arma::Mat<int>& labels,
			   size_t num_classes,
			   string leaf_class_membership_file = "") 
  {
    // tag the leaves with numbers
    int num_leaves = dtree->TagTree(0);
    
    arma::Mat<size_t> table(num_leaves, num_classes);
    table.zeros();

    for (size_t i = 0; i < data.n_cols; i++) {
      arma::Col<eT> test_p = data.unsafe_col(i);
      int leaf_tag = dtree->FindBucket(&test_p);
      int label = labels[i];
      table(leaf_tag, label) += 1;
    } // end for

    if (leaf_class_membership_file == "") {
      Log::Warn << "Leaf Membership: Classes in each leaf" << endl
		<< table << endl;
    } else {
      // create a stream for the file
      ofstream outfile(leaf_class_membership_file.c_str());
      if (outfile.good()) {
	Log::Warn << "Leaf Membership: Classes in each leaf" 
		  << " printed in '" << leaf_class_membership_file
		  << "'" << endl;
	outfile << table;
      } else {
	Log::Warn << "Can't open '" << leaf_class_membership_file
		  << "'" << endl;
      }
      outfile.close();
    }

    return;
    // maybe print some more statistics if these work out well
  } // PrintLeafMembership


  template<typename eT>
  void PrintVariableImportance(DTree<eT> *dtree,
			       size_t num_dims,
			       string vi_file = "")
  {
    arma::Col<double> *imps 
      = new arma::Col<double>(num_dims);

    for (size_t i = 0; i < imps->n_elem; i++)
      (*imps)[i] = 0.0;
    
    dtree->ComputeVariableImportance(imps);
    double max = 0.0;
    for (size_t i = 0; i < imps->n_elem; i++)
      if ((*imps)[i] > max)
	max = (*imps)[i];
    Log::Warn << "Max. variable importance: " << max << endl;


    if (vi_file == "") {
      Log::Warn << "Variable importance: " << endl
		<< imps->t();
    } else {
      ofstream outfile(vi_file.c_str());
      if (outfile.good()) {
	Log::Warn << "Variable importance printed in '"
		  << vi_file << "'" << endl;
	outfile << *imps;
      } else {
	Log::Warn << "Can't open '" << vi_file
		  << "'" << endl;
      }
      outfile.close();
    }    

    return;
  } // PrintVariableImportance


  // This function trains the optimal decision tree
  // using the given number of folds
  template<typename eT>
  DTree<eT> *Trainer(arma::Mat<eT>* dataset, 
		     size_t folds,
		     bool useVolumeReg = false,
		     size_t maxLeafSize = 10,
		     size_t minLeafSize = 5,
		     string unprunedTreeOutput = "") 
  {
    // Initializing the tree
    DTree<eT> *dtree = new DTree<eT>(dataset);

    // Getting ready to grow the tree
    arma::Col<size_t> old_from_new(dataset->n_cols);
    for (size_t i = 0; i < old_from_new.n_elem; i++) {
      old_from_new[i] = i;
    }

    // Saving the dataset since it would be modified
    // while growing the tree
    arma::Mat<eT>* new_dataset = new arma::Mat<eT>(*dataset);

    // Growing the tree
    long double old_alpha = 0.0;
    long double alpha = dtree->Grow(new_dataset, &old_from_new,
				    useVolumeReg, maxLeafSize, 
				    minLeafSize);
    // clear the data set
    delete new_dataset;

    Log::Info << dtree->subtree_leaves() 
	      << " leaf nodes in the tree with full data, min_alpha: "
	      << alpha << endl;

    // computing densities for the train points in the
    // full tree if asked for.
    if (unprunedTreeOutput != "") {

      ofstream outfile(unprunedTreeOutput.c_str());
      if (outfile.good()) {
	for (size_t i = 0; i < dataset->n_cols; i++) {
	  arma::Col<eT> test_p = dataset->unsafe_col(i);
	  outfile << dtree->ComputeValue(&test_p) << endl;
	} // end for
      } else {
	Log::Warn << "Can't open '" << unprunedTreeOutput
		  << "'" << endl;
      }

      outfile.close();

    } // if unprunedTreeOutput

    // sequential pruning and saving the alpha vals and the
    // values of c_t^2*r_t
    std::vector<std::pair<long double, long double> > pruned_sequence;
    while (dtree->subtree_leaves() > 1) {

      std::pair<long double, long double> tree_seq
	(old_alpha, -1.0 * dtree->subtree_leaves_error());
      pruned_sequence.push_back(tree_seq);
      old_alpha = alpha;
      alpha = dtree->PruneAndUpdate(old_alpha, useVolumeReg);

      // some checks
      assert((alpha < std::numeric_limits<long double>::max())
	     ||(dtree->subtree_leaves() == 1));
      assert(alpha > old_alpha);
      assert(dtree->subtree_leaves_error() >= -1.0 * tree_seq.second);

    } // end while
    
    std::pair<long double, long double> tree_seq 
      (old_alpha, -1.0 * dtree->subtree_leaves_error());
    pruned_sequence.push_back(tree_seq);

    Log::Info << pruned_sequence.size()
	      << " trees in the sequence, max_alpha: "
	      << old_alpha << endl;

    delete dtree;

    arma::Mat<eT>* cvdata = new arma::Mat<eT>(*dataset);

    size_t test_size = dataset->n_cols / folds;

    // Go through each fold
    for (size_t fold = 0; fold < folds; fold++) {
      
      // break up data into train and test set
      size_t start = fold * test_size,
	end = std::min((fold + 1) * test_size, (size_t) cvdata->n_cols);
      arma::Mat<eT> test = cvdata->cols(start, end - 1);
      arma::Mat<eT>* train 
	= new arma::Mat<eT>(cvdata->n_rows, 
			    cvdata->n_cols - test.n_cols);

      if (start == 0 && end < cvdata->n_cols) {
	assert(train->n_cols == cvdata->n_cols - end);
	train->cols(0, train->n_cols - 1) 
	  = cvdata->cols(end, cvdata->n_cols - 1);


      } else if (start > 0 && end == cvdata->n_cols) {
	assert(train->n_cols == start);
	train->cols(0, train->n_cols - 1) = cvdata->cols(0, start - 1);

      } else {
	assert(train->n_cols == start + cvdata->n_cols - end);

	train->cols(0, start - 1) = cvdata->cols(0, start - 1);
	train->cols(start, train->n_cols - 1) 
	  = cvdata->cols(end, cvdata->n_cols - 1);
      }

      assert(train->n_cols + test.n_cols == cvdata->n_cols);

      // Initializing the tree
      DTree<eT> *dtree_cv = new DTree<eT>(train);

      // Getting ready to grow the tree
      arma::Col<size_t> old_from_new_cv(train->n_cols);
      for (size_t i = 0; i < old_from_new_cv.n_elem; i++) {
	old_from_new_cv[i] = i;
      }

      // Growing the tree
      old_alpha = 0.0;
      alpha = dtree_cv->Grow(train, &old_from_new_cv,
			     useVolumeReg, maxLeafSize, 
			     minLeafSize);

      // sequential pruning with all the values of available
      // alphas and adding values for test values
      std::vector<std::pair<long double, long double> >::iterator it;
      for (it = pruned_sequence.begin();
	   it < pruned_sequence.end() -2; ++it) {
      
	// compute test values for this state of the tree
	long double val_cv = 0.0;
	for (size_t i = 0; i < test.n_cols; i++) {
	  arma::Col<eT> test_point = test.unsafe_col(i);
	  val_cv += dtree_cv->ComputeValue(&test_point);
	}

	// update the cv error value
	it->second -= 2.0 * val_cv / (long double) dataset->n_cols;

	// getting the new alpha value and pruning accordingly
	old_alpha = sqrt(((it+1)->first) * ((it+2)->first));
	alpha = dtree_cv->PruneAndUpdate(old_alpha, useVolumeReg);
      } // end for

      // compute test values for this state of the tree
      long double val_cv = 0.0;
      for (size_t i = 0; i < test.n_cols; i++) {
	arma::Col<eT> test_point = test.unsafe_col(i);
	val_cv += dtree_cv->ComputeValue(&test_point);
      }
      // update the cv error value
      it->second -= 2.0 * val_cv / (long double) dataset->n_cols;

      test.reset();
      delete train;

      delete dtree_cv;

    } // end for loop for number of cv-folds

    delete cvdata;

    long double optimal_alpha = -1.0, 
      best_cv_error = numeric_limits<long double>::max();
    std::vector<std::pair<long double, long double> >::iterator it;

    for (it = pruned_sequence.begin();
	 it < pruned_sequence.end() -1; ++it) {

      if (it->second < best_cv_error) {
	best_cv_error = it->second;
	optimal_alpha = it->first;
      } // end if
    } // end for

    Log::Info << "Optimal alpha: " << optimal_alpha << endl;

    // Initializing the tree
    DTree<eT> *dtree_opt = new DTree<eT>(dataset);
    // Getting ready to grow the tree
    for (size_t i = 0; i < old_from_new.n_elem; i++) {
      old_from_new[i] = i;
    }

    // Saving the dataset since it would be modified
    // while growing the tree
    new_dataset = new arma::Mat<eT>(*dataset);

    // Growing the tree
    old_alpha = 0.0;
    alpha = dtree_opt->Grow(new_dataset, &old_from_new,
			    useVolumeReg, maxLeafSize, 
			    minLeafSize);

    // Pruning with optimal alpha
    while (old_alpha < optimal_alpha 
	   && dtree_opt->subtree_leaves() > 1) {
      old_alpha = alpha;
      alpha = dtree_opt->PruneAndUpdate(old_alpha, useVolumeReg);

      // some checks
      assert((alpha < numeric_limits<long double>::max())
	     ||(dtree_opt->subtree_leaves() == 1));
      assert(alpha > old_alpha);
    } // end while

    Log::Info << dtree_opt->subtree_leaves() 
	      << " leaf nodes in the optimally pruned tree,"
	      << " optimal alpha: "
	      << old_alpha << endl;

    delete new_dataset;

    return dtree_opt;
  } // Trainer

}; // namespace det
}; // namespace mlpack

#endif // __MLPACK_METHODS_DET_DT_UTILS_HPP
