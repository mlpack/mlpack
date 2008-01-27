/** @file ortho_range_search.h
 *
 *  This file contains an implementation of a tree-based and a naive
 *  algorithm for orthogonal range search.
 *
 *  @author Dongryeol Lee (dongryel)
 */
#ifndef ORTHO_RANGE_SEARCH_H
#define ORTHO_RANGE_SEARCH_H

#include "range_reader.h"
#include "fastlib/fastlib.h"

/** @brief Naive orthogonal range search class.
 *
 *  @code
 *    NaiveOrthoRangeSearch search;
 *    search.Init(dataset);
 *    search.Compute(low_coord_limits, high_coord_limits);
 *
 *    Vector naive_search_results;
 *
 *    // Make sure that the vector is uninitialized before passing.
 *    search.get_results(&naive_search_results);
 *  @endcode
 */
class NaiveOrthoRangeSearch {
  
  // This class object cannot be copied!
  FORBID_ACCIDENTAL_COPIES(NaiveOrthoRangeSearch);

 private:

  /** @brief The i-th position of this array tells whether the i-th
   *         point is in the specified orthogonal range.
   */
  ArrayList<bool> in_range_;

  /** @brief The dataset. 
   */
  Matrix data_;
  
 public:
  
  ////////// Constructor/Destructor //////////

  /** @brief Constructor which does not do anything.
   */
  NaiveOrthoRangeSearch() {}

  /** @brief Destructor which does not do anything.
   */
  ~NaiveOrthoRangeSearch() {}

  ////////// Getters/Setters //////////

  /** @brief Retrieve the result of the search.
   *
   *  @param results An uninitialized vector which will have the boolean 
   *                 results representing the search results.
   */
  void get_results(ArrayList<bool> *results) const {
    results->Init(in_range_.size());

    for(index_t i = 0; i < in_range_.size(); i++) {
      (*results)[i] = in_range_[i];
    }
  }

  ////////// User-level Functions //////////

  /** @brief Initialize the computation object.
   *
   *  @param data The data used for orthogonal range search.
   */
  void Init(Matrix &data) {

    // copy the incoming data
    data_.Copy(data);
    
    // re-initialize boolean flag
    in_range_.Init(data_.n_cols());
    for(index_t i = 0; i < data_.n_cols(); i++) {
      in_range_[i] = false;
    }
  }

  /** @brief The main computation of naive orthogonal range search.
   *
   *  @param low_coord_limits The lower coordinate range of the search window.
   *  @param high_coord_limits The upper coordinate range of the search
   *                           window.
   */
  void Compute(Vector &low_coord_limits, Vector &high_coord_limits) {

    // Start the search.
    fx_timer_start(NULL, "naive_search");
    for(index_t i = 0; i < data_.n_cols(); i++) {

      Vector pt;
      bool flag = true;
      data_.MakeColumnVector(i, &pt);
      
      // Determine which one of the two cases we have: EXCLUDE, SUBSUME
      // first the EXCLUDE case: when dist is above the upper bound distance
      // of this dimension, or dist is below the lower bound distance of
      // this dimension
      for(index_t d = 0; d < data_.n_rows(); d++) {
	if(pt[d] < low_coord_limits[d] || pt[d] > high_coord_limits[d]) {
	  flag = false;
	  break;
	}
      }
      in_range_[i] = flag;
    }
    fx_timer_stop(NULL, "naive_search");
    
    // Search is now finished.
    
  }

};

/** @brief Faster orthogonal range search class using a tree.
 *
 *  @code
 *    OrthoRangeSearch search;
 *    search.Init(dataset, NULL);
 *    search.Compute(low_coord_limits, high_coord_limits);
 *
 *    Vector search_results;
 *
 *    // Make sure that the vector is uninitialized before passing.
 *    search.get_results(&search_results);
 *  @endcode
 */
class OrthoRangeSearch {

 public:
  
  ////////// Constructor/Destructor //////////

  /** @brief Constructor that initializes pointers to NULL.
   */
  OrthoRangeSearch() {
    tree_buffer_ = NULL;
    old_from_new_buffer_ = NULL;
    new_from_old_buffer_ = NULL;
    root_ = NULL;
  }

  /** @brief Destructor that frees up memory.
   */
  ~OrthoRangeSearch() {
    if(tree_buffer_ != NULL) {
      mem::Free(tree_buffer_);
      mem::Free(old_from_new_buffer_);
      mem::Free(new_from_old_buffer_);
    }
    else {
      delete root_;
    }
  }

  ////////// Getters/Setters //////////

  /** @brief Retrieve the result of the search.
   *
   *  @param results An uninitialized vector which will have the boolean
   *                 results representing the search results.
   */
  void get_results(ArrayList<bool> *results) {
    results->Init(candidate_points_.size());
    
    for(index_t i = 0; i < candidate_points_.size(); i++) {
      (*results)[i] = candidate_points_[i];
    }
  }

  ////////// User-level Functions //////////

  /** @brief Perform the orthogonal range search.
   *
   *  @param low_coord_limits The lower coordinate range of the search window.
   *  @param high_coord_limits The upper coordinate range of the search
   *                           window.
   */
  void Compute(Vector &low_coord_limits, Vector &high_coord_limits) {

    fx_timer_start(NULL, "tree_range_search");
    ortho_range_search(root_, 0, low_coord_limits, high_coord_limits);
    fx_timer_stop(NULL, "tree_range_search");

    // reshuffle the results to account for shuffling during tree construction
    ArrayList<bool> tmp_results;
    tmp_results.Init(candidate_points_.size());

    for(index_t i = 0; i < candidate_points_.size(); i++) {
      tmp_results[old_from_new_[i]] = candidate_points_[i];
    }
    for(index_t i = 0; i < candidate_points_.size(); i++) {
      candidate_points_[i] = tmp_results[i];
    }
  }

  /** @brief Save the tree to the file.
   *
   *  @param save_tree_file_name The tree is serialized to the file whose
   *                             name is given as the argument.
   */
  void SaveTree(const char *save_tree_file_name) {

    printf("Serializing the tree data structure...\n");

    FILE *output = fopen(save_tree_file_name, "w+");

    // first serialize the total amount of bytes needed for serializing the
    // tree and the tree itself
    int tree_size = ot::PointerFrozenSize(*root_);
    printf("Tree occupies %d bytes...\n", tree_size);

    fwrite((const void *) &tree_size, sizeof(int), 1, output);
    char *tmp_root = (char *) mem::AllocBytes<Tree>(tree_size);
    ot::PointerFreeze(*root_, tmp_root);
    fwrite((const void *) tmp_root, tree_size, 1, output);
    mem::Free(tmp_root);

    // then serialize the permutation of the points due to tree construction
    // along with its sizes
    int old_from_new_size = ot::PointerFrozenSize(old_from_new_);
    int new_from_old_size = ot::PointerFrozenSize(new_from_old_);
    char *tmp_array = 
      (char *) mem::AllocBytes<ArrayList<index_t> >(old_from_new_size);

    fwrite((const void *) &old_from_new_size, sizeof(int), 1, output);    
    ot::PointerFreeze(old_from_new_, tmp_array);
    fwrite((const void *) tmp_array, old_from_new_size, 1, output);
    
    fwrite((const void*) &new_from_old_size, sizeof(int), 1, output);
    ot::PointerFreeze(new_from_old_, tmp_array);
    fwrite((const void *) tmp_array, new_from_old_size, 1, output);
    
    mem::Free(tmp_array);

    printf("Tree is serialized...\n");
  }

  /** @brief Initialization function - to read the data and to construct tree.
   *
   *  @param dataset The dataset for orthogonal range searching.
   *  @param load_tree_file_name If NULL, the tree is built from scratch.
   *                             If not NULL, the tree is loaded from the
   *                             file whose name is given as the argument.
   */
  void Init(Matrix &dataset, const char *load_tree_file_name) {

    int leaflen = fx_param_int(NULL, "leaflen", 20);

    // Make a copy of the dataset.
    data_.Copy(dataset);

    fx_timer_start(NULL, "tree_d");

    // If the user wants to load the tree from a file,
    if(load_tree_file_name != NULL) {
      LoadTree(load_tree_file_name);
    }

    // Otherwise, construct one from scratch.
    else {
      root_ = tree::MakeKdTreeMidpoint<Tree>(data_, leaflen,
					     &old_from_new_,
					     &new_from_old_);
    }
    fx_timer_stop(NULL, "tree_d");
    
    // initialize candidate nodes and points */
    candidate_points_.Init(data_.n_cols());
    for(index_t i = 0; i < data_.n_cols(); i++) {
      candidate_points_[i] = false;
    }
  }

 private:

  /** @brief This defines the type of the tree used in this algorithm. */
  typedef BinarySpaceTree<DHrectBound<2>, Matrix> Tree;

  /** @brief Flag determining a prune */
  enum PruneStatus {SUBSUME, INCONCLUSIVE, EXCLUDE};

  ////////// Private Member Variables //////////

  /** @brief Pointer to the dataset */
  Matrix data_;

  /** @brief Buffer for loading up old_from_new mapping. */
  ArrayList<index_t> *old_from_new_buffer_;

  /** @brief Buffer for loading up new_from_old mapping. */
  ArrayList<index_t> *new_from_old_buffer_;

  /** @brief Temporary pointer used for loading the tree from a file. */
  Tree *tree_buffer_;

  ArrayList<index_t> old_from_new_;
  
  ArrayList<index_t> new_from_old_;

  /** @brief The root of the tree */
  Tree *root_;

  /** @brief List of candidate points. The i-th position tells whether
   *         the i-th data point was in the orthogonal search window.
   */
  ArrayList<bool> candidate_points_;

  ////////// Private Member Functions //////////

  /** @brief Load the tree from the file.
   *
   *  @param load_tree_file_name Loads up the saved tree from the file whose
   *                             name is given as the argument.
   */
  void LoadTree(const char *load_tree_file_name) {
    
    //const char *tfname = fx_param_str(NULL, "load_tree_file", "savedtree");
    FILE *input = fopen(load_tree_file_name, "r");
    
    // read the tree size
    int tree_size, old_from_new_size, new_from_old_size;
    fread((void *) &tree_size, sizeof(int), 1, input);

    printf("Tree file: %s occupies %d bytes...\n", load_tree_file_name, 
	   tree_size);
    tree_buffer_ = mem::AllocBytes<Tree>(tree_size);
    fread((void *) tree_buffer_, 1, tree_size, input);
    root_ = ot::PointerThaw<Tree>((char *) tree_buffer_);
    
    // read old_from_new
    fread((void *) &old_from_new_size, sizeof(int), 1, input);
    old_from_new_buffer_ = 
    mem::AllocBytes<ArrayList<index_t> >(old_from_new_size);
    fread((void *) old_from_new_buffer_, old_from_new_size, 1, input);
    old_from_new_.Copy(*(ot::PointerThaw<ArrayList<index_t> >
			 ((char *) old_from_new_buffer_)));

    // read new_from_old
    fread((void *) &new_from_old_size, sizeof(int), 1, input);
    new_from_old_buffer_ = 
    mem::AllocBytes<ArrayList<index_t> >(new_from_old_size);
    fread((void *) new_from_old_buffer_, new_from_old_size, 1, input);
    new_from_old_.Copy(*(ot::PointerThaw<ArrayList<index_t> >
			 ((char *) new_from_old_buffer_)));

    printf("Tree has been loaded...\n");

    // apply permutation to the dataset
    Matrix tmp_data;
    tmp_data.Init(data_.n_rows(), data_.n_cols());

    for(index_t i = 0; i < data_.n_cols(); i++) {
      Vector source, dest;
      data_.MakeColumnVector(i, &source);
      tmp_data.MakeColumnVector(new_from_old_[i], &dest);
      dest.CopyValues(source);
    }
    data_.Destruct();
    data_.Own(&tmp_data);
  }
  
  /** @brief The base case.
   *
   *  @param node The tree node currently under consideration.
   *  @param start_dim The dimension currently under consideration.
   *  @param low_coord_limits The lower coordinate limits of the search.
   *  @param high_coord_limits The upper coordinate limits of the search.
   */
  void ortho_slow_range_search(Tree *node, int start_dim,
			       const Vector &low_coord_limits,
			       const Vector &high_coord_limits) {
    PruneStatus prune_flag;

    for(index_t row = node->begin(); row < node->end(); row++) {
      prune_flag = SUBSUME;

      for(index_t d = start_dim; d < data_.n_rows(); d++) {
	// determine which one of the two cases we have: EXCLUDE, SUBSUME

	// first the EXCLUDE case: when dist is above the upper bound distance
	// of this dimension, or dist is below the lower bound distance of
	// this dimension
	if(data_.get(d, row) > high_coord_limits[d] ||
	   data_.get(d, row) < low_coord_limits[d]) {
	  prune_flag = EXCLUDE;
	  break;
	}
      }

      if(prune_flag == SUBSUME) {
	candidate_points_[row] = true;
      }
      else {
	candidate_points_[row] = false;
      }
    }
  }

  /** @brief The workhorse algorithm for fast orthgonal range
   *         search.
   *
   *  @param node The current tree node that is under consideration.
   *  @param start_dim The dimension currently under consideration.
   *  @param low_coord_limits The lower coordinate limits of the search.
   *  @param high_coord_limits The upper coordinate limits of the search
   *                           window.
   */
  void ortho_range_search(Tree *node, int start_dim,
			  const Vector &low_coord_limits, 
			  const Vector &high_coord_limits) {

    PruneStatus prune_flag = SUBSUME;
    
    // loop over each dimension to determine inclusion/exclusion by 
    // determining the lower and the upper bound distance per each dimension 
    // for the given reference node, kn
    for(index_t d = start_dim; d < data_.n_rows(); d++) {

      DRange node_dir_range = node->bound().get(d);

      // determine which one of the three cases we have: EXCLUDE, SUBSUME, or
      // INCONCLUSIVE.
      
      // first the EXCLUDE case: when mindist is above the upper bound 
      // distance of this dimension,  or maxdist is below the lower bound 
      // distance of this dimension
      if(node_dir_range.lo > high_coord_limits[d] ||
	 node_dir_range.hi < low_coord_limits[d]) {
	for(index_t i = node->begin(); i < node->end(); i++) {
	  candidate_points_[i] = false;
	}
	return;
      }
      // otherwise, check for SUBSUME case
      else if(low_coord_limits[d] <= node_dir_range.lo &&
	      node_dir_range.hi <= high_coord_limits[d]) {
      }
      // if any dimension turns out to be inconclusive, then break.
      else {
	start_dim = d;
	prune_flag = INCONCLUSIVE;
	break;
      }
    }
    
    // in case of subsume, then add all points owned by this node to
    // candidates
    if(prune_flag == SUBSUME) {
      for(index_t i = node->begin(); i < node->end(); i++) {
	candidate_points_[i] = true;
      }
      return;
    }
    else if(node->is_leaf()) {
      ortho_slow_range_search(node, start_dim, low_coord_limits, 
			      high_coord_limits);
    }
    else {
      ortho_range_search(node->left(), start_dim, low_coord_limits,
			 high_coord_limits);
      ortho_range_search(node->right(), start_dim, low_coord_limits,
			 high_coord_limits);
    }
  }
};

#endif
