/** @file ortho_range_search.h
 *
 *  This file contains an implementation of a tree-based algorithm for
 *  orthogonal range search.
 *
 *  @author Dongryeol Lee (dongryel)
 */
#ifndef ORTHO_RANGE_SEARCH_H
#define ORTHO_RANGE_SEARCH_H

#include "fastlib/fastlib.h"
#include "contrib/dongryel/proximity_project/gen_kdtree.h"
#include "contrib/dongryel/proximity_project/gen_kdtree_hyper.h"
#include "contrib/dongryel/proximity_project/general_type_bounds.h"

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
template<typename T>
class OrthoRangeSearch {

  // This class object cannot be copied!
  FORBID_ACCIDENTAL_COPIES(OrthoRangeSearch);

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

  ////////// User-level Functions //////////

  /** @brief Performs the multiple orthogonal range searches
   *         simultaneously.
   *
   *  @param set_of_low_coord_limits
   *  @param set_of_high_coord_limits
   */
  void Compute(GenMatrix<T> &set_of_low_coord_limits,
	       GenMatrix<T> &set_of_high_coord_limits, 
	       GenMatrix<bool> *candidate_points) {

    // Allocate space for storing candidate points found during
    // search.
    candidate_points->Init(data_.n_cols(), set_of_low_coord_limits.n_cols());
    for(index_t j = 0; j < set_of_low_coord_limits.n_cols(); j++) {
      for(index_t i = 0; i < data_.n_cols(); i++) {
	(*candidate_points).set(i, j, false);
      }
    }

    fx_timer_start(NULL, "tree_range_search");

    // First build a tree out of the set of range windows.
    ArrayList<index_t> old_from_new_windows;
    ArrayList<index_t> new_from_old_windows;

    Tree *tree_of_windows = 
      proximity::MakeGenKdTree<T, Tree, proximity::GenKdTreeMedianSplitter>
      (set_of_low_coord_limits, set_of_high_coord_limits, 2,
       &old_from_new_windows, &new_from_old_windows);

    ortho_range_search(tree_of_windows, set_of_low_coord_limits, 
		       set_of_high_coord_limits, old_from_new_windows,
		       new_from_old_windows,
		       root_, 0, data_.n_rows() - 1, *candidate_points);
    
    fx_timer_stop(NULL, "tree_range_search");

    // Delete the tree of search windows...
    delete tree_of_windows;

    // Reshuffle the search windows.
    GenMatrix<T> tmp_matrix;
    tmp_matrix.Init(set_of_low_coord_limits.n_rows(),
		    set_of_low_coord_limits.n_cols());
    for(index_t i = 0; i < tmp_matrix.n_cols(); i++) {
      GenVector<T> dest;
      GenVector<T> src;
      tmp_matrix.MakeColumnVector(old_from_new_windows[i], &dest);
      set_of_low_coord_limits.MakeColumnVector(i, &src);
      dest.CopyValues(src);
    }
    set_of_low_coord_limits.CopyValues(tmp_matrix);

    for(index_t i = 0; i < tmp_matrix.n_cols(); i++) {
      GenVector<T> dest;
      GenVector<T> src;
      tmp_matrix.MakeColumnVector(old_from_new_windows[i], &dest);
      set_of_high_coord_limits.MakeColumnVector(i, &src);
      dest.CopyValues(src);
    }
    set_of_high_coord_limits.CopyValues(tmp_matrix);
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
    int tree_size = ot::FrozenSize(*root_);
    printf("Tree occupies %d bytes...\n", tree_size);

    fwrite((const void *) &tree_size, sizeof(int), 1, output);
    char *tmp_root = (char *) mem::AllocBytes<Tree>(tree_size);
    ot::Freeze(tmp_root, *root_);
    fwrite((const void *) tmp_root, tree_size, 1, output);
    mem::Free(tmp_root);

    // then serialize the permutation of the points due to tree construction
    // along with its sizes
    int old_from_new_size = ot::FrozenSize(old_from_new_);
    int new_from_old_size = ot::FrozenSize(new_from_old_);
    char *tmp_array = 
      (char *) mem::AllocBytes<ArrayList<index_t> >(old_from_new_size);

    fwrite((const void *) &old_from_new_size, sizeof(int), 1, output);    
    ot::Freeze(tmp_array, old_from_new_);
    fwrite((const void *) tmp_array, old_from_new_size, 1, output);
    
    fwrite((const void*) &new_from_old_size, sizeof(int), 1, output);
    ot::Freeze(tmp_array, new_from_old_);
    fwrite((const void *) tmp_array, new_from_old_size, 1, output);
    
    mem::Free(tmp_array);

    printf("Tree is serialized...\n");
  }

  /** @brief Initialization function - to read the data and to construct tree.
   *
   *  @param dataset The dataset for orthogonal range searching.
   *  @param make_copy Whether to make the copy of the incoming dataset or
   *                   not. If true, a copy is made. Otherwise, the object
   *                   will "steal" the incoming matrix.
   *  @param load_tree_file_name If NULL, the tree is built from scratch.
   *                             If not NULL, the tree is loaded from the
   *                             file whose name is given as the argument.
   */
  void Init(GenMatrix<T> &dataset, bool make_copy, 
	    const char *load_tree_file_name) {

    int leaflen = fx_param_int(NULL, "leaflen", 20);

    // decide whether to make a copy or not.
    if(make_copy) {
      data_.StaticCopy(dataset);
    }
    else {
      data_.StaticOwn(&dataset);
    }

    fx_timer_start(NULL, "tree_d");

    // If the user wants to load the tree from a file,
    if(load_tree_file_name != NULL) {
      LoadTree(load_tree_file_name);
    }

    // Otherwise, construct one from scratch.
    else {
      root_ = proximity::MakeGenKdTree
	<T, Tree, proximity::GenKdTreeMedianSplitter>(data_, leaflen, 
						      &old_from_new_, 
						      &new_from_old_);
    }
    fx_timer_stop(NULL, "tree_d");
  }

 private:

  /** @brief This defines the type of the tree used in this algorithm. */
  typedef GeneralBinarySpaceTree<GenHrectBound<T, 2>, GenMatrix<T> > Tree;

  /** @brief Flag determining a prune */
  enum PruneStatus {SUBSUME, INCONCLUSIVE, EXCLUDE};

  ////////// Private Member Variables //////////

  /** @brief Pointer to the dataset */
  GenMatrix<T> data_;

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
    root_ = ot::SemiThaw<Tree>((char *) tree_buffer_);
    
    // read old_from_new
    fread((void *) &old_from_new_size, sizeof(int), 1, input);
    old_from_new_buffer_ = 
    mem::AllocBytes<ArrayList<index_t> >(old_from_new_size);
    fread((void *) old_from_new_buffer_, old_from_new_size, 1, input);
    old_from_new_.InitCopy(*(ot::SemiThaw<ArrayList<index_t> >
			     ((char *) old_from_new_buffer_)));

    // read new_from_old
    fread((void *) &new_from_old_size, sizeof(int), 1, input);
    new_from_old_buffer_ = 
    mem::AllocBytes<ArrayList<index_t> >(new_from_old_size);
    fread((void *) new_from_old_buffer_, new_from_old_size, 1, input);
    new_from_old_.InitCopy(*(ot::SemiThaw<ArrayList<index_t> >
			     ((char *) new_from_old_buffer_)));

    printf("Tree has been loaded...\n");

    // apply permutation to the dataset
    GenMatrix<T> tmp_data;
    tmp_data.Init(data_.n_rows(), data_.n_cols());

    for(index_t i = 0; i < data_.n_cols(); i++) {
      GenVector<T> source, dest;
      data_.MakeColumnVector(i, &source);
      tmp_data.MakeColumnVector(new_from_old_[i], &dest);
      dest.CopyValues(source);
    }
    data_.CopyValues(tmp_data);
  }
  
  /** @brief The base case.
   *
   *  @param search_window_node The node containing the search windows.
   *  @param low_coord_limits The lower coordinate limits of the search for
   *                          the current search window node.
   *  @param high_coord_limits The upper coordinate limits of the search for
   *                           the current search window node.
   *  @param reference_node The reference tree node currently under 
   *                        consideration.
   *  @param start_dim The starting dimension currently under consideration.
   *  @param end_dim The ending index of the dimension currently under 
   *                 consideration.
   *  @param candiate_points Records the membership of each point for each
   *                         search window.
   */
  void ortho_slow_range_search(Tree *search_window_node,
			       GenMatrix<T> &low_coord_limits,
			       GenMatrix<T> &high_coord_limits,
			       const ArrayList<index_t> &old_from_new_windows,
			       const ArrayList<index_t> &new_from_old_windows,
			       Tree *reference_node, 
			       index_t start_dim, index_t end_dim,
			       GenMatrix<bool> &candidate_points) {
    PruneStatus prune_flag;

    // Loop over each search window...
    for(index_t window = search_window_node->begin();
	window < search_window_node->end(); window++) {

      // Loop over each reference point...
      for(index_t row = reference_node->begin(); row < reference_node->end(); 
	  row++) {
	prune_flag = SUBSUME;
	
	// loop over each dimension...
	for(index_t d = start_dim; d <= end_dim; d++) {
	  // determine which one of the two cases we have: EXCLUDE,
	  // SUBSUME.
	  
	  // first the EXCLUDE case: when dist is above the upper
	  // bound distance of this dimension, or dist is below the
	  // lower bound distance of this dimension
	  if(data_.get(d, row) > high_coord_limits.get(d, window) ||
	     data_.get(d, row) < low_coord_limits.get(d, window)) {
	    prune_flag = EXCLUDE;
	    break;
	  }
	} // end of looping over dimensions...

	// Set each point result depending on the flag...
	candidate_points.set(old_from_new_[row], old_from_new_windows[window],
			     (prune_flag == SUBSUME));

      } // end of iterating over reference points...
    } // end of iterating over search window...
  }

  /** @brief The workhorse algorithm for fast orthgonal range
   *         search.
   *
   *  @param search_window_node The node containing the search windows.
   *  @param low_coord_limits The lower coordinate limits of the search for
   *                          the current search window node.
   *  @param high_coord_limits The upper coordinate limits of the search for
   *                           the current search window node.
   *  @param reference_node The reference tree node currently under
   *                        consideration.
   *  @param start_dim The starting dimension currently under consideration.
   *  @param end_dim The ending index of the dimension currently under
   *                 consideration.
   *  @param candiate_points Records the membership of each point for each
   *                         search window.
   */
  void ortho_range_search(Tree *search_window_node, 
			  GenMatrix<T> &low_coord_limits, 
			  GenMatrix<T> &high_coord_limits,
			  const ArrayList<index_t> &old_from_new_windows,
			  const ArrayList<index_t> &new_from_old_windows,
			  Tree *reference_node, index_t start_dim,
			  index_t end_dim, GenMatrix<bool> &candidate_points) {

    PruneStatus prune_flag = SUBSUME;
    
    // loop over each dimension to determine inclusion/exclusion by
    // determining the lower and the upper bound distance per each
    // dimension for the given reference node, kn
    for(index_t d = start_dim; d <= end_dim; d++) {

      const GenRange<T> &reference_node_dir_range = 
	reference_node->bound().get(d);
      const GenRange<T> &search_window_node_dir_range =
	search_window_node->bound().get(d);
      
      // determine which one of the three cases we have: EXCLUDE,
      // SUBSUME, or INCONCLUSIVE.
      
      // First the EXCLUDE case: when mindist is above the upper bound
      // distance of this dimension, or maxdist is below the lower
      // bound distance of this dimension
      if(reference_node_dir_range.lo > search_window_node_dir_range.hi ||
	 reference_node_dir_range.hi < search_window_node_dir_range.lo) {
	return;
      }
      // otherwise, check for SUBSUME case
      else if(search_window_node_dir_range.lo <= reference_node_dir_range.lo &&
	      reference_node_dir_range.hi <= search_window_node_dir_range.hi) {
      }
      // if any dimension turns out to be inconclusive, then break.
      else {
	if(search_window_node->count() == 1) {
	  start_dim = d;
	}
	prune_flag = INCONCLUSIVE;
	break;
      }
    } // end of iterating over each dimension.

    // In case of subsume, then add all points owned by this node to
    // candidates - note that subsume prunes cannot be performed
    // always in batch query.
    if(search_window_node->count() == 1 && prune_flag == SUBSUME) {
      for(index_t j = search_window_node->begin(); 
	  j < search_window_node->end(); j++) {
	for(index_t i = reference_node->begin(); 
	    i < reference_node->end(); i++) {
	  candidate_points.set(old_from_new_[i], old_from_new_windows[j],
			       true);
	}
      }
      return;
    }
    else {
      if(search_window_node->is_leaf()) {

	// If both the search window and the reference nodes are
	// leaves, then compute exhaustively.
	if(reference_node->is_leaf()) {
	  ortho_slow_range_search(search_window_node, low_coord_limits,
				  high_coord_limits, old_from_new_windows,
				  new_from_old_windows, reference_node,
				  start_dim, end_dim, candidate_points);
	}
	// If the reference node can be expanded, then do so.
	else {
	  ortho_range_search(search_window_node, low_coord_limits,
			     high_coord_limits, old_from_new_windows,
			     new_from_old_windows, reference_node->left(),
			     start_dim, end_dim, candidate_points);
	  ortho_range_search(search_window_node, low_coord_limits,
			     high_coord_limits, old_from_new_windows,
			     new_from_old_windows, reference_node->right(),
			     start_dim, end_dim, candidate_points);
	}
      }
      else {

	// In this case, expand the query side.
	if(reference_node->is_leaf()) {
	  ortho_range_search(search_window_node->left(), low_coord_limits,
                             high_coord_limits, old_from_new_windows,
			     new_from_old_windows, reference_node,
                             start_dim, end_dim, candidate_points);
          ortho_range_search(search_window_node->right(), low_coord_limits,
                             high_coord_limits, old_from_new_windows,
			     new_from_old_windows, reference_node,
                             start_dim, end_dim, candidate_points);
	}
	// Otherwise, expand both query and the reference sides.
	else {
          ortho_range_search(search_window_node->left(), low_coord_limits,
                             high_coord_limits, old_from_new_windows,
			     new_from_old_windows, reference_node->left(),
                             start_dim, end_dim, candidate_points);
          ortho_range_search(search_window_node->left(), low_coord_limits,
                             high_coord_limits, old_from_new_windows,
			     new_from_old_windows, reference_node->right(),
                             start_dim, end_dim, candidate_points);
          ortho_range_search(search_window_node->right(), low_coord_limits,
                             high_coord_limits, old_from_new_windows,
			     new_from_old_windows, reference_node->left(),
                             start_dim, end_dim, candidate_points);
          ortho_range_search(search_window_node->right(), low_coord_limits,
                             high_coord_limits, old_from_new_windows,
			     new_from_old_windows, reference_node->right(),
                             start_dim, end_dim, candidate_points);
	}
      }
    }
  }
};

#endif
