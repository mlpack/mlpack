/**
 * @author Hua Ouyang
 *
 * @file opt_hcy.h
 *
 * This head file contains functions for performing Hierarchical Propergative Optimization for SVM
 *
 * The algorithms in the following papers are implemented:
 *
 *
 * @see svm.h
 */

#ifndef U_SVM_OPT_HCY_H
#define U_SVM_OPT_HCY_H

#include "fastlib/fastlib.h"
#include "fastlib/base/test.h"
#include "general_spacetree.h"
#include "gen_kdtree.h"

// max imum # of iterations for HCY training
const index_t MAX_NUM_ITER_HCY = 1000000;
// after # of iterations to do shrinking
const index_t HCY_NUM_FOR_SHRINKING = 1000;
// threshold that determines whether need to do unshrinking
const double HCY_UNSHRINKING_FACTOR = 10;
// threshold that determines whether an alpha is a SV or not
const double HCY_ALPHA_ZERO = 1.0e-4;
// inital alpha for the root of the positive tree
const double INIT_ROOT_ALPHA_POS = 1.0;
// inital alpha for the root of the negative tree
const double INIT_ROOT_ALPHA_NEG = 1.0;

const double HCY_ZERO = 1.0e-3;

const double HCY_ID_LOWER_BOUNDED = -1;
const double HCY_ID_UPPER_BOUNDED = 1;
const double HCY_ID_FREE = 0;


template<typename TKernel>
class HCY {
  FORBID_ACCIDENTAL_COPIES(HCY);


  //////////////////////////// Nested Classes ///////////////////////////////////////////////
  /**
  * Extra data for each node in the tree.
  */
  class StatkdTree {
    // Defines many useful things for a class, including a pretty printer and copy constructor
    OT_DEF_BASIC(StatkdTree) {
      // Include this line for all non-pointer members
      // There are other versions for arrays and pointers, see base/otrav.h
      OT_MY_OBJECT(max_distance_so_far_); 
    }
    
   private:
    // upper bound on the node's nearest neighbor distances.
    double max_distance_so_far_;
    
   public:
    double max_distance_so_far() {
      return max_distance_so_far_; 
    } 
    
    void set_max_distance_so_far(double new_dist) {
      max_distance_so_far_ = new_dist; 
    }
    
    // In addition to any member variables for the statistic, all stat 
    // classes need two Init 
    // functions, one for leaves and one for non-leaves. 
    
    /**
     * Initialization function used in tree-building when initializing 
     * a leaf node.  For allnn, needs no additional information 
     * at the time of tree building.  
     */
    void Init(const Matrix& matrix, index_t start, index_t count) {
      // The bound starts at infinity
      max_distance_so_far_ = DBL_MAX;
    } 
    
    /**
     * Initialization function used in tree-building when initializing a non-leaf node.  For other algorithms,
     * node statistics can be built using information from the children.  
     */
    void Init(const Matrix& matrix, index_t start, index_t count, 
        const StatkdTree& left, const StatkdTree& right) {
      // For allnn, non-leaves can be initialized in the same way as leaves
      Init(matrix, start, count);
    } 
  };

 public:
  typedef TKernel Kernel;
  // Euclidean distance based kd tree
  typedef GeneralBinarySpaceTree<DHrectBound<2>, Matrix, StatkdTree> TreeType;

 private:
  int learner_typeid_;
  index_t ct_iter_; /* counter for the number of iterations */
  index_t ct_shrinking_; /* counter for doing shrinking  */

  Kernel kernel_;
  index_t n_data_; /* number of data samples */
  index_t n_features_; /* # of features == # of row - 1, exclude the last row (for labels) */
  Matrix datamatrix_; /* alias for the matrix of all data, including last label row */
  
  index_t n_data_pos_; /* number of samples with label 1 */
  index_t n_data_neg_; /* number of samples with label -1 */

  Vector alpha_; /* the alphas, to be optimized */
  Vector alpha_status_; /*  ID_LOWER_BOUND (-1), ID_UPPER_BOUND (1), ID_FREE (0) */
  index_t n_sv_; /* number of support vectors */

  //Vector alpha_pos_; /* the alphas for class 1, to be optimized; same order as reordered positive dataset*/
  //Vector alpha_status_pos_; /*  ID_LOWER_BOUND (-1), ID_UPPER_BOUND (1), ID_FREE (0) */
  //Vector alpha_neg_; /* the alphas for class -1, to be optimized; same order as reordered negative dataset*/
  //Vector alpha_status_neg_; /*  ID_LOWER_BOUND (-1), ID_UPPER_BOUND (1), ID_FREE (0) */

  index_t max_n_alpha_; /* max number of alphas == n_data(SVC) or 2*n_data(SVR) */
  index_t n_used_alpha_; /* number of samples used in this level == number of variables to optimize in this level*/
  index_t n_active_; /* number of samples in the active set (not been shrinked) of this level */
  // n_active + n_inactive == n_used_alpha;

  ArrayList<index_t> active_set_; /* list that stores the indices of active alphas. == old_from_new*/
  ArrayList<index_t> new_from_old_;

  ArrayList< GenVector<index_t> > kdtree_o_f_n_maps_pos_;
  ArrayList< GenVector<index_t> > kdtree_o_f_n_maps_neg_;

  bool unshrinked_; /* indicator: where unshrinking has be carried out  */
  index_t i_cache_, j_cache_; /* indices for the most recently cached kernel value */
  double cached_kernel_value_; /* cache */

  ArrayList<int> y_; /* list that stores "labels" */

  double bias_;

  Vector grad_; /* gradient value */
  Vector grad_bar_; /* gradient value when treat free variables as 0 */

  index_t leaf_size_;
  TreeType* tree_pos_; /* tree for the positive class 1 */
  TreeType* tree_neg_; /* tree for the negative class -1 */

  bool b_end_of_recursion_; /* sign of reaching the end for recursion (the last level) */

  // parameters
  int budget_;
  double Cp_; // C_+, for SVM_C, y==1
  double Cn_; // C_-, for SVM_C, y==-1
  double epsilon_; // for SVM_R
  int wss_; // working set selection scheme, 1 for 1st order expansion; 2 for 2nd order expansion
  index_t n_iter_; // number of iterations
  double accuracy_; // accuracy for stopping creterion

 public:
  HCY() {}
  ~HCY() {}

  /**
   * Initialization for parameters
   */
  void InitPara(int learner_typeid, ArrayList<double> &param_) {
    // init parameters
    budget_ = (int)param_[0];
    wss_ = (int) param_[3];
    n_iter_ = (index_t) param_[4];
    n_iter_ = n_iter_ < MAX_NUM_ITER_HCY ? n_iter_: MAX_NUM_ITER_HCY;
    accuracy_ = param_[5];
    n_data_pos_ = (index_t)param_[6];
    if (learner_typeid == 0) { // SVM_C
      Cp_ = param_[1];
      Cn_ = param_[2];
    }
    else if (learner_typeid == 1) { // SVM_R
      // TODO
    }
  }

  void Train(int learner_typeid, const Dataset* dataset_in);

  Kernel& kernel() {
    return kernel_;
  }

  double Bias() const {
    return bias_;
  }

  void GetSV(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator);

 private:

  void SwapValues(index_t idx_1, index_t idx_2);

  void LearnersInit_(int learner_typeid);

  void TreeDescentRecursion_(ArrayList<TreeType*> &node_pool_pos, ArrayList<TreeType*> &node_pool_neg,
			     index_t n_samples_for_opt,
			     index_t n_splitted_node_pos, index_t n_splitted_node_neg,
			     index_t n_leaf_node_pos, index_t n_leaf_node_neg,
			     index_t n_not_splitted_node_pos, index_t n_not_splitted_node_neg,
			     ArrayList<index_t> &idx_not_splitted_node_pos, ArrayList<index_t> &new_idx_not_splitted_node_pos,
			     ArrayList<index_t> &idx_not_splitted_node_neg, ArrayList<index_t> &new_idx_not_splitted_node_neg,
			     ArrayList<index_t> &idx_leaf_node_pos, ArrayList<index_t> &idx_leaf_node_neg);

  void SplitNodePropagatePos_(ArrayList<TreeType*> &node_pool,
			      index_t &n_samples_for_opt, index_t &new_n_samples_for_opt,
			      index_t &n_splitted_node, index_t &new_n_splitted_node,
			      index_t &n_leaf_node, index_t &new_n_leaf_node,
			      index_t &n_not_splitted_node, index_t &new_n_not_splitted_node,
			      ArrayList<double> &changed_values, ArrayList<index_t> &changed_idx_old,
			      ArrayList<index_t> &idx_not_splitted_node, ArrayList<index_t> &new_idx_not_splitted_node);
  
  void SplitNodePropagateNeg_(ArrayList<TreeType*> &node_pool,
			   index_t &n_samples_for_opt, index_t &new_n_samples_for_opt,
			   index_t &n_splitted_node, index_t &new_n_splitted_node,
			   index_t &n_leaf_node, index_t &new_n_leaf_node,
			   index_t &n_not_splitted_node, index_t &new_n_not_splitted_node,
			   ArrayList<double> &changed_values, ArrayList<index_t> &changed_idx_old,
			   ArrayList<index_t> &idx_not_splitted_node, ArrayList<index_t> &new_idx_not_splitted_node);
  
  double DirectPropagatePos_(ArrayList<TreeType*> &node_pool,
			  index_t &n_samples_for_opt, index_t &new_n_samples_for_opt,
			  index_t &n_splitted_node, index_t &new_n_splitted_node,
			  index_t &n_leaf_node, index_t &new_n_leaf_node,
			  index_t &n_not_splitted_node, index_t &new_n_not_splitted_node,
			  ArrayList<double> &changed_values, ArrayList<index_t> &changed_idx_old,
			  ArrayList<index_t> &idx_not_splitted_node, ArrayList<index_t> &new_idx_not_splitted_node,
			  ArrayList<index_t> &idx_leaf_node);

  double DirectPropagateNeg_(ArrayList<TreeType*> &node_pool,
			  index_t &n_samples_for_opt, index_t &new_n_samples_for_opt,
			  index_t &n_splitted_node, index_t &new_n_splitted_node,
			  index_t &n_leaf_node, index_t &new_n_leaf_node,
			  index_t &n_not_splitted_node, index_t &new_n_not_splitted_node,
			  ArrayList<double> &changed_values, ArrayList<index_t> &changed_idx_old,
			  ArrayList<index_t> &idx_not_splitted_node, ArrayList<index_t> &new_idx_not_splitted_node,
			  ArrayList<index_t> &idx_leaf_node);

  int TrainIteration_();

  void ReconstructGradient_(int learner_typeid);
  
  bool TestShrink_(index_t i, double y_grad_max, double y_grad_min);

  void Shrinking_();

  bool WorkingSetSelection_(index_t &i, index_t &j);

  void UpdateGradientAlpha_(index_t i, index_t j);

  void CalcBias_();

  /**
   * Instead of C, we use C_+ and C_- to handle unbalanced data
   */
  double GetC_(index_t i) {
    return (y_[i] > 0 ? Cp_ : Cn_);
  }

  void UpdateAlphaStatus_(index_t i) {
    if (alpha_[i] >= GetC_(i)) {
      alpha_status_[i] = HCY_ID_UPPER_BOUNDED;
    }
    else if (alpha_[i] <= 0) {
      alpha_status_[i] = HCY_ID_LOWER_BOUNDED;
    }
    else { // 0 < alpha_[i] < C
      alpha_status_[i] = HCY_ID_FREE;
    }
  }

  bool IsUpperBounded(index_t i) {
    return alpha_status_[i] == HCY_ID_UPPER_BOUNDED;
  }
  bool IsLowerBounded(index_t i) {
    return alpha_status_[i] == HCY_ID_LOWER_BOUNDED;
  }

  /**
   * Calculate kernel values
   */
  double CalcKernelValue_(index_t ii, index_t jj) {
    index_t i = active_set_[ii]; // ii/jj: index in the new permuted set
    index_t j = active_set_[jj]; // i/j: index in the old set

    // for SVM_R where max_n_alpha_==2*n_data_
    /*
    if (learner_typeid_ == 1) {
      i = i >= n_data_ ? (i-n_data_) : i;
      j = j >= n_data_ ? (j-n_data_) : j;
      }*/

    // Check cache
    //if (i == i_cache_ && j == j_cache_) {
    //  return cached_kernel_value_;
    //}

    double *v_i, *v_j;
    v_i = datamatrix_.GetColumnPtr(i);
    v_j = datamatrix_.GetColumnPtr(j);

    // Do Caching. Store the recently caculated kernel values.
    //i_cache_ = i;
    //j_cache_ = j;
    cached_kernel_value_ = kernel_.Eval(v_i, v_j, n_features_);
    return cached_kernel_value_;
  }
};


template<typename TKernel>
void HCY<TKernel>::SwapValues(index_t idx_1, index_t idx_2) {
  if (idx_1 != idx_2) {
    swap(new_from_old_[active_set_[idx_1]], new_from_old_[active_set_[idx_2]]);
    swap(active_set_[idx_1], active_set_[idx_2]);
    swap(alpha_[idx_1], alpha_[idx_2]);
    swap(alpha_status_[idx_1], alpha_status_[idx_2]);
    swap(y_[idx_1], y_[idx_2]);
    swap(grad_[idx_1], grad_[idx_2]);
    swap(grad_bar_[idx_1], grad_bar_[idx_2]);
  }
}



/**
* Reconstruct inactive elements of G from G_bar and free variables 
*
* @param: learner type id
*/
template<typename TKernel>
void HCY<TKernel>::ReconstructGradient_(int learner_typeid) {
  index_t i, j;
  if (n_active_ == n_used_alpha_)
    return;
  if (learner_typeid == 0) { // SVM_C
    for (i=n_active_; i<n_used_alpha_; i++) {
      grad_[i] = 1 - grad_bar_[i];
    }
  }
  else if (learner_typeid == 1) { // SVM_R
    for (i=n_active_; i<n_used_alpha_; i++) {
      j = i >= n_data_ ? (i-n_data_) : i;
      grad_[j] = grad_bar_[j] + datamatrix_.get(datamatrix_.n_rows()-1, active_set_[j]) - epsilon_; // TODO
    }
  }

  for (i=0; i<n_active_; i++) {
    if (alpha_status_[i] == HCY_ID_FREE) {
      for (j=n_active_; j<n_used_alpha_; j++) {
	grad_[j] = grad_[j] - y_[j] * alpha_[i] * y_[i] * CalcKernelValue_(i,j);
      }
    }
  }
}

/**
 * Test whether need to do shrinking for provided index and y_grad_max, y_grad_min
 * 
 */
template<typename TKernel>
bool HCY<TKernel>::TestShrink_(index_t i, double y_grad_max, double y_grad_min) {
  if (IsUpperBounded(i)) { // alpha_[i] = C
    if (y_[i] == 1) {
      return (grad_[i] > y_grad_max);
    }
    else { // y_[i] == -1
      return (grad_[i] + y_grad_min > 0); // -grad_[i]<y_grad_min
    }
  }
  else if (IsLowerBounded(i)) {
    if (y_[i] == 1) {
      return (grad_[i] < y_grad_min);
    }
    else { // y_[i] == -1
      return (grad_[i] + y_grad_max < 0); // -grad_[i]>y_grad_max
    }
  }
  else
    return false;
}

/**
 * Do Shrinking. Temporarily remove alphas (from the active set) that are 
 * unlikely to be selected in the working set, since they have reached their 
 * lower/upper bound.
 * 
 */
template<typename TKernel>
void HCY<TKernel>::Shrinking_() {
  index_t t;

  // Find m(a) == y_grad_max(i\in I_up) and M(a) == y_grad_min(j\in I_down)
  double y_grad_max = -INFINITY;
  double y_grad_min =  INFINITY;
  for (t=0; t<n_active_; t++) { // find argmax(y*grad), t\in I_up
    if (y_[t] == 1) {
      if (!IsUpperBounded(t)) // t\in I_up, y==1: y[t]alpha[t] < C
	if (grad_[t] > y_grad_max) { // y==1
	  y_grad_max = grad_[t];
	}
    }
    else { // y[t] == -1
      if (!IsLowerBounded(t)) // t\in I_up, y==-1: y[t]alpha[t] < 0
	if (grad_[t] + y_grad_max < 0) { // y==-1... <=> -grad_[t] > y_grad_max
	  y_grad_max = -grad_[t];
	}
    }
  }
  for (t=0; t<n_active_; t++) { // find argmin(y*grad), t\in I_down
    if (y_[t] == 1) {
      if (!IsLowerBounded(t)) // t\in I_down, y==1: y[t]alpha[t] > 0
	if (grad_[t] < y_grad_min) { // y==1
	  y_grad_min = grad_[t];
	}
    }
    else { // y[t] == -1
      if (!IsUpperBounded(t)) // t\in I_down, y==-1: y[t]alpha[t] > -C
	if (grad_[t] + y_grad_min > 0) { // y==-1...<=>  -grad_[t] < y_grad_min
	  y_grad_min = -grad_[t];
	}
    }
  }

  // Find the alpha to be shrunk
  printf("Shrinking...\n");
  for (t=0; t<n_active_; t++) {
    // Shrinking: put inactive alphas behind the active set
    if (TestShrink_(t, y_grad_max, y_grad_min)) {
      n_active_ --;
      while (n_active_ > t) {
	if (!TestShrink_(n_active_, y_grad_max, y_grad_min)) {
	  SwapValues(t, n_active_);
	  break;
	}
	n_active_ --;
      }
    }
  }

  // Determine whether need to do Unshrinking
  if ( unshrinked_==false && y_grad_max - y_grad_min <= SMO_UNSHRINKING_FACTOR * accuracy_ ) {
    printf("Unshrinking...\n");
    // Unshrinking: put shrinked alphas back to active set
    // 1.recover gradient
    ReconstructGradient_(learner_typeid_);
    // 2.recover active status
    for (t=n_used_alpha_-1; t>n_active_; t--) {
      if (!TestShrink_(t, y_grad_max, y_grad_min)) {
	while (n_active_ < t) {
	  if (TestShrink_(n_active_, y_grad_max, y_grad_min)) {
	    SwapValues(t, n_active_);
	    break;
	  }
	  n_active_ ++;
	}
	n_active_ ++;
      }
    }
    
    unshrinked_ = true; // indicator: unshrinking has been carried out in this round
  }
}

/**
 * Initialization according to different SVM learner types
 *
 * @param: learner type id 
 */
template<typename TKernel>
void HCY<TKernel>::LearnersInit_(int learner_typeid) {
  index_t i;
  learner_typeid_ = learner_typeid;
  
  if (learner_typeid_ == 0) { // SVM_C
    max_n_alpha_ = n_data_;

    alpha_.Init(max_n_alpha_);
    alpha_.SetZero();

    // initialize gradient
    grad_.Init(max_n_alpha_);
    grad_.SetAll(1.0);

    y_.Init(n_data_);
    for (i = 0; i < n_data_; i++) {
      y_[i] = datamatrix_.get(datamatrix_.n_rows()-1, i) > 0 ? 1 : -1;
    }
  }
  else if (learner_typeid_ == 1) { // SVM_R
    // TODO
  }
  else if (learner_typeid_ == 2) { // SVM_DE
    // TODO
  }
  
}


/**
* Hierarchical SVM training for 2-classes
*
* @param: input 2-classes data matrix with labels {1,-1} in the last row
*/
template<typename TKernel>
void HCY<TKernel>::Train(int learner_typeid, const Dataset* dataset_in) {
  index_t i, j;
  // Load data
  datamatrix_.Alias(dataset_in->matrix());
  n_data_ = datamatrix_.n_cols();
  n_data_neg_ = n_data_ - n_data_pos_;
  n_features_ = datamatrix_.n_rows() - 1;

  // Learners initialization
  LearnersInit_(learner_typeid);

  // General learner-independent initializations
  bias_ = 0.0;
  // minimum size of the leaf node, if smaller than this, do not split the node
  leaf_size_= fx_param_int(NULL,"leaf_size", min(10, index_t(n_data_/2)-1)); 

  active_set_.Init(max_n_alpha_); // it is actually old_from_new
  new_from_old_.Init(max_n_alpha_);
  for (i=0; i<max_n_alpha_; i++) {
    active_set_[i] = i;
    new_from_old_[i] = i;
  }
  alpha_status_.Init(max_n_alpha_);
  for (i=0; i<max_n_alpha_; i++)
    UpdateAlphaStatus_(i);

  grad_bar_.Init(max_n_alpha_);
  grad_bar_.SetZero();

  /* Building kd trees. Here we choose balanced median-split kd-trees */

  // Only copy data, but not labels
  // CAUTION: THESE 2 DATA MATRICES of the bi-classes dataset_in WILL BE REARRANGED AFTER BUILDING TREES
  Matrix datamatrix_pos; // alias for the data matrix of the positive class 1, excluding last label row
  Matrix datamatrix_neg; // alias for the data matrix of the negative class -1, excluding last label row
  ArrayList<index_t> kdtree_old_from_new_pos; // not used
  ArrayList<index_t> kdtree_new_from_old_pos;
  ArrayList<index_t> kdtree_old_from_new_neg;
  ArrayList<index_t> kdtree_new_from_old_neg;

  // copy data. TODO: avoid these memory allocations
  datamatrix_pos.Copy(dataset_in->matrix().ptr(), n_features_, n_data_pos_);
  datamatrix_neg.Copy(dataset_in->matrix().GetColumnPtr(n_data_pos_), n_features_, n_data_neg_);
  // build trees
  fx_timer_start(NULL, "tree_build");
  printf("Building kd tree for class 1...\n");
  tree_pos_ = proximity::MakeGenKdTree<double, TreeType, proximity::GenKdTreeMedianSplitter>(datamatrix_pos, 
                         leaf_size_, kdtree_old_from_new_pos, kdtree_new_from_old_pos, kdtree_o_f_n_maps_pos_);
  printf("Building kd tree for class -1...\n");
  tree_neg_ = proximity::MakeGenKdTree<double, TreeType, proximity::GenKdTreeMedianSplitter>(datamatrix_neg, 
			 leaf_size_, kdtree_old_from_new_neg, kdtree_new_from_old_neg, kdtree_o_f_n_maps_neg_);
  fx_timer_stop(NULL, "tree_build");

  /* Hierarchical optimization with tree descent */
  ArrayList<TreeType*> node_pool_pos; // node pool that stores the nodes of the positive tree
  ArrayList<TreeType*> node_pool_neg; // node pool that stores the nodes of the negative tree
  node_pool_pos.Init();
  node_pool_neg.Init();
  node_pool_pos.PushBack() = tree_pos_; // the top level only contains one node (sample)
  node_pool_neg.PushBack() = tree_neg_;
  
  index_t idx_tmp;
  idx_tmp = tree_pos_->get_split_point_idx_old(); // index in the original dataset
  idx_tmp = new_from_old_[idx_tmp];
  SwapValues(0, idx_tmp);

  idx_tmp = tree_neg_->get_split_point_idx_old() + n_data_pos_; // index in the original dataset
  idx_tmp = new_from_old_[idx_tmp];
  SwapValues(1, idx_tmp);

  // Assign initial alphas to the two data samples in the top level
  alpha_[0] = INIT_ROOT_ALPHA_POS;
  alpha_[1] = INIT_ROOT_ALPHA_NEG;

  n_used_alpha_ = 2;
  n_active_ = 2;

  ArrayList<index_t> idx_not_splitted_node_pos; // stores the indices in the node_pool_pos of the non_splitted node
  ArrayList<index_t> new_idx_not_splitted_node_pos;
  ArrayList<index_t> idx_not_splitted_node_neg; // stores the indices in the node_pool_neg of the non_splitted node
  ArrayList<index_t> new_idx_not_splitted_node_neg;
  ArrayList<index_t> idx_leaf_node_pos;
  ArrayList<index_t> idx_leaf_node_neg;
  idx_not_splitted_node_pos.Init(1);
  idx_not_splitted_node_pos[0] = 0;
  new_idx_not_splitted_node_pos.Init();
  idx_not_splitted_node_neg.Init(1);
  idx_not_splitted_node_neg[0] = 0;
  new_idx_not_splitted_node_neg.Init();
  idx_leaf_node_pos.Init();
  idx_leaf_node_neg.Init();

  // Initialize gradient (already set to init values 1)
  for (i=0; i<n_used_alpha_; i++) {
    if(!IsLowerBounded(i)) { // alpha_i > 0
      for(j=0; j<n_used_alpha_; j++)
	grad_[i] = grad_[i] - y_[i] * y_[j] * alpha_[j] * CalcKernelValue_(i,j);
    }
  }
  /*
  // Initialize gradient_bar
  for (i=0; i<max_n_alpha_; i++) {
    for(j=0; j<max_n_alpha_; j++) {
      if(IsUpperBounded(j)) // alpha_j >= C
	grad_bar_[i] = grad_bar_[i] + GetC_(j) * y_[j] * CalcKernelValue_(i,j);
    }
    grad_bar_[i] = y_[i] * grad_bar_[i];
  }
  */

  // Begin recursive hierarchical optimization
  b_end_of_recursion_ = false;
  TreeDescentRecursion_(node_pool_pos, node_pool_neg, 2, 0, 0, 0, 0, 1, 1, 
			idx_not_splitted_node_pos, new_idx_not_splitted_node_pos,
			idx_not_splitted_node_neg, new_idx_not_splitted_node_neg,
			idx_leaf_node_pos, idx_leaf_node_neg);

  // Calculate the bias term
  CalcBias_();

}


/* Generate pool of children for the next level. Propagate alphas and gradients.
   * - For already splitted nodes: directly propagate alphas,
   *     i.e. those alphas remain unchanged.
   * - For leaf nodes: directly propagate alphas,
   *     i.e. those alphas remain unchanged.
   * - For not-splitted nodes, divide alphas properly according how many samples they have in their children,
   *     and propagate alphas.
   */
template<typename TKernel>
void HCY<TKernel>::SplitNodePropagatePos_(ArrayList<TreeType*> &node_pool,
				       index_t &n_samples_for_opt, index_t &new_n_samples_for_opt,
				       index_t &n_splitted_node, index_t &new_n_splitted_node,
				       index_t &n_leaf_node, index_t &new_n_leaf_node,
				       index_t &n_not_splitted_node, index_t &new_n_not_splitted_node,
				       ArrayList<double> &changed_values,  ArrayList<index_t> &changed_idx_old,
				       ArrayList<index_t> &idx_not_splitted_node, ArrayList<index_t> &new_idx_not_splitted_node) {
  
  double alpha_tmp, two_alpha_tmp;
  index_t old_idx_tmp, new_idx_tmp;
  index_t leaf_node_id;

  /**** Handle not-splitted nodes of the new level ****/
  //for (index_t k=n_splitted_node; k<n_splitted_node+n_not_splitted_node; k++) {
  for (index_t i=0; i<n_not_splitted_node; i++) {
    index_t k = idx_not_splitted_node[i];
    new_n_splitted_node ++;
    TreeType *left_node, *right_node;
    left_node = node_pool[k]->left();
    right_node = node_pool[k]->right();

    if (!node_pool[k]->is_leaf()) {
      //printf("l:%d\n", left_node->get_split_point_idx_old());
      //printf("r:%d\n", right_node->get_split_point_idx_old());
      // left child has a splitting sample
      if(left_node->get_split_point_idx_old() != -1) {
	b_end_of_recursion_ = false;
	new_idx_not_splitted_node.PushBack() = node_pool.size();
	node_pool.PushBack() = node_pool[k]->left();
	new_n_not_splitted_node ++;
	new_n_samples_for_opt ++;
	// right child also has a splitting sample
	if(right_node->get_split_point_idx_old() != -1) {
	  new_idx_not_splitted_node.PushBack() = node_pool.size();
	  node_pool.PushBack() = node_pool[k]->right();
	  new_n_not_splitted_node ++;
	  new_n_samples_for_opt ++;
	  
	  // divide alpha into 3 and propagate
	  old_idx_tmp = node_pool[k]->get_split_point_idx_old();
	  // printf("s_old_idx_tmp=%d\n", old_idx_tmp);
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_tmp = alpha_[new_idx_tmp];
	  alpha_tmp = alpha_tmp / 3.0;
	  two_alpha_tmp = 2.0 * alpha_tmp;
	  
	  /* Current node itselft */
	  // update alpha
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = y_[new_idx_tmp] * two_alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  
	  /* Left child */
	  // update alpha
	  old_idx_tmp = node_pool[k]->left()->get_split_point_idx_old();
	  //printf("l_old_idx_tmp=%d\n", old_idx_tmp);
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  // update active_set etc.
	  //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	  SwapValues(n_active_, new_idx_tmp);
	  n_used_alpha_ ++;
	  n_active_ ++;
	  
	  /* Right child */
	  // update alpha
	  old_idx_tmp = node_pool[k]->right()->get_split_point_idx_old();
	  //printf("r_old_idx_tmp=%d\n", old_idx_tmp);
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  // update active_set etc.
	  //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	  SwapValues(n_active_, new_idx_tmp);
	  n_used_alpha_ ++;
	  n_active_ ++;

	}
	// only left child has a splitting sample; right child is a leaf node
	else {
	  // Number of samples in the right child
	  index_t n_samples_leaf = node_pool[k]->right()->count();
	  //printf("right_n_sample_leaf=%d\n", n_samples_leaf);
	  // divide alpha into (2+#leaf) and propagate
	  old_idx_tmp = node_pool[k]->get_split_point_idx_old();
	  //printf("s_old_idx_tmp=%d\n", old_idx_tmp);
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_tmp = alpha_[new_idx_tmp];
	  alpha_tmp = alpha_tmp / (2.0 + n_samples_leaf);
	  
	  /* current node itselft */
	  // update alpha
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = y_[new_idx_tmp] * (1.0 + n_samples_leaf) * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  
	  /* left child */
	  // update alpha
	  old_idx_tmp = node_pool[k]->left()->get_split_point_idx_old();
	  //printf("l_old_idx_tmp=%d\n", old_idx_tmp);
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  // update active_set etc.
	  //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	  SwapValues(n_active_, new_idx_tmp);
	  n_used_alpha_ ++;
	  n_active_ ++;
	  
	  /* right child (a leaf node) */
	  new_n_leaf_node ++;
	  new_n_samples_for_opt += n_samples_leaf;
	  node_pool.PushBack() = node_pool[k]->right();
	  leaf_node_id = node_pool[k]->right()->node_id();
	  for (index_t t=0; t<n_samples_leaf; t++) {
	    // update alpha
	    old_idx_tmp = (kdtree_o_f_n_maps_pos_[leaf_node_id])[node_pool[k]->right()->begin() + t];
	    //printf("r_leaf_old_idx_tmp=%d\n", old_idx_tmp);
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    //printf("new_idx_tmp=%d\n", new_idx_tmp);
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	  }
	}

	/*printf("intrim\n");
      for (index_t i=0; i<n_used_alpha_; i++)
	printf("%f.\n", y_[i]*alpha_[i]);
      printf("\n\n");
	*/
      }

      // only right child has a splitting sample; left child is a leaf node
      else if(right_node->get_split_point_idx_old() != -1) {
	b_end_of_recursion_ = false;
	new_idx_not_splitted_node.PushBack() = node_pool.size();
	node_pool.PushBack() = node_pool[k]->right();
	new_n_not_splitted_node ++;
	new_n_samples_for_opt ++;
	
	// Number of samples in the left child;
	index_t n_samples_leaf = node_pool[k]->left()->count();
	//printf("left_n_sample_leaf=%d\n", n_samples_leaf);
	// divide alpha into (2+#leaf) and propagate
	old_idx_tmp = node_pool[k]->get_split_point_idx_old();
	new_idx_tmp = new_from_old_[old_idx_tmp];
	alpha_tmp = alpha_[new_idx_tmp];
	alpha_tmp = alpha_tmp / (2.0 + n_samples_leaf);
	
	/* current node itselft */
	// update alpha
	alpha_[new_idx_tmp] = alpha_tmp;
	// pre-update for gradients
	changed_values.PushBack() = y_[new_idx_tmp] * (1.0 + n_samples_leaf) * alpha_tmp;
	changed_idx_old.PushBack() = old_idx_tmp;
	
	/* right child */
	// update alpha
	old_idx_tmp = node_pool[k]->right()->get_split_point_idx_old();
	new_idx_tmp = new_from_old_[old_idx_tmp];
	alpha_[new_idx_tmp] = alpha_tmp;
	// pre-update for gradients
	changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	changed_idx_old.PushBack() = old_idx_tmp;
	// update active_set etc.
	//printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	SwapValues(n_active_, new_idx_tmp);
	n_used_alpha_ ++;
	n_active_ ++;

	/* left child (a leaf node) */
	new_n_leaf_node ++;
	new_n_samples_for_opt += n_samples_leaf;
	node_pool.PushBack() = node_pool[k]->left();
	leaf_node_id = node_pool[k]->left()->node_id();
	for (index_t t=0; t<n_samples_leaf; t++) {
	  // update alpha
	  old_idx_tmp = (kdtree_o_f_n_maps_pos_[leaf_node_id])[node_pool[k]->left()->begin() + t];
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  // update active_set etc.
	  //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	  SwapValues(n_active_, new_idx_tmp);
	  n_used_alpha_ ++;
	  n_active_ ++;
	}
      }
      // this node has neither left splitting sample, nor right splitting sample
      else {
	// both left and right children of this node are leaf nodes
	if ( (node_pool[k]->left()->is_leaf()) && (node_pool[k]->right()->is_leaf())  ) {
	  node_pool.PushBack() = node_pool[k]->left();
	  node_pool.PushBack() = node_pool[k]->right();

	  index_t n_samples_left_leaf = node_pool[k]->left()->count();
	  index_t n_samples_right_leaf = node_pool[k]->right()->count();
	  //printf("both_node_idx=%d\n",node_pool[k]->get_split_point_idx_old());
	  //printf("both_n_sample_left_leaf=%d\n", n_samples_left_leaf);
	  //printf("both_n_sample_right_leaf=%d\n", n_samples_right_leaf);
	  new_n_samples_for_opt += n_samples_left_leaf;
	  new_n_samples_for_opt += n_samples_right_leaf;

	  // divide alpha into (1+#left_leaf+#right_leaf) and propagate
	  old_idx_tmp = node_pool[k]->get_split_point_idx_old();
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_tmp = alpha_[new_idx_tmp];
	  alpha_tmp = alpha_tmp / (1.0 + n_samples_left_leaf + n_samples_right_leaf);
	  
	  /* current node itselft */
	  // update alpha
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = y_[new_idx_tmp] * (n_samples_left_leaf + n_samples_right_leaf) * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  
	  /* left child (a leaf node) */
	  new_n_leaf_node ++;
	  leaf_node_id = node_pool[k]->left()->node_id();
	  for (index_t t=0; t<n_samples_left_leaf; t++) {
	    // update alpha
	    old_idx_tmp = (kdtree_o_f_n_maps_pos_[leaf_node_id])[node_pool[k]->left()->begin() + t];
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	  }
	  /* right child (a leaf node) */
	  new_n_leaf_node ++;
	  leaf_node_id = node_pool[k]->right()->node_id();
	  for (index_t t=0; t<n_samples_right_leaf; t++) {
	    // update alpha
	    old_idx_tmp = (kdtree_o_f_n_maps_pos_[leaf_node_id])[node_pool[k]->right()->begin() + t];
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	  } // for
	} // if
      } // else
    } // if
  } // for

}




template<typename TKernel>
void HCY<TKernel>::SplitNodePropagateNeg_(ArrayList<TreeType*> &node_pool,
				       index_t &n_samples_for_opt, index_t &new_n_samples_for_opt,
				       index_t &n_splitted_node, index_t &new_n_splitted_node,
				       index_t &n_leaf_node, index_t &new_n_leaf_node,
				       index_t &n_not_splitted_node, index_t &new_n_not_splitted_node,
				       ArrayList<double> &changed_values, ArrayList<index_t> &changed_idx_old,
				       ArrayList<index_t> &idx_not_splitted_node, ArrayList<index_t> &new_idx_not_splitted_node) {
  
  double alpha_tmp, two_alpha_tmp;
  index_t old_idx_tmp, new_idx_tmp;
  index_t leaf_node_id;

  /**** Handle not-splitted nodes of the new level ****/
  //for (index_t k=n_splitted_node; k<n_splitted_node+n_not_splitted_node; k++) {
  for (index_t i=0; i<n_not_splitted_node; i++) {
    index_t k = idx_not_splitted_node[i];
    new_n_splitted_node ++;
    TreeType *left_node, *right_node;
    left_node = node_pool[k]->left();
    right_node = node_pool[k]->right();

    if (!node_pool[k]->is_leaf()) {
      //printf("l:%d\n", left_node->get_split_point_idx_old());
      //printf("r:%d\n", right_node->get_split_point_idx_old());
      // left child has a splitting sample
      if(left_node->get_split_point_idx_old() != -1) {
	b_end_of_recursion_ = false;
	new_idx_not_splitted_node.PushBack() = node_pool.size();
	node_pool.PushBack() = node_pool[k]->left();
	new_n_not_splitted_node ++;
	new_n_samples_for_opt ++;
	// right child also has a splitting sample
	if(right_node->get_split_point_idx_old() != -1) {
	  new_idx_not_splitted_node.PushBack() = node_pool.size();
	  node_pool.PushBack() = node_pool[k]->right();
	  new_n_not_splitted_node ++;
	  new_n_samples_for_opt ++;
	  
	  // divide alpha into 3 and propagate
	  old_idx_tmp = node_pool[k]->get_split_point_idx_old()+ n_data_pos_;
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_tmp = alpha_[new_idx_tmp];
	  alpha_tmp = alpha_tmp / 3.0;
	  two_alpha_tmp = 2.0 * alpha_tmp;
	  
	  /* Current node itselft */
	  // update alpha
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = y_[new_idx_tmp] * two_alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  
	  /* Left child */
	  // update alpha
	  old_idx_tmp = node_pool[k]->left()->get_split_point_idx_old()+ n_data_pos_;
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  // update active_set etc.
	  //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	  SwapValues(n_active_, new_idx_tmp);
	  n_used_alpha_ ++;
	  n_active_ ++;
	  
	  /* Right child */
	  // update alpha
	  old_idx_tmp = node_pool[k]->right()->get_split_point_idx_old()+ n_data_pos_;
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  // update active_set etc.
	  //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	  SwapValues(n_active_, new_idx_tmp);
	  n_used_alpha_ ++;
	  n_active_ ++;

	  /*
printf("intrim\n");
      for (index_t i=0; i<n_used_alpha_; i++)
	printf("%f.\n", y_[i]*alpha_[i]);
      printf("\n\n");
	  */

	}
	// only left child has a splitting sample; right child is a leaf node
	else {
	  // Number of samples in the right child
	  index_t n_samples_leaf = node_pool[k]->right()->count();
	  //printf("right_n_sample_leaf=%d\n", n_samples_leaf);
	  // divide alpha into (2+#leaf) and propagate
	  old_idx_tmp = node_pool[k]->get_split_point_idx_old()+ n_data_pos_;
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_tmp = alpha_[new_idx_tmp];
	  alpha_tmp = alpha_tmp / (2.0 + n_samples_leaf);
	  
	  /* current node itselft */
	  // update alpha
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = y_[new_idx_tmp] * (1.0 + n_samples_leaf) * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  
	  /* left child */
	  // update alpha
	  old_idx_tmp = node_pool[k]->left()->get_split_point_idx_old()+ n_data_pos_;
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  // update active_set etc.
	  //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	  SwapValues(n_active_, new_idx_tmp);
	  n_used_alpha_ ++;
	  n_active_ ++;
	  
	  /* right child (a leaf node) */
	  new_n_leaf_node ++;
	  new_n_samples_for_opt += n_samples_leaf;
	  node_pool.PushBack() = node_pool[k]->right();
	  leaf_node_id = node_pool[k]->right()->node_id();
	  for (index_t t=0; t<n_samples_leaf; t++) {
	    // update alpha
	    old_idx_tmp = (kdtree_o_f_n_maps_neg_[leaf_node_id])[node_pool[k]->right()->begin() + t] + n_data_pos_;
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	  }
	}
      }

      // only right child has a splitting sample; left child is a leaf node
      else if(right_node->get_split_point_idx_old() != -1) {
	b_end_of_recursion_ = false;
	new_idx_not_splitted_node.PushBack() = node_pool.size();
	node_pool.PushBack() = node_pool[k]->right();
	new_n_not_splitted_node ++;
	new_n_samples_for_opt ++;
	
	// Number of samples in the left child;
	index_t n_samples_leaf = node_pool[k]->left()->count();
	//printf("left_n_sample_leaf=%d\n", n_samples_leaf);
	// divide alpha into (2+#leaf) and propagate
	old_idx_tmp = node_pool[k]->get_split_point_idx_old()+ n_data_pos_;
	new_idx_tmp = new_from_old_[old_idx_tmp];
	alpha_tmp = alpha_[new_idx_tmp];
	alpha_tmp = alpha_tmp / (2.0 + n_samples_leaf);
	
	/* current node itselft */
	// update alpha
	alpha_[new_idx_tmp] = alpha_tmp;
	// pre-update for gradients
	changed_values.PushBack() = y_[new_idx_tmp] * (1.0 + n_samples_leaf) * alpha_tmp;
	changed_idx_old.PushBack() = old_idx_tmp;
	
	/* right child */
	// update alpha
	old_idx_tmp = node_pool[k]->right()->get_split_point_idx_old()+ n_data_pos_;
	new_idx_tmp = new_from_old_[old_idx_tmp];
	alpha_[new_idx_tmp] = alpha_tmp;
	// pre-update for gradients
	changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	changed_idx_old.PushBack() = old_idx_tmp;
	// update active_set etc.
	//printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	SwapValues(n_active_, new_idx_tmp);
	n_used_alpha_ ++;
	n_active_ ++;

	/* left child (a leaf node) */
	new_n_leaf_node ++;
	new_n_samples_for_opt += n_samples_leaf;
	node_pool.PushBack() = node_pool[k]->left();
	leaf_node_id = node_pool[k]->left()->node_id();
	for (index_t t=0; t<n_samples_leaf; t++) {
	  // update alpha
	  old_idx_tmp = (kdtree_o_f_n_maps_neg_[leaf_node_id])[node_pool[k]->left()->begin() + t] + n_data_pos_;
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  // update active_set etc.
	  //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	  SwapValues(n_active_, new_idx_tmp);
	  n_used_alpha_ ++;
	  n_active_ ++;
	}
      }
      // this node has neither left splitting sample, nor right splitting sample
      else {
	// both left and right children of this node are leaf nodes
	if ( (node_pool[k]->left()->is_leaf()) && (node_pool[k]->right()->is_leaf())  ) {
	  node_pool.PushBack() = node_pool[k]->left();
	  node_pool.PushBack() = node_pool[k]->right();

	  index_t n_samples_left_leaf = node_pool[k]->left()->count();
	  index_t n_samples_right_leaf = node_pool[k]->right()->count();
	  //printf("both_node_idx=%d\n",node_pool[k]->get_split_point_idx_old());
	  //printf("both_n_sample_left_leaf=%d\n", n_samples_left_leaf);
	  //printf("both_n_sample_right_leaf=%d\n", n_samples_right_leaf);
	  new_n_samples_for_opt += n_samples_left_leaf;
	  new_n_samples_for_opt += n_samples_right_leaf;

	  // divide alpha into (1+#left_leaf+#right_leaf) and propagate
	  old_idx_tmp = node_pool[k]->get_split_point_idx_old()+ n_data_pos_;
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_tmp = alpha_[new_idx_tmp];
	  alpha_tmp = alpha_tmp / (1.0 + n_samples_left_leaf + n_samples_right_leaf);
	  
	  /* current node itselft */
	  // update alpha
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = y_[new_idx_tmp] * (n_samples_left_leaf + n_samples_right_leaf) * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  
	  /* left child (a leaf node) */
	  new_n_leaf_node ++;
	  leaf_node_id = node_pool[k]->left()->node_id();
	  for (index_t t=0; t<n_samples_left_leaf; t++) {
	    // update alpha
	    old_idx_tmp = (kdtree_o_f_n_maps_neg_[leaf_node_id])[node_pool[k]->left()->begin() + t] + n_data_pos_;
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	  }
	  /* right child (a leaf node) */
	  new_n_leaf_node ++;
	  leaf_node_id = node_pool[k]->right()->node_id();
	  for (index_t t=0; t<n_samples_right_leaf; t++) {
	    // update alpha
	    old_idx_tmp = (kdtree_o_f_n_maps_neg_[leaf_node_id])[node_pool[k]->right()->begin() + t] + n_data_pos_;
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    //printf("swaping %d_%d\n", n_active_, new_idx_tmp);
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	  } // for
	} // if
      } // else
    } // if
  } // for

}



/**** Handle not-splitted nodes of the positive tree ****/
template<typename TKernel>
double HCY<TKernel>::DirectPropagatePos_(ArrayList<TreeType*> &node_pool,
				      index_t &n_samples_for_opt, index_t &new_n_samples_for_opt,
				      index_t &n_splitted_node, index_t &new_n_splitted_node,
				      index_t &n_leaf_node, index_t &new_n_leaf_node,
				      index_t &n_not_splitted_node, index_t &new_n_not_splitted_node,
				      ArrayList<double> &changed_values, ArrayList<index_t> &changed_idx_old,
				      ArrayList<index_t> &idx_not_splitted_node, ArrayList<index_t> &new_idx_not_splitted_node,
				      ArrayList<index_t> &idx_leaf_node) {
  
  double alpha_tmp;
  index_t old_idx_tmp, new_idx_tmp;
  index_t leaf_node_id;

  //printf("begin_n_active=%d\n",n_active_);
  /**** Handle not-splitted nodes of the tree ****/
  //for (index_t k=n_splitted_node; k<n_splitted_node+n_not_splitted_node; k++) {
  for (index_t i=0; i<n_not_splitted_node; i++) {
    index_t k = idx_not_splitted_node[i];

    old_idx_tmp = node_pool[k]->get_split_point_idx_old();
    new_idx_tmp = new_from_old_[old_idx_tmp];
    if (alpha_[new_idx_tmp]>HCY_ZERO) {

      new_n_splitted_node ++;

      TreeType *left_node, *right_node;
      left_node = node_pool[k]->left();
      right_node = node_pool[k]->right();
      
      if (!node_pool[k]->is_leaf()) {
	//printf("l:%d\n", left_node->get_split_point_idx_old());
	//printf("r:%d\n", right_node->get_split_point_idx_old());
	// left child has a splitting sample
	if(left_node->get_split_point_idx_old() != -1) {
	  b_end_of_recursion_ = false;
	  new_idx_not_splitted_node.PushBack() = node_pool.size();
	  node_pool.PushBack() = node_pool[k]->left();
	  new_n_not_splitted_node ++;
	  new_n_samples_for_opt ++;
	  // right child also has a splitting sample
	  if(right_node->get_split_point_idx_old() != -1) {
	    new_idx_not_splitted_node.PushBack() = node_pool.size();
	    node_pool.PushBack() = node_pool[k]->right();
	    new_n_not_splitted_node ++;
	    new_n_samples_for_opt ++;
	    
	    // direct propagation
	    old_idx_tmp = node_pool[k]->get_split_point_idx_old();
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_tmp = alpha_[new_idx_tmp];
	    
	    /* Current node itselft */
	    // update alpha
	    //alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    //changed_values.PushBack() = y_[new_idx_tmp] * two_alpha_tmp;
	    //changed_idx_old.PushBack() = old_idx_tmp;
	    
	    /* Left child */
	    // update alpha
	    old_idx_tmp = node_pool[k]->left()->get_split_point_idx_old();
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	    
	    /* Right child */
	    // update alpha
	    old_idx_tmp = node_pool[k]->right()->get_split_point_idx_old();
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	  }
	  // only left child has a splitting sample; right child is a leaf node
	  else {
	    // Number of samples in the right child
	    index_t n_samples_leaf = node_pool[k]->right()->count();
	    // direct propagation
	    old_idx_tmp = node_pool[k]->get_split_point_idx_old();
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_tmp = alpha_[new_idx_tmp];
	    
	    /* current node itselft */
	    // update alpha
	    //alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    //changed_values.PushBack() = y_[new_idx_tmp] * (1.0 + n_samples_leaf) * alpha_tmp;
	    //changed_idx_old.PushBack() = old_idx_tmp;
	    
	    /* left child */
	    // update alpha
	    old_idx_tmp = node_pool[k]->left()->get_split_point_idx_old();
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	  
	    /* right child (a leaf node) */
	    new_n_leaf_node ++;
	    new_n_samples_for_opt += n_samples_leaf;
	    idx_leaf_node.PushBack() = node_pool.size();
	    node_pool.PushBack() = node_pool[k]->right();
	    leaf_node_id = node_pool[k]->right()->node_id();
	    for (index_t t=0; t<n_samples_leaf; t++) {
	      // update alpha
	      old_idx_tmp = (kdtree_o_f_n_maps_pos_[leaf_node_id])[node_pool[k]->right()->begin() + t];
	      new_idx_tmp = new_from_old_[old_idx_tmp];
	      alpha_[new_idx_tmp] = alpha_tmp;
	      // pre-update for gradients
	      changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	      changed_idx_old.PushBack() = old_idx_tmp;
	      // update active_set etc.
	      SwapValues(n_active_, new_idx_tmp);
	      n_used_alpha_ ++;
	      n_active_ ++;
	    }
	  }
	}
	// only right child has a splitting sample; left child is a leaf node
	else if(right_node->get_split_point_idx_old() != -1) {
	  b_end_of_recursion_ = false;
	  new_idx_not_splitted_node.PushBack() = node_pool.size();
	  node_pool.PushBack() = node_pool[k]->right();
	  new_n_not_splitted_node ++;
	  new_n_samples_for_opt ++;
	  
	  // Number of samples in the left child;
	  index_t n_samples_leaf = node_pool[k]->left()->count();
	  //printf("n_sample_leaf=%d\n", n_samples_leaf);
	  // direct propagation
	  old_idx_tmp = node_pool[k]->get_split_point_idx_old();
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_tmp = alpha_[new_idx_tmp];
	  
	  /* current node itselft */
	  // update alpha
	  //alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  //changed_values.PushBack() = y_[new_idx_tmp] * (1.0 + n_samples_leaf) * alpha_tmp;
	  //changed_idx_old.PushBack() = old_idx_tmp;
	  
	  /* right child */
	  // update alpha
	  old_idx_tmp = node_pool[k]->right()->get_split_point_idx_old();
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  // update active_set etc.
	  SwapValues(n_active_, new_idx_tmp);
	  n_used_alpha_ ++;
	  n_active_ ++;
	  
	  /* left child (a leaf node) */
	  new_n_leaf_node ++;
	  new_n_samples_for_opt += n_samples_leaf;
	  idx_leaf_node.PushBack() = node_pool.size();
	  node_pool.PushBack() = node_pool[k]->left();
	  leaf_node_id = node_pool[k]->left()->node_id();
	  for (index_t t=0; t<n_samples_leaf; t++) {
	  // update alpha
	    old_idx_tmp = (kdtree_o_f_n_maps_pos_[leaf_node_id])[node_pool[k]->left()->begin() + t];
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	  }
	}
	// this node has neither left splitting sample, nor right splitting sample
	else {
	  // both left and right children of this node are leaf nodes
	  if ( (node_pool[k]->left()->is_leaf()) && (node_pool[k]->right()->is_leaf())  ) {
	    idx_leaf_node.PushBack() = node_pool.size();
	    node_pool.PushBack() = node_pool[k]->left();
	    idx_leaf_node.PushBack() = node_pool.size();
	    node_pool.PushBack() = node_pool[k]->right();
	    
	    index_t n_samples_left_leaf = node_pool[k]->left()->count();
	    index_t n_samples_right_leaf = node_pool[k]->right()->count();
	    //printf("idx=%d\n",node_pool[k]->get_split_point_idx_old());
	    //printf("n_sample_left_leaf=%d\n", n_samples_left_leaf);
	    //printf("n_sample_right_leaf=%d\n", n_samples_right_leaf);
	    new_n_samples_for_opt += n_samples_left_leaf;
	    new_n_samples_for_opt += n_samples_right_leaf;
	    
	    // direct propagation
	    old_idx_tmp = node_pool[k]->get_split_point_idx_old();
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_tmp = alpha_[new_idx_tmp];
	    
	    /* current node itselft */
	    // update alpha
	    //alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    //changed_values.PushBack() = y_[new_idx_tmp] * (n_samples_left_leaf + n_samples_right_leaf) * alpha_tmp;
	    //changed_idx_old.PushBack() = old_idx_tmp;
	    
	    /* left child (a leaf node) */
	    new_n_leaf_node ++;
	    leaf_node_id = node_pool[k]->left()->node_id();
	    for (index_t t=0; t<n_samples_left_leaf; t++) {
	      // update alpha
	      old_idx_tmp = (kdtree_o_f_n_maps_pos_[leaf_node_id])[node_pool[k]->left()->begin() + t];
	      new_idx_tmp = new_from_old_[old_idx_tmp];
	      alpha_[new_idx_tmp] = alpha_tmp;
	      // pre-update for gradients
	      changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	      changed_idx_old.PushBack() = old_idx_tmp;
	      // update active_set etc.
	      SwapValues(n_active_, new_idx_tmp);
	      n_used_alpha_ ++;
	      n_active_ ++;
	    }
	    /* right child (a leaf node) */
	    new_n_leaf_node ++;
	    leaf_node_id = node_pool[k]->right()->node_id();
	    for (index_t t=0; t<n_samples_right_leaf; t++) {
	      // update alpha
	      old_idx_tmp = (kdtree_o_f_n_maps_pos_[leaf_node_id])[node_pool[k]->right()->begin() + t];
	      new_idx_tmp = new_from_old_[old_idx_tmp];
	      alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	      changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	      changed_idx_old.PushBack() = old_idx_tmp;
	      // update active_set etc.
	      SwapValues(n_active_, new_idx_tmp);
	      n_used_alpha_ ++;
	      n_active_ ++;
	    } // for
	  } // if
	} // else
      } // if
    }// if
    else { // alpha_[new_idx_tmp] <= HCY_ZERO
      if (!node_pool[k]->is_leaf()) {
	TreeType *left_node, *right_node;
	left_node = node_pool[k]->left();
	right_node = node_pool[k]->right();
	if(left_node->get_split_point_idx_old() != -1 && right_node->get_split_point_idx_old() != -1) {
	  new_idx_not_splitted_node.PushBack() = k;
	  new_n_not_splitted_node ++;
	}
      } 
    }
  } // for

  // calculate the sum of alphas for this tree
  double alphas_sum = 0.0;
  
  if (idx_leaf_node.size() > 0) { // there exist leaf nodes
    index_t leaf_i = 0;
    for (index_t i=0; i<node_pool.size(); i++) {
      if (i != idx_leaf_node[leaf_i]) { // not a leaf node
	old_idx_tmp = node_pool[i]->get_split_point_idx_old();
	new_idx_tmp = new_from_old_[old_idx_tmp];
	//printf("+a%f*_nl.\n", alpha_[new_idx_tmp]);
	alphas_sum += alpha_[new_idx_tmp];
      }
      else { // leaf node
	leaf_node_id = node_pool[i]->node_id();
	for (index_t t=0; t<node_pool[i]->count(); t++) {
	  old_idx_tmp = (kdtree_o_f_n_maps_pos_[leaf_node_id])[node_pool[i]->begin() + t];
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  //printf("+a%f*_lf\n", alpha_[new_idx_tmp]);
	  alphas_sum += alpha_[new_idx_tmp];
	}
	leaf_i ++;
      }
    }
  }
  else { // no leaf node exists
    for (index_t i=0; i<node_pool.size(); i++) {
      old_idx_tmp = node_pool[i]->get_split_point_idx_old();
      new_idx_tmp = new_from_old_[old_idx_tmp];
      //printf("+a%f\n", alpha_[new_idx_tmp]);
      alphas_sum += alpha_[new_idx_tmp];
    }
  }
  
  return alphas_sum;
}




/**** Handle not-splitted nodes of the negative tree ****/
template<typename TKernel>
double HCY<TKernel>::DirectPropagateNeg_(ArrayList<TreeType*> &node_pool,
				      index_t &n_samples_for_opt, index_t &new_n_samples_for_opt,
				      index_t &n_splitted_node, index_t &new_n_splitted_node,
				      index_t &n_leaf_node, index_t &new_n_leaf_node,
				      index_t &n_not_splitted_node, index_t &new_n_not_splitted_node,
				      ArrayList<double> &changed_values, ArrayList<index_t> &changed_idx_old,
				      ArrayList<index_t> &idx_not_splitted_node, ArrayList<index_t> &new_idx_not_splitted_node,
				      ArrayList<index_t> &idx_leaf_node) {
  
  double alpha_tmp;
  index_t old_idx_tmp, new_idx_tmp;
  index_t leaf_node_id;

  //printf("begin_n_active=%d\n",n_active_);
  /**** Handle not-splitted nodes of the tree ****/
  //for (index_t k=n_splitted_node; k<n_splitted_node+n_not_splitted_node; k++) {
  for (index_t i=0; i<n_not_splitted_node; i++) {
    index_t k = idx_not_splitted_node[i];

    old_idx_tmp = node_pool[k]->get_split_point_idx_old()+ n_data_pos_;
    new_idx_tmp = new_from_old_[old_idx_tmp];
    if (alpha_[new_idx_tmp]>HCY_ZERO) {
      
      new_n_splitted_node ++;

      TreeType *left_node, *right_node;
      left_node = node_pool[k]->left();
      right_node = node_pool[k]->right();
      
      if (!node_pool[k]->is_leaf()) {
	//printf("l:%d\n", left_node->get_split_point_idx_old());
	//printf("r:%d\n", right_node->get_split_point_idx_old());
	// left child has a splitting sample
	if(left_node->get_split_point_idx_old() != -1) {
	  b_end_of_recursion_ = false;
	  new_idx_not_splitted_node.PushBack() = node_pool.size();
	  node_pool.PushBack() = node_pool[k]->left();
	  new_n_not_splitted_node ++;
	  new_n_samples_for_opt ++;
	  // right child also has a splitting sample
	  if(right_node->get_split_point_idx_old() != -1) {
	    new_idx_not_splitted_node.PushBack() = node_pool.size();
	    node_pool.PushBack() = node_pool[k]->right();
	    new_n_not_splitted_node ++;
	    new_n_samples_for_opt ++;
	    
	    // direct propagation
	    old_idx_tmp = node_pool[k]->get_split_point_idx_old()+ n_data_pos_;
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_tmp = alpha_[new_idx_tmp];
	    
	    /* Current node itselft */
	    // update alpha
	    //alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    //changed_values.PushBack() = y_[new_idx_tmp] * two_alpha_tmp;
	    //changed_idx_old.PushBack() = old_idx_tmp;
	    
	    /* Left child */
	    // update alpha
	    old_idx_tmp = node_pool[k]->left()->get_split_point_idx_old()+ n_data_pos_;
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	    
	    /* Right child */
	    // update alpha
	    old_idx_tmp = node_pool[k]->right()->get_split_point_idx_old()+ n_data_pos_;
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	  }
	  // only left child has a splitting sample; right child is a leaf node
	  else {
	    // Number of samples in the right child
	    index_t n_samples_leaf = node_pool[k]->right()->count();
	    // direct propagation
	    old_idx_tmp = node_pool[k]->get_split_point_idx_old()+ n_data_pos_;
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_tmp = alpha_[new_idx_tmp];
	    
	    /* current node itselft */
	    // update alpha
	    //alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    //changed_values.PushBack() = y_[new_idx_tmp] * (1.0 + n_samples_leaf) * alpha_tmp;
	    //changed_idx_old.PushBack() = old_idx_tmp;
	    
	    /* left child */
	    // update alpha
	    old_idx_tmp = node_pool[k]->left()->get_split_point_idx_old()+ n_data_pos_;
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	    
	    /* right child (a leaf node) */
	    new_n_leaf_node ++;
	    new_n_samples_for_opt += n_samples_leaf;
	    idx_leaf_node.PushBack() = node_pool.size();
	    node_pool.PushBack() = node_pool[k]->right();
	    leaf_node_id = node_pool[k]->right()->node_id();
	    for (index_t t=0; t<n_samples_leaf; t++) {
	      // update alpha
	      old_idx_tmp = (kdtree_o_f_n_maps_neg_[leaf_node_id])[node_pool[k]->right()->begin() + t] + n_data_pos_;
	      new_idx_tmp = new_from_old_[old_idx_tmp];
	      alpha_[new_idx_tmp] = alpha_tmp;
	      // pre-update for gradients
	      changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	      changed_idx_old.PushBack() = old_idx_tmp;
	      // update active_set etc.
	      SwapValues(n_active_, new_idx_tmp);
	      n_used_alpha_ ++;
	      n_active_ ++;
	    }
	  }
	}
	// only right child has a splitting sample; left child is a leaf node
	else if(right_node->get_split_point_idx_old() != -1) {
	  b_end_of_recursion_ = false;
	  new_idx_not_splitted_node.PushBack() = node_pool.size();
	  node_pool.PushBack() = node_pool[k]->right();
	  new_n_not_splitted_node ++;
	  new_n_samples_for_opt ++;
	  
	  // Number of samples in the left child;
	  index_t n_samples_leaf = node_pool[k]->left()->count();
	  //printf("n_sample_leaf=%d\n", n_samples_leaf);
	  // direct propagation
	  old_idx_tmp = node_pool[k]->get_split_point_idx_old()+ n_data_pos_;
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_tmp = alpha_[new_idx_tmp];
	
	  /* current node itselft */
	  // update alpha
	  //alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  //changed_values.PushBack() = y_[new_idx_tmp] * (1.0 + n_samples_leaf) * alpha_tmp;
	  //changed_idx_old.PushBack() = old_idx_tmp;
	  
	  /* right child */
	  // update alpha
	  old_idx_tmp = node_pool[k]->right()->get_split_point_idx_old()+ n_data_pos_;
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_tmp;
	  // pre-update for gradients
	  changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	  changed_idx_old.PushBack() = old_idx_tmp;
	  // update active_set etc.
	  SwapValues(n_active_, new_idx_tmp);
	  n_used_alpha_ ++;
	  n_active_ ++;
	  
	  /* left child (a leaf node) */
	  new_n_leaf_node ++;
	  new_n_samples_for_opt += n_samples_leaf;
	  idx_leaf_node.PushBack() = node_pool.size();
	  node_pool.PushBack() = node_pool[k]->left();
	  leaf_node_id = node_pool[k]->left()->node_id();
	  for (index_t t=0; t<n_samples_leaf; t++) {
	    // update alpha
	    old_idx_tmp = (kdtree_o_f_n_maps_neg_[leaf_node_id])[node_pool[k]->left()->begin() + t] + n_data_pos_;
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	    changed_idx_old.PushBack() = old_idx_tmp;
	    // update active_set etc.
	    SwapValues(n_active_, new_idx_tmp);
	    n_used_alpha_ ++;
	    n_active_ ++;
	  }
	}
	// this node has neither left splitting sample, nor right splitting sample
	else {
	  // both left and right children of this node are leaf nodes
	  if ( (node_pool[k]->left()->is_leaf()) && (node_pool[k]->right()->is_leaf())  ) {
	    idx_leaf_node.PushBack() = node_pool.size();
	    node_pool.PushBack() = node_pool[k]->left();
	    idx_leaf_node.PushBack() = node_pool.size();
	    node_pool.PushBack() = node_pool[k]->right();
	    
	    index_t n_samples_left_leaf = node_pool[k]->left()->count();
	    index_t n_samples_right_leaf = node_pool[k]->right()->count();
	    //printf("idx=%d\n",node_pool[k]->get_split_point_idx_old());
	    //printf("n_sample_left_leaf=%d\n", n_samples_left_leaf);
	    //printf("n_sample_right_leaf=%d\n", n_samples_right_leaf);
	    new_n_samples_for_opt += n_samples_left_leaf;
	    new_n_samples_for_opt += n_samples_right_leaf;
	    
	    // direct propagation
	    old_idx_tmp = node_pool[k]->get_split_point_idx_old()+ n_data_pos_;
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_tmp = alpha_[new_idx_tmp];
	    
	    /* current node itselft */
	    // update alpha
	    //alpha_[new_idx_tmp] = alpha_tmp;
	    // pre-update for gradients
	    //changed_values.PushBack() = y_[new_idx_tmp] * (n_samples_left_leaf + n_samples_right_leaf) * alpha_tmp;
	    //changed_idx_old.PushBack() = old_idx_tmp;
	    
	    /* left child (a leaf node) */
	    new_n_leaf_node ++;
	    leaf_node_id = node_pool[k]->left()->node_id();
	    for (index_t t=0; t<n_samples_left_leaf; t++) {
	      // update alpha
	      old_idx_tmp = (kdtree_o_f_n_maps_neg_[leaf_node_id])[node_pool[k]->left()->begin() + t] + n_data_pos_;
	      new_idx_tmp = new_from_old_[old_idx_tmp];
	      alpha_[new_idx_tmp] = alpha_tmp;
	      // pre-update for gradients
	      changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	      changed_idx_old.PushBack() = old_idx_tmp;
	      // update active_set etc.
	      SwapValues(n_active_, new_idx_tmp);
	      n_used_alpha_ ++;
	      n_active_ ++;
	    }
	    /* right child (a leaf node) */
	    new_n_leaf_node ++;
	    leaf_node_id = node_pool[k]->right()->node_id();
	    for (index_t t=0; t<n_samples_right_leaf; t++) {
	      // update alpha
	      old_idx_tmp = (kdtree_o_f_n_maps_neg_[leaf_node_id])[node_pool[k]->right()->begin() + t] + n_data_pos_;
	      new_idx_tmp = new_from_old_[old_idx_tmp];
	      alpha_[new_idx_tmp] = alpha_tmp;
	      // pre-update for gradients
	      changed_values.PushBack() = - y_[new_idx_tmp] * alpha_tmp;
	      changed_idx_old.PushBack() = old_idx_tmp;
	      // update active_set etc.
	      SwapValues(n_active_, new_idx_tmp);
	      n_used_alpha_ ++;
	      n_active_ ++;
	  } // for
	  } // if
	} // else
      } // if
    } // if
    else { // alpha_[new_idx_tmp] <= HCY_ZERO
      if (!node_pool[k]->is_leaf()) {
	TreeType *left_node, *right_node;
	left_node = node_pool[k]->left();
	right_node = node_pool[k]->right();
	if(left_node->get_split_point_idx_old() != -1 && right_node->get_split_point_idx_old() != -1) {
	  new_idx_not_splitted_node.PushBack() = k;
	  new_n_not_splitted_node ++;
	}
      } 
    }
  } // for
  
  // calculate the sum of alphas for this tree
  double alphas_sum = 0.0;
  
  if (idx_leaf_node.size() > 0) { // there exist leaf nodes
    index_t leaf_i = 0;
    for (index_t i=0; i<node_pool.size(); i++) {
      if (i != idx_leaf_node[leaf_i]) { // not a leaf node
	old_idx_tmp = node_pool[i]->get_split_point_idx_old()+ n_data_pos_;
	new_idx_tmp = new_from_old_[old_idx_tmp];
	//printf("-a%f*_nl\n", alpha_[new_idx_tmp]);
	alphas_sum += alpha_[new_idx_tmp];
      }
      else { // leaf node
	leaf_node_id = node_pool[i]->node_id();
	for (index_t t=0; t<node_pool[i]->count(); t++) {
	  old_idx_tmp = (kdtree_o_f_n_maps_neg_[leaf_node_id])[node_pool[i]->begin() + t] + n_data_pos_;
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  //printf("-a%f*_lf\n", alpha_[new_idx_tmp]);
	  alphas_sum += alpha_[new_idx_tmp];
	}
	leaf_i ++;
      }
    }
  }
  else { // no leaf node exists
    for (index_t i=0; i<node_pool.size(); i++) {
      old_idx_tmp = node_pool[i]->get_split_point_idx_old()+ n_data_pos_;
      new_idx_tmp = new_from_old_[old_idx_tmp];
      //printf("-a%f\n", alpha_[new_idx_tmp]);
      alphas_sum += alpha_[new_idx_tmp];
    }
  }
  
  return alphas_sum;
}









template<typename TKernel>
void HCY<TKernel>::TreeDescentRecursion_(ArrayList<TreeType*> &node_pool_pos, ArrayList<TreeType*> &node_pool_neg,
					 index_t n_samples_for_opt,
					 index_t n_splitted_node_pos, index_t n_splitted_node_neg,
					 index_t n_leaf_node_pos, index_t n_leaf_node_neg,
					 index_t n_not_splitted_node_pos, index_t n_not_splitted_node_neg,
					 ArrayList<index_t> &idx_not_splitted_node_pos, ArrayList<index_t> &new_idx_not_splitted_node_pos,
					 ArrayList<index_t> &idx_not_splitted_node_neg, ArrayList<index_t> &new_idx_not_splitted_node_neg,
					 ArrayList<index_t> &idx_leaf_node_pos, ArrayList<index_t> &idx_leaf_node_neg) {
  printf("iteration %d\n", ct_iter_);

  index_t i, j;

  index_t n_node_pos = node_pool_pos.size();
  index_t n_node_neg = node_pool_neg.size();
  //printf("n_node_pos=%d, n_node_neg=%d\n", n_node_pos, n_node_neg);

  DEBUG_ASSERT_MSG( (n_splitted_node_pos+n_leaf_node_pos+n_not_splitted_node_pos)== n_node_pos, 
		    "Pool of nodes for positive tree not match!!!");
  DEBUG_ASSERT_MSG( (n_splitted_node_neg+n_leaf_node_neg+n_not_splitted_node_neg)== n_node_neg, 
		    "Pool of nodes for negative tree not match!!!");

printf("\n");
printf("n_samples_for_opt=%d\n", n_samples_for_opt);

  /*** SMO Optimization of alphas for this level ***/
  n_used_alpha_ = n_samples_for_opt;
  n_active_ = n_used_alpha_;
  n_sv_ = 0;
  unshrinked_ = false;
  i_cache_ = -1; j_cache_ = -1;
  cached_kernel_value_ = INFINITY;

  
  // update alpha status
  for (i=0; i<n_used_alpha_; i++)
    UpdateAlphaStatus_(i);
  // update gradient bar
  for (i=0; i<n_used_alpha_; i++) {
    for(j=0; j<n_used_alpha_; j++) {
      if(IsUpperBounded(j)) // alpha_j >= C
	grad_bar_[i] = grad_bar_[i] + GetC_(j) * y_[j] * CalcKernelValue_(i,j);
    }
    grad_bar_[i] = y_[i] * grad_bar_[i];
  }

  ct_iter_ = 0;
  ct_shrinking_ = min( HCY_NUM_FOR_SHRINKING, max(max_n_alpha_, HCY_NUM_FOR_SHRINKING));
  //ct_shrinking_ = 1;




/*
printf("before smo:\n");
  for (i=0; i<n_used_alpha_; i++)
    printf("%f-\n", y_[i]*alpha_[i]);
*/

/*
  double check_ya = 0.0;
  for (i=0; i<n_used_alpha_; i++)
    check_ya += y_[i]*alpha_[i];
  printf("check_sum_ya_before_SMO=%f\n", check_ya);
*/

  int stop_condition;
fx_timer_start(NULL, "hcy_smo");
  while (1) {

    /*
    // for every min(n_data_, 1000) iterations, do shrinking
    if (--ct_shrinking_ == 0) {
      Shrinking_();
      ct_shrinking_ = min( HCY_NUM_FOR_SHRINKING, max(max_n_alpha_, HCY_NUM_FOR_SHRINKING));
    }
    */

    // Find working set, check stopping criterion, update gradient and alphas
    ct_iter_ ++;
    if (WorkingSetSelection_(i,j) == true) {
      /*
      ReconstructGradient_(learner_typeid_); // restore the inactive alphas and reconstruct gradients
      n_active_ = n_used_alpha_;
      if (WorkingSetSelection_(i,j) == true) { // optimality reached
	stop_condition = 1;
      }
      else {
	ct_shrinking_ = 1; // do shrinking in the next iteration
	stop_condition = 0;
      }
      */
      
      stop_condition = 1;
      
    }
    else if (ct_iter_ >= n_iter_) { // number of iterations exceeded
      stop_condition = 2;
    }
    else{ // update gradients and alphas, and continue iterations
      UpdateGradientAlpha_(i, j);
      stop_condition = 0;
    }

    // termination check, stop_condition==1 or 2->terminate
    if (stop_condition == 1) {// optimality reached
      printf("Accuracy %f achieved. Number of iterations: %d.\n", accuracy_, ct_iter_);
      break;
    }
    else if (stop_condition == 2) {// max num of iterations exceeded
      fprintf(stderr, "Number of iterations %d exceeded !!!\n", n_iter_);
      break;
    }

  }

/*
printf("after smo:\n");
for (i=0; i<n_used_alpha_; i++)
    printf("%f+\n", y_[i]*alpha_[i]);
*/ 

/*
  check_ya = 0.0;
  for (i=0; i<n_used_alpha_; i++)
    check_ya += y_[i]*alpha_[i];
  printf("check_sum_ya_after_SMO=%f\n", check_ya);
*/
 

fx_timer_stop(NULL, "hcy_smo");
  /*** SMO finished for this level. Begin alpha propagation for next level ***/


 int n_stop= fx_param_int(NULL,"n_stop", n_samples_for_opt);  // alpha propagation method: 
 if (n_samples_for_opt > n_stop)
   return;


  /*** if reached the last level, end recursion ***/
  if (b_end_of_recursion_)
    return;

  /* Generate pool of children for the next level. Propagate alphas and gradients */
  b_end_of_recursion_ = true;

  ArrayList<double> changed_values_pos;
  ArrayList<double> changed_values_neg;
  ArrayList<index_t> changed_idx_old_pos;
  ArrayList<index_t> changed_idx_old_neg;
  changed_values_pos.Init();
  changed_values_neg.Init();
  changed_idx_old_pos.Init();
  changed_idx_old_neg.Init();

  // alphas of already splitted nodes of this level (either active or inactive) will be re-optimized in the next level;
  // besides, all new alphas will be used
  index_t new_n_samples_for_opt = n_samples_for_opt;
  //printf("---new_n_samples_for_opt=%d\n", new_n_samples_for_opt);
  index_t new_n_splitted_node_pos = n_splitted_node_pos; // only contain the splitting sample
  index_t new_n_splitted_node_neg = n_splitted_node_neg;
  index_t new_n_leaf_node_pos = n_leaf_node_pos; // contains a bunch of samples
  index_t new_n_leaf_node_neg = n_leaf_node_neg;
  index_t new_n_not_splitted_node_pos = 0; // only contain the splitting sample
  index_t new_n_not_splitted_node_neg = 0;

  index_t leaf_node_id;


  int pm= fx_param_int(NULL,"pm", 1);  // alpha propagation method:

  // 1: Split and propagation: \alpha/3
  // 2: Naive direct propagation and sacle with eta_1
  // 3: Optimized direct propagation and scale with opt_eta_1, opt_eta_2

  if (pm == 1) {
    /**** Handle not-splitted nodes of the positive tree ****/
    SplitNodePropagatePos_(node_pool_pos,
			   n_samples_for_opt, new_n_samples_for_opt,
			   n_splitted_node_pos, new_n_splitted_node_pos,
			   n_leaf_node_pos, new_n_leaf_node_pos,
			   n_not_splitted_node_pos, new_n_not_splitted_node_pos,
			   changed_values_pos, changed_idx_old_pos,
			   idx_not_splitted_node_pos, new_idx_not_splitted_node_pos);
    idx_not_splitted_node_pos.Swap(&new_idx_not_splitted_node_pos); // new_index -> idx
    new_idx_not_splitted_node_pos.Clear(); // idx->new_idx and cleared

    /**** Handle not-splitted nodes of the negative tree ****/
    SplitNodePropagateNeg_(node_pool_neg,
			   n_samples_for_opt, new_n_samples_for_opt,
			   n_splitted_node_neg, new_n_splitted_node_neg,
			   n_leaf_node_neg, new_n_leaf_node_neg,
			   n_not_splitted_node_neg, new_n_not_splitted_node_neg,
			   changed_values_neg, changed_idx_old_neg,
			   idx_not_splitted_node_neg, new_idx_not_splitted_node_neg);
    idx_not_splitted_node_neg.Swap(&new_idx_not_splitted_node_neg); // new_index -> idx
    new_idx_not_splitted_node_neg.Clear(); // idx->new_idx and cleared
  }
  else if (pm == 2) {
    double alphas_sum_pos, alphas_sum_neg;
    /**** Handle not-splitted nodes of the positive tree ****/
    alphas_sum_pos = DirectPropagatePos_(node_pool_pos,
					 n_samples_for_opt, new_n_samples_for_opt,
					 n_splitted_node_pos, new_n_splitted_node_pos,
					 n_leaf_node_pos, new_n_leaf_node_pos,
					 n_not_splitted_node_pos, new_n_not_splitted_node_pos,
					 changed_values_pos, changed_idx_old_pos,
					 idx_not_splitted_node_pos, new_idx_not_splitted_node_pos,
					 idx_leaf_node_pos);
    idx_not_splitted_node_pos.Swap(&new_idx_not_splitted_node_pos); // new_index -> idx
    new_idx_not_splitted_node_pos.Clear(); // idx->new_idx and cleared

    /**** Handle not-splitted nodes of the negative tree ****/
    alphas_sum_neg = DirectPropagateNeg_(node_pool_neg,
					 n_samples_for_opt, new_n_samples_for_opt,
					 n_splitted_node_neg, new_n_splitted_node_neg,
					 n_leaf_node_neg, new_n_leaf_node_neg,
					 n_not_splitted_node_neg, new_n_not_splitted_node_neg,
					 changed_values_neg, changed_idx_old_neg,
					 idx_not_splitted_node_neg, new_idx_not_splitted_node_neg,
					 idx_leaf_node_neg);
    idx_not_splitted_node_neg.Swap(&new_idx_not_splitted_node_neg); // new_index -> idx
    new_idx_not_splitted_node_neg.Clear(); // idx->new_idx and cleared
    // calculate eta
    printf("a_sum_pos=%f, a_sum_neg=%f\n", alphas_sum_pos, alphas_sum_neg);
    double eta;
    eta = alphas_sum_pos / alphas_sum_neg;
    printf("eta=%f\n", eta);

    index_t old_idx_tmp, new_idx_tmp;
    if (eta > 1) { // scale pos
      if (idx_leaf_node_pos.size() > 0) { // there exist leaf nodes
	index_t leaf_i = 0;
	for (i=0; i<node_pool_pos.size(); i++) {
	  if (i != idx_leaf_node_pos[leaf_i]) { // not a leaf node
	    old_idx_tmp = node_pool_pos[i]->get_split_point_idx_old();
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_[new_idx_tmp] / eta;
	  }
	  else { // leaf node
	    leaf_node_id = node_pool_pos[i]->node_id();
	    for (index_t t=0; t<node_pool_pos[i]->count(); t++) {
	      old_idx_tmp = (kdtree_o_f_n_maps_pos_[leaf_node_id])[node_pool_pos[i]->begin() + t];
	      new_idx_tmp = new_from_old_[old_idx_tmp];
	      alpha_[new_idx_tmp] = alpha_[new_idx_tmp] / eta;
	    }
	    leaf_i ++;
	  }
	}
      }
      else { // no leaf node exists
	for (index_t i=0; i<node_pool_pos.size(); i++) {
	  old_idx_tmp = node_pool_pos[i]->get_split_point_idx_old();
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_[new_idx_tmp] / eta;
	}
      }
      // scale pre-update of gradients
      for (i=0; i<changed_values_pos.size(); i++) {
	changed_values_pos[i] = changed_values_pos[i] / eta;
      }
    }
    else if (eta < 1) { // scale neg
      if (idx_leaf_node_neg.size() > 0) { // there exist leaf nodes
	index_t leaf_i = 0;
	for (i=0; i<node_pool_neg.size(); i++) {
	  if (i != idx_leaf_node_neg[leaf_i]) { // not a leaf node
	    old_idx_tmp = node_pool_neg[i]->get_split_point_idx_old()+ n_data_pos_;
	    new_idx_tmp = new_from_old_[old_idx_tmp];
	    alpha_[new_idx_tmp] = alpha_[new_idx_tmp] * eta;
	  }
	  else { // leaf node
	    leaf_node_id = node_pool_neg[i]->node_id();
	    for (index_t t=0; t<node_pool_neg[i]->count(); t++) {
	      old_idx_tmp = (kdtree_o_f_n_maps_neg_[leaf_node_id])[node_pool_neg[i]->begin() + t] + n_data_pos_;
	      new_idx_tmp = new_from_old_[old_idx_tmp];
	      alpha_[new_idx_tmp] = alpha_[new_idx_tmp] * eta;
	    }
	    leaf_i ++;
	  }
	}
      }
      else { // no leaf node exists
	for (index_t i=0; i<node_pool_neg.size(); i++) {
	  old_idx_tmp = node_pool_neg[i]->get_split_point_idx_old()+ n_data_pos_;
	  new_idx_tmp = new_from_old_[old_idx_tmp];
	  alpha_[new_idx_tmp] = alpha_[new_idx_tmp] * eta;
	}
      }
      // scale pre-update of gradients
      for (i=0; i<changed_values_neg.size(); i++) {
	changed_values_neg[i] = changed_values_neg[i] * eta;
      }
    }
    else { // eta == 1
      
    }
    
  }
  else if (pm == 3) {
    
  }

  
fx_timer_start(NULL, "alpha_propagate");
  /* Update and propagate gradients
   */
  //printf("+++new_n_samples_for_opt=%d\n", new_n_samples_for_opt);
  for(index_t nl=0; nl<new_n_samples_for_opt; nl++) {
    for (index_t nl_p=0; nl_p<changed_values_pos.size(); nl_p++) {
      grad_[nl] = grad_[nl] + y_[nl] * changed_values_pos[nl_p] * CalcKernelValue_(new_from_old_[changed_idx_old_pos[nl_p]], nl);
    }
    for (index_t nl_p=0; nl_p<changed_values_neg.size(); nl_p++) {
      grad_[nl] = grad_[nl] + y_[nl] * changed_values_neg[nl_p] * CalcKernelValue_(new_from_old_[changed_idx_old_neg[nl_p]], nl);
    }
  }
  
fx_timer_stop(NULL, "alpha_propagate");

  // the bias term ramains unchanged in propagations
  
  // recursion for the next level
  return TreeDescentRecursion_(node_pool_pos, node_pool_neg, 
			       new_n_samples_for_opt,
			       new_n_splitted_node_pos, new_n_splitted_node_neg,
			       new_n_leaf_node_pos, new_n_leaf_node_neg,
			       new_n_not_splitted_node_pos, new_n_not_splitted_node_neg,
			       idx_not_splitted_node_pos, new_idx_not_splitted_node_pos,
			       idx_not_splitted_node_neg, new_idx_not_splitted_node_neg,
			       idx_leaf_node_pos, idx_leaf_node_neg);
}


/**
* Try to find a working set (i,j). Both 1st order and 2nd order approximations of 
* the objective function Z(\alpha+\lambda u_ij)-Z(\alpha) are implemented.
*
* @param: working set (i, j)
*
* @return: indicator of whether the optimal solution is reached (true:reached)
*/
template<typename TKernel>
bool HCY<TKernel>::WorkingSetSelection_(index_t &out_i, index_t &out_j) {
  double y_grad_max = -INFINITY;
  double y_grad_min =  INFINITY;
  int idx_i = -1;
  int idx_j = -1;
  
  // Find i using maximal violating pair scheme
  index_t t;
  for (t=0; t<n_active_; t++) { // find argmax(y*grad), t\in I_up
    if (y_[t] == 1) {
      if (!IsUpperBounded(t)) // t\in I_up, y==1: y[t]alpha[t] < C
	if (grad_[t] > y_grad_max) { // y==1
	  y_grad_max = grad_[t];
	  idx_i = t;
	}
    }
    else { // y[t] == -1
      if (!IsLowerBounded(t)) // t\in I_up, y==-1: y[t]alpha[t] < 0
	if (grad_[t] + y_grad_max < 0) { // y==-1... <=> -grad_[t] > y_grad_max
	  y_grad_max = -grad_[t];
	  idx_i = t;
	}
    }
  }
  out_i = idx_i; // i found

  /*  Find j using maximal violating pair scheme (1st order approximation) */
  if (wss_ == 1) {
    for (t=0; t<n_active_; t++) { // find argmin(y*grad), t\in I_down
      if (y_[t] == 1) {
	if (!IsLowerBounded(t)) // t\in I_down, y==1: y[t]alpha[t] > 0
	  if (grad_[t] < y_grad_min) { // y==1
	    y_grad_min = grad_[t];
	    idx_j = t;
	  }
      }
      else { // y[t] == -1
	if (!IsUpperBounded(t)) // t\in I_down, y==-1: y[t]alpha[t] > -C
	  if (grad_[t] + y_grad_min > 0) { // y==-1...<=>  -grad_[t] < y_grad_min
	    y_grad_min = -grad_[t];
	    idx_j = t;
	  }
      }
    }
    out_j = idx_j; // j found
  }
  /* Find j using 2nd order working set selection scheme; need to calc kernels, but faster convergence */
  else if (wss_ == 2) {
    double K_ii = CalcKernelValue_(out_i, out_i);
    double opt_gain_max = -INFINITY;
    double grad_diff;
    double quad_kernel;
    double opt_gain = -INFINITY;
    for (t=0; t<n_active_; t++) {
      double K_it = CalcKernelValue_(out_i, t);
      double K_tt = CalcKernelValue_(t, t);
      if (y_[t] == 1) {
	if (!IsLowerBounded(t)) { // t\in I_down, y==1: y[t]alpha[t] > 0
	  // calculate y_grad_min for Stopping Criterion
	  if (grad_[t] < y_grad_min) // y==1
	    y_grad_min = grad_[t];
	  // find j
	  grad_diff = y_grad_max - grad_[t]; // max(y_i*grad_i) - y_t*grad_t
	  if (grad_diff > 0) {
	    quad_kernel = K_ii + K_tt - 2 * K_it;
	    if (quad_kernel > 0) // for positive definite kernels
	      opt_gain = ( grad_diff * grad_diff ) / quad_kernel; // actually ../2*quad_kernel
	    else // handle non-positive definite kernels
	      opt_gain = ( grad_diff * grad_diff ) / TAU;
	    // find max(opt_gain)
	    if (opt_gain > opt_gain_max) {
	      idx_j = t;
	      opt_gain_max = opt_gain;
	    }
	  }
	}
      }
      else { // y[t] == -1
	if (!IsUpperBounded(t)) {// t\in I_down, y==-1: y[t]alpha[t] > -C
	  // calculate y_grad_min for Stopping Criterion
	  if (grad_[t] + y_grad_min > 0) // y==-1, -grad_[t] < y_grad_min
	    y_grad_min = -grad_[t];
	  // find j
	  grad_diff = y_grad_max + grad_[t]; // max(y_i*grad_i) - y_t*grad_t
	  if (grad_diff > 0) {
	    quad_kernel = K_ii + K_tt - 2 * K_it;
	    if (quad_kernel > 0) // for positive definite kernels
	      opt_gain = ( grad_diff * grad_diff ) / quad_kernel; // actually ../2*quad_kernel
	    else // handle non-positive definite kernels
	      opt_gain = ( grad_diff * grad_diff ) / TAU;
	    // find max(opt_gain)
	    if (opt_gain > opt_gain_max) {
	      idx_j = t;
	      opt_gain_max = opt_gain;
	    }
	  }
	}
      }
    }
  }
  out_j = idx_j; // j found
  
  // Stopping Criterion check
  if (y_grad_max - y_grad_min <= accuracy_)
    return true; // optimality reached

  return false;
}



/**
* Search direction; Update gradient and alphas
* 
* @param: a working set (i,j) found by working set selection
*
*/
template<typename TKernel>
void HCY<TKernel>::UpdateGradientAlpha_(index_t i, index_t j) {
  index_t t;

  double a_i = alpha_[i]; // old alphas
  double a_j = alpha_[j];
  int y_i = y_[i];
  int y_j = y_[j];
  double C_i = GetC_(i); // can be Cp (for y==1) or Cn (for y==-1)
  double C_j = GetC_(j);

  // cached kernel values
  double K_ii, K_ij, K_jj;
  K_ii = CalcKernelValue_(i, i);
  K_ij = CalcKernelValue_(i, j);
  K_jj = CalcKernelValue_(j, j);

  double first_order_diff = y_i * grad_[i] - y_j * grad_[j];
  double second_order_diff = K_ii + K_jj - 2 * K_ij;
  if (second_order_diff <= 0) // handle non-positive definite kernels
    second_order_diff = TAU;
  double newton_step = first_order_diff / second_order_diff;

  /*
  double step_B, step_A;
  if (y_i == 1) {
    step_B = C_i - a_i;
  }
  else { // y_i == -1
    step_B = a_i; // 0-(-1)a_i
  }
  if (y_j == 1) {
    step_A = a_j;
  }
  else { // y_j == -1
    step_A = C_j - a_j; // (-1)a_j - (-C_j)
  }
  double min_step_temp = min(step_B, step_A);
  double min_step = min(min_step_temp, newton_step);
  */

  // Update alphas
  alpha_[i] = a_i + y_i * newton_step;
  alpha_[j] = a_j - y_j * newton_step;
  
  // Update alphas and handle bounds for updated alphas
  /*
  if (y_i != y_j) {
    double alpha_old_diff = a_i - a_j;
    if (alpha_old_diff > 0) {
      if (alpha_[i] < alpha_old_diff) {
	alpha_[i] = alpha_old_diff;
      }
      else if (alpha_[i] > C_i) {
	alpha_[i] = C_i;
      }
    }
    else { // alpha_old_diff <= 0
      if (alpha_[i] < 0) {
	alpha_[i] = 0;
      }
      else if (alpha_[i] > C_i + alpha_old_diff) {
	alpha_[i] = C_i + alpha_old_diff;
      }
    }
  }
  else { // y_i == y_j
    double alpha_old_sum = a_i + a_j;
    if (alpha_old_sum > C_i) {
      if (alpha_[i] < alpha_old_sum - C_i) {
	alpha_[i] =  alpha_old_sum - C_i;
      }
      else if (alpha_[i] > C_i) {
	alpha_[i] = C_i;
      }
    }
    else { //alpha_old_sum <= C_i
      if (alpha_[i] < 0) {
	alpha_[i] = 0;
      }
      else if (alpha_[i] > alpha_old_sum) {
	alpha_[i] = alpha_old_sum;
      }
    }
  }
  alpha_[j] = a_j + y_i * y_j * (a_i - alpha_[i]);
  */

  // Handle bounds for updated alphas
  if (y_i != y_j) {
    double alpha_old_diff = a_i - a_j;
    if (alpha_old_diff > 0) {
      if (alpha_[j] < 0) {
	alpha_[j] = 0;
	alpha_[i] = alpha_old_diff;
      }
    }
    else { // alpha_old_diff <= 0
      if (alpha_[i] < 0) {
	alpha_[i] = 0;
	alpha_[j] = - alpha_old_diff;
      }
    }
    if (alpha_old_diff > C_i - C_j) {
      if (alpha_[i] > C_i) {
	alpha_[i] = C_i;
	alpha_[j] = C_i - alpha_old_diff;
      }
    }
    else {
      if (alpha_[j] > C_j) {
	alpha_[j] = C_j;
	alpha_[i] = C_j + alpha_old_diff;
      }
    }
  }
  else { // y_i == y_j
    double alpha_old_sum = a_i + a_j;
    if (alpha_old_sum > C_i) {
      if (alpha_[i] > C_i) {
	alpha_[i] = C_i;
	alpha_[j] = alpha_old_sum - C_i;
      }
    }
    else {
      if (alpha_[j] < 0) {
	alpha_[j] = 0;
	alpha_[i] = alpha_old_sum;
      }
    }
    if (alpha_old_sum > C_j) {
      if (alpha_[j] > C_j) {
	alpha_[j] = C_j;
	alpha_[i] = alpha_old_sum - C_j;
      }
    }
    else {
      if (alpha_[i] < 0) {
	alpha_[i] = 0;
	alpha_[j] = alpha_old_sum;
      }
    }
  }

  // Update gradient
  double diff_i = alpha_[i] - a_i;
  double diff_j = alpha_[j] - a_j;
  for (t=0; t<n_active_; t++) {
    grad_[t] = grad_[t] - y_[t] * (y_[i] * diff_i * CalcKernelValue_(i, t) + y_[j] * diff_j * CalcKernelValue_(j, t));
  }

  bool ub_i = IsUpperBounded(i);
  bool ub_j = IsUpperBounded(j);
  
  // Update alpha active status
  UpdateAlphaStatus_(i);
  UpdateAlphaStatus_(j);

  // Update gradient_bar
  if( ub_i != IsUpperBounded(i) ) { // updated_alpha_i >= C
      if(ub_i) // old_alpha_i >= C, new_alpha_i < C
	for(t=0; t<n_used_alpha_; t++)
	  grad_bar_[t] = grad_bar_[t] - C_i * y_[i] * y_[t] * CalcKernelValue_(i, t);
      else // old_alpha_i < C, new_alpha_i >= C
	for(t=0; t<n_used_alpha_; t++)
	  grad_bar_[t] = grad_bar_[t] + C_i * y_[i] * y_[t] * CalcKernelValue_(i, t);
  }
  
  if( ub_j != IsUpperBounded(j) ) {
    if(ub_j) // old_alpha_j >= C, new_alpha_j < C
      for(t=0; t<n_used_alpha_; t++)
	grad_bar_[t] = grad_bar_[t] - C_j * y_[j] * y_[t] * CalcKernelValue_(j, t);
    else // old_alpha_j < C, new_alpha_j >= C
      for(t=0; t<n_used_alpha_; t++)
	grad_bar_[t] = grad_bar_[t] + C_j * y_[j] * y_[t] * CalcKernelValue_(j, t);
  }
  
}

/**
* Calcualte bias term
* 
* @return: the bias
*
*/
template<typename TKernel>
void HCY<TKernel>::CalcBias_() {
  double b;
  index_t n_free_alpha = 0;
  double ub = INFINITY, lb = -INFINITY, sum_free_yg = 0.0;
  
  for (index_t i=0; i<n_active_; i++){
    double yg = y_[i] * grad_[i];
      
    if (IsUpperBounded(i)) { // bounded: alpha_i >= C
      if(y_[i] == 1)
	lb = max(lb, yg);
      else
	ub = min(ub, yg);
    }
    else if (IsLowerBounded(i)) { // bounded: alpha_i <= 0
      if(y_[i] == -1)
	lb = max(lb, yg);
      else
	ub = min(ub, yg);
    }
    else { // free: 0< alpha_i <C
      n_free_alpha++;
      sum_free_yg += yg;
    }
  }
  
  if(n_free_alpha>0)
    b = sum_free_yg / n_free_alpha;
  else
    b = (ub + lb) / 2;
  
  bias_ = b;
}

/* Get SVM results:coefficients, number and indecies of SVs
*
* @param: sample indices of the training (sub)set in the total training set
* @param: support vector coefficients: alpha*y
* @param: bool indicators  FOR THE TRAINING SET: is/isn't a support vector
*
*/
template<typename TKernel>
void HCY<TKernel>::GetSV(ArrayList<index_t> &dataset_index, ArrayList<double> &coef, ArrayList<bool> &sv_indicator) {
  ArrayList<index_t> new_from_old; // it's used to retrieve the permuted new index from old index
  new_from_old.Init(max_n_alpha_);
  for (index_t i = 0; i < max_n_alpha_; i++) {
    new_from_old[active_set_[i]] = i;
  }
  if (learner_typeid_ == 0) {// SVM_C
    for (index_t ii = 0; ii < n_data_; ii++) {
      index_t i = new_from_old[ii]; // retrive the index of permuted vector
      if (alpha_[i] >= SMO_ALPHA_ZERO) { // support vectors found
	//printf("%f\n",alpha_[i] );
	coef.PushBack() = alpha_[i] * y_[i];
	sv_indicator[dataset_index[ii]] = true;
	n_sv_++;
      }
      else {
	coef.PushBack() = 0;
      }
    }
  }
  /*
  else if (learner_typeid_ == 1) {// SVM_R
    for (index_t i = 0; i < n_data_; i++) {
      double alpha_diff = -alpha_[i] + alpha_[i+n_data_]; // alpha_i^* - alpha_i
      if (fabs(alpha_diff) >= HCY_ALPHA_ZERO) { // support vectors found
	coef.PushBack() = alpha_diff; 
	sv_indicator[dataset_index[i]] = true;
	n_sv_++;
      }
      else {
	coef.PushBack() = 0;
      }
    }
  }
  */
}

#endif
