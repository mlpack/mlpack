#ifndef MULTIBODY_H
#define MULTIBODY_H

#include <values.h>

#include "fastlib/fastlib.h"
#include "mlpack/kde/dataset_scaler.h"
#include "mlpack/series_expansion/farfield_expansion.h"
#include "mlpack/series_expansion/local_expansion.h"
#include "mlpack/series_expansion/series_expansion_aux.h"
#include "mlpack/allknn/allknn.h"
#include "contrib/nvasil/allkfn/allkfn.h"
#include "multibody_kernel.h"

#define INSIDE_MULTIBODY_H
#include "multibody_stat.h"



template<typename TMultibodyKernel, typename TTree>
class MultitreeMultibody {

  FORBID_ACCIDENTAL_COPIES(MultitreeMultibody);

 public:
  
  typedef TMultibodyKernel MultibodyKernel;

  ////////// Constructor/Destructor //////////

  /** @brief The default constructor.
   */
  MultitreeMultibody() {
  }
  
  /** @brief The default destructor.
   */
  ~MultitreeMultibody() { 
    delete root_;
  }

  ////////// User-level Functions //////////

  /** @brief Copies the computed vectors.
   */
  void get_force_vectors(Matrix *destination) {
    destination->Copy(total_force_e_);
  }

  /** @brief The computes the maximum L1 norm error of approximations
   *         against the true values (both relative and absolute).
   */
  static void MaxL1NormError(const Matrix &approximations,
			     const Matrix &exact_values, 
			     double *max_relative_l1_norm_error,
			     int *relative_error_under_threshold,
			     double *max_absolute_l1_norm_error,
			     int *absolute_error_under_threshold,
			     double relative_error, double threshold) {
    
    *max_relative_l1_norm_error = 0;
    *relative_error_under_threshold = 0;
    *max_absolute_l1_norm_error = 0;
    *absolute_error_under_threshold = 0;

    for(index_t i = 0; i < approximations.n_cols(); i++) {
      
      // Get both vectors.
      const double *approximated_vector = approximations.GetColumnPtr(i);
      const double *exact_vector = exact_values.GetColumnPtr(i);
      double error_l1_norm = 0;
      double exact_l1_norm = 0;
      
      for(index_t j = 0; j < approximations.n_rows(); j++) {
	error_l1_norm += fabs(approximated_vector[j] - exact_vector[j]);
	exact_l1_norm += fabs(exact_vector[j]);
      }
      *max_relative_l1_norm_error = 
	std::max(*max_relative_l1_norm_error, error_l1_norm / exact_l1_norm);
      if(error_l1_norm / exact_l1_norm <= relative_error) {
	(*relative_error_under_threshold)++;
      }
      *max_absolute_l1_norm_error =
	std::max(*max_absolute_l1_norm_error, error_l1_norm);
      if(error_l1_norm <= threshold) {
	(*absolute_error_under_threshold)++;
      }
    }
  }

  /** @brief The main naive computation procedure.
   */
  void NaiveCompute() {

    ArrayList<TTree *> root_nodes;
    root_nodes.Init(mkernel_.order());

    // Set the root node pointers for starting the computation.
    for(index_t i = 0; i < mkernel_.order(); i++) {
      root_nodes[i] = root_;
    }
    
    total_num_tuples_ = math::BinomialCoefficient(data_.n_cols(),
						  mkernel_.order());
    total_n_minus_one_num_tuples_ = 
      math::BinomialCoefficient(data_.n_cols() - 1, mkernel_.order() - 1);

    // Initialize intermediate computation spaces to zero.
    negative_force1_e_.SetZero();
    negative_force1_u_.SetZero();
    negative_force1_used_error_.SetZero();
    positive_force1_l_.SetZero();
    positive_force1_e_.SetZero();
    positive_force1_used_error_.SetZero();
    negative_force2_e_.SetZero();
    negative_force2_u_.SetZero();
    negative_force2_used_error_.SetZero();
    positive_force2_l_.SetZero();
    positive_force2_e_.SetZero();
    positive_force2_used_error_.SetZero();
    n_pruned_.SetZero();
    total_force_e_.SetZero();

    // Run and do timing for multitree multibody
    MTMultibodyBase(root_nodes, 0);
    PostProcess(root_);
  }

  /** @brief The main computation procedure.
   */
  void Compute(double relative_error, double threshold, 
	       double centered_percentile_coverage, double probability) {
    
    ArrayList<TTree *> root_nodes;
    root_nodes.Init(mkernel_.order());

    // Set the root node pointers for starting the computation.
    for(index_t i = 0; i < mkernel_.order(); i++) {
      root_nodes[i] = root_;
    }

    num_prunes_ = 0;
    total_num_tuples_ = math::BinomialCoefficient(data_.n_cols(),
						  mkernel_.order());
    total_n_minus_one_num_tuples_ = 
      math::BinomialCoefficient(data_.n_cols() - 1, mkernel_.order() - 1);

    relative_error_ = relative_error;

    // Convert the absolute error threshold to an internal tolerance
    // level. We divide by 4 here because the force vector is composed
    // of 4 different parts and we bound errors on each separately.
    threshold_ = threshold / 4.0;

    // Initialize intermediate computation spaces to zero.
    negative_force1_e_.SetZero();
    negative_force1_u_.SetZero();
    negative_force1_used_error_.SetZero();
    positive_force1_l_.SetZero();
    positive_force1_e_.SetZero();
    positive_force1_used_error_.SetZero();
    negative_force2_e_.SetZero();
    negative_force2_u_.SetZero();
    negative_force2_used_error_.SetZero();
    positive_force2_l_.SetZero();
    positive_force2_e_.SetZero();
    positive_force2_used_error_.SetZero();
    n_pruned_.SetZero();
    total_force_e_.SetZero();


    // Set the sample multiple.
    sample_multiple_ = 10;

    // Compute the coverage probability table.
    double lower_percentile = (100 - centered_percentile_coverage) / 
      100.0;
    for(index_t j = 0; j < coverage_probabilities_.length(); j++) {
      coverage_probabilities_[j] = 
	mkernel_.OuterConfidenceInterval
	(ceil(total_num_tuples_), ceil(sample_multiple_ * (j + 1)), 1,
	 ceil(total_num_tuples_ * lower_percentile));
    }
    coverage_probabilities_.PrintDebug();

    // Run and do timing for multitree multibody
    MTMultibody(root_nodes, total_num_tuples_, probability_);
    PostProcess(root_);

    printf("%d n-tuples have been pruned...\n", num_prunes_);
  }

  /** @brief Initialize the kernel object, and build the tree.
   */
  void Init(double bandwidth) {

    const char *fname = fx_param_str(NULL, "data", NULL);
    int leaflen = fx_param_int(NULL, "leaflen", 20);
     
    // Read in the dataset and translate it to be in the positive
    // quadrant. This is due to the limitation of the pruning
    // rule. Then build a kd-tree.
    fx_timer_start(NULL, "tree_d");
    Dataset dataset_;
    dataset_.InitFromFile(fname);
    data_.Own(&(dataset_.matrix()));
    DatasetScaler::TranslateDataByMin(data_, data_, true);
    
    root_ = tree::MakeKdTreeMidpoint<TTree>(data_, leaflen, 
					    (GenVector<index_t> *)NULL, NULL);

    // Initialize the multibody kernel.
    mkernel_.Init(bandwidth);

    fx_timer_stop(NULL, "tree_d");

    // Compute the (k - 1) nearest neighbors of each point, where k is
    // the order of the multibody kernel.
    AllkNN all_knn;
    all_knn.Init(data_, 20, 1);
    ArrayList<index_t> resulting_neighbors;
    ArrayList<double> squared_distances;

    fx_timer_start(fx_root, "k_nn_dsqd_initialization");
    all_knn.ComputeNeighbors(&resulting_neighbors, &squared_distances);
    fx_timer_stop(fx_root, "k_nn_dsqd_initialization");

    // Compute the (k - 1) farthest neighbors of each point, where k
    // is the order of the multibody kernel.
    AllkFN all_kfn;
    all_kfn.Init(data_, 20, 1);
    ArrayList<index_t> resulting_farthest_neighbors;
    ArrayList<double> squared_farthest_distances;

    fx_timer_start(fx_root, "k_fn_dsqd_initialization");
    all_kfn.ComputeNeighbors(&resulting_farthest_neighbors, 
			     &squared_farthest_distances);
    PreProcess_(root_, squared_distances, squared_farthest_distances);
    fx_timer_stop(fx_root, "k_fn_dsqd_initialization");

    // More temporary variables initialization.
    non_leaf_indices_.Init(mkernel_.order());
    distmat_.Init(mkernel_.order(), mkernel_.order());
    coverage_probabilities_.Init(20);    
    exhaustive_indices_.Init(mkernel_.order());
    num_leave_one_out_tuples_.Init(mkernel_.order());

    // Initialize space for computation values.
    negative_force1_e_.Init(data_.n_cols());
    negative_force1_u_.Init(data_.n_cols());
    negative_force1_used_error_.Init(data_.n_cols());
    positive_force1_l_.Init(data_.n_cols());
    positive_force1_e_.Init(data_.n_cols());
    positive_force1_used_error_.Init(data_.n_cols());
    negative_force2_e_.Init(data_.n_rows(), data_.n_cols());
    negative_force2_u_.Init(data_.n_rows(), data_.n_cols());
    negative_force2_used_error_.Init(data_.n_cols());
    positive_force2_l_.Init(data_.n_rows(), data_.n_cols());
    positive_force2_e_.Init(data_.n_rows(), data_.n_cols());
    positive_force2_used_error_.Init(data_.n_cols());
    n_pruned_.Init(data_.n_cols());
    total_force_e_.Init(data_.n_rows(), data_.n_cols());
  }

  /** @brief Outputs the force vectors to the file.
   */
  void PrintDebug(bool naive_print_out) {
    
    FILE *stream = NULL;

    if(naive_print_out) {
      stream = fopen("naive_force_vectors.txt", "w+");
    }
    else {
      stream = fopen("force_vectors.txt", "w+");
    }
    for(index_t q = 0; q < data_.n_cols(); q++) {
      for(index_t d = 0; d < data_.n_rows(); d++) {
	fprintf(stream, "%g ", total_force_e_.get(d, q));
      }
      fprintf(stream, "\n");
    }
  }

private:

  ////////// Private Member Variables //////////

  /** @brief The total number of n-tuples.
   */
  double total_num_tuples_;

  /** @brief The unit multiple of the number of samples to take.
   */
  int sample_multiple_;

  /** @brief The total number of n-tuple  prunes.
   */
  int num_prunes_;

  /** @brief The total number of (n - 1) tuples.
   */
  double total_n_minus_one_num_tuples_;

  /** @brief The multibody kernel function.
   */
  MultibodyKernel mkernel_;

  /** @brief The probability requirement.
   */
  double probability_;

  /** @brief The accuracy requirement: componentwise relative error
   *         bound.
   */
  double relative_error_;

  /** @brief The absolute threshold: quantities below this will be
   *         approximated with absolute error.
   */
  double threshold_;

  /** @brief The current list of non-leaf indices. */
  ArrayList<int> non_leaf_indices_;

  /** @brief The temporary space for storing indices selected for
   *         exhaustive computation.
   */
  ArrayList<int> exhaustive_indices_;
 
  /** @brief The number of leave-one-out (n - 1) tuples for each node
   *         in the n-tuple sequence.
   */
  Vector num_leave_one_out_tuples_;

  /** @brief The temporary space for storing pairwise distances.
   */
  Matrix distmat_;

  /** @brief The probabiliy lookup table for sample statistics
   *         coverage.
   */
  Vector coverage_probabilities_;

  /** @brief The pointer to the root of the tree.
   */
  TTree *root_;

  /** @brief The dataset for the tree.
   */
  Matrix data_;

  /** @brief The negative force due to the multibody potential on each
   *         particle. Each column is a force vector on each particle.
   */
  Vector negative_force1_e_;

  /** @brief The upper bound on the negative force due to the
   *         multibody potential on each particle. Each column is a
   *         force vector on each particle.
   */
  Vector negative_force1_u_;

  Vector negative_force1_used_error_;

  /** @brief The lower bound on the positive force due to the
   *         multibody potential on each particle. Each column is a
   *         force vector on each particle.
   */
  Vector positive_force1_l_;

  /** @brief The positive force due to the multibody potential on each
   *         particle. Each column is a force vector on each particle.
   */
  Vector positive_force1_e_;

  Vector positive_force1_used_error_;

  /** @brief The negative force due to the multibody potential on each
   *         particle. Each column is a force vector on each particle.
   */
  Matrix negative_force2_e_;

  /** @brief The upper bound on the negative force due to the
   *         multibody potential on each particle. Each column is a
   *         force vector on each particle.
   */
  Matrix negative_force2_u_;

  Vector negative_force2_used_error_;

  /** @brief The lower bound on the positive force due to the
   *         multibody potential on each particle. Each column is a
   *         force vector on each particle.
   */
  Matrix positive_force2_l_;

  /** @brief The positive force due to the multibody potential on each
   *         particle. Each column is a force vector on each particle.
   */
  Matrix positive_force2_e_;

  Vector positive_force2_used_error_;

  /** @brief The number of pruned (n - 1) tuples for each particle.
   */
  Vector n_pruned_;

  /** @brief The total estimated force due to the multibody potential
   *         on each particle. Each column is a force vector on each
   *         particle.
   */
  Matrix total_force_e_;
  

  /////////// Helper Functions //////////

  void PreProcess_(TTree *node, const ArrayList<double> &knn_dsqds,
		   const ArrayList<double> &kfn_dsqds);

  void RefineStatistics_(int point_index, TTree *destination_node);

  void RefineStatistics_(TTree *internal_node);

  /** @brief Adds the postponed information from a leaf node to the
   *         point's contribution.
   */
  void AddPostponed(TTree *node, index_t destination);

  /** @brief Adds the postponed information from a node to another.
   */
  void AddPostponed(TTree *source_node, TTree *destination_node);

  /** @brief Tests whether node a is an ancestor node of node b.
   */
  int as_indexes_strictly_surround_bs(TTree *a, TTree *b);

  /** @brief Compute the total number of n-tuples by recursively
   *         splitting up the i-th node
   */
  double two_ttn(int b, ArrayList<TTree *> &nodes, int i);

  /** @brief Compute the total number of n-tuples.
   */
  double ttn(int b, ArrayList<TTree *> &nodes);

  /** @brief Heuristic for node splitting - find the node with most
   *         points.
   */
  int FindSplitNode(ArrayList<TTree *> &nodes);

  /** Pruning rule */
  bool Prunable(ArrayList<TTree *> &nodes, double num_tuples,
		double required_probability, bool *pruned_with_exact_method);

  /** @brief The base exhaustive computations.
   */
  void MTMultibodyBase(const ArrayList<TTree *> &nodes, int level);
  
  /** @brief The post-processing function to push down all unclaimed
   *         approximations.
   */
  void PostProcess(TTree *node);  

  /** @brief The post-processing function to push down all unclaimed
   *         approximations (for naive algorithm).
   */
  void PostProcessNaive_(TTree *node);

  /** @brief The main multitree recursion.
   */
  bool MTMultibody(ArrayList<TTree *> &nodes, double num_tuples,
		   double required_probability);

};

#include "multibody_impl.h"
#undef INSIDE_MULTIBODY_H

#endif
