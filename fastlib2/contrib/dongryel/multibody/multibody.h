#ifndef MULTIBODY_H
#define MULTIBODY_H

#include <values.h>

#include "fastlib/fastlib.h"
#include "mlpack/kde/dataset_scaler.h"
#include "mlpack/series_expansion/farfield_expansion.h"
#include "mlpack/series_expansion/local_expansion.h"
#include "mlpack/series_expansion/series_expansion_aux.h"
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
    positive_force1_l_.SetZero();
    positive_force1_e_.SetZero();
    negative_force2_e_.SetZero();
    negative_force2_u_.SetZero();
    positive_force2_l_.SetZero();
    positive_force2_e_.SetZero();
    total_force_e_.SetZero();

    // Run and do timing for multitree multibody
    MTMultibodyBase(root_nodes, 0);
    PostProcessNaive_(root_);
  }

  /** @brief The main computation procedure.
   */
  void Compute(double relative_error) {
    
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
   
    // Set the probability requirement to 90 %, which gives the
    // standard z-score to be the following.
    z_score_ = 1.28;

    // Initialize intermediate computation spaces to zero.
    negative_force1_e_.SetZero();
    negative_force1_u_.SetZero();
    positive_force1_l_.SetZero();
    positive_force1_e_.SetZero();
    negative_force2_e_.SetZero();
    negative_force2_u_.SetZero();
    positive_force2_l_.SetZero();
    positive_force2_e_.SetZero();
    total_force_e_.SetZero();

    // Run and do timing for multitree multibody
    MTMultibody(root_nodes, total_num_tuples_);
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

    // More temporary variables initialization.
    non_leaf_indices_.Init(mkernel_.order());
    distmat_.Init(mkernel_.order(), mkernel_.order());
    exhaustive_indices_.Init(mkernel_.order());

    // Initialize space for computation values.
    negative_force1_e_.Init(data_.n_cols());
    negative_force1_u_.Init(data_.n_cols());
    positive_force1_l_.Init(data_.n_cols());
    positive_force1_e_.Init(data_.n_cols());
    negative_force2_e_.Init(data_.n_rows(), data_.n_cols());
    negative_force2_u_.Init(data_.n_rows(), data_.n_cols());
    positive_force2_l_.Init(data_.n_rows(), data_.n_cols());
    positive_force2_e_.Init(data_.n_rows(), data_.n_cols());
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

  /** @brief The total number of n-tuple  prunes.
   */
  int num_prunes_;

  /** @brief The total number of (n - 1) tuples.
   */
  double total_n_minus_one_num_tuples_;

  /** @brief The multibody kernel function.
   */
  MultibodyKernel mkernel_;

  /** @brief The probability requirement that the componentwise
   *         relative error bound holds.
   */
  double z_score_;

  /** @brief The accuracy requirement: componentwise relative error
   *         bound.
   */
  double relative_error_;

  /** The current list of non-leaf indices */
  ArrayList<int> non_leaf_indices_;

  /** @brief The temporary space for storing indices selected for
   *         exhaustive computation.
   */
  ArrayList<int> exhaustive_indices_;

  /** @brief The temporary space for storing pairwise distances.
   */
  Matrix distmat_;

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

  /** @brief The lower bound on the positive force due to the
   *         multibody potential on each particle. Each column is a
   *         force vector on each particle.
   */
  Vector positive_force1_l_;

  /** @brief The positive force due to the multibody potential on each
   *         particle. Each column is a force vector on each particle.
   */
  Vector positive_force1_e_;

  /** @brief The negative force due to the multibody potential on each
   *         particle. Each column is a force vector on each particle.
   */
  Matrix negative_force2_e_;

  /** @brief The upper bound on the negative force due to the
   *         multibody potential on each particle. Each column is a
   *         force vector on each particle.
   */
  Matrix negative_force2_u_;

  /** @brief The lower bound on the positive force due to the
   *         multibody potential on each particle. Each column is a
   *         force vector on each particle.
   */
  Matrix positive_force2_l_;

  /** @brief The positive force due to the multibody potential on each
   *         particle. Each column is a force vector on each particle.
   */
  Matrix positive_force2_e_;

  /** @brief The total estimated force due to the multibody potential
   *         on each particle. Each column is a force vector on each
   *         particle.
   */
  Matrix total_force_e_;
  

  /////////// Helper Functions //////////

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
  bool Prunable(ArrayList<TTree *> &nodes, double num_tuples);

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
  void MTMultibody(ArrayList<TTree *> &nodes, double num_tuples);

};

#include "multibody_impl.h"
#undef INSIDE_MULTIBODY_H

#endif
