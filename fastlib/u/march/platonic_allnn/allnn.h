/**
 * @file allnn.h
 *
 * This file contains a "platonic" example of FASTlib code for a
 * linkable library component.  It implements a rudimentary dual-tree
 * algorithm.  For more details, see accompanying file allnn_main.cc.
 *
 * @see allnn_main.cc
 */

// Header files should always have inclusion guards.  It's a good idea
// to "sign" these guards with the containing folder or project name,
// in the off chance that someone else has a file with the same name.
#ifndef PLATONIC_ALLNN_H
#define PLATONIC_ALLNN_H

// You can include all core FASTlib components at once as follows.
// Your "deplibs" entry in build.py should mirror your includes.
#include <fastlib/fastlib.h>

/**
 * A computation class for dual-tree and naive all-nearest-neighbors.
 *
 * This class builds trees for (assumed distinct) input query and
 * reference sets on Init.  The all-nearest-neighbors computation is
 * then performed by calling ComputeNeighbors or ComputeNaive.
 *
 * This class is only intended to compute once per instantiation.
 *
 * Example use:
 *
 * @code
 *   AllNN allnn;
 *   struct datanode* allnn_module;
 *   ArrayList<index_t> results;
 *
 *   allnn_module = fx_submodule(NULL, "allnn", "allnn");
 *   allnn.Init(query_set, reference_set, allnn_module);
 *   allnn.ComputeNeighbors(&results);
 * @endcode
 */
class AllNN {

  ////////// Nested Classes //////////////////////////////////////////

 private:
  /**
   * Additional data stored at each node of a BinarySpaceTree, used by
   * our QueryTree type.
   *
   * Dual-tree all-nearest-neighbors maintains an upper bound on
   * nearest neighbor distance for each query node.
   */
  class QueryStat {

   private:
    /**
     * An upper bound on nearest neighbor distance for points within
     * the node, to be modified after tree formation.
     */
    double max_distance_so_far_;

    // The object traversal macros establish a FASTlib-complient
    // storage class, providing many tools including pretty printing
    // and copy construction.  See base/otrav.h for more details.
    //
    // OT_DEF_BASIC is suitable for when you want pretty printing, but
    // don't need a special destructor (your object has no pointers).
    // Otherwise, you should use OT_DEF.
    OT_DEF_BASIC(QueryStat) {
      // Declare a non-pointer/array member variable to be traversed.
      // See base/otrav.h for other kinds of declarations.
      OT_MY_OBJECT(max_distance_so_far_);
    }

   public:
    double max_distance_so_far() {
      return max_distance_so_far_;
    }

    void set_max_distance_so_far(double new_dist) {
      max_distance_so_far_ = new_dist;
    }

    /**
     * An Init function required by BinarySpaceTree to build
     * statistics for a leaf node.
     *
     * All-nearest-neighbors fills statistics during computation, but
     * we must set it to an appropriate starting value here.
     */
    void Init(const Matrix& matrix, index_t start, index_t count) {
      // The bound starts at infinity
      max_distance_so_far_ = DBL_MAX;
    }

    /**
     * An Init function required by BinarySpaceTree to build
     * statistics from two child nodes.
     *
     * For all-nearest-neighbors, we reuse initialization for leaves.
     */
    void Init(const Matrix& matrix, index_t start, index_t count,
	      const QueryStat& left, const QueryStat& right) {
      Init(matrix, start, count);
    }

  }; /* class AllNNStat */

  // The tree directory defines several tools for the creation of
  // custom tree types, especially for kd-trees.  The DHrectBound<2>
  // gives us the normal kind of kd-tree bounding boxes, using the
  // 2-norm, and Matrix specifies the storage type of our data.

  /** kd-tree (binary with hrect bounds) with stats for queries. */
  typedef BinarySpaceTree<DHrectBound<2>, Matrix, QueryStat> QueryTree;
  /** kd-tree without stats for references. */
  typedef BinarySpaceTree<DHrectBound<2>, Matrix> ReferenceTree;

  ////////// Members Variables ///////////////////////////////////////

 private:
  /** Module used to pass parameters into the AllNN object. */
  struct datanode* module_;

  /** Copy of the query matrix given in Init. */
  Matrix queries_;
  /** Copy of the reference matrix given in Init. */
  Matrix references_;

  /** Root of a tree formed on queries_. */
  QueryTree* query_tree_;
  /** Root of a tree formed on references_. */
  ReferenceTree* reference_tree_;

  /** Maximum number of points in either tree's leaves. */
  index_t leaf_size_;

  /** Permutation mapping indices of queries_ to original order. */
  ArrayList<index_t> old_from_new_queries_;
  /** Permutation mapping indices of references_ to original order. */
  ArrayList<index_t> old_from_new_references_;

  /**
   * Candidate nearest neighbor distances, modified during
   * compuatation.  Later, true nearest neighbor distances.
   */
  Vector neighbor_distances_;
  /**
   * Candidate nearest neighbor indicies, modified during
   * compuatation.  Later, true nearest neighbor indices.
   */
  ArrayList<index_t> neighbor_indices_;

  /** Number of node-pairs pruned by the dual-tree algorithm. */
  index_t number_of_prunes_;

  /** Debug-mode test whether an AllNN object is initialized. */
  bool initialized_;
  /** Debug-mode test whether an AllNN object is used twice. */
  bool already_used_;


  ////////// Constructors ////////////////////////////////////////////

  // It is easy to accidently call copy constructors in C++.  The most
  // common mistake is to define functions with object arguments
  // passed by value:
  //
  //   void foo(HugeObject x) {...}
  //
  // This recursively copies each member variable of the object, which
  // would be disasterous, for instance, if stored query and reference
  // matrices are huge.  Core FASTlib components usually mitigate this
  // by passing objects by const reference or by pointer:
  //
  //   void bar(const HugeObject& x, HugeObject* y) {...}
  //
  // Non-const pointers are used when the outside object is modified.
  //
  // The following disables copy construction and assignment for
  // objects of this class, which prevents functions like foo from
  // compiling, saving you from poor performance and strange bugs.
  FORBID_ACCIDENTAL_COPIES(AllNN);

 public:
  // Default constructors should be kept very simple and should never
  // allocate memory.  Their two responsibilities are to ensure that
  // it's safe to destroy the object without having otherwise used it
  // (e.g. to set pointers to NULL) and to poison memory when in debug
  // mode with BIG_BAD_NUMBER = 2146666666 = NaN as a double and
  // BIG_BAD_POINTER = 0xdeadbeef.
  AllNN() {
    query_tree_ = NULL;
    reference_tree_ = NULL;

    DEBUG_POISON_PTR(module_);
    DEBUG_ONLY(leaf_size_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(number_of_prunes_ = BIG_BAD_NUMBER);

    DEBUG_ONLY(initialized_ = false);
    DEBUG_ONLY(already_used_ = false);
  }

  // Note that we don't delete the fx module; it's managed externally.
  ~AllNN() {
    if (query_tree_ != NULL) {
      delete query_tree_;
    }
    if (reference_tree_ != NULL) {
      delete reference_tree_;
    }
  }

  ////////// Helper Functions ////////////////////////////////////////

  /**
   * Computes the minimum squared distance between the bounding boxes
   * of two nodes.
   */
  double MinNodeDistSq_(QueryTree* query_node, ReferenceTree* reference_node) {
    return query_node->bound().MinDistanceSq(reference_node->bound());
  }


  /**
   * Performs exhaustive computation between two nodes.
   *
   * Note that naive also makes use of this function.
   */
  void GNPBaseCase_(QueryTree* query_node, ReferenceTree* reference_node) {

    // Debug checks should be used frequently.  They incur no overhead
    // when compiled in --mode=fast and very little otherwise.

    /* Make sure we didn't try to split children */
    DEBUG_ASSERT(query_node != NULL);
    DEBUG_ASSERT(reference_node != NULL);

    /* Make sure we should be in the base case */
    DEBUG_WARN_IF(!query_node->is_leaf());
    DEBUG_WARN_IF(!reference_node->is_leaf());

    /* Used to find the query node's new upper bound */
    double max_nearest_neighbor_distance = -1.0;

    /* Loop over all query-reference pairs */

    // Trees don't store their points, but instead give index ranges.
    // To make this feasible, they have to rearrange their input
    // matrices, which is why we were sure to make copies.
    for (index_t query_index = query_node->begin();
        query_index < query_node->end(); query_index++) {

      // MakeColumnVector aliases (i.e. points to but does not copy) a
      // column from the matrix.
      //
      // A brief aside: BLAS/LAPACK is coded in Fortran and thus
      // expects matrices to be column major.  We side with their
      // format for compatiblity, and accordingly, it is more cache
      // friendly to store data points along columns, as is common in
      // statistics, than along rows, as is more conventional.
      Vector query_point;
      queries_.MakeColumnVector(query_index, &query_point);

      // It's not terrible form to leave TODO statements in code you
      // intend to maintain, especially when coding under a deadline.
      // These are easy to search for, though for some reason, Garry
      // was more partial to "where's WALDO".  More memorable, maybe?

      /* TODO: try pruning query points vs reference node */

      for (index_t reference_index = reference_node->begin();
          reference_index < reference_node->end(); reference_index++) {

        Vector reference_point;
        references_.MakeColumnVector(reference_index, &reference_point);

        // BLAS can perform many vectors ops more quickly than C/C++.
        double distance =
            la::DistanceSqEuclidean(query_point, reference_point);

	/* Record points found to be closer than the best so far */
        if (distance < neighbor_distances_[query_index]) {
          neighbor_distances_[query_index] = distance;
          neighbor_indices_[query_index] = reference_index;
        }

      } /* for reference_index */

      /* Find the upper bound nn distance for this node */
      if (neighbor_distances_[query_index] > max_nearest_neighbor_distance) {
        max_nearest_neighbor_distance = neighbor_distances_[query_index];
      }

    } /* for query_index */

    /* Update the upper bound nn distance for the node */
    query_node->stat().set_max_distance_so_far(max_nearest_neighbor_distance);

  } /* GNPBaseCase_ */


  /**
   * Performs one node-node comparison in the GNP algorithm and
   * recurses upon all child combinations if no prune.
   */
  void GNPRecursion_(QueryTree* query_node, ReferenceTree* reference_node,
		     double lower_bound_distance) {

    /* Make sure we didn't try to split children */
    DEBUG_ASSERT(query_node != NULL);
    DEBUG_ASSERT(reference_node != NULL);

    // The following asserts equality of two doubles and prints their
    // values if it fails.  Note that this *isn't* a particularly fast
    // debug check, though; it negates the benefit of passing ahead a
    // precomputed distance entirely.  That's why we have --mode=fast.

    /* Make sure the precomputed bounding information is correct */
    DEBUG_SAME_DBL(lower_bound_distance,
        MinNodeDistSq_(query_node, reference_node));

    if (lower_bound_distance > query_node->stat().max_distance_so_far()) {

      /*
       * A reference node with lower-bound distance greater than this
       * query node's upper-bound nearest neighbor distance cannot
       * contribute a reference closer than any of the queries'
       * current neighbors, hence prune
       */
      number_of_prunes_++;

    } else if (query_node->is_leaf() && reference_node->is_leaf()) {

      /* Cannot further split leaves, so process exhaustively */
      GNPBaseCase_(query_node, reference_node);

    } else if (query_node->is_leaf()) {

      /* Query node's a leaf, but we can split references */
      double left_distance =
	  MinNodeDistSq_(query_node, reference_node->left());
      double right_distance =
	  MinNodeDistSq_(query_node, reference_node->right());

      /*
       * Nearer reference node more likely to contribute neighbors
       * (and thus tighten bounds), so visit it first
       */
      if (left_distance < right_distance) {
        GNPRecursion_(query_node, reference_node->left(), left_distance);
        GNPRecursion_(query_node, reference_node->right(), right_distance);
      } else {
        GNPRecursion_(query_node, reference_node->right(), right_distance);
        GNPRecursion_(query_node, reference_node->left(), left_distance);
      }

    } else if (reference_node->is_leaf()) {

      /* Reference node's a leaf, but we can split queries */
      double left_distance =
          MinNodeDistSq_(query_node->left(), reference_node);
      double right_distance =
          MinNodeDistSq_(query_node->right(), reference_node);

      /* Order of recursion does not matter */
      GNPRecursion_(query_node->left(), reference_node, left_distance);
      GNPRecursion_(query_node->right(), reference_node, right_distance);

      /* Update upper bound nn distance base new child bounds */
      query_node->stat().set_max_distance_so_far(
          max(query_node->left()->stat().max_distance_so_far(),
              query_node->right()->stat().max_distance_so_far()));

    } else {

      /*
       * Neither node is a leaf, so split both
       *
       * The order we process the query node's children doesn't
       * matter, but for each we should visit their nearer reference
       * node first.
       */

      double left_distance =
          MinNodeDistSq_(query_node->left(), reference_node->left());
      double right_distance =
          MinNodeDistSq_(query_node->left(), reference_node->right());

      if (left_distance < right_distance) {
        GNPRecursion_(query_node->left(),
            reference_node->left(), left_distance);
        GNPRecursion_(query_node->left(),
            reference_node->right(), right_distance);
      } else {
        GNPRecursion_(query_node->left(),
            reference_node->right(), right_distance);
        GNPRecursion_(query_node->left(),
            reference_node->left(), left_distance);
      }

      left_distance =
          MinNodeDistSq_(query_node->right(), reference_node->left());
      right_distance =
          MinNodeDistSq_(query_node->right(), reference_node->right());

      if (left_distance < right_distance) {
        GNPRecursion_(query_node->right(),
            reference_node->left(), left_distance);
        GNPRecursion_(query_node->right(),
            reference_node->right(), right_distance);
      } else {
        GNPRecursion_(query_node->right(),
            reference_node->right(), right_distance);
        GNPRecursion_(query_node->right(),
            reference_node->left(), left_distance);
      }

      /* Update upper bound nn distance base new child bounds */
      query_node->stat().set_max_distance_so_far(
          max(query_node->left()->stat().max_distance_so_far(),
              query_node->right()->stat().max_distance_so_far()));

    }

  } /* GNPRecursion_ */

  ////////// Public Functions ////////////////////////////////////////

  // Note that we initialize with const references below to keep from
  // copying data until we want to.  By the way, which side you put
  // the &'s and *'s on is on the level of deep-seated religious
  // belief: some people get real angry if you defy them, but you're
  // really no worse a person either way.  The compiler is agnostic.

  /**
   * Read parameters, copy data into the class, and build the trees.
   */
  void Init(const Matrix& queries_in, const Matrix& references_in,
	    struct datanode* module_in) {

    // It's a good idea to make sure the object isn't initialized a
    // second time, as this is almost certainly mistaken.
    DEBUG_ASSERT(initialized_ == false);
    DEBUG_ONLY(initialized_ = true);

    module_ = module_in;

    /* The data sets need to have the same dimensionality */
    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());

    /* Copy input matrices as they will be rearranged */
    queries_.Copy(queries_in);
    references_.Copy(references_in);

    leaf_size_ = fx_param_int(module_, "leaf_size", 20);
    DEBUG_ASSERT(leaf_size_ > 0);

    // Timers are another handy tool provided by FASTexec.  These are
    // emitted automatically once you call fx_done.
    fx_timer_start(module_, "tree_building");

    // Input matrices are rearranged to an in-order traversal of
    // either tree.  To help in iterpretting results, the third
    // argument is Init'd to a mapping from rearranged indices to the
    // original order.  The fourth argument, if provided, would
    // initialize the reverse of said.

    /* Build the trees */
    query_tree_ = tree::MakeKdTreeMidpoint<QueryTree>(
	queries_, leaf_size_, &old_from_new_queries_, NULL);
    reference_tree_ = tree::MakeKdTreeMidpoint<ReferenceTree>(
        references_, leaf_size_, &old_from_new_references_, NULL);

    // While we don't make use of this here, it is possible to start
    // timers after stopping them.  They continue where they left off.
    fx_timer_stop(module_, "tree_building");

    /* Ready the list of nearest neighbor candidates to be filled. */
    neighbor_indices_.Init(queries_.n_cols());

    /* Ready the vector of upper bound nn distances for use. */
    neighbor_distances_.Init(queries_.n_cols());
    neighbor_distances_.SetAll(DBL_MAX);

    number_of_prunes_ = 0;

  } /* Init */


  /**
   * Initializes the AllNN structure for naive computation.
   *
   * We have no need to build trees for naive.
   */
  void InitNaive(const Matrix& queries_in, const Matrix& references_in,
		 struct datanode* module_in){

    DEBUG_ASSERT(initialized_ == false);
    DEBUG_ONLY(initialized_ = true);

    module_ = module_in;

    /* The data sets need to have the same dimensionality */
    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());

    /* Copy input matrices */
    queries_.Copy(queries_in);
    references_.Copy(references_in);

    /*
     * A bit of a trick so we can still use BaseCase_: we'll expand
     * the leaf size so that our trees only have one node.
     */
    leaf_size_ = max(queries_.n_cols(), references_.n_cols());

    /* Build the (single node) trees */
    query_tree_ = tree::MakeKdTreeMidpoint<QueryTree>(
	queries_, leaf_size_, &old_from_new_queries_, NULL);
    reference_tree_ = tree::MakeKdTreeMidpoint<ReferenceTree>(
        references_, leaf_size_, &old_from_new_references_, NULL);

    /* Ready the list of nearest neighbor candidates to be filled. */
    neighbor_indices_.Init(queries_.n_cols());

    /* Ready the vector of upper bound nn distances for use. */
    neighbor_distances_.Init(queries_.n_cols());
    neighbor_distances_.SetAll(DBL_MAX);

    number_of_prunes_ = 0;

  } /* InitNaive */


  /**
   * Computes the nearest neighbors and stores them in results if
   * provided.
   */
  void ComputeNeighbors(ArrayList<index_t>* results) {

    // In addition to confirming the object's been initialized, we
    // want to make sure we aren't asking it to compute a second time.
    DEBUG_ASSERT(initialized_ == true);
    DEBUG_ASSERT(already_used_ == false);
    DEBUG_ONLY(already_used_ = true);

    fx_timer_start(module_, "dual_tree_computation");

    /* Start recursion on the roots of either tree */
    GNPRecursion_(query_tree_, reference_tree_,
        MinNodeDistSq_(query_tree_, reference_tree_));

    fx_timer_stop(module_, "dual_tree_computation");

    // Save the total number of prunes to the FASTexec module; this
    // will printed after calling fx_done or can be read back later.
    fx_format_result(module_, "number_of_prunes", "%d", number_of_prunes_);

    if (results) {
      EmitResults(results);
    }

  } /* ComputeNeighbors */


  /**
   * Computes the nearest neighbors naively.
   */
  void ComputeNaive(ArrayList<index_t>* results) {

    DEBUG_ASSERT(initialized_ == true);
    DEBUG_ASSERT(already_used_ == false);
    DEBUG_ONLY(already_used_ = true);

    fx_timer_start(module_, "naive_time");

    /* BaseCase_ on the roots is equivalent to naive */
    GNPBaseCase_(query_tree_, reference_tree_);

    fx_timer_stop(module_, "naive_time");

    if (results) {
      EmitResults(results);
    }

  } /* ComputeNaive */

  /**
   * Initialize and fill an ArrayList of results.
   */
  void EmitResults(ArrayList<index_t>* results) {

    DEBUG_ASSERT(initialized_ == true);

    results->Init(neighbor_indices_.size());

    /* Map the indices back from how they have been permuted. */
    for (index_t i = 0; i < neighbor_indices_.size(); i++) {
      (*results)[old_from_new_queries_[i]] =
          old_from_new_references_[neighbor_indices_[i]];
    }

  } /* EmitResults */

}; /* class AllNN */

#endif /* PLATONIC_ALLNN_H */
