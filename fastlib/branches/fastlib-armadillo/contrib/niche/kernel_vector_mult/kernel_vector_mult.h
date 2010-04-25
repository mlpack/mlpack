/**
 * @file kernel_vector_mult.h
 *
 * This file is derived from the interspectacular platonic allnn code.
 * It implements a rudimentary dual-tree algorithm for fast kernel
 * matrix vector multiplication using a simple zero-support pruning
 * rule for Epanechnikov kernel.
 *
 * @see kernel_vector_mult.cc (to be coded!)
 */

// Header files should always have inclusion guards.  It's a good idea
// to "sign" these guards with the containing folder or project name,
// in the off chance that someone else has a file with the same name.
#ifndef KERNEL_VECTOR_MULT_H
#define KERNEL_VECTOR_MULT_H

// You can include all core FASTlib components at once as follows.
// Your "deplibs" entry in build.py should mirror your includes.
#include <fastlib/fastlib.h>

/**
 * A computation class for dual-tree kernel matrix vector multiplication
 *
 * This class builds 2 monochromatic trees from a reference set on Init.
 * The kernel matrix vector multiplication computation is then performed
 * by calling ComputeProduct.
 *
 * This class is only intended to compute once per instantiation.
 *
 * Example use:
 *
 * @code
 *   KernelVectorMult kernel_vector_mult;
 *   struct datanode* kernel_vector_mult_module;
 *   ArrayList<index_t> results;
 *
 *   kernel_vector_mult_module = fx_submodule(NULL, "kernel_vector_mult");
 *   kernel_vector_mult.Init(query_set, reference_set, kernel_vector_mult_module);
 *   kernel_vector_mult.ComputeKernelMatrixVectorMultiplication(&results);
 * @endcode
 */
class KernelVectorMult {

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

    // The object traversal macros establish a FASTlib-complient
    // storage class, providing many tools including pretty printing
    // and copy construction.  See base/otrav.h for more details.
    //
    // OT_DEF_BASIC is suitable for when you want pretty printing, but
    // don't need a special destructor (your object has no pointers).
    // Otherwise, you should use OT_DEF.
    OBJECT_TRAVERSAL_SHALLOW(QueryStat) {
      // Declare a non-pointer/array member variable to be traversed.
      // See base/otrav.h for other kinds of declarations.
    }

   public:


    /**
     * An Init function required by BinarySpaceTree to build
     * statistics for a leaf node.
     *
     * Kernel matrix vector multiplication fills statistics during computation, but
     * we must set it to an appropriate starting value here.
     */
    void Init(const Matrix& matrix, index_t start, index_t count) {
      // nothing to do here
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

  }; /* class QueryStat */

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
  /** Module used to pass parameters into the KernelVectorMul object. */
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


  // kernel function
  GaussianKernel kernel;

  double cutoff_dist_;




  /**
   * Weighted sums computed so far for each query point, modified during computation.
   * Later, true nearest weighted kernel sum for each query point.
   */
  Vector weighted_sums_;

  /**
   * The weights vector v such that we compute K * v in the overall computation
   */
  Vector weights_;

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
  FORBID_ACCIDENTAL_COPIES(KernelVectorMult);

 public:
  // Default constructors should be kept very simple and should never
  // allocate memory.  Their two responsibilities are to ensure that
  // it's safe to destroy the object without having otherwise used it
  // (e.g. to set pointers to NULL) and to poison memory when in debug
  // mode with BIG_BAD_NUMBER = 2146666666 = NaN as a double and
  // BIG_BAD_POINTER = 0xdeadbeef.
  KernelVectorMult() {
    query_tree_ = NULL;
    reference_tree_ = NULL;

    DEBUG_POISON_PTR(module_);
    DEBUG_ONLY(leaf_size_ = BIG_BAD_NUMBER);
    DEBUG_ONLY(number_of_prunes_ = BIG_BAD_NUMBER);

    DEBUG_ONLY(initialized_ = false);
    DEBUG_ONLY(already_used_ = false);
  }

  // Note that we don't delete the fx module; it's managed externally.
  ~KernelVectorMult() {
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

	weighted_sums_[query_index] +=
	  weights_[reference_index] * kernel.EvalUnnormOnSq(distance);

      } /* for reference_index */

    } /* for query_index */

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
    DEBUG_SAME_DOUBLE(lower_bound_distance,
        MinNodeDistSq_(query_node, reference_node));

    if (0) {//lower_bound_distance > cutoff_dist_) {

      // execute prune

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
  void Init(const Matrix& references_in, double bandwidth, struct datanode* module_in) {

    // It's a good idea to make sure the object isn't initialized a
    // second time, as this is almost certainly mistaken.
    DEBUG_ASSERT(initialized_ == false);
    DEBUG_ONLY(initialized_ = true);

    module_ = module_in;

    /* The data sets need to have the same dimensionality */
    DEBUG_SAME_SIZE(queries_.n_rows(), references_.n_rows());

    /* Copy input matrices as they will be rearranged */
    queries_.Copy(references_in);
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

    /* Ready the vector of weighted sums so far for use. */
    weighted_sums_.Init(queries_.n_cols());
    weighted_sums_.SetZero();


    // Init weights once and for all here, but we can overwrite it with
    // the weights vector passed into ComputeKernelMatrixVectorMultiplication
    weights_.Init(queries_.n_cols());





    number_of_prunes_ = 0;


    DEBUG_ASSERT(bandwidth > 0);

    // kernel function
    kernel.Init(bandwidth);
    
    cutoff_dist_ = bandwidth*bandwidth;
    
    DEBUG_ONLY(printf("cutoff_dist = %f\n", cutoff_dist_));
  

  } /* Init */




  /**
   * Computes the kernel matrix vector multiplication and store the result in
   * the results vector.
   */
  void ComputeKernelMatrixVectorMultiplication(Vector weights_in, Vector* results) {

    // In addition to confirming the object's been initialized, we
    // want to make sure we aren't asking it to compute a second time.
    DEBUG_ASSERT(initialized_ == true);
    DEBUG_ASSERT(already_used_ == false);
    DEBUG_ONLY(already_used_ = true);

    // permute weights around to match the new reference point ordering
    for (index_t i = 0; i < weights_.length(); i++) {
      weights_[i] = weights_in[old_from_new_references_[i]];
    }



    fx_timer_start(module_, "dual_tree_computation");

    /* Start recursion on the roots of either tree */
    GNPRecursion_(query_tree_, reference_tree_,
        MinNodeDistSq_(query_tree_, reference_tree_));
    //printf("queries_.n_rows() = %d\n", queries_.n_rows());
    la::Scale(1 / kernel.CalcNormConstant(queries_.n_rows()), &weighted_sums_);

    fx_timer_stop(module_, "dual_tree_computation");

    // Save the total number of prunes to the FASTexec module; this
    // will printed after calling fx_done or can be read back later.
    fx_format_result(module_, "number_of_prunes", "%d", number_of_prunes_);

    if (results) {
      EmitResults(results);
    }

  } /* ComputeNeighbors */


  /**
   * Computes the kernel matrix vector multiplication naively.
   */
  void ComputeNaive(Vector* results) {

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
   * Reset things so that we can use the same tree for another computation
   */
  void Reset() {

    weighted_sums_.SetZero();

    already_used_ = false;

  } /* ResetWeightedSums */

  /**
   * Initialize and fill an ArrayList of results.
   */
  void EmitResults(Vector* results) {

    DEBUG_ASSERT(initialized_ == true);

    /* Map the indices back from how they have been permuted. */
    for (index_t i = 0; i < weighted_sums_.length(); i++) {

      (*results)[old_from_new_queries_[i]] = weighted_sums_[i];
    }

  } /* EmitResults */

}; /* class KernelVectorMult */

#endif /* KERNEL_VECTOR_MULT_H */
