### mlpack 2.1.0
###### 2016-10-31
  * Fixed CoverTree to properly handle single-point datasets.

  * Fixed a bug in CosineTree (and thus QUIC-SVD) that caused split failures for
    some datasets (#717).

  * Added mlpack_preprocess_describe program, which can be used to print
    statistics on a given dataset (#742).

  * Fix prioritized recursion for k-furthest-neighbor search (mlpack_kfn and the
    KFN class), leading to orders-of-magnitude speedups in some cases.

  * Bump minimum required version of Armadillo to 4.200.0.

  * Added simple Gradient Descent optimizer, found in
    src/mlpack/core/optimizers/gradient_descent/ (#792).

  * Added approximate furthest neighbor search algorithms QDAFN and
    DrusillaSelect in src/mlpack/methods/approx_kfn/, with command-line program
    mlpack_approx_kfn.

### mlpack 2.0.3
###### 2016-07-21
  * Added multiprobe LSH (#691).  The parameter 'T' to LSHSearch::Search() can
    now be used to control the number of extra bins that are probed, as can the
    -T (--num_probes) option to mlpack_lsh.

  * Added the Hilbert R tree to src/mlpack/core/tree/rectangle_tree/ (#664).  It
    can be used as the typedef HilbertRTree, and it is now an option in the
    mlpack_knn, mlpack_kfn, mlpack_range_search, and mlpack_krann command-line
    programs.

  * Added the mlpack_preprocess_split and mlpack_preprocess_binarize programs,
    which can be used for preprocessing code (#650, #666).

  * Added OpenMP support to LSHSearch and mlpack_lsh (#700).

### mlpack 2.0.2
###### 2016-06-20
  * Added the function LSHSearch::Projections(), which returns an arma::cube
    with each projection table in a slice (#663).  Instead of Projection(i), you
    should now use Projections().slice(i).

  * A new constructor has been added to LSHSearch that creates objects using
    projection tables provided in an arma::cube (#663).

  * Handle zero-variance dimensions in DET (#515).

  * Add MiniBatchSGD optimizer (src/mlpack/core/optimizers/minibatch_sgd/) and
    allow its use in mlpack_logistic_regression and mlpack_nca programs.

  * Add better backtrace support from Grzegorz Krajewski for Log::Fatal messages
    when compiled with debugging and profiling symbols.  This requires libbfd
    and libdl to be present during compilation.

  * CosineTree test fix from Mikhail Lozhnikov (#358).

  * Fixed HMM initial state estimation (#600).

  * Changed versioning macros __MLPACK_VERSION_MAJOR, __MLPACK_VERSION_MINOR,
    and __MLPACK_VERSION_PATCH to MLPACK_VERSION_MAJOR, MLPACK_VERSION_MINOR,
    and MLPACK_VERSION_PATCH.  The old names will remain in place until
    mlpack 3.0.0.

  * Renamed mlpack_allknn, mlpack_allkfn, and mlpack_allkrann to mlpack_knn,
    mlpack_kfn, and mlpack_krann.  The mlpack_allknn, mlpack_allkfn, and
    mlpack_allkrann programs will remain as copies until mlpack 3.0.0.

  * Add --random_initialization option to mlpack_hmm_train, for use when no
    labels are provided.

  * Add --kill_empty_clusters option to mlpack_kmeans and KillEmptyClusters
    policy for the KMeans class (#595, #596).

### mlpack 2.0.1
###### 2016-02-04
  * Fix CMake to properly detect when MKL is being used with Armadillo.

  * Minor parameter handling fixes to mlpack_logistic_regression (#504, #505).

  * Properly install arma_config.hpp.

  * Memory handling fixes for Hoeffding tree code.

  * Add functions that allow changing training-time parameters to HoeffdingTree
    class.

  * Fix infinite loop in sparse coding test.

  * Documentation spelling fixes (#501).

  * Properly handle covariances for Gaussians with large condition number
    (#496), preventing GMMs from filling with NaNs during training (and also
    HMMs that use GMMs).

  * CMake fixes for finding LAPACK and BLAS as Armadillo dependencies when ATLAS
    is used.

  * CMake fix for projects using mlpack's CMake configuration from elsewhere
    (#512).

### mlpack 2.0.0
###### 2015-12-24
  * Removed overclustering support from k-means because it is not well-tested,
    may be buggy, and is (I think) unused.  If this was support you were using,
    open a bug or get in touch with us; it would not be hard for us to
    reimplement it.

  * Refactored KMeans to allow different types of Lloyd iterations.

  * Added implementations of k-means: Elkan's algorithm, Hamerly's algorithm,
    Pelleg-Moore's algorithm, and the DTNN (dual-tree nearest neighbor)
    algorithm.

  * Significant acceleration of LRSDP via the use of accu(a % b) instead of
    trace(a * b).

  * Added MatrixCompletion class (matrix_completion), which performs nuclear
    norm minimization to fill unknown values of an input matrix.

  * No more dependence on Boost.Random; now we use C++11 STL random support.

  * Add softmax regression, contributed by Siddharth Agrawal and QiaoAn Chen.

  * Changed NeighborSearch, RangeSearch, FastMKS, LSH, and RASearch API; these
    classes now take the query sets in the Search() method, instead of in the
    constructor.

  * Use OpenMP, if available.  For now OpenMP support is only available in the
    DET training code.

  * Add support for predicting new test point values to LARS and the
    command-line 'lars' program.

  * Add serialization support for Perceptron and LogisticRegression.

  * Refactor SoftmaxRegression to predict into an arma::Row<size_t> object, and
    add a softmax_regression program.

  * Refactor LSH to allow loading and saving of models.

  * ToString() is removed entirely (#487).

  * Add --input_model_file and --output_model_file options to appropriate
    machine learning algorithms.

  * Rename all executables to start with an "mlpack" prefix (#229).

  * Add HoeffdingTree and mlpack_hoeffding_tree, an implementation of the
    streaming decision tree methodology from Domingos and Hulten in 2000.

### mlpack 1.0.12
###### 2015-01-07
  * Switch to 3-clause BSD license (from LGPL).

### mlpack 1.0.11
###### 2014-12-11
  * Proper handling of dimension calculation in PCA.

  * Load parameter vectors properly for LinearRegression models.

  * Linker fixes for AugLagrangian specializations under Visual Studio.

  * Add support for observation weights to LinearRegression.

  * MahalanobisDistance<> now takes root of the distance by default and
    therefore satisfies the triangle inequality (TakeRoot now defaults to true).

  * Better handling of optional Armadillo HDF5 dependency.

  * Fixes for numerous intermittent test failures.

  * math::RandomSeed() now sets the random seed for recent (>=3.930) Armadillo
    versions.

  * Handle Newton method convergence better for
    SparseCoding::OptimizeDictionary() and make maximum iterations a parameter.

  * Known bug: CosineTree construction may fail in some cases on i386 systems
    (#358).

### mlpack 1.0.10
###### 2014-08-29
  * Bugfix for NeighborSearch regression which caused very slow allknn/allkfn.
    Speeds are now restored to approximately 1.0.8 speeds, with significant
    improvement for the cover tree (#347).

  * Detect dependencies correctly when ARMA_USE_WRAPPER is not being defined
    (i.e., libarmadillo.so does not exist).

  * Bugfix for compilation under Visual Studio (#348).

### mlpack 1.0.9
###### 2014-07-28
  * GMM initialization is now safer and provides a working GMM when constructed
    with only the dimensionality and number of Gaussians (#301).

  * Check for division by 0 in Forward-Backward Algorithm in HMMs (#301).

  * Fix MaxVarianceNewCluster (used when re-initializing clusters for k-means)
    (#301).

  * Fixed implementation of Viterbi algorithm in HMM::Predict() (#303).

  * Significant speedups for dual-tree algorithms using the cover tree (#235,
    #314) including a faster implementation of FastMKS.

  * Fix for LRSDP optimizer so that it compiles and can be used (#312).

  * CF (collaborative filtering) now expects users and items to be zero-indexed,
    not one-indexed (#311).

  * CF::GetRecommendations() API change: now requires the number of
    recommendations as the first parameter.  The number of users in the local
    neighborhood should be specified with CF::NumUsersForSimilarity().

  * Removed incorrect PeriodicHRectBound (#58).

  * Refactor LRSDP into LRSDP class and standalone function to be optimized
    (#305).

  * Fix for centering in kernel PCA (#337).

  * Added simulated annealing (SA) optimizer, contributed by Zhihao Lou.

  * HMMs now support initial state probabilities; these can be set in the
    constructor, trained, or set manually with HMM::Initial() (#302).

  * Added Nyström method for kernel matrix approximation by Marcus Edel.

  * Kernel PCA now supports using Nyström method for approximation.

  * Ball trees now work with dual-tree algorithms, via the BallBound<> bound
    structure (#307); fixed by Yash Vadalia.

  * The NMF class is now AMF<>, and supports far more types of factorizations,
    by Sumedh Ghaisas.

  * A QUIC-SVD implementation has returned, written by Siddharth Agrawal and
    based on older code from Mudit Gupta.

  * Added perceptron and decision stump by Udit Saxena (these are weak learners
    for an eventual AdaBoost class).

  * Sparse autoencoder added by Siddharth Agrawal.

### mlpack 1.0.8
###### 2014-01-06
  * Memory leak in NeighborSearch index-mapping code fixed (#298).

  * GMMs can be trained using the existing model as a starting point by
    specifying an additional boolean parameter to GMM::Estimate() (#296).

  * Logistic regression implementation added in methods/logistic_regression (see
    also #293).

  * L-BFGS optimizer now returns its function via Function().

  * Version information is now obtainable via mlpack::util::GetVersion() or the
    __MLPACK_VERSION_MAJOR, __MLPACK_VERSION_MINOR, and  __MLPACK_VERSION_PATCH
    macros (#297).

  * Fix typos in allkfn and allkrann output.

### mlpack 1.0.7
###### 2013-10-04
  * Cover tree support for range search (range_search), rank-approximate nearest
    neighbors (allkrann), minimum spanning tree calculation (emst), and FastMKS
    (fastmks).

  * Dual-tree FastMKS implementation added and tested.

  * Added collaborative filtering package (cf) that can provide recommendations
    when given users and items.

  * Fix for correctness of Kernel PCA (kernel_pca) (#270).

  * Speedups for PCA and Kernel PCA (#198).

  * Fix for correctness of Neighborhood Components Analysis (NCA) (#279).

  * Minor speedups for dual-tree algorithms.

  * Fix for Naive Bayes Classifier (nbc) (#269).

  * Added a ridge regression option to LinearRegression (linear_regression)
    (#286).

  * Gaussian Mixture Models (gmm::GMM<>) now support arbitrary covariance matrix
    constraints (#283).

  * MVU (mvu) removed because it is known to not work (#183).

  * Minor updates and fixes for kernels (in mlpack::kernel).

### mlpack 1.0.6
###### 2013-06-13
  * Minor bugfix so that FastMKS gets built.

### mlpack 1.0.5
###### 2013-05-01
  * Speedups of cover tree traversers (#235).

  * Addition of rank-approximate nearest neighbors (RANN), found in
    src/mlpack/methods/rann/.

  * Addition of fast exact max-kernel search (FastMKS), found in
    src/mlpack/methods/fastmks/.

  * Fix for EM covariance estimation; this should improve GMM training time.

  * More parameters for GMM estimation.

  * Force GMM and GaussianDistribution covariance matrices to be positive
    definite, so that training converges much more often.

  * Add parameter for the tolerance of the Baum-Welch algorithm for HMM
    training.

  * Fix for compilation with clang compiler.

  * Fix for k-furthest-neighbor-search.

### mlpack 1.0.4
###### 2013-02-08
  * Force minimum Armadillo version to 2.4.2.

  * Better output of class types to streams; a class with a ToString() method
    implemented can be sent to a stream with operator<<.

  * Change return type of GMM::Estimate() to double (#257).

  * Style fixes for k-means and RADICAL.

  * Handle size_t support correctly with Armadillo 3.6.2 (#258).

  * Add locality-sensitive hashing (LSH), found in src/mlpack/methods/lsh/.

  * Better tests for SGD (stochastic gradient descent) and NCA (neighborhood
    components analysis).

### mlpack 1.0.3
###### 2012-09-16

  * Remove internal sparse matrix support because Armadillo 3.4.0 now includes
    it.  When using Armadillo versions older than 3.4.0, sparse matrix support
    is not available.

  * NCA (neighborhood components analysis) now support an arbitrary optimizer
    (#245), including stochastic gradient descent (#249).

### mlpack 1.0.2
###### 2012-08-15
  * Added density estimation trees, found in src/mlpack/methods/det/.

  * Added non-negative matrix factorization, found in src/mlpack/methods/nmf/.

  * Added experimental cover tree implementation, found in
    src/mlpack/core/tree/cover_tree/ (#157).

  * Better reporting of boost::program_options errors (#225).

  * Fix for timers on Windows (#212, #211).

  * Fix for allknn and allkfn output (#204).

  * Sparse coding dictionary initialization is now a template parameter (#220).

### mlpack 1.0.1
###### 2012-03-03
  * Added kernel principal components analysis (kernel PCA), found in
    src/mlpack/methods/kernel_pca/ (#74).

  * Fix for Lovasz-Theta AugLagrangian tests (#182).

  * Fixes for allknn output (#185, #186).

  * Added range search executable (#192).

  * Adapted citations in documentation to BiBTeX; no citations in -h output
    (#195).

  * Stop use of 'const char*' and prefer 'std::string' (#176).

  * Support seeds for random numbers (#177).

### mlpack 1.0.0
###### 2011-12-17
  * Initial release.  See any resolved tickets numbered less than #196 or
    execute this query:
    http://www.mlpack.org/trac/query?status=closed&milestone=mlpack+1.0.0
