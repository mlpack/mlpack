# mlpack changelog

## mlpack ?.?.?

_????-??-??_

 * Distribute STB headers as part of R package (#3724, #3726).

 * Added OpenMP support for fast approximation (#3685).

 * Implemented the Find and Fill algorithm into the Dropout Layer and added OpenMP support (#3684).

 * Update Python bindings to support NumPy 2.x (#3752).
 
 * Bump minimum Armadillo version to 10.8 (#3760).

## mlpack 4.4.0

_2024-05-26_

  * Add `print_training_accuracy` option to LogisticRegression bindings (#3552).

  * Fix `preprocess_split()` call in documentation for `LinearRegression` and
    `AdaBoost` Python classes (#3563).

  * Added `Repeat` ANN layer type (#3565).

  * Remove `round()` implementation for old MSVC compilers (#3570).

  * (R) Added inline plugin to the R bindings to allow for other R packages to
    link to headers (#3626, h/t @cgiachalis).

  * (R) Removed extra gcc-specific options from `Makevars.win`  (#3627, h/t
    @kalibera).

  * (R) Changed roxygen package-level documentation from using
    `@docType package` to `"_PACKAGE"`. (#3636)

  * Fix floating-point accuracy issue for decision trees that sometimes caused
    crashes (#3595).

  * Use templates metaprog to distinguish between a matrix and a cube type
    (#3602), (#3585).

  * Use `MatType` instead of `arma::Mat<eT>`, (#3567), (#3607), (#3608),
    (#3609), (#3568).

  * Generalize matrix operations for armadillo and bandicoot, (#3619), (#3617),
    (#3610), (#3643), (#3600), (#3605), (#3629).

  * Change `arma::conv_to` to `ConvTo` using a local shim for bandicoot support
    (#3614).

  * Fix a bug for the stddev and mean in `RandNormal()` #(3651).

  * Allow PCA to take different matrix types (#3677).

  * Fix usage of precompiled headers; remove cotire (#3635).

  * Fix non-working `verbose` option for R bindings (#3691), and add global
    `mlpack.verbose` option (#3706).

  * Fix divide-by-zero edge case for LARS (#3701).

  * Templatize `SparseCoding` and `LocalCoordinateCoding` to allow different
    matrix types (#3709, #3711).

  * Fix handling of unused atoms in `LocalCoordinateCoding` (#3711).

  * Move minimum required C++ version from C++14 to C++17 (#3704).

## mlpack 4.3.0

_2023-11-27_

  * Fix include ordering issue for `LinearRegression` (#3541).

  * Fix L1 regularization in case where weight is zero (#3545).

  * Use HTTPS for all auto-downloaded dependencies (#3550).

  * More robust detection of C++17 mode in the MSVC "compiler" (#3555, #3557).

  * Fix setting number of classes correctly in `SoftmaxRegression::Train()`
    (#3553).

  * Adapt `MultiheadAttention` and `LayerNorm` ANN layers to new Layer interface
    (#3547).

  * Fix inconsistent use of the "input" parameter to the Backward method in ANNs
    (#3551).

  * Allow passing weak learner hyperparameters directly to AdaBoost (#3560).

## mlpack 4.2.1

_2023-09-05_

  * Reinforcement Learning: Gaussian noise (#3515).

  * Reinforcement Learning: Twin Delayed Deep Deterministic Policy Gradient
    (#3512).

  * Reinforcement Learning: Ornstein-Uhlenbeck noise (#3499).

  * Reinforcement Learning: Deep Deterministic Policy Gradient (#3494).

  * Add `ClassProbabilities()` member to `DecisionTree` so that the internal
    details of trees can be more easily inspected (#3511).

  * Bipolar sigmoid activation function added and invertible functions 
    fixed (#3506).

  * Add auto-configured `mlpack/config.hpp` to contain configuration details of
    mlpack that are required at compile time.  STB detection is now done in this
    file with the `MLPACK_HAS_STB` macro (#3519).

  * Fix CRAN package alias for R bindings (#3543).

## mlpack 4.2.0

_2023-06-14_

  * Adapt `C_ReLU`, `ReLU6`, `FlexibleReLU` layers for the new neural network
    API (#3445).

  * Fix PReLU, add integration test to it (#3473).

  * Fix bug in LogSoftMax derivative (#3469).

  * Add `serialize` method to `GaussianInitialization`,
    `LecunNormalInitialization`,
    `KathirvalavakumarSubavathiInitialization`, `NguyenWidrowInitialization`,
    and `OrthogonalInitialization` (#3483).

  * Allow categorical features to `preprocess_one_hot_encode` (#3487).

  * Install mlpack and cereal headers as part of R package (#3488).

  * Add intercept and normalization support to LARS (#3493).

  * Allow adding two features simultaneously to LARS models (#3493).

  * Adapt FTSwish activation function (#3485).

  * Adapt Hyper-Sinh activation function (#3491).

## mlpack 4.1.0

_2023-04-26_

  * Adapt HardTanH layer (#3454).

  * Adapt Softmin layer for new neural network API (#3437).

  * Adapt PReLU layer for new neural network API (#3420).

  * Add CF decomposition methods: `QUIC_SVDPolicy` and `BlockKrylovSVDPolicy`
    (#3413, #3404).

  * Update outdated code in tutorials (#3398, #3401).

  * Bugfix for non-square convolution kernels (#3376).

  * Fix a few missing includes in `<mlpack.hpp>` (#3374).

  * Fix DBSCAN handling of non-core points (#3346).

  * Avoid deprecation warnings in Armadillo 11.4.4+ (#3405).

  * Issue runtime error when serialization of neural networks is attempted but
    `MLPACK_ENABLE_ANN_SERIALIZATION` is not defined (#3451).

## mlpack 4.0.1

_2022-12-23_

  * Fix mapping of categorical data for Julia bindings (#3305).

  * Bugfix: catch all exceptions when running bindings from Julia, instead of
    crashing (#3304).

  * Various Python configuration fixes for Windows and OS X (#3312, #3313,
    #3311, #3309, #3308, #3297, #3302).

  * Optimize and strip compiled Python bindings when possible, resulting in
    significant size minimization (#3310).

  * The `/std:c++17` and `/Zc:__cplusplus` options are now required when using
    Visual Studio (#3318).  Documentation and compile-time checks added.

  * Set `BUILD_TESTS` to `OFF` by default.  If you want to build tests, like
    `mlpack_test`, manually set `BUILD_TESTS` to `ON` in your CMake
    configuration step (#3316).

  * Fix handling of transposed matrix parameters in Python, Julia, R, and Go
    bindings (#3327).

  * Comment out definition of ARMA_NO DEBUG. This allows various Armadillo
    run-time checks such as non-conforming matrices and out-of-bounds
    element access. In turn this helps tracking down bugs and incorrect
    usage (#3322).

## mlpack 4.0.0

_2022-10-23_

  * Bump C++ standard requirement to C++14 (#3233).

  * Fix `Perceptron` to work with cross-validation framework (#3190).

  * Migrate from boost tests to Catch2 framework (#2523), (#2584).

  * Bump minimum armadillo version from 8.400 to 9.800 (#3043), (#3048).

  * Adding a copy constructor in the Convolution layer (#3067).

  * Replace `boost::spirit` parser by a local efficient implementation (#2942).

  * Disable correctly the autodownloader + fix tests stability (#3076).

  * Replace `boost::any` with `core::v2::any` or `std::any` if available (#3006).

  * Remove old non used Boost headers (#3005).

  * Replace `boost::enable_if` with `std::enable_if` (#2998).

  * Replace `boost::is_same` with `std::is_same` (#2993).

  * Remove invalid option for emsmallen and STB (#2960).

  * Check for armadillo dependencies before downloading armadillo (#2954).

  * Disable the usage of autodownloader by default (#2953).

  * Install dependencies downloaded with the autodownloader (#2952).

  * Download older Boost if the compiler is old (#2940).

  * Add support for embedded systems (#2531).

  * Build mlpack executable statically if the library is statically linked (#2931).

  * Fix cover tree loop bug on embedded arm systems (#2869).

  * Fix a LAPACK bug in `FindArmadillo.cmake` (#2929).

  * Add an autodownloader to get mlpack dependencies (#2927).

  * Remove Coverage files and configurations from CMakeLists (#2866).

  * Added `Multi Label Soft Margin Loss` loss function for neural networks
   (#2345).

  * Added Decision Tree Regressor (#2905). It can be used using the class
    `mlpack::tree::DecisionTreeRegressor`. It is accessible only though C++.

  * Added dict-style inspection of mlpack models in python bindings (#2868).

  * Added Extra Trees Algorithm (#2883). Currently, it can be used using the
    class `mlpack::tree::ExtraTrees`, but only through C++.

  * Add Flatten T Swish activation function (`flatten-t-swish.hpp`)

  * Added warm start feature to Random Forest (#2881); this feature is
    accessible from mlpack's bindings to different languages.

  * Added Pixel Shuffle layer (#2563).

  * Add "check_input_matrices" option to python bindings that checks
    for NaN and inf values in all the input matrices (#2787).

  * Add Adjusted R squared functionality to R2Score::Evaluate (#2624).

  * Disabled all the bindings by default in CMake (#2782).

  * Added an implementation to Stratify Data (#2671).

  * Add `BUILD_DOCS` CMake option to control whether Doxygen documentation is
    built (default ON) (#2730).

  * Add Triplet Margin Loss function (#2762).

  * Add finalizers to Julia binding model types to fix memory handling (#2756).

  * HMM: add functions to calculate likelihood for data stream with/without
    pre-calculated emission probability (#2142).

  * Replace Boost serialization library with Cereal (#2458).

  * Add `PYTHON_INSTALL_PREFIX` CMake option to specify installation root for
    Python bindings (#2797).

  * Removed `boost::visitor` from model classes for `knn`, `kfn`, `cf`,
    `range_search`, `krann`, and `kde` bindings (#2803).

  * Add k-means++ initialization strategy (#2813).

  * `NegativeLogLikelihood<>` now expects classes in the range `0` to
    `numClasses - 1` (#2534).

  * Add `Lambda1()`, `Lambda2()`, `UseCholesky()`, and `Tolerance()` members to
    `LARS` so parameters for training can be modified (#2861).

  * Remove unused `ElemType` template parameter from `DecisionTree` and
    `RandomForest` (#2874).

  * Fix Python binding build when the CMake variable `USE_OPENMP` is set to
    `OFF` (#2884).

  * The `mlpack_test` target is no longer built as part of `make all`.  Use
    `make mlpack_test` to build the tests.

  * Fixes to `HoeffdingTree`: ensure that training still works when empty
    constructor is used (#2964).

  * Fix Julia model serialization bug (#2970).

  * Fix `LoadCSV()` to use pre-populated `DatasetInfo` objects (#2980).

  * Add `probabilities` option to softmax regression binding, to get class
    probabilities for test points (#3001).

  * Fix thread safety issues in mlpack bindings to other languages (#2995).

  * Fix double-free of model pointers in R bindings (#3034).

  * Fix Julia, Python, R, and Go handling of categorical data for
    `decision_tree()` and `hoeffding_tree()` (#2971).

  * Depend on `pkgbuild` for R bindings (#3081).

  * Replaced Numpy deprecated code in Python bindings (#3126).

## mlpack 3.4.2

_2020-10-26_

  * Added Mean Absolute Percentage Error.

  * Added Softmin activation function as layer in ann/layer.

  * Fix spurious ARMA_64BIT_WORD compilation warnings on 32-bit systems (#2665).

## mlpack 3.4.1

_2020-09-07_

  * Fix incorrect parsing of required matrix/model parameters for command-line
    bindings (#2600).

  * Add manual type specification support to `data::Load()` and `data::Save()`
    (#2084, #2135, #2602).

  * Remove use of internal Armadillo functionality (#2596, #2601, #2602).

## mlpack 3.4.0

_2020-09-01_

  * Issue warnings when metrics produce NaNs in KFoldCV (#2595).

  * Added bindings for _R_ during Google Summer of Code (#2556).

  * Added common striptype function for all bindings (#2556).

  * Refactored common utility function of bindings to bindings/util (#2556).

  * Renamed InformationGain to HoeffdingInformationGain in
    methods/hoeffding_trees/information_gain.hpp (#2556).

  * Added macro for changing stream of printing and warnings/errors (#2556).

  * Added Spatial Dropout layer (#2564).

  * Force CMake to show error when it didn't find Python/modules (#2568).

  * Refactor `ProgramInfo()` to separate out all the different
    information (#2558).

  * Add bindings for one-hot encoding (#2325).

  * Added Soft Actor-Critic to RL methods (#2487).

  * Added Categorical DQN to q_networks (#2454).

  * Added N-step DQN to q_networks (#2461).

  * Add Silhoutte Score metric and Pairwise Distances (#2406).

  * Add Go bindings for some missed models (#2460).

  * Replace boost program_options dependency with CLI11 (#2459).

  * Additional functionality for the ARFF loader (#2486); use case sensitive
    categories (#2516).

  * Add `bayesian_linear_regression` binding for the command-line, Python,
    Julia, and Go.  Also called "Bayesian Ridge", this is equivalent to a
    version of linear regression where the regularization parameter is
    automatically tuned (#2030).

  * Fix defeatist search for spill tree traversals (#2566, #1269).

  * Fix incremental training of logistic regression models (#2560).

  * Change default configuration of `BUILD_PYTHON_BINDINGS` to `OFF` (#2575).

## mlpack 3.3.2

_2020-06-18_

  * Added Noisy DQN to q_networks (#2446).

  * Add Go bindings (#1884).

  * Added Dueling DQN to q_networks, Noisy linear layer to ann/layer
    and Empty loss to ann/loss_functions (#2414).

  * Storing and adding accessor method for action in q_learning (#2413).

  * Added accessor methods for ANN layers (#2321).

  * Addition of `Elliot` activation function (#2268).

  * Add adaptive max pooling and adaptive mean pooling layers (#2195).

  * Add parameter to avoid shuffling of data in preprocess_split (#2293).

  * Add `MatType` parameter to `LSHSearch`, allowing sparse matrices to be used
    for search (#2395).

  * Documentation fixes to resolve Doxygen warnings and issues (#2400).

  * Add Load and Save of Sparse Matrix (#2344).

  * Add Intersection over Union (IoU) metric for bounding boxes (#2402).

  * Add Non Maximal Supression (NMS) metric for bounding boxes (#2410).

  * Fix `no_intercept` and probability computation for linear SVM bindings
    (#2419).

  * Fix incorrect neighbors for `k > 1` searches in `approx_kfn` binding, for
    the `QDAFN` algorithm (#2448).

  * Fix serialization of kernels with state for FastMKS (#2452).

  * Add `RBF` layer in ann module to make `RBFN` architecture (#2261).

## mlpack 3.3.1

_2020-04-29_

  * Minor Julia and Python documentation fixes (#2373).

  * Updated terminal state and fixed bugs for Pendulum environment (#2354,
    #2369).

  * Added `EliSH` activation function (#2323).

  * Add L1 Loss function (#2203).

  * Pass CMAKE_CXX_FLAGS (compilation options) correctly to Python build
    (#2367).

  * Expose ensmallen Callbacks for sparseautoencoder (#2198).

  * Bugfix for LARS class causing invalid read (#2374).

  * Add serialization support from Julia; use `mlpack.serialize()` and
    `mlpack.deserialize()` to save and load from `IOBuffer`s.

## mlpack 3.3.0

_2020-04-07_

  * Added `Normal Distribution` to `ann/dists` (#2382).

  * Templated return type of `Forward function` of loss functions (#2339).

  * Added `R2 Score` regression metric (#2323).

  * Added `poisson negative log likelihood` loss function (#2196).

  * Added `huber` loss function (#2199).

  * Added `mean squared logarithmic error` loss function for neural networks
    (#2210).

  * Added `mean bias loss function` for neural networks (#2210).

  * The DecisionStump class has been marked deprecated; use the `DecisionTree`
    class with `NoRecursion=true` or use `ID3DecisionStump` instead (#2099).

  * Added `probabilities_file` parameter to get the probabilities matrix of
    AdaBoost classifier (#2050).

  * Fix STB header search paths (#2104).

  * Add `DISABLE_DOWNLOADS` CMake configuration option (#2104).

  * Add padding layer in TransposedConvolutionLayer (#2082).

  * Fix pkgconfig generation on non-Linux systems (#2101).

  * Use log-space to represent HMM initial state and transition probabilities
    (#2081).

  * Add functions to access parameters of `Convolution` and `AtrousConvolution`
    layers (#1985).

  * Add Compute Error function in lars regression and changing Train function to
    return computed error (#2139).

  * Add Julia bindings (#1949).  Build settings can be controlled with the
    `BUILD_JULIA_BINDINGS=(ON/OFF)` and `JULIA_EXECUTABLE=/path/to/julia` CMake
    parameters.

  * CMake fix for finding STB include directory (#2145).

  * Add bindings for loading and saving images (#2019); `mlpack_image_converter`
    from the command-line, `mlpack.image_converter()` from Python.

  * Add normalization support for CF binding (#2136).

  * Add Mish activation function (#2158).

  * Update `init_rules` in AMF to allow users to merge two initialization
    rules (#2151).

  * Add GELU activation function (#2183).

  * Better error handling of eigendecompositions and Cholesky decompositions
    (#2088, #1840).

  * Add LiSHT activation function (#2182).

  * Add Valid and Same Padding for Transposed Convolution layer (#2163).

  * Add CELU activation function (#2191)

  * Add Log-Hyperbolic-Cosine Loss function (#2207).

  * Change neural network types to avoid unnecessary use of rvalue references
    (#2259).

  * Bump minimum Boost version to 1.58 (#2305).

  * Refactor STB support so `HAS_STB` macro is not needed when compiling against
    mlpack (#2312).

  * Add Hard Shrink Activation Function (#2186).

  * Add Soft Shrink Activation Function (#2174).

  * Add Hinge Embedding Loss Function (#2229).

  * Add Cosine Embedding Loss Function (#2209).

  * Add Margin Ranking Loss Function (#2264).

  * Bugfix for incorrect parameter vector sizes in logistic regression and
    softmax regression (#2359).

## mlpack 3.2.2

_2019-11-26_

  * Add `valid` and `same` padding option in `Convolution` and `Atrous
    Convolution` layer (#1988).

  * Add Model() to the FFN class to access individual layers (#2043).

  * Update documentation for pip and conda installation packages (#2044).

  * Add bindings for linear SVM (#1935); `mlpack_linear_svm` from the
    command-line, `linear_svm()` from Python.

  * Add support to return the layer name as `std::string` (#1987).

  * Speed and memory improvements for the Transposed Convolution layer (#1493).

  * Fix Windows Python build configuration (#1885).

  * Validate md5 of STB library after download (#2087).

  * Add `__version__` to `__init__.py` (#2092).

  * Correctly handle RNN sequences that are shorter than the value of rho (#2102).

## mlpack 3.2.1

_2019-10-01_

  * Enforce CMake version check for ensmallen (#2032).

  * Fix CMake check for Armadillo version (#2029).

  * Better handling of when STB is not installed (#2033).

  * Fix Naive Bayes classifier computations in high dimensions (#2022).

## mlpack 3.2.0

_2019-09-25_

  * Fix some potential infinity errors in Naive Bayes Classifier (#2022).

  * Fix occasionally-failing RADICAL test (#1924).

  * Fix gcc 9 OpenMP compilation issue (#1970).

  * Added support for loading and saving of images (#1903).

  * Add Multiple Pole Balancing Environment (#1901, #1951).

  * Added functionality for scaling of data (#1876); see the command-line
    binding `mlpack_preprocess_scale` or Python binding `preprocess_scale()`.

  * Add new parameter `maximum_depth` to decision tree and random forest
    bindings (#1916).

  * Fix prediction output of softmax regression when test set accuracy is
    calculated (#1922).

  * Pendulum environment now checks for termination. All RL environments now
    have an option to terminate after a set number of time steps (no limit
    by default) (#1941).

  * Add support for probabilistic KDE (kernel density estimation) error bounds
    when using the Gaussian kernel (#1934).

  * Fix negative distances for cover tree computation (#1979).

  * Fix cover tree building when all pairwise distances are 0 (#1986).

  * Improve KDE pruning by reclaiming not used error tolerance (#1954, #1984).

  * Optimizations for sparse matrix accesses in z-score normalization for CF
    (#1989).

  * Add `kmeans_max_iterations` option to GMM training binding `gmm_train_main`.

  * Bump minimum Armadillo version to 8.400.0 due to ensmallen dependency
    requirement (#2015).

## mlpack 3.1.1

_2019-05-26_

  * Fix random forest bug for numerical-only data (#1887).

  * Significant speedups for random forest (#1887).

  * Random forest now has `minimum_gain_split` and `subspace_dim` parameters
    (#1887).

  * Decision tree parameter `print_training_error` deprecated in favor of
    `print_training_accuracy`.

  * `output` option changed to `predictions` for adaboost and perceptron
    binding. Old options are now deprecated and will be preserved until mlpack
    4.0.0 (#1882).

  * Concatenated ReLU layer (#1843).

  * Accelerate NormalizeLabels function using hashing instead of linear search
    (see `src/mlpack/core/data/normalize_labels_impl.hpp`) (#1780).

  * Add `ConfusionMatrix()` function for checking performance of classifiers
    (#1798).

  * Install ensmallen headers when it is downloaded during build (#1900).

## mlpack 3.1.0

_2019-04-25_

  * Add DiagonalGaussianDistribution and DiagonalGMM classes to speed up the
    diagonal covariance computation and deprecate DiagonalConstraint (#1666).

  * Add kernel density estimation (KDE) implementation with bindings to other
    languages (#1301).

  * Where relevant, all models with a `Train()` method now return a `double`
    value representing the goodness of fit (i.e. final objective value, error,
    etc.) (#1678).

  * Add implementation for linear support vector machine (see
    `src/mlpack/methods/linear_svm`).

  * Change DBSCAN to use PointSelectionPolicy and add OrderedPointSelection (#1625).

  * Residual block support (#1594).

  * Bidirectional RNN (#1626).

  * Dice loss layer (#1674, #1714) and hard sigmoid layer (#1776).

  * `output` option changed to `predictions` and `output_probabilities` to
    `probabilities` for Naive Bayes binding (`mlpack_nbc`/`nbc()`).  Old options
    are now deprecated and will be preserved until mlpack 4.0.0 (#1616).

  * Add support for Diagonal GMMs to HMM code (#1658, #1666).  This can provide
    large speedup when a diagonal GMM is acceptable as an emission probability
    distribution.

  * Python binding improvements: check parameter type (#1717), avoid copying
    Pandas dataframes (#1711), handle Pandas Series objects (#1700).

## mlpack 3.0.4

_2018-11-13_

  * Bump minimum CMake version to 3.3.2.

  * CMake fixes for Ninja generator by Marc Espie.

## mlpack 3.0.3

_2018-07-27_

  * Fix Visual Studio compilation issue (#1443).

  * Allow running local_coordinate_coding binding with no initial_dictionary
    parameter when input_model is not specified (#1457).

  * Make use of OpenMP optional via the CMake 'USE_OPENMP' configuration
    variable (#1474).

  * Accelerate FNN training by 20-30% by avoiding redundant calculations
    (#1467).

  * Fix math::RandomSeed() usage in tests (#1462, #1440).

  * Generate better Python setup.py with documentation (#1460).

## mlpack 3.0.2

_2018-06-08_

  * Documentation generation fixes for Python bindings (#1421).

  * Fix build error for man pages if command-line bindings are not being built
    (#1424).

  * Add 'shuffle' parameter and Shuffle() method to KFoldCV (#1412).  This will
    shuffle the data when the object is constructed, or when Shuffle() is
    called.

  * Added neural network layers: AtrousConvolution (#1390), Embedding (#1401),
    and LayerNorm (layer normalization) (#1389).

  * Add Pendulum environment for reinforcement learning (#1388) and update
    Mountain Car environment (#1394).

## mlpack 3.0.1

_2018-05-10_

  * Fix intermittently failing tests (#1387).

  * Add big-batch SGD (BBSGD) optimizer in
    src/mlpack/core/optimizers/bigbatch_sgd/ (#1131).

  * Fix simple compiler warnings (#1380, #1373).

  * Simplify NeighborSearch constructor and Train() overloads (#1378).

  * Add warning for OpenMP setting differences (#1358/#1382).  When mlpack is
    compiled with OpenMP but another application is not (or vice versa), a
    compilation warning will now be issued.

  * Restructured loss functions in src/mlpack/methods/ann/ (#1365).

  * Add environments for reinforcement learning tests (#1368, #1370, #1329).

  * Allow single outputs for multiple timestep inputs for recurrent neural
    networks (#1348).

  * Add He and LeCun normal initializations for neural networks (#1342).
    Neural networks: add He and LeCun normal initializations (#1342), add FReLU
    and SELU activation functions (#1346, #1341), add alpha-dropout (#1349).

## mlpack 3.0.0

_2018-03-30_

  * Speed and memory improvements for DBSCAN.  --single_mode can now be used for
    situations where previously RAM usage was too high.

  * Bump minimum required version of Armadillo to 6.500.0.

  * Add automatically generated Python bindings.  These have the same interface
    as the command-line programs.

  * Add deep learning infrastructure in src/mlpack/methods/ann/.

  * Add reinforcement learning infrastructure in
    src/mlpack/methods/reinforcement_learning/.

  * Add optimizers: AdaGrad, CMAES, CNE, FrankeWolfe, GradientDescent,
    GridSearch, IQN, Katyusha, LineSearch, ParallelSGD, SARAH, SCD, SGDR,
    SMORMS3, SPALeRA, SVRG.

  * Add hyperparameter tuning infrastructure and cross-validation infrastructure
    in src/mlpack/core/cv/ and src/mlpack/core/hpt/.

  * Fix bug in mean shift.

  * Add random forests (see src/mlpack/methods/random_forest).

  * Numerous other bugfixes and testing improvements.

  * Add randomized Krylov SVD and Block Krylov SVD.

## mlpack 2.2.5

_2017-08-25_

  * Compilation fix for some systems (#1082).

  * Fix PARAM_INT_OUT() (#1100).

## mlpack 2.2.4

_2017-07-18_

  * Speed and memory improvements for DBSCAN. --single_mode can now be used for
    situations where previously RAM usage was too high.

  * Fix bug in CF causing incorrect recommendations.

## mlpack 2.2.3

_2017-05-24_

  * Bug fix for --predictions_file in mlpack_decision_tree program.

## mlpack 2.2.2

_2017-05-04_

  * Install backwards-compatibility mlpack_allknn and mlpack_allkfn programs;
    note they are deprecated and will be removed in mlpack 3.0.0 (#992).

  * Fix RStarTree bug that surfaced on OS X only (#964).

  * Small fixes for MiniBatchSGD and SGD and tests.

## mlpack 2.2.1

_2017-04-13_

  * Compilation fix for mlpack_nca and mlpack_test on older Armadillo versions
    (#984).

## mlpack 2.2.0

_2017-03-21_

  * Bugfix for mlpack_knn program (#816).

  * Add decision tree implementation in methods/decision_tree/.  This is very
    similar to a C4.5 tree learner.

  * Add DBSCAN implementation in methods/dbscan/.

  * Add support for multidimensional discrete distributions (#810, #830).

  * Better output for Log::Debug/Log::Info/Log::Warn/Log::Fatal for Armadillo
    objects (#895, #928).

  * Refactor categorical CSV loading with boost::spirit for faster loading
    (#681).

## mlpack 2.1.1

_2016-12-22_

  * HMMs now use random initialization; this should fix some convergence issues
    (#828).

  * HMMs now initialize emissions according to the distribution of observations
    (#833).

  * Minor fix for formatted output (#814).

  * Fix DecisionStump to properly work with any input type.

## mlpack 2.1.0

_2016-10-31_

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

## mlpack 2.0.3

_2016-07-21_

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

## mlpack 2.0.2

_2016-06-20_

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

## mlpack 2.0.1

_2016-02-04_

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

## mlpack 2.0.0

_2015-12-24_

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
    command-line `lars` program.

  * Add serialization support for `Perceptron` and `LogisticRegression`.

  * Refactor SoftmaxRegression to predict into an `arma::Row<size_t>` object,
    and add a `softmax_regression` program.

  * Refactor LSH to allow loading and saving of models.

  * ToString() is removed entirely (#487).

  * Add `--input_model_file` and `--output_model_file` options to appropriate
    machine learning algorithms.

  * Rename all executables to start with an "mlpack" prefix (#229).

  * Add HoeffdingTree and `mlpack_hoeffding_tree`, an implementation of the
    streaming decision tree methodology from Domingos and Hulten in 2000.

## mlpack 1.0.12

_2015-01-07_

  * Switch to 3-clause BSD license (from LGPL).

## mlpack 1.0.11

_2014-12-11_

  * Proper handling of dimension calculation in PCA.

  * Load parameter vectors properly for LinearRegression models.

  * Linker fixes for AugLagrangian specializations under Visual Studio.

  * Add support for observation weights to LinearRegression.

  * `MahalanobisDistance<>` now takes the root of the distance by default and
    therefore satisfies the triangle inequality (TakeRoot now defaults to true).

  * Better handling of optional Armadillo HDF5 dependency.

  * Fixes for numerous intermittent test failures.

  * math::RandomSeed() now sets the random seed for recent (>=3.930) Armadillo
    versions.

  * Handle Newton method convergence better for
    SparseCoding::OptimizeDictionary() and make maximum iterations a parameter.

  * Known bug: CosineTree construction may fail in some cases on i386 systems
    (#358).

## mlpack 1.0.10

_2014-08-29_

  * Bugfix for NeighborSearch regression which caused very slow allknn/allkfn.
    Speeds are now restored to approximately 1.0.8 speeds, with significant
    improvement for the cover tree (#347).

  * Detect dependencies correctly when ARMA_USE_WRAPPER is not being defined
    (i.e., libarmadillo.so does not exist).

  * Bugfix for compilation under Visual Studio (#348).

## mlpack 1.0.9

_2014-07-28_

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

## mlpack 1.0.8

_2014-01-06_

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

## mlpack 1.0.7

_2013-10-04_

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

## mlpack 1.0.6

_2013-06-13_

  * Minor bugfix so that FastMKS gets built.

## mlpack 1.0.5

_2013-05-01_

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

## mlpack 1.0.4

_2013-02-08_

  * Force minimum Armadillo version to 2.4.2.

  * Better output of class types to streams; a class with a ToString() method
    implemented can be sent to a stream with operator<<.

  * Change return type of GMM::Estimate() to double (#257).

  * Style fixes for k-means and RADICAL.

  * Handle size_t support correctly with Armadillo 3.6.2 (#258).

  * Add locality-sensitive hashing (LSH), found in src/mlpack/methods/lsh/.

  * Better tests for SGD (stochastic gradient descent) and NCA (neighborhood
    components analysis).

## mlpack 1.0.3

_2012-09-16_

  * Remove internal sparse matrix support because Armadillo 3.4.0 now includes
    it.  When using Armadillo versions older than 3.4.0, sparse matrix support
    is not available.

  * NCA (neighborhood components analysis) now support an arbitrary optimizer
    (#245), including stochastic gradient descent (#249).

## mlpack 1.0.2

_2012-08-15_

  * Added density estimation trees, found in src/mlpack/methods/det/.

  * Added non-negative matrix factorization, found in src/mlpack/methods/nmf/.

  * Added experimental cover tree implementation, found in
    src/mlpack/core/tree/cover_tree/ (#157).

  * Better reporting of boost::program_options errors (#225).

  * Fix for timers on Windows (#212, #211).

  * Fix for allknn and allkfn output (#204).

  * Sparse coding dictionary initialization is now a template parameter (#220).

## mlpack 1.0.1

_2012-03-03_

  * Added kernel principal components analysis (kernel PCA), found in
    src/mlpack/methods/kernel_pca/ (#74).

  * Fix for Lovasz-Theta AugLagrangian tests (#182).

  * Fixes for allknn output (#185, #186).

  * Added range search executable (#192).

  * Adapted citations in documentation to BibTeX; no citations in -h output
    (#195).

  * Stop use of 'const char*' and prefer 'std::string' (#176).

  * Support seeds for random numbers (#177).

## mlpack 1.0.0

_2011-12-17_

  * Initial release.  See any resolved tickets numbered less than #196 or
    execute this query:
    http://www.mlpack.org/trac/query?status=closed&milestone=mlpack+1.0.0
