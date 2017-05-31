# mlpack test executable.
add_executable(mlpack_test
  activation_functions_test.cpp
  adaboost_test.cpp
  adam_test.cpp
  ada_delta_test.cpp
  ada_grad_test.cpp
  akfn_test.cpp
  aknn_test.cpp
  ann_layer_test.cpp
  arma_extend_test.cpp
  armadillo_svd_test.cpp
  aug_lagrangian_test.cpp
  binarize_test.cpp
  block_krylov_svd_test.cpp
  cf_test.cpp
  cli_test.cpp
  convolution_test.cpp
  convolutional_network_test.cpp
  cosine_tree_test.cpp
  dbscan_test.cpp
  decision_stump_test.cpp
  decision_tree_test.cpp
  det_test.cpp
  distribution_test.cpp
  drusilla_select_test.cpp
  emst_test.cpp
  fastmks_test.cpp
  feedforward_network_test.cpp
  gmm_test.cpp
  gradient_descent_test.cpp
  hmm_test.cpp
  hoeffding_tree_test.cpp
  hyperplane_test.cpp
  imputation_test.cpp
  ind2sub_test.cpp
  init_rules_test.cpp
  kernel_test.cpp
  kernel_pca_test.cpp
  kernel_traits_test.cpp
  kfn_test.cpp
  kmeans_test.cpp
  knn_test.cpp
  krann_search_test.cpp
  ksinit_test.cpp
  lars_test.cpp
  lbfgs_test.cpp
  lin_alg_test.cpp
  linear_regression_test.cpp
  load_save_test.cpp
  local_coordinate_coding_test.cpp
  log_test.cpp
  logistic_regression_test.cpp
  lrsdp_test.cpp
  lsh_test.cpp
  math_test.cpp
  matrix_completion_test.cpp
  maximal_inputs_test.cpp
  mean_shift_test.cpp
  metric_test.cpp
  minibatch_sgd_test.cpp
  mlpack_test.cpp
  momentum_sgd_test.cpp
  nbc_test.cpp
  nca_test.cpp
  nmf_test.cpp
  nystroem_method_test.cpp
  octree_test.cpp
  pca_test.cpp
  perceptron_test.cpp
  qdafn_test.cpp
  quic_svd_test.cpp
  radical_test.cpp
  randomized_svd_test.cpp
  range_search_test.cpp
  recurrent_network_test.cpp
  rectangle_tree_test.cpp
  regularized_svd_test.cpp
  rl_components_test.cpp
  rmsprop_test.cpp
  sa_test.cpp
  sdp_primal_dual_test.cpp
  sgd_test.cpp
  sgdr_test.cpp
  snapshot_ensembles.cpp
  serialization.hpp
  serialization.cpp
  serialization_test.cpp
  smorms3_test.cpp
  softmax_regression_test.cpp
  sort_policy_test.cpp
  sparse_autoencoder_test.cpp
  sparse_coding_test.cpp
  spill_tree_test.cpp
  split_data_test.cpp
  svd_batch_test.cpp
  svd_incremental_test.cpp
  termination_policy_test.cpp
  tree_test.cpp
  tree_traits_test.cpp
  union_find_test.cpp
  svd_batch_test.cpp
  svd_incremental_test.cpp
  nystroem_method_test.cpp
  armadillo_svd_test.cpp
  ub_tree_test.cpp
  vantage_point_tree_test.cpp
  prefixedoutstream_test.cpp
  timer_test.cpp
)
# Link dependencies of test executable.
target_link_libraries(mlpack_test
  mlpack
  ${BOOST_unit_test_framework_LIBRARY}
)

# Copy test data into right place.
add_custom_command(TARGET mlpack_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/data/
      ${PROJECT_BINARY_DIR}
)
add_custom_command(TARGET mlpack_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E tar xjpf mnist_first250_training_4s_and_9s.tar.bz2
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

# List of tests in the testing suite
add_test(NAME   ActivationFunctionsTest     COMMAND   mlpack_test   -t   ActivationFunctionsTest     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   AdaBoostTest                COMMAND   mlpack_test   -t   AdaBoostTest                WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   AdamTest                    COMMAND   mlpack_test   -t   AdamTest                    WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   AdaDeltaTest                COMMAND   mlpack_test   -t   AdaDeltaTest                WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   AdaGradTest                 COMMAND   mlpack_test   -t   AdaGradTest                 WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   AKFNTest                    COMMAND   mlpack_test   -t   AKFNTest                    WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   AKNNTest                    COMMAND   mlpack_test   -t   AKNNTest                    WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   ANNLayerTest                COMMAND   mlpack_test   -t   ANNLayerTest                WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   ArmaExtendTest              COMMAND   mlpack_test   -t   ArmaExtendTest              WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   ArmadilloSVDTest            COMMAND   mlpack_test   -t   ArmadilloSVDTest            WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   AugLagrangianTest           COMMAND   mlpack_test   -t   AugLagrangianTest           WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   BinarizeTest                COMMAND   mlpack_test   -t   BinarizeTest                WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   BlockKrylovSVDTest          COMMAND   mlpack_test   -t   BlockKrylovSVDTest          WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   CFTest                      COMMAND   mlpack_test   -t   CFTest                      WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   CLITest                     COMMAND   mlpack_test   -t   CLITest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   ConvolutionTest             COMMAND   mlpack_test   -t   ConvolutionTest             WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   ConvolutionalNetworkTest    COMMAND   mlpack_test   -t   ConvolutionalNetworkTest    WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   CosineTreeTest              COMMAND   mlpack_test   -t   CosineTreeTest              WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   DBSCANTest                  COMMAND   mlpack_test   -t   DBSCANTest                  WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   DecisionStumpTest           COMMAND   mlpack_test   -t   DecisionStumpTest           WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   DecisionTreeTest            COMMAND   mlpack_test   -t   DecisionTreeTest            WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   DETTest                     COMMAND   mlpack_test   -t   DETTest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   DistributionTest            COMMAND   mlpack_test   -t   DistributionTest            WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   DrusillaSelectTest          COMMAND   mlpack_test   -t   DrusillaSelectTest          WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   EMSTTest                    COMMAND   mlpack_test   -t   EMSTTest                    WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   FastMKSTest                 COMMAND   mlpack_test   -t   FastMKSTest                 WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   FeedForwardNetworkTest      COMMAND   mlpack_test   -t   FeedForwardNetworkTest      WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   GMMTest                     COMMAND   mlpack_test   -t   GMMTest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   GradientDescentTest         COMMAND   mlpack_test   -t   GradientDescentTest         WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   HMMTest                     COMMAND   mlpack_test   -t   HMMTest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   HoeffdingTreeTest           COMMAND   mlpack_test   -t   HoeffdingTreeTest           WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   HyperplaneTest              COMMAND   mlpack_test   -t   HyperplaneTest              WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   ImputationTest              COMMAND   mlpack_test   -t   ImputationTest              WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   ind2subTest                 COMMAND   mlpack_test   -t   ind2subTest                 WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   InitRulesTest               COMMAND   mlpack_test   -t   InitRulesTest               WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   KernelTest                  COMMAND   mlpack_test   -t   KernelTest                  WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   KernelPCATest               COMMAND   mlpack_test   -t   KernelPCATest               WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   KernelTraitsTest            COMMAND   mlpack_test   -t   KernelTraitsTest            WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   KFNTest                     COMMAND   mlpack_test   -t   KFNTest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   KMeansTest                  COMMAND   mlpack_test   -t   KMeansTest                  WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   KNNTest                     COMMAND   mlpack_test   -t   KNNTest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   KRANNTest                   COMMAND   mlpack_test   -t   KRANNTest                   WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   KSInitialization            COMMAND   mlpack_test   -t   KSInitialization            WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   LARSTest                    COMMAND   mlpack_test   -t   LARSTest                    WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   LBFGSTest                   COMMAND   mlpack_test   -t   LBFGSTest                   WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   LinAlgTest                  COMMAND   mlpack_test   -t   LinAlgTest                  WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   LinearRegressionTest        COMMAND   mlpack_test   -t   LinearRegressionTest        WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   LoadSaveTest                COMMAND   mlpack_test   -t   LoadSaveTest                WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   LocalCoordinateCodingTest   COMMAND   mlpack_test   -t   LocalCoordinateCodingTest   WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   LogTest                     COMMAND   mlpack_test   -t   LogTest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   LogisticRegressionTest      COMMAND   mlpack_test   -t   LogisticRegressionTest      WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   LRSDPTest                   COMMAND   mlpack_test   -t   LRSDPTest                   WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   LSHTest                     COMMAND   mlpack_test   -t   LSHTest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   MathTest                    COMMAND   mlpack_test   -t   MathTest                    WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   MatrixCompletionTest        COMMAND   mlpack_test   -t   MatrixCompletionTest        WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   MaximalInputsTest           COMMAND   mlpack_test   -t   MaximalInputsTest           WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   MeanShiftTest               COMMAND   mlpack_test   -t   MeanShiftTest               WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   LMetricTest                 COMMAND   mlpack_test   -t   LMetricTest                 WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   MiniBatchSGDTest            COMMAND   mlpack_test   -t   MiniBatchSGDTest            WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   MomentumSGDTest             COMMAND   mlpack_test   -t   MomentumSGDTest             WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   NBCTest                     COMMAND   mlpack_test   -t   NBCTest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   NCATest                     COMMAND   mlpack_test   -t   NCATest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   NMFTest                     COMMAND   mlpack_test   -t   NMFTest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   NystroemMethodTest          COMMAND   mlpack_test   -t   NystroemMethodTest          WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   OctreeTest                  COMMAND   mlpack_test   -t   OctreeTest                  WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   PCATest                     COMMAND   mlpack_test   -t   PCATest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   PerceptronTest              COMMAND   mlpack_test   -t   PerceptronTest              WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   QDAFNTest                   COMMAND   mlpack_test   -t   QDAFNTest                   WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   QUICSVDTest                 COMMAND   mlpack_test   -t   QUICSVDTest                 WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   RadicalTest                 COMMAND   mlpack_test   -t   RadicalTest                 WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   RandomizedSVDTest           COMMAND   mlpack_test   -t   RandomizedSVDTest           WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   RangeSearchTest             COMMAND   mlpack_test   -t   RangeSearchTest             WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   RecurrentNetworkTest        COMMAND   mlpack_test   -t   RecurrentNetworkTest        WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   RectangleTreeTest           COMMAND   mlpack_test   -t   RectangleTreeTest           WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   RegularizedSVDTest          COMMAND   mlpack_test   -t   RegularizedSVDTest          WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   RLComponentsTest            COMMAND   mlpack_test   -t   RLComponentsTest            WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   RMSPropTest                 COMMAND   mlpack_test   -t   RMSPropTest                 WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SATest                      COMMAND   mlpack_test   -t   SATest                      WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SdpPrimalDualTest           COMMAND   mlpack_test   -t   SdpPrimalDualTest           WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SGDTest                     COMMAND   mlpack_test   -t   SGDTest                     WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SGDRTest                    COMMAND   mlpack_test   -t   SGDRTest                    WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SnapshotEnsemblesTest       COMMAND   mlpack_test   -t   SnapshotEnsemblesTest       WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SerializationTest           COMMAND   mlpack_test   -t   SerializationTest           WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SMORMS3Test                 COMMAND   mlpack_test   -t   SMORMS3Test                 WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SoftmaxRegressionTest       COMMAND   mlpack_test   -t   SoftmaxRegressionTest       WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SortPolicyTest              COMMAND   mlpack_test   -t   SortPolicyTest              WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SparseAutoencoderTest       COMMAND   mlpack_test   -t   SparseAutoencoderTest       WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SparseCodingTest            COMMAND   mlpack_test   -t   SparseCodingTest            WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SpillTreeTest               COMMAND   mlpack_test   -t   SpillTreeTest               WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SplitDataTest               COMMAND   mlpack_test   -t   SplitDataTest               WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SVDBatchTest                COMMAND   mlpack_test   -t   SVDBatchTest                WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   SVDIncrementalTest          COMMAND   mlpack_test   -t   SVDIncrementalTest          WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   TerminationPolicyTest       COMMAND   mlpack_test   -t   TerminationPolicyTest       WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   TreeTest                    COMMAND   mlpack_test   -t   TreeTest                    WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   TreeTraitsTest              COMMAND   mlpack_test   -t   TreeTraitsTest              WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   UnionFindTest               COMMAND   mlpack_test   -t   UnionFindTest               WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   UBTreeTest                  COMMAND   mlpack_test   -t   UBTreeTest                  WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   VantagePointTreeTest        COMMAND   mlpack_test   -t   VantagePointTreeTest        WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   PrefixedOutStreamTest       COMMAND   mlpack_test   -t   PrefixedOutStreamTest       WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
add_test(NAME   TimerTest                   COMMAND   mlpack_test   -t   TimerTest                   WORKING_DIRECTORY   ${CMAKE_BINARY_DIR})
