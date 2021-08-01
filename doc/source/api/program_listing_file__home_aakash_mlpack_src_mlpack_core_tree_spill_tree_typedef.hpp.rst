
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_typedef.hpp:

Program Listing for File typedef.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_spill_tree_typedef.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/spill_tree/typedef.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_SPILL_TREE_TYPEDEF_HPP
   #define MLPACK_CORE_TREE_SPILL_TREE_TYPEDEF_HPP
   
   #include "../space_split/mean_space_split.hpp"
   #include "../space_split/midpoint_space_split.hpp"
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using SPTree = SpillTree<MetricType,
                            StatisticType,
                            MatType,
                            AxisOrthogonalHyperplane,
                            MidpointSpaceSplit>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using MeanSPTree = SpillTree<MetricType,
                                StatisticType,
                                MatType,
                                AxisOrthogonalHyperplane,
                                MeanSpaceSplit>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using NonOrtSPTree = SpillTree<MetricType,
                                  StatisticType,
                                  MatType,
                                  Hyperplane,
                                  MidpointSpaceSplit>;
   
   template<typename MetricType, typename StatisticType, typename MatType>
   using NonOrtMeanSPTree = SpillTree<MetricType,
                                      StatisticType,
                                      MatType,
                                      Hyperplane,
                                      MeanSpaceSplit>;
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
