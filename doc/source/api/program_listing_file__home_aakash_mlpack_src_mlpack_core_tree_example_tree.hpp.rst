
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_tree_example_tree.hpp:

Program Listing for File example_tree.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_tree_example_tree.hpp>` (``/home/aakash/mlpack/src/mlpack/core/tree/example_tree.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_TREE_EXAMPLE_TREE_HPP
   #define MLPACK_CORE_TREE_EXAMPLE_TREE_HPP
   
   namespace mlpack {
   namespace tree {
   
   template<typename MetricType = metric::LMetric<2, true>,
            typename StatisticType = EmptyStatistic,
            typename MatType = arma::mat>
   class ExampleTree
   {
    public:
     ExampleTree(const MatType& dataset,
                 MetricType& metric);
   
     size_t NumChildren() const;
   
     const ExampleTree& Child(const size_t i) const;
     ExampleTree& Child(const size_t i);
   
     ExampleTree* Parent() const;
   
     size_t NumPoints() const;
   
     size_t Point(const size_t i) const;
   
     size_t NumDescendants() const;
   
     size_t Descendant(const size_t i) const;
   
     const StatisticType& Stat() const;
     StatisticType& Stat();
   
     const MetricType& Metric() const;
     MetricType& Metric();
   
     double MinDistance(const MatType& point) const;
   
     double MinDistance(const ExampleTree& other) const;
   
     double MaxDistance(const MatType& point) const;
   
     double MaxDistance(const ExampleTree& other) const;
   
     math::Range RangeDistance(const MatType& point) const;
   
     math::Range RangeDistance(const ExampleTree& other) const;
   
     void Centroid(arma::vec& centroid) const;
   
     double FurthestDescendantDistance() const;
   
     double ParentDistance() const;
   
    private:
     StatisticType stat;
   
     MetricType& metric;
   };
   
   } // namespace tree
   } // namespace mlpack
   
   #endif
