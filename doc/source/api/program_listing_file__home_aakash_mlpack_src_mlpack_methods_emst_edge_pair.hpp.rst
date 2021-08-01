
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_emst_edge_pair.hpp:

Program Listing for File edge_pair.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_emst_edge_pair.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/emst/edge_pair.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_EMST_EDGE_PAIR_HPP
   #define MLPACK_METHODS_EMST_EDGE_PAIR_HPP
   
   #include <mlpack/prereqs.hpp>
   
   #include "union_find.hpp"
   
   namespace mlpack {
   namespace emst {
   
   class EdgePair
   {
    private:
     size_t lesser;
     size_t greater;
     double distance;
   
    public:
     EdgePair(const size_t lesser, const size_t greater, const double dist) :
         lesser(lesser), greater(greater), distance(dist)
     {
       Log::Assert(lesser != greater,
           "EdgePair::EdgePair(): indices cannot be equal.");
     }
   
     size_t Lesser() const { return lesser; }
     size_t& Lesser() { return lesser; }
   
     size_t Greater() const { return greater; }
     size_t& Greater() { return greater; }
   
     double Distance() const { return distance; }
     double& Distance() { return distance; }
   }; // class EdgePair
   
   } // namespace emst
   } // namespace mlpack
   
   #endif // MLPACK_METHODS_EMST_EDGE_PAIR_HPP
