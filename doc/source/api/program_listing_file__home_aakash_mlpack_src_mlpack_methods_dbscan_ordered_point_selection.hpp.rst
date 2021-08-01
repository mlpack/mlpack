
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_dbscan_ordered_point_selection.hpp:

Program Listing for File ordered_point_selection.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_dbscan_ordered_point_selection.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/dbscan/ordered_point_selection.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DBSCAN_ORDERED_POINT_SELECTION_HPP
   #define MLPACK_METHODS_DBSCAN_ORDERED_POINT_SELECTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace dbscan {
   
   class OrderedPointSelection
   {
    public:
     template<typename MatType>
     static size_t Select(const size_t point,
                          const MatType& /* data */)
     {
       return point; // Just return point.
     }
   };
   
   } // namespace dbscan
   } // namespace mlpack
   
   #endif
