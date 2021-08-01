
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_dbscan_random_point_selection.hpp:

Program Listing for File random_point_selection.hpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_dbscan_random_point_selection.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/dbscan/random_point_selection.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_DBSCAN_RANDOM_POINT_SELECTION_HPP
   #define MLPACK_METHODS_DBSCAN_RANDOM_POINT_SELECTION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace dbscan {
   
   class RandomPointSelection
   {
    public:
     template<typename MatType>
     size_t Select(const size_t /* point */,
                   const MatType& data)
     {
       // Initialize the length of the unvisited bitset.
       size_t size = data.n_cols; // Get the size of points.
       if (unvisited.size() != size)
         unvisited.resize(size, true); // Resize & Set bitset to one.
   
       // Count the unvisited points and generate nth index randomly.
       const size_t max = std::count(unvisited.begin(), unvisited.end(), true);
       const size_t index = math::RandInt(max);
   
       // Select the index'th unvisited point.
       size_t found = 0;
       for (size_t i = 0; i < unvisited.size(); ++i)
       {
         if (unvisited[i])
           ++found;
   
         if (found > index)
         {
           unvisited[i].flip(); // Set unvisited point to visited point.
           return i;
         }
       }
       return 0; // Not sure if it is possible to get here.
     }
   
    private:
     // Bitset for unvisited points. If true, mean unvisited.
     std::vector<bool> unvisited;
   };
   
   } // namespace dbscan
   } // namespace mlpack
   
   #endif
