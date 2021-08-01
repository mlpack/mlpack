
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_mvu_mvu.hpp:

Program Listing for File mvu.hpp
================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_mvu_mvu.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/mvu/mvu.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_MVU_MVU_HPP
   #define MLPACK_METHODS_MVU_MVU_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace mvu {
   
   class MVU
   {
    public:
     MVU(const arma::mat& dataIn);
   
     void Unfold(const size_t newDim,
                 const size_t numNeighbors,
                 arma::mat& outputCoordinates);
   
    private:
     const arma::mat& data;
   };
   
   } // namespace mvu
   } // namespace mlpack
   
   #endif
