
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_interpolation_policies_average_interpolation.hpp:

Program Listing for File average_interpolation.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_interpolation_policies_average_interpolation.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/interpolation_policies/average_interpolation.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_AVERAGE_INTERPOLATION_HPP
   #define MLPACK_METHODS_CF_AVERAGE_INTERPOLATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace cf {
   
   class AverageInterpolation
   {
    public:
     // Empty constructor.
     AverageInterpolation() { }
   
     AverageInterpolation(const arma::sp_mat& /* cleanedData */) { }
   
     template <typename VectorType,
               typename DecompositionPolicy>
     void GetWeights(VectorType&& weights,
                     const DecompositionPolicy& /* decomposition */,
                     const size_t /* queryUser */,
                     const arma::Col<size_t>& neighbors,
                     const arma::vec& /* similarities */,
                     const arma::sp_mat& /* cleanedData */)
     {
       if (neighbors.n_elem == 0)
       {
         Log::Fatal << "Require: neighbors.n_elem > 0. There should be at "
             << "least one neighbor!" << std::endl;
       }
   
       if (weights.n_elem != neighbors.n_elem)
       {
         Log::Fatal << "The size of the first parameter (weights) should "
             << "be set to the number of neighbors before calling GetWeights()."
             << std::endl;
       }
   
       weights.fill(1.0 / neighbors.n_elem);
     }
   };
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
