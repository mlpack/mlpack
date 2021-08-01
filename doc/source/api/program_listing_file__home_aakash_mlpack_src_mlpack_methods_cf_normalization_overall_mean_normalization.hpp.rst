
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_normalization_overall_mean_normalization.hpp:

Program Listing for File overall_mean_normalization.hpp
=======================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_normalization_overall_mean_normalization.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/normalization/overall_mean_normalization.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_NORMALIZATION_OVERALL_MEAN_NORMALIZATION_HPP
   #define MLPACK_METHODS_CF_NORMALIZATION_OVERALL_MEAN_NORMALIZATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace cf {
   
   class OverallMeanNormalization
   {
    public:
     // Empty constructor.
     OverallMeanNormalization() : mean(0) { }
   
     void Normalize(arma::mat& data)
     {
       mean = arma::mean(data.row(2));
       data.row(2) -= mean;
       // The algorithm omits rating of zero. If normalized rating equals zero,
       // it is set to the smallest positive float value.
       data.row(2).for_each([](double& x)
       {
         if (x == 0)
           x = std::numeric_limits<double>::min();
       });
     }
   
     void Normalize(arma::sp_mat& cleanedData)
     {
       // Caculate mean of all non zero ratings.
       if (cleanedData.n_nonzero != 0)
       {
         mean = arma::accu(cleanedData) / cleanedData.n_nonzero;
         // Subtract mean from all non zero ratings.
         arma::sp_mat::iterator it = cleanedData.begin();
         arma::sp_mat::iterator it_end = cleanedData.end();
         for (; it != it_end; ++it)
         {
           double tmp = *it - mean;
   
           // The algorithm omits rating of zero. If normalized rating equals zero,
           // it is set to the smallest positive float value.
           if (tmp == 0)
             tmp = std::numeric_limits<float>::min();
   
           *it = tmp;
         }
       }
       else
       {
         mean = 0;
         // cleanedData remains the same when mean == 0.
       }
     }
   
     double Denormalize(const size_t /* user */,
                        const size_t /* item */,
                        const double rating) const
     {
       return rating + mean;
     }
   
     void Denormalize(const arma::Mat<size_t>& /* combinations */,
                      arma::vec& predictions) const
     {
       predictions += mean;
     }
   
     double Mean() const
     {
       return mean;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(mean));
     }
   
    private:
     double mean;
   };
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
