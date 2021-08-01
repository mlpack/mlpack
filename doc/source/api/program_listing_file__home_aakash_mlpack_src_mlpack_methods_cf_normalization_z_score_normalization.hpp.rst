
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_normalization_z_score_normalization.hpp:

Program Listing for File z_score_normalization.hpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_normalization_z_score_normalization.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/normalization/z_score_normalization.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_NORMALIZATION_Z_SCORE_NORMALIZATION_HPP
   #define MLPACK_METHODS_CF_NORMALIZATION_Z_SCORE_NORMALIZATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace cf {
   
   class ZScoreNormalization
   {
    public:
     // Empty constructor.
     ZScoreNormalization() : mean(0), stddev(1) { }
   
     void Normalize(arma::mat& data)
     {
       mean = arma::mean(data.row(2));
       stddev = arma::stddev(data.row(2));
   
       if (std::fabs(stddev) < 1e-14)
       {
         Log::Fatal << "Standard deviation of all existing ratings is 0! "
             << "This may indicate that all existing ratings are the same."
             << std::endl;
       }
   
       data.row(2) = (data.row(2) - mean) / stddev;
       // The algorithm omits rating of zero. If normalized rating equals zero,
       // it is set to the smallest positive float value.
       data.row(2).for_each([](double& x)
       {
         if (x == 0)
           x = std::numeric_limits<float>::min();
       });
     }
   
     void Normalize(arma::sp_mat& cleanedData)
     {
       // Caculate mean and stdev of all non zero ratings.
       arma::vec ratings = arma::nonzeros(cleanedData);
       mean = arma::mean(ratings);
       stddev = arma::stddev(ratings);
   
       if (std::fabs(stddev) < 1e-14)
       {
         Log::Fatal << "Standard deviation of all existing ratings is 0! "
             << "This may indicate that all existing ratings are the same."
             << std::endl;
       }
   
       // Subtract mean from existing rating and divide it by stddev.
       // TODO: consider using spmat::transform() instead of spmat iterators
       // TODO: http://arma.sourceforge.net/docs.html#transform
       arma::sp_mat::iterator it = cleanedData.begin();
       arma::sp_mat::iterator it_end = cleanedData.end();
       for (; it != it_end; ++it)
       {
         double tmp = (*it - mean) / stddev;
   
         // The algorithm omits rating of zero. If normalized rating equals zero,
         // it is set to the smallest positive float value.
         if (tmp == 0)
           tmp = std::numeric_limits<float>::min();
   
         *it = tmp;
       }
     }
   
     double Denormalize(const size_t /* user */,
                        const size_t /* item */,
                        const double rating) const
     {
       return rating * stddev + mean;
     }
   
     void Denormalize(const arma::Mat<size_t>& /* combinations */,
                      arma::vec& predictions) const
     {
       predictions = predictions * stddev + mean;
     }
   
     double Mean() const
     {
       return mean;
     }
   
     double Stddev() const
     {
       return stddev;
     }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(mean));
       ar(CEREAL_NVP(stddev));
     }
   
    private:
     double mean;
     double stddev;
   };
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
