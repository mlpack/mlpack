
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_cf_normalization_item_mean_normalization.hpp:

Program Listing for File item_mean_normalization.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_cf_normalization_item_mean_normalization.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/cf/normalization/item_mean_normalization.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_CF_NORMALIZATION_ITEM_MEAN_NORMALIZATION_HPP
   #define MLPACK_METHODS_CF_NORMALIZATION_ITEM_MEAN_NORMALIZATION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace cf {
   
   class ItemMeanNormalization
   {
    public:
     // Empty constructor.
     ItemMeanNormalization() { }
   
     void Normalize(arma::mat& data)
     {
       const size_t itemNum = arma::max(data.row(1)) + 1;
       itemMean = arma::vec(itemNum, arma::fill::zeros);
       // Number of ratings for each item.
       arma::Row<size_t> ratingNum(itemNum, arma::fill::zeros);
   
       // Sum ratings for each item.
       data.each_col([&](arma::vec& datapoint)
       {
         const size_t item = (size_t) datapoint(1);
         const double rating = datapoint(2);
         itemMean(item) += rating;
         ratingNum(item) += 1;
       });
   
       // Calculate item mean and subtract item mean from ratings.
       // Set item mean to 0 if the item has no rating.
       for (size_t i = 0; i < itemNum; ++i)
       {
         if (ratingNum(i) != 0)
           itemMean(i) /= ratingNum(i);
       }
   
       data.each_col([&](arma::vec& datapoint)
       {
         const size_t item = (size_t) datapoint(1);
         datapoint(2) -= itemMean(item);
         // The algorithm omits rating of zero. If normalized rating equals zero,
         // it is set to the smallest positive float value.
         if (datapoint(2) == 0)
           datapoint(2) = std::numeric_limits<float>::min();
       });
     }
   
     void Normalize(arma::sp_mat& cleanedData)
     {
       // Calculate itemMean.
       itemMean = arma::vec(cleanedData.n_rows, arma::fill::zeros);
       arma::Col<size_t> ratingNum(cleanedData.n_rows, arma::fill::zeros);
       arma::sp_mat::iterator it = cleanedData.begin();
       arma::sp_mat::iterator it_end = cleanedData.end();
       for (; it != it_end; ++it)
       {
         itemMean(it.row()) += *it;
         ratingNum(it.row()) += 1;
       }
       for (size_t i = 0; i < itemMean.n_elem; ++i)
       {
         if (ratingNum(i) != 0)
           itemMean(i) /= ratingNum(i);
       }
   
       // Normalize the data.
       it = cleanedData.begin();
       for (; it != cleanedData.end(); ++it)
       {
         double tmp = *it - itemMean(it.row());
   
         // The algorithm omits rating of zero. If normalized rating equals zero,
         // it is set to the smallest positive double value.
         if (tmp == 0)
           tmp = std::numeric_limits<float>::min();
   
         *it = tmp;
       }
     }
   
     double Denormalize(const size_t /* user */,
                        const size_t item,
                        const double rating) const
     {
       return rating + itemMean(item);
     }
   
     void Denormalize(const arma::Mat<size_t>& combinations,
                      arma::vec& predictions) const
     {
       for (size_t i = 0; i < predictions.n_elem; ++i)
       {
         const size_t item = combinations(1, i);
         predictions(i) += itemMean(item);
       }
     }
   
     const arma::vec& Mean() const { return itemMean; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(itemMean));
     }
   
    private:
     arma::vec itemMean;
   };
   
   } // namespace cf
   } // namespace mlpack
   
   #endif
