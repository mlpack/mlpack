
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_image_info.hpp:

Program Listing for File image_info.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_image_info.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/image_info.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_IMAGE_INFO_HPP
   #define MLPACK_CORE_DATA_IMAGE_INFO_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "extension.hpp"
   
   namespace mlpack {
   namespace data {
   
   inline bool ImageFormatSupported(const std::string& fileName,
                                    const bool save = false);
   
   class ImageInfo
   {
    public:
     ImageInfo(const size_t width = 0,
               const size_t height = 0,
               const size_t channels = 3,
               const size_t quality = 90);
   
     const size_t& Width() const { return width; }
     size_t& Width() { return width; }
   
     const size_t& Height() const { return height; }
     size_t& Height() { return height; }
   
     const size_t& Channels() const { return channels; }
     size_t& Channels() { return channels; }
   
     const size_t& Quality() const { return quality; }
     size_t& Quality() { return quality; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(width));
       ar(CEREAL_NVP(channels));
       ar(CEREAL_NVP(height));
       ar(CEREAL_NVP(quality));
     }
   
    private:
     // To store the image width.
     size_t width;
   
     // To store the image height.
     size_t height;
   
     // To store the number of channels in the image.
     size_t channels;
   
     // Compression of the image if saved as jpg (0 - 100).
     size_t quality;
   };
   
   } // namespace data
   } // namespace mlpack
   
   // Include implementation of Image.
   #include "image_info_impl.hpp"
   
   #endif
