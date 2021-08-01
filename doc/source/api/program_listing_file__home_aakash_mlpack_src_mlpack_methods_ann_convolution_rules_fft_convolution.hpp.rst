
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_convolution_rules_fft_convolution.hpp:

Program Listing for File fft_convolution.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_convolution_rules_fft_convolution.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/convolution_rules/fft_convolution.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_CONVOLUTION_RULES_FFT_CONVOLUTION_HPP
   #define MLPACK_METHODS_ANN_CONVOLUTION_RULES_FFT_CONVOLUTION_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "border_modes.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename BorderMode = FullConvolution, const bool padLastDim = false>
   class FFTConvolution
   {
    public:
     /*
      * Perform a convolution through fft (valid mode). This method only supports
      * input which is even on the last dimension. In case of an odd input width, a
      * user can manually pad the input or specify the padLastDim parameter which
      * takes care of the padding. The filter instead can have any size. When using
      * the valid mode the filter has to be smaller than the input.
      *
      * @param input Input used to perform the convolution.
      * @param filter Filter used to perform the convolution.
      * @param output Output data that contains the results of the convolution.
      */
     template<typename eT, typename Border = BorderMode>
     static typename std::enable_if<
         std::is_same<Border, ValidConvolution>::value, void>::type
     Convolution(const arma::Mat<eT>& input,
                 const arma::Mat<eT>& filter,
                 arma::Mat<eT>& output)
     {
       arma::Mat<eT> inputPadded = input;
       arma::Mat<eT> filterPadded = filter;
   
       if (padLastDim)
         inputPadded.resize(inputPadded.n_rows, inputPadded.n_cols + 1);
   
       // Pad filter and input to the output shape.
       filterPadded.resize(inputPadded.n_rows, inputPadded.n_cols);
   
       arma::Mat<eT> temp = arma::real(ifft2(arma::fft2(inputPadded) % arma::fft2(
           filterPadded)));
   
       // Extract the region of interest. We don't need to handle the padLastDim in
       // a special way we just cut it out from the output matrix.
       output = temp.submat(filter.n_rows - 1, filter.n_cols - 1,
           input.n_rows - 1, input.n_cols - 1);
     }
   
     /*
      * Perform a convolution through fft (full mode). This method only supports
      * input which is even on the last dimension. In case of an odd input width, a
      * user can manually pad the input or specify the padLastDim parameter which
      * takes care of the padding. The filter instead can have any size.
      *
      * @param input Input used to perform the convolution.
      * @param filter Filter used to perform the convolution.
      * @param output Output data that contains the results of the convolution.
      */
     template<typename eT, typename Border = BorderMode>
     static typename std::enable_if<
         std::is_same<Border, FullConvolution>::value, void>::type
     Convolution(const arma::Mat<eT>& input,
                 const arma::Mat<eT>& filter,
                 arma::Mat<eT>& output)
     {
       // In case of the full convolution outputRows and outputCols doesn't
       // represent the true output size when the padLastDim parameter is set,
       // instead it's the working size.
       const size_t outputRows = input.n_rows + 2 * (filter.n_rows - 1);
       size_t outputCols = input.n_cols + 2 * (filter.n_cols - 1);
   
       if (padLastDim)
           outputCols++;
   
       // Pad filter and input to the working output shape.
       arma::Mat<eT> inputPadded = arma::zeros<arma::Mat<eT> >(outputRows,
           outputCols);
       inputPadded.submat(filter.n_rows - 1, filter.n_cols - 1,
             filter.n_rows - 1 + input.n_rows - 1,
             filter.n_cols - 1 + input.n_cols - 1) = input;
   
       arma::Mat<eT> filterPadded = filter;
       filterPadded.resize(outputRows, outputCols);
   
       // Perform FFT and IFFT
       arma::Mat<eT> temp = arma::real(ifft2(arma::fft2(inputPadded) % arma::fft2(
           filterPadded)));
   
       // Extract the region of interest. We don't need to handle the padLastDim
       // parameter in a special way we just cut it out from the output matrix.
       output = temp.submat(filter.n_rows - 1, filter.n_cols - 1,
           2 * (filter.n_rows - 1) + input.n_rows - 1,
           2 * (filter.n_cols - 1) + input.n_cols - 1);
     }
   
     /*
      * Perform a convolution through fft using 3rd order tensors. This method only
      * supports input which is even on the last dimension. In case of an odd input
      * width, a user can manually pad the input or specify the padLastDim
      * parameter which takes care of the padding. The filter instead can have any
      * size.
      *
      * @param input Input used to perform the convolution.
      * @param filter Filter used to perform the convolution.
      * @param output Output data that contains the results of the convolution.
      */
     template<typename eT>
     static void Convolution(const arma::Cube<eT>& input,
                             const arma::Cube<eT>& filter,
                             arma::Cube<eT>& output)
     {
       arma::Mat<eT> convOutput;
       FFTConvolution<BorderMode>::Convolution(input.slice(0), filter.slice(0),
           convOutput);
   
       output = arma::Cube<eT>(convOutput.n_rows, convOutput.n_cols,
           input.n_slices);
       output.slice(0) = convOutput;
   
       for (size_t i = 1; i < input.n_slices; ++i)
       {
         FFTConvolution<BorderMode>::Convolution(input.slice(i), filter.slice(i),
             output.slice(i));
       }
     }
   
     /*
      * Perform a convolution through fft using dense matrix as input and a 3rd
      * order tensors as filter and output. This method only supports input which
      * is even on the last dimension. In case of an odd input width, a user can
      * manually pad the input or specify the padLastDim parameter which takes care
      * of the padding. The filter instead can have any size.
      *
      * @param input Input used to perform the convolution.
      * @param filter Filter used to perform the convolution.
      * @param output Output data that contains the results of the convolution.
      */
     template<typename eT>
     static void Convolution(const arma::Mat<eT>& input,
                             const arma::Cube<eT>& filter,
                             arma::Cube<eT>& output)
     {
       arma::Mat<eT> convOutput;
       FFTConvolution<BorderMode>::Convolution(input, filter.slice(0),
           convOutput);
   
       output = arma::Cube<eT>(convOutput.n_rows, convOutput.n_cols,
           filter.n_slices);
       output.slice(0) = convOutput;
   
       for (size_t i = 1; i < filter.n_slices; ++i)
       {
         FFTConvolution<BorderMode>::Convolution(input, filter.slice(i),
             output.slice(i));
       }
     }
   
     /*
      * Perform a convolution using a 3rd order tensors as input and output and a
      * dense matrix as filter.
      *
      * @param input Input used to perform the convolution.
      * @param filter Filter used to perform the convolution.
      * @param output Output data that contains the results of the convolution.
      */
     template<typename eT>
     static void Convolution(const arma::Cube<eT>& input,
                             const arma::Mat<eT>& filter,
                             arma::Cube<eT>& output)
     {
       arma::Mat<eT> convOutput;
       FFTConvolution<BorderMode>::Convolution(input.slice(0), filter,
           convOutput);
   
       output = arma::Cube<eT>(convOutput.n_rows, convOutput.n_cols,
           input.n_slices);
       output.slice(0) = convOutput;
   
       for (size_t i = 1; i < input.n_slices; ++i)
       {
         FFTConvolution<BorderMode>::Convolution(input.slice(i), filter,
             output.slice(i));
       }
     }
   };  // class FFTConvolution
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
