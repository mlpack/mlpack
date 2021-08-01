
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_math_shuffle_data.hpp:

Program Listing for File shuffle_data.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_math_shuffle_data.hpp>` (``/home/aakash/mlpack/src/mlpack/core/math/shuffle_data.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_MATH_SHUFFLE_DATA_HPP
   #define MLPACK_CORE_MATH_SHUFFLE_DATA_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace math {
   
   template<typename MatType, typename LabelsType>
   void ShuffleData(const MatType& inputPoints,
                    const LabelsType& inputLabels,
                    MatType& outputPoints,
                    LabelsType& outputLabels,
                    const std::enable_if_t<!arma::is_SpMat<MatType>::value>* = 0,
                    const std::enable_if_t<!arma::is_Cube<MatType>::value>* = 0)
   {
     // Generate ordering.
     arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
         inputPoints.n_cols - 1, inputPoints.n_cols));
   
     outputPoints = inputPoints.cols(ordering);
     outputLabels = inputLabels.cols(ordering);
   }
   
   template<typename MatType, typename LabelsType>
   void ShuffleData(const MatType& inputPoints,
                    const LabelsType& inputLabels,
                    MatType& outputPoints,
                    LabelsType& outputLabels,
                    const std::enable_if_t<arma::is_SpMat<MatType>::value>* = 0,
                    const std::enable_if_t<!arma::is_Cube<MatType>::value>* = 0)
   {
     // Generate ordering.
     arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
         inputPoints.n_cols - 1, inputPoints.n_cols));
   
     // Extract coordinate list representation.
     arma::umat locations(2, inputPoints.n_nonzero);
     arma::Col<typename MatType::elem_type> values(inputPoints.n_nonzero);
     typename MatType::const_iterator it = inputPoints.begin();
     size_t index = 0;
     while (it != inputPoints.end())
     {
       locations(0, index) = it.row();
       locations(1, index) = ordering[it.col()];
       values(index) = (*it);
       ++it;
       ++index;
     }
   
     if (&inputPoints == &outputPoints || &inputLabels == &outputLabels)
     {
       MatType newOutputPoints(locations, values, inputPoints.n_rows,
           inputPoints.n_cols, true);
       LabelsType newOutputLabels(inputLabels.n_elem);
       newOutputLabels.cols(ordering) = inputLabels;
   
       outputPoints = std::move(newOutputPoints);
       outputLabels = std::move(newOutputLabels);
     }
     else
     {
       outputPoints = MatType(locations, values, inputPoints.n_rows,
           inputPoints.n_cols, true);
       outputLabels.set_size(inputLabels.n_elem);
       outputLabels.cols(ordering) = inputLabels;
     }
   }
   
   template<typename MatType, typename LabelsType>
   void ShuffleData(const MatType& inputPoints,
                    const LabelsType& inputLabels,
                    MatType& outputPoints,
                    LabelsType& outputLabels,
                    const std::enable_if_t<!arma::is_SpMat<MatType>::value>* = 0,
                    const std::enable_if_t<arma::is_Cube<MatType>::value>* = 0,
                    const std::enable_if_t<arma::is_Cube<LabelsType>::value>* = 0)
   {
     // Generate ordering.
     arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
         inputPoints.n_cols - 1, inputPoints.n_cols));
   
     // Properly handle the case where the input and output data are the same
     // object.
     MatType* outputPointsPtr = &outputPoints;
     LabelsType* outputLabelsPtr = &outputLabels;
     if (&inputPoints == &outputPoints)
       outputPointsPtr = new MatType();
     if (&inputLabels == &outputLabels)
       outputLabelsPtr = new LabelsType();
   
     outputPointsPtr->set_size(inputPoints.n_rows, inputPoints.n_cols,
         inputPoints.n_slices);
     outputLabelsPtr->set_size(inputLabels.n_rows, inputLabels.n_cols,
         inputLabels.n_slices);
     for (size_t i = 0; i < ordering.n_elem; ++i)
     {
       outputPointsPtr->tube(0, ordering[i], outputPointsPtr->n_rows - 1,
           ordering[i]) = inputPoints.tube(0, i, inputPoints.n_rows - 1, i);
       outputLabelsPtr->tube(0, ordering[i], outputLabelsPtr->n_rows - 1,
           ordering[i]) = inputLabels.tube(0, i, inputLabels.n_rows - 1, i);
     }
   
     // Clean up memory if needed.
     if (&inputPoints == &outputPoints)
     {
       outputPoints = std::move(*outputPointsPtr);
       delete outputPointsPtr;
     }
   
     if (&inputLabels == &outputLabels)
     {
       outputLabels = std::move(*outputLabelsPtr);
       delete outputLabelsPtr;
     }
   }
   
   template<typename MatType, typename LabelsType, typename WeightsType>
   void ShuffleData(const MatType& inputPoints,
                    const LabelsType& inputLabels,
                    const WeightsType& inputWeights,
                    MatType& outputPoints,
                    LabelsType& outputLabels,
                    WeightsType& outputWeights,
                    const std::enable_if_t<!arma::is_SpMat<MatType>::value>* = 0,
                    const std::enable_if_t<!arma::is_Cube<MatType>::value>* = 0)
   {
     // Generate ordering.
     arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
         inputPoints.n_cols - 1, inputPoints.n_cols));
   
     outputPoints = inputPoints.cols(ordering);
     outputLabels = inputLabels.cols(ordering);
     outputWeights = inputWeights.cols(ordering);
   }
   
   template<typename MatType, typename LabelsType, typename WeightsType>
   void ShuffleData(const MatType& inputPoints,
                    const LabelsType& inputLabels,
                    const WeightsType& inputWeights,
                    MatType& outputPoints,
                    LabelsType& outputLabels,
                    WeightsType& outputWeights,
                    const std::enable_if_t<arma::is_SpMat<MatType>::value>* = 0,
                    const std::enable_if_t<!arma::is_Cube<MatType>::value>* = 0)
   {
     // Generate ordering.
     arma::uvec ordering = arma::shuffle(arma::linspace<arma::uvec>(0,
         inputPoints.n_cols - 1, inputPoints.n_cols));
   
     // Extract coordinate list representation.
     arma::umat locations(2, inputPoints.n_nonzero);
     arma::Col<typename MatType::elem_type> values(inputPoints.n_nonzero);
     typename MatType::const_iterator it = inputPoints.begin();
     size_t index = 0;
     while (it != inputPoints.end())
     {
       locations(0, index) = it.row();
       locations(1, index) = ordering[it.col()];
       values(index) = (*it);
       ++it;
       ++index;
     }
   
     if (&inputPoints == &outputPoints || &inputLabels == &outputLabels ||
         &inputWeights == &outputWeights)
     {
       MatType newOutputPoints(locations, values, inputPoints.n_rows,
           inputPoints.n_cols, true);
       LabelsType newOutputLabels(inputLabels.n_elem);
       WeightsType newOutputWeights(inputWeights.n_elem);
       newOutputLabels.cols(ordering) = inputLabels;
       newOutputWeights.cols(ordering) = inputWeights;
   
       outputPoints = std::move(newOutputPoints);
       outputLabels = std::move(newOutputLabels);
       outputWeights = std::move(newOutputWeights);
     }
     else
     {
       outputPoints = MatType(locations, values, inputPoints.n_rows,
           inputPoints.n_cols, true);
       outputLabels.set_size(inputLabels.n_elem);
       outputLabels.cols(ordering) = inputLabels;
       outputWeights.set_size(inputWeights.n_elem);
       outputWeights.cols(ordering) = inputWeights;
     }
   }
   
   } // namespace math
   } // namespace mlpack
   
   #endif
