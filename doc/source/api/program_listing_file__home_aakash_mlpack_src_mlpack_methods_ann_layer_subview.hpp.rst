
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_subview.hpp:

Program Listing for File subview.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_subview.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/subview.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_SUBVIEW_HPP
   #define MLPACK_METHODS_ANN_LAYER_SUBVIEW_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/activation_functions/identity_function.hpp>
   
   namespace mlpack {
   namespace ann {
   
   template <
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class Subview
   {
    public:
     Subview(const size_t inSize = 1,
             const size_t beginRow = 0,
             const size_t endRow = 0,
             const size_t beginCol = 0,
             const size_t endCol = 0) :
         inSize(inSize),
         beginRow(beginRow),
         endRow(endRow),
         beginCol(beginCol),
         endCol(endCol)
     {
       /* Nothing to do here */
     }
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output)
     {
       size_t batchSize = input.n_cols / inSize;
   
       // Check if subview parameters are within the indices of input sample.
       endRow = ((endRow < input.n_rows) && (endRow >= beginRow))?
           endRow : (input.n_rows - 1);
       endCol = ((endCol < inSize) && (endCol >= beginCol)) ?
           endCol : (inSize - 1);
   
       output.set_size(
           (endRow - beginRow + 1) * (endCol - beginCol + 1), batchSize);
   
       size_t batchBegin = beginCol;
       size_t batchEnd = endCol;
   
       // Check whether the input is already in desired form.
       if ((input.n_rows != ((endRow - beginRow + 1) *
           (endCol - beginCol + 1))) || (input.n_cols != batchSize))
       {
         for (size_t i = 0; i < batchSize; ++i)
         {
           output.col(i) = arma::vectorise(
               input.submat(beginRow, batchBegin, endRow, batchEnd));
   
           // Move to next batch.
           batchBegin += inSize;
           batchEnd += inSize;
         }
       }
       else
       {
         output = input;
       }
     }
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& /* input */,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g)
     {
       g = gy;
     }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     size_t InSize() const { return inSize; }
   
     size_t const& BeginRow() const { return beginRow; }
     size_t& BeginRow() { return beginRow; }
   
     size_t const& EndRow() const { return endRow; }
     size_t& EndRow() { return endRow; }
   
     size_t const& BeginCol() const { return beginCol; }
     size_t& BeginCol() { return beginCol; }
   
     size_t const& EndCol() const { return endCol; }
     size_t& EndCol() { return endCol; }
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(inSize));
       ar(CEREAL_NVP(beginRow));
       ar(CEREAL_NVP(endRow));
       ar(CEREAL_NVP(beginCol));
       ar(CEREAL_NVP(endCol));
     }
   
    private:
     size_t inSize;
   
     size_t beginRow;
   
     size_t endRow;
   
     size_t beginCol;
   
     size_t endCol;
   
     OutputDataType delta;
   
     OutputDataType outputParameter;
   }; // class Subview
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
