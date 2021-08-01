
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_split_data.hpp:

Program Listing for File split_data.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_split_data.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/split_data.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_SPLIT_DATA_HPP
   #define MLPACK_CORE_DATA_SPLIT_DATA_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   
   template<typename InputType>
   void SplitHelper(const InputType& input,
                    InputType& train,
                    InputType& test,
                    const double testRatio,
                    const arma::uvec& order = arma::uvec())
   {
     const size_t testSize = static_cast<size_t>(input.n_cols * testRatio);
     const size_t trainSize = input.n_cols - testSize;
   
     // Initialising the sizes of outputs if not already initialized.
     train.set_size(input.n_rows, trainSize);
     test.set_size(input.n_rows, testSize);
   
     // Shuffling and spliting simultaneously.
     if (!order.is_empty())
     {
       if (trainSize > 0)
       {
         for (size_t i = 0; i < trainSize; ++i)
           train.col(i) = input.col(order(i));
       }
       if (trainSize < input.n_cols)
       {
         for (size_t i = trainSize; i < input.n_cols; ++i)
           test.col(i - trainSize) = input.col(order(i));
       }
     }
     // Spliting only.
     else
     {
       if (trainSize > 0)
         train = input.cols(0, trainSize - 1);
   
       if (trainSize < input.n_cols)
         test = input.cols(trainSize, input.n_cols - 1);
     }
   }
   
   template<typename T, typename LabelsType,
            typename = std::enable_if_t<arma::is_arma_type<LabelsType>::value> >
   void StratifiedSplit(const arma::Mat<T>& input,
                        const LabelsType& inputLabel,
                        arma::Mat<T>& trainData,
                        arma::Mat<T>& testData,
                        LabelsType& trainLabel,
                        LabelsType& testLabel,
                        const double testRatio,
                        const bool shuffleData = true)
   {
     const bool typeCheck = (arma::is_Row<LabelsType>::value)
         || (arma::is_Col<LabelsType>::value);
     if (!typeCheck)
       throw std::runtime_error("data::Split(): when stratified sampling is done, "
           "labels must have type `arma::Row<>`!");
     size_t trainIdx = 0;
     size_t testIdx = 0;
     size_t trainSize = 0;
     size_t testSize = 0;
     arma::uvec labelCounts;
     arma::uvec testLabelCounts;
     typename LabelsType::elem_type maxLabel = inputLabel.max();
   
     labelCounts.zeros(maxLabel+1);
     testLabelCounts.zeros(maxLabel+1);
   
     for (typename LabelsType::elem_type label : inputLabel)
       ++labelCounts[label];
   
     for (arma::uword labelCount : labelCounts)
     {
       testSize += floor(labelCount * testRatio);
       trainSize += labelCount - floor(labelCount * testRatio);
     }
   
     trainData.set_size(input.n_rows, trainSize);
     testData.set_size(input.n_rows, testSize);
     trainLabel.set_size(inputLabel.n_rows, trainSize);
     testLabel.set_size(inputLabel.n_rows, testSize);
   
     if (shuffleData)
     {
       arma::uvec order = arma::shuffle(
           arma::linspace<arma::uvec>(0, input.n_cols - 1, input.n_cols));
   
       for (arma::uword i : order)
       {
         typename LabelsType::elem_type label = inputLabel[i];
         if (testLabelCounts[label] < floor(labelCounts[label] * testRatio))
         {
           testLabelCounts[label] += 1;
           testData.col(testIdx) = input.col(i);
           testLabel[testIdx] = inputLabel[i];
           testIdx += 1;
         }
         else
         {
           trainData.col(trainIdx) = input.col(i);
           trainLabel[trainIdx] = inputLabel[i];
           trainIdx += 1;
         }
       }
     }
     else
     {
       for (arma::uword i = 0; i < input.n_cols; i++)
       {
         typename LabelsType::elem_type label = inputLabel[i];
         if (testLabelCounts[label] < floor(labelCounts[label] * testRatio))
         {
           testLabelCounts[label] += 1;
           testData.col(testIdx) = input.col(i);
           testLabel[testIdx] = inputLabel[i];
           testIdx += 1;
         }
         else
         {
           trainData.col(trainIdx) = input.col(i);
           trainLabel[trainIdx] = inputLabel[i];
           trainIdx += 1;
         }
       }
     }
   }
   
   template<typename T, typename LabelsType,
            typename = std::enable_if_t<arma::is_arma_type<LabelsType>::value> >
   void Split(const arma::Mat<T>& input,
              const LabelsType& inputLabel,
              arma::Mat<T>& trainData,
              arma::Mat<T>& testData,
              LabelsType& trainLabel,
              LabelsType& testLabel,
              const double testRatio,
              const bool shuffleData = true)
   {
     if (shuffleData)
     {
       arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(0,
           input.n_cols - 1, input.n_cols));
       SplitHelper(input, trainData, testData, testRatio, order);
       SplitHelper(inputLabel, trainLabel, testLabel, testRatio, order);
     }
     else
     {
       SplitHelper(input, trainData, testData, testRatio);
       SplitHelper(inputLabel, trainLabel, testLabel, testRatio);
     }
   }
   
   template<typename T>
   void Split(const arma::Mat<T>& input,
              arma::Mat<T>& trainData,
              arma::Mat<T>& testData,
              const double testRatio,
              const bool shuffleData = true)
   {
     if (shuffleData)
     {
       arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(0,
           input.n_cols - 1, input.n_cols));
       SplitHelper(input, trainData, testData, testRatio, order);
     }
     else
     {
       SplitHelper(input, trainData, testData, testRatio);
     }
   }
   
   template<typename T, typename LabelsType,
            typename = std::enable_if_t<arma::is_arma_type<LabelsType>::value> >
   std::tuple<arma::Mat<T>, arma::Mat<T>, LabelsType, LabelsType>
   Split(const arma::Mat<T>& input,
         const LabelsType& inputLabel,
         const double testRatio,
         const bool shuffleData = true,
         const bool stratifyData = false)
   {
     arma::Mat<T> trainData;
     arma::Mat<T> testData;
     LabelsType trainLabel;
     LabelsType testLabel;
   
     if (stratifyData)
     {
       StratifiedSplit(input, inputLabel, trainData, testData, trainLabel,
           testLabel, testRatio, shuffleData);
     }
     else
     {
       Split(input, inputLabel, trainData, testData, trainLabel, testLabel,
           testRatio, shuffleData);
     }
   
     return std::make_tuple(std::move(trainData),
                            std::move(testData),
                            std::move(trainLabel),
                            std::move(testLabel));
   }
   
   template<typename T>
   std::tuple<arma::Mat<T>, arma::Mat<T>>
   Split(const arma::Mat<T>& input,
         const double testRatio,
         const bool shuffleData = true)
   {
     arma::Mat<T> trainData;
     arma::Mat<T> testData;
     Split(input, trainData, testData, testRatio, shuffleData);
   
     return std::make_tuple(std::move(trainData),
                            std::move(testData));
   }
   
   template <typename FieldType, typename T,
             typename = std::enable_if_t<
                 arma::is_Col<typename FieldType::object_type>::value ||
                 arma::is_Mat_only<typename FieldType::object_type>::value>>
   void Split(const FieldType& input,
              const arma::field<T>& inputLabel,
              FieldType& trainData,
              arma::field<T>& trainLabel,
              FieldType& testData,
              arma::field<T>& testLabel,
              const double testRatio,
              const bool shuffleData = true)
   {
     if (shuffleData)
     {
       arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(0,
           input.n_cols - 1, input.n_cols));
       SplitHelper(input, trainData, testData, testRatio, order);
       SplitHelper(inputLabel, trainLabel, testLabel, testRatio, order);
     }
     else
     {
       SplitHelper(input, trainData, testData, testRatio);
       SplitHelper(inputLabel, trainLabel, testLabel, testRatio);
     }
   }
   
   template <class FieldType,
             class = std::enable_if_t<
                 arma::is_Col<typename FieldType::object_type>::value ||
                 arma::is_Mat_only<typename FieldType::object_type>::value>>
   void Split(const FieldType& input,
              FieldType& trainData,
              FieldType& testData,
              const double testRatio,
              const bool shuffleData = true)
   {
     if (shuffleData)
     {
       arma::uvec order = arma::shuffle(arma::linspace<arma::uvec>(0,
           input.n_cols - 1, input.n_cols));
       SplitHelper(input, trainData, testData, testRatio, order);
     }
     else
     {
       SplitHelper(input, trainData, testData, testRatio);
     }
   }
   
   template <class FieldType, typename T,
             class = std::enable_if_t<
                 arma::is_Col<typename FieldType::object_type>::value ||
                 arma::is_Mat_only<typename FieldType::object_type>::value>>
   std::tuple<FieldType, FieldType, arma::field<T>, arma::field<T>>
   Split(const FieldType& input,
         const arma::field<T>& inputLabel,
         const double testRatio,
         const bool shuffleData = true)
   {
     FieldType trainData;
     FieldType testData;
     arma::field<T> trainLabel;
     arma::field<T> testLabel;
   
     Split(input, inputLabel, trainData, trainLabel, testData, testLabel,
         testRatio, shuffleData);
   
     return std::make_tuple(std::move(trainData),
                            std::move(testData),
                            std::move(trainLabel),
                            std::move(testLabel));
   }
   
   template <class FieldType,
             class = std::enable_if_t<
                 arma::is_Col<typename FieldType::object_type>::value ||
                 arma::is_Mat_only<typename FieldType::object_type>::value>>
   std::tuple<FieldType, FieldType>
   Split(const FieldType& input,
         const double testRatio,
         const bool shuffleData = true)
   {
     FieldType trainData;
     FieldType testData;
     Split(input, trainData, testData, testRatio, shuffleData);
   
     return std::make_tuple(std::move(trainData),
                            std::move(testData));
   }
   
   } // namespace data
   } // namespace mlpack
   
   #endif
