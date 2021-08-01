
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_cv_meta_info_extractor.hpp:

Program Listing for File meta_info_extractor.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_cv_meta_info_extractor.hpp>` (``/home/aakash/mlpack/src/mlpack/core/cv/meta_info_extractor.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_CV_META_INFO_EXTRACTOR_HPP
   #define MLPACK_CORE_CV_META_INFO_EXTRACTOR_HPP
   
   #include <type_traits>
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/util/sfinae_utility.hpp>
   
   namespace mlpack {
   namespace cv {
   
   template<typename MatType,
            typename PredictionsType,
            typename WeightsType,
            bool DatasetInfo,
            bool NumClasses>
   struct TrainForm;
   
   #if _MSC_VER <= 1916
   // Visual Studio 2017 version 15.9 or older.
   // Due to an internal MSVC compiler bug (MSVC ) we can't use two parameter
   // packs. So we have to write multiple TrainFormBase forms.
   template<typename PT, typename WT, typename T1, typename T2>
   struct TrainFormBase4
   {
     using PredictionsType = PT;
     using WeightsType = WT;
   
     /* A minimum number of parameters that should be inferred */
     static const size_t MinNumberOfAdditionalArgs = 1;
   
     template<typename Class, typename RT, typename... Ts>
     using Type = RT(Class::*)(T1, T2, Ts...);
   };
   
   template<typename PT, typename WT, typename T1, typename T2, typename T3>
   struct TrainFormBase5
   {
     using PredictionsType = PT;
     using WeightsType = WT;
   
     /* A minimum number of parameters that should be inferred */
     static const size_t MinNumberOfAdditionalArgs = 1;
   
     template<typename Class, typename RT, typename... Ts>
     using Type = RT(Class::*)(T1, T2, T3, Ts...);
   };
   
   template<typename PT, typename WT, typename T1, typename T2, typename T3,
       typename T4>
   struct TrainFormBase6
   {
     using PredictionsType = PT;
     using WeightsType = WT;
   
     /* A minimum number of parameters that should be inferred */
     static const size_t MinNumberOfAdditionalArgs = 1;
   
     template<typename Class, typename RT, typename... Ts>
     using Type = RT(Class::*)(T1, T2, T3, T4, Ts...);
   };
   
   template<typename PT, typename WT, typename T1, typename T2, typename T3,
       typename T4, typename T5>
   struct TrainFormBase7
   {
     using PredictionsType = PT;
     using WeightsType = WT;
   
     /* A minimum number of parameters that should be inferred */
     static const size_t MinNumberOfAdditionalArgs = 1;
   
     template<typename Class, typename RT, typename... Ts>
     using Type = RT(Class::*)(T1, T2, T3, T4, T5, Ts...);
   };
   
   template<typename MT, typename PT>
   struct TrainForm<MT, PT, void, false, false> : public TrainFormBase4<PT, void,
       const MT&, const PT&> {};
   
   template<typename MT, typename PT>
   struct TrainForm<MT, PT, void, true, false> : public TrainFormBase5<PT, void,
       const MT&, const data::DatasetInfo&, const PT&> {};
   
   template<typename MT, typename PT, typename WT>
   struct TrainForm<MT, PT, WT, false, false> : public TrainFormBase5<PT, WT,
       const MT&, const PT&, const WT&> {};
   
   template<typename MT, typename PT, typename WT>
   struct TrainForm<MT, PT, WT, true, false> : public TrainFormBase6<PT, WT,
       const MT&, const data::DatasetInfo&, const PT&, const WT&> {};
   
   template<typename MT, typename PT>
   struct TrainForm<MT, PT, void, false, true> : public TrainFormBase5<PT, void,
       const MT&, const PT&, const size_t> {};
   
   template<typename MT, typename PT>
   struct TrainForm<MT, PT, void, true, true> : public TrainFormBase6<PT, void,
       const MT&, const data::DatasetInfo&, const PT&, const size_t> {};
   
   template<typename MT, typename PT, typename WT>
   struct TrainForm<MT, PT, WT, false, true> : public TrainFormBase6<PT, WT,
       const MT&, const PT&, const size_t, const WT&> {};
   
   template<typename MT, typename PT, typename WT>
   struct TrainForm<MT, PT, WT, true, true> : public TrainFormBase7<PT, WT,
       const MT&, const data::DatasetInfo&, const PT&,
       const size_t, const WT&> {};
   #else
   template<typename PT, typename WT, typename... SignatureParams>
   struct TrainFormBase
   {
     using PredictionsType = PT;
     using WeightsType = WT;
   
     /* A minimum number of parameters that should be inferred */
     static const size_t MinNumberOfAdditionalArgs = 1;
   
     template<typename Class, typename RT, typename... Ts>
     using Type = RT(Class::*)(SignatureParams..., Ts...);
   };
   
   template<typename MT, typename PT>
   struct TrainForm<MT, PT, void, false, false> : public TrainFormBase<PT, void,
       const MT&, const PT&> {};
   
   template<typename MT, typename PT>
   struct TrainForm<MT, PT, void, true, false> : public TrainFormBase<PT, void,
       const MT&, const data::DatasetInfo&, const PT&> {};
   
   template<typename MT, typename PT, typename WT>
   struct TrainForm<MT, PT, WT, false, false> : public TrainFormBase<PT, WT,
       const MT&, const PT&, const WT&> {};
   
   template<typename MT, typename PT, typename WT>
   struct TrainForm<MT, PT, WT, true, false> : public TrainFormBase<PT, WT,
       const MT&, const data::DatasetInfo&, const PT&, const WT&> {};
   
   template<typename MT, typename PT>
   struct TrainForm<MT, PT, void, false, true> : public TrainFormBase<PT, void,
       const MT&, const PT&, const size_t> {};
   
   template<typename MT, typename PT>
   struct TrainForm<MT, PT, void, true, true> : public TrainFormBase<PT, void,
       const MT&, const data::DatasetInfo&, const PT&, const size_t> {};
   
   template<typename MT, typename PT, typename WT>
   struct TrainForm<MT, PT, WT, false, true> : public TrainFormBase<PT, WT,
       const MT&, const PT&, const size_t, const WT&> {};
   
   template<typename MT, typename PT, typename WT>
   struct TrainForm<MT, PT, WT, true, true> : public TrainFormBase<PT, WT,
       const MT&, const data::DatasetInfo&, const PT&,
       const size_t, const WT&> {};
   #endif
   
   /* A struct for indication that a right method form can't be found */
   struct NotFoundMethodForm
   {
     using PredictionsType = void*;
     using WeightsType = void*;
   };
   
   template<typename MLAlgorithm,
            template<class, template<class...> class, size_t> class... HMFs>
   struct SelectMethodForm;
   
   template<typename MLAlgorithm,
            template<class, template<class...> class, size_t> class HasMethodForm,
            template<class, template<class...> class, size_t> class... HMFs>
   struct SelectMethodForm<MLAlgorithm, HasMethodForm, HMFs...>
   {
     template<typename... Forms>
     class From
     {
       /* Declaration and definition of Implementation for the case when
        * RemainingForms are empty */
       template<typename... RemainingForms>
       struct Implementation
       {
         using NextSMF = SelectMethodForm<MLAlgorithm, HMFs...>;
         using Type = typename NextSMF::template From<Forms...>::Type;
       };
   
       /* The case when there is at least one remaining form */
       template<typename Form, typename... RemainingForms>
       struct Implementation<Form, RemainingForms...>
       {
         using Type = typename std::conditional<
             HasMethodForm<MLAlgorithm, Form::template Type,
                 Form::MinNumberOfAdditionalArgs>::value,
             Form,
             typename Implementation<RemainingForms...>::Type>::type;
       };
   
      public:
       using Type = typename Implementation<Forms...>::Type;
     };
   };
   
   template<typename MLAlgorithm>
   struct SelectMethodForm<MLAlgorithm>
   {
     template<typename... Forms>
     struct From
     {
       using Type = NotFoundMethodForm;
     };
   };
   
   template<typename MLAlgorithm,
            typename MT = arma::mat,
            typename PT = arma::Row<size_t>,
            typename WT = arma::rowvec>
   class MetaInfoExtractor
   {
     /* Defining type functions that check presence of Train methods of a given
      * form. Defining such functions for templated and non-templated Train
      * methods. */
     HAS_METHOD_FORM(Train, HasTrain);
     HAS_METHOD_FORM(template Train<>, HasTTrain);
     HAS_METHOD_FORM(template Train<const MT&>, HasMTrain);
     HAS_METHOD_FORM(SINGLE_ARG(template Train<const MT&, const PT&>), HasMPTrain);
     HAS_METHOD_FORM(SINGLE_ARG(template Train<const MT&, const PT&, const WT&>),
         HasMPWTrain);
   
     /* Forms of Train for regression algorithms */
     using TF1 = TrainForm<MT, arma::rowvec, void, false, false>;
     using TF2 = TrainForm<MT, arma::mat, void, false, false>;
     using TF3 = TrainForm<MT, PT, void, false, false>;
   
     /* Forms of Train for classification algorithms */
     using TF4 = TrainForm<MT, PT, void, false, true>;
     using TF5 = TrainForm<MT, PT, void, true, true>;
   
     /* Forms of Train with weights for regression algorithms */
     using WTF1 = TrainForm<MT, arma::rowvec, WT, false, false>;
     using WTF2 = TrainForm<MT, arma::mat, WT, false, false>;
     using WTF3 = TrainForm<MT, PT, WT, false, false>;
   
     /* Forms of Train with weights for classification algorithms */
     using WTF4 = TrainForm<MT, PT, WT, false, true>;
     using WTF5 = TrainForm<MT, PT, WT, true, true>;
   
     /* A short alias for a type function that selects a right method form */
     template<typename... MethodForms>
     using Select = typename SelectMethodForm<MLAlgorithm, HasTrain, HasTTrain,
         HasMTrain, HasMPTrain, HasMPWTrain>::template From<MethodForms...>;
   
     /* An indication whether a method form is selected */
     template<typename... MFs /* MethodForms */>
     using Selects = typename std::conditional<
         std::is_same<typename Select<MFs...>::Type, NotFoundMethodForm>::value,
         std::false_type, std::true_type>::type;
   
    public:
     using PredictionsType =
         typename Select<TF1, TF2, TF3, TF4, TF5>::Type::PredictionsType;
   
     using WeightsType =
         typename Select<WTF1, WTF2, WTF3, WTF4, WTF5>::Type::WeightsType;
   
     static const bool IsSupported = !std::is_same<PredictionsType, void*>::value;
   
     static const bool SupportsWeights = !std::is_same<WeightsType, void*>::value;
   
     static const bool TakesDatasetInfo = Selects<TF5>::value;
   
     static const bool TakesNumClasses = Selects<TF4, TF5>::value;
   };
   
   } // namespace cv
   } // namespace mlpack
   
   #endif
