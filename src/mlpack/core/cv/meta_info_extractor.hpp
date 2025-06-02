/**
 * @file core/cv/meta_info_extractor.hpp
 * @author Kirill Mishchenko
 *
 * Tools for extracting meta information about machine learning algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_META_INFO_EXTRACTOR_HPP
#define MLPACK_CORE_CV_META_INFO_EXTRACTOR_HPP

#include <type_traits>

#include <mlpack/core.hpp>
#include <mlpack/core/util/sfinae_utility.hpp>

namespace mlpack {

/**
 * A wrapper struct for holding a Train form.
 *
 * @tparam MatType The type of data.
 * @tparam PredictionsType The type of predictions.
 * @tparam WeightsType The type of weights.
 * @tparam DatasetInfo An indicator whether a data::DatasetInfo parameter should
 *   be present.
 * @tparam NumClasses An indicator whether the numClasses parameter should be
 *   present.
 */
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

/**
 * A type function that selects a right method form. As parameters it takes a
 * machine learning algorithm, a set of HasMethodForm structs, and a set of
 * method forms. Method forms are passed to the internal struct From. The result
 * type can be accessed through the Type member of the From struct.
 *
 * The implementation basically loops through all combinations of the
 * HasMethodForm structs and the method forms. It stops when a right
 * combination succeeds, or when there are no more combinations.
 */
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
      using Type = std::conditional_t<
          HasMethodForm<MLAlgorithm, Form::template Type,
              Form::MinNumberOfAdditionalArgs>::value,
          Form,
          typename Implementation<RemainingForms...>::Type>;
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

/**
 * MetaInfoExtractor is a tool for extracting meta information about a given
 * machine learning algorithm. It can be used to automatically extract the type
 * of predictions and weights (if weighted learning is supported), whether the
 * machine learning algorithm takes a DatasetInfo parameter or a numClasses
 * parameter.
 *
 * The following assumptions are made about the machine learning algorithm.
 * 1. All needed information can be extracted from signatures of Train methods.
 * 2. The machine learning algorithm contains either only non-templated Train
 *   methods or only templated ones.
 * 3. Train methods that can be used for extraction of needed information should
 *   be distinguishable by a number of arguments (for more information read
 *   discussion in https://github.com/mlpack/mlpack/issues/929).
 * 4. If weighted learning is supported, passing weights is an option rather
 *   than a requirement.
 *
 * @tparam MLAlgorithm A machine learning algorithm to investigate.
 * @tparam MT The type of data.
 * @tparam PT The type of predictions (should be passed when the
 *   predictions type is a template parameter in Train methods of MLAlgorithm).
 * @tparam WT The type of weights (should be passed when weighted learning is
 *   supported, and the weights type is a template parameter in Train methods of
 *   MLAlgorithm).
 */
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
  using Selects = std::conditional_t<
      std::is_same_v<typename Select<MFs...>::Type, NotFoundMethodForm>,
      std::false_type, std::true_type>;

 public:
  /**
   * The type of predictions used in MLAlgorithm. It is equal to void* if the
   * extraction fails.
   */
  using PredictionsType =
      typename Select<TF1, TF2, TF3, TF4, TF5>::Type::PredictionsType;

  /**
   * The type of weights used in MLAlgorithm. It is equal to void* if the
   * extraction fails.
   */
  using WeightsType =
      typename Select<WTF1, WTF2, WTF3, WTF4, WTF5>::Type::WeightsType;

  /**
   * An indication whether PredictionsType has been identified (i.e. MLAlgorithm
   * is supported by MetaInfoExtractor).
   */
  static const bool IsSupported = !std::is_same_v<PredictionsType, void*>;

  /**
   * An indication whether MLAlgorithm supports weighted learning.
   */
  static const bool SupportsWeights = !std::is_same_v<WeightsType, void*>;

  /**
   * An indication whether MLAlgorithm takes a data::DatasetInfo parameter.
   */
  static const bool TakesDatasetInfo = Selects<TF5>::value;

  /**
   * An indication whether MLAlgorithm takes the numClasses (size_t) parameter.
   */
  static const bool TakesNumClasses = Selects<TF4, TF5>::value;
};

} // namespace mlpack

#endif
