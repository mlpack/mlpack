/**
 * @file core/data/text_options.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Text options, all possible options to load different data types and format
 * with specific settings into mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_TEXT_OPTIONS_HPP
#define MLPACK_CORE_DATA_TEXT_OPTIONS_HPP

#include <mlpack/prereqs.hpp>

#include "matrix_options.hpp"

namespace mlpack {
namespace data {

class TextOptions : public MatrixOptionsBase<TextOptions>
{
 public:
  // TODO: pass through noTranspose option?
  TextOptions(std::optional<bool> hasHeaders = std::nullopt,
              std::optional<bool> semicolon = std::nullopt,
              std::optional<bool> missingToNan = std::nullopt,
              std::optional<bool> categorical = std::nullopt) :
      MatrixOptionsBase<TextOptions>(),
      hasHeaders(hasHeaders),
      semicolon(semicolon),
      missingToNan(missingToNan),
      categorical(categorical)
  {
    // Do Nothing.
  }

  //
  // Handling for copy and move operations on other TextOptions objects.
  //

  TextOptions(const DataOptionsBase<MatrixOptionsBase<TextOptions>>& opts) :
      MatrixOptionsBase<TextOptions>()
  {
    // Delegate to copy operator.
    *this = opts;
  }

  TextOptions(DataOptionsBase<MatrixOptionsBase<TextOptions>>&& opts) :
      MatrixOptionsBase<TextOptions>()
  {
    // Delegate to move operator.
    *this = std::move(opts);
  }

  TextOptions& operator=(
      const DataOptionsBase<MatrixOptionsBase<TextOptions>>& otherIn)
  {
    const TextOptions& other = static_cast<const TextOptions&>(otherIn);

    if (&other == this)
      return *this;

    if (other.hasHeaders.has_value())
      hasHeaders = *other.hasHeaders;
    if (other.semicolon.has_value())
      semicolon = *other.semicolon;
    if (other.missingToNan.has_value())
      missingToNan = *other.missingToNan;
    if (other.categorical.has_value())
      categorical = *other.categorical;

    headers = other.headers;
    datasetInfo = other.datasetInfo;

    // Copy base members.
    MatrixOptionsBase<TextOptions>::operator=(other);

    return *this;
  }

  TextOptions& operator=(
      DataOptionsBase<MatrixOptionsBase<TextOptions>>&& otherIn)
  {
    TextOptions&& other = static_cast<TextOptions&&>(otherIn);

    if (&other == this)
      return *this;

    hasHeaders = std::move(other.hasHeaders);
    semicolon = std::move(other.semicolon);
    missingToNan = std::move(other.missingToNan);
    categorical = std::move(other.categorical);

    headers = std::move(other.headers);
    datasetInfo = std::move(other.datasetInfo);

    // Move base members.
    MatrixOptionsBase<TextOptions>::operator=(std::move(other));

    return *this;
  }

  //
  // Handling for copy and move operations on other DataOptionsBase types.
  //

  // Conversions must be explicit.
  template<typename Derived2>
  explicit TextOptions(const DataOptionsBase<Derived2>& other) :
      MatrixOptionsBase<TextOptions>(other) { }

  template<typename Derived2>
  explicit TextOptions(DataOptionsBase<Derived2>&& other) :
      MatrixOptionsBase<TextOptions>(std::move(other)) { }

  template<typename Derived2>
  TextOptions& operator=(const DataOptionsBase<Derived2>& other)
  {
    return static_cast<TextOptions&>(
        MatrixOptionsBase<TextOptions>::operator=(other));
  }

  template<typename Derived2>
  TextOptions& operator=(DataOptionsBase<Derived2>&& other)
  {
    return static_cast<TextOptions&>(
        MatrixOptionsBase<TextOptions>::operator=(std::move(other)));
  }

  void Combine(const TextOptions& other)
  {
    // Combine all boolean options.
    hasHeaders =
        DataOptionsBase<MatrixOptionsBase<TextOptions>>::CombineBooleanOption(
        hasHeaders, other.hasHeaders, "HasHeaders()");
    missingToNan =
        DataOptionsBase<MatrixOptionsBase<TextOptions>>::CombineBooleanOption(
        missingToNan, other.missingToNan, "MissingToNan()");
    categorical =
        DataOptionsBase<MatrixOptionsBase<TextOptions>>::CombineBooleanOption(
        categorical, other.categorical, "Categorical()");
    semicolon =
        DataOptionsBase<MatrixOptionsBase<TextOptions>>::CombineBooleanOption(
        semicolon, other.semicolon, "Semicolon()");

    // Whenever we combine two TextOptions, we reset the headers and
    // datasetInfo.
    headers.clear();
    datasetInfo = DatasetInfo();
  }

  // Print warnings for any members that cannot be represented by a
  // DataOptionsBase<void>.
  void WarnBaseConversion(const char* dataDescription) const
  {
    if (missingToNan.has_value() && missingToNan != defaultMissingToNan)
      this->WarnOptionConversion("missingToNan", dataDescription);
    if (semicolon.has_value() && semicolon != defaultSemicolon)
      this->WarnOptionConversion("semicolon", dataDescription);
    if (categorical.has_value() && categorical != defaultCategorical)
      this->WarnOptionConversion("categorical", dataDescription);
    if (hasHeaders.has_value() && hasHeaders != defaultHasHeaders)
      this->WarnOptionConversion("hasHeaders", dataDescription);

    // If either headers or datasetInfo are non-empty, then we take it that the
    // user has manually modified them.
    if (!headers.is_empty())
      this->WarnOptionConversion("headers", dataDescription);
    if (datasetInfo.Dimensionality() > 0)
      this->WarnOptionConversion("datasetInfo", dataDescription);
  }

  static const char* DataDescription() { return "text-file matrix data"; }

  void Reset()
  {
    hasHeaders.reset();
    semicolon.reset();
    missingToNan.reset();
    categorical.reset();

    headers.clear();
    datasetInfo = DatasetInfo();
  }

  // Get if the dataset has headers or not.
  bool HasHeaders() const
  {
    return this->AccessMember(hasHeaders, defaultHasHeaders);
  }
  // Modify if the dataset has headers.
  bool& HasHeaders()
  {
    return this->ModifyMember(hasHeaders, defaultHasHeaders);
  }

  // Get if the separator is a semicolon in the data file.
  bool Semicolon() const
  {
    return this->AccessMember(semicolon, defaultSemicolon);
  }
  // Modify the separator type in the matrix.
  bool& Semicolon()
  {
    return this->ModifyMember(semicolon, defaultSemicolon);
  }

  // Get whether missing values are converted to NaN values.
  bool MissingToNan() const
  {
    return this->AccessMember(missingToNan, defaultMissingToNan);
  }
  // Modify whether missing values are converted to NaN values.
  bool& MissingToNan()
  {
    return this->ModifyMember(missingToNan, defaultMissingToNan);
  }

  // Get whether the data should be interpreted as categorical when columns are
  // not numeric.
  bool Categorical() const
  {
    return this->AccessMember(categorical, defaultCategorical);
  }
  // Modify whether the data should be interpreted as categorical when columns
  // are not numeric.
  bool& Categorical()
  {
    return this->ModifyMember(categorical, defaultCategorical);
  }

  // Get the headers.
  const arma::field<std::string>& Headers() const { return headers; }
  // Modify the headers.
  arma::field<std::string>& Headers() { return headers; }

  // Get the DatasetInfo for categorical data.
  const data::DatasetInfo& DatasetInfo() const { return datasetInfo; }
  // Modify the DatasetInfo.
  data::DatasetInfo& DatasetInfo() { return datasetInfo; }

 private:
  std::optional<bool> hasHeaders;
  std::optional<bool> semicolon;
  std::optional<bool> missingToNan;
  std::optional<bool> categorical;

  // These are not optional, but if either is specified, then it should be taken
  // to mean that `hasHeaders` or `categorical` has been specified as true.
  arma::field<std::string> headers;
  data::DatasetInfo datasetInfo;

  constexpr static const bool defaultHasHeaders = false;
  constexpr static const bool defaultSemicolon = false;
  constexpr static const bool defaultMissingToNan = false;
  constexpr static const bool defaultCategorical = false;
};

// Boolean options
static const TextOptions HasHeaders   = TextOptions(true);
static const TextOptions Semicolon    = TextOptions(std::nullopt, true);
static const TextOptions MissingToNan = TextOptions(std::nullopt, std::nullopt,
      true);
static const TextOptions Categorical  = TextOptions(std::nullopt,
      std::nullopt, std::nullopt, true);

template<>
struct IsDataOptions<TextOptions>
{
  constexpr static bool value = true;
};

} // namespace data
} // namespace mlpack

#endif
