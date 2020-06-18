/**
 * @file bindings/go/strip_type.hpp
 * @author Yasmine Dumouchel
 *
 * Given a C++ typename that may have template parameters, return stripped and
 * printable versions to be used in Go bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_STRIP_TYPE_HPP
#define MLPACK_BINDINGS_GO_STRIP_TYPE_HPP

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Given an input type like, e.g., "LogisticRegression<>", return four types
 * that can be used in Go code. goStrippedType will be a type with lower first
 * letter of parameter name so it is of unexported type in GoDoc
 * (e.g. "logisticRegression"), strippedType will be a type with no
 * template parameters (e.g. "LogisticRegression"), printedType will be a
 * printable type with the template parameters (e.g. "LogisticRegression<>"),
 * and defaultsType will be a printable type with a default template parameter
 * (e.g. "LogisticRegression\<T\>") that can be used for class definitions.
 */
inline void StripType(const std::string& inputType,
                      std::string& goStrippedType,
                      std::string& strippedType,
                      std::string& printedType,
                      std::string& defaultsType)
{
  // First, we have to parse the type.  If we have something like, e.g.,
  // 'LogisticRegression<>', we must convert this to 'LogisticRegression<T>.'
  goStrippedType = inputType;
  printedType = inputType;
  strippedType = inputType;
  defaultsType = inputType;

  // Lower the first letter of parameter name so it is
  // of unexported type in GoDoc.
  for (size_t i = 0; i < goStrippedType.size(); i++)
  {
    goStrippedType[i] = std::tolower(goStrippedType[i]);
    if (std::isupper(goStrippedType[i+1]) && !std::isupper(goStrippedType[i+2]))
    {
      if (isalpha(goStrippedType[i+2]))
        break;
    }
  }

  if (printedType.find("<") != std::string::npos)
  {
    // Are there any template parameters?  Or is it the default?
    const size_t loc = printedType.find("<>");
    if (loc != std::string::npos)
    {
      // Convert it from "<>".
      goStrippedType.replace(loc, 2, "");
      strippedType.replace(loc, 2, "");
      printedType.replace(loc, 2, "<>");
      defaultsType.replace(loc, 2, "<T>");
    }
  }
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
