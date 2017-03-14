/**
 * @file prefixedoutstream.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Implementation of templated PrefixedOutStream member functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_PREFIXEDOUTSTREAM_IMPL_HPP
#define MLPACK_CORE_UTIL_PREFIXEDOUTSTREAM_IMPL_HPP

// Just in case it hasn't been included.
#include "prefixedoutstream.hpp"

#ifdef HAS_BFD_DL
  #include "backtrace.hpp"
#endif

#include <iostream>

namespace mlpack {
namespace util {

template<typename T>
PrefixedOutStream& PrefixedOutStream::operator<<(const T& s)
{
  BaseLogic<T>(s);
  return *this;
}

// For non-Armadillo types.
template<typename T>
typename std::enable_if<!arma::is_arma_type<T>::value>::type
PrefixedOutStream::BaseLogic(const T& val)
{
  // We will use this to track whether or not we need to terminate at the end of
  // this call (only for streams which terminate after a newline).
  bool newlined = false;
  std::string line;

  // If we need to, output the prefix.
  PrefixIfNeeded();

  std::ostringstream convert;
  // Sync flags and precision with destination stream
  convert.setf(destination.flags());
  convert.precision(destination.precision());
  convert << val;

  if (convert.fail())
  {
    PrefixIfNeeded();
    if (!ignoreInput)
    {
      destination << "Failed lexical_cast<std::string>(T) for output; output"
          " not shown." << std::endl;
      newlined = true;
    }
  }
  else
  {
    line = convert.str();

    // If the length of the casted thing was 0, it may have been a stream
    // manipulator, so send it directly to the stream and don't ask questions.
    if (line.length() == 0)
    {
      // The prefix cannot be necessary at this point.
      if (!ignoreInput) // Only if the user wants it.
        destination << val;

      return;
    }

    // Now, we need to check for newlines in the output and print it.
    size_t nl;
    size_t pos = 0;
    while ((nl = line.find('\n', pos)) != std::string::npos)
    {
      PrefixIfNeeded();

      // Only output if the user wants it.
      if (!ignoreInput)
      {
        destination << line.substr(pos, nl - pos);
        destination << std::endl;
        newlined = true;
      }

      carriageReturned = true; // Regardless of whether or not we display it.

      pos = nl + 1;
    }

    if (pos != line.length()) // We need to display the rest.
    {
      PrefixIfNeeded();
      if (!ignoreInput)
        destination << line.substr(pos);
    }
  }

  // If we displayed a newline and we need to throw afterwards, do that.
  if (fatal && newlined)
  {
    if (!ignoreInput)
      destination << std::endl;

    // Print a backtrace, if we can.
#ifdef HAS_BFD_DL
    if (fatal)
    {
      size_t nl;
      size_t pos = 0;

      Backtrace bt;
      std::string btLine = bt.ToString();
      while ((nl = btLine.find('\n', pos)) != std::string::npos)
      {
        PrefixIfNeeded();

        destination << btLine.substr(pos, nl - pos);
        destination << std::endl;

        carriageReturned = true; // Regardless of whether or not we display it.

        pos = nl + 1;
      }
    }
#endif

    throw std::runtime_error("fatal error; see Log::Fatal output");
  }
}

// For Armadillo types.
template<typename T>
typename std::enable_if<arma::is_arma_type<T>::value>::type
PrefixedOutStream::BaseLogic(const T& val)
{
  // Extract printable object from the input.
  const arma::Mat<typename T::elem_type>& printVal(val);

  // We will use this to track whether or not we need to terminate at the end of
  // this call (only for streams which terminate after a newline).
  bool newlined = false;
  std::string line;

  // If we need to, output the prefix.
  PrefixIfNeeded();

  std::ostringstream convert;

  // Check if the stream is in the default state.
  if (destination.flags() == convert.flags() &&
      destination.precision() == convert.precision())
  {
    printVal.print(convert);
  }
  else
  {
    // Sync flags and precision with destination stream
    convert.setf(destination.flags());
    convert.precision(destination.precision());

    // Set width of the convert stream.
    const arma::Mat<typename T::elem_type>& absVal(arma::abs(printVal));
    double maxVal = absVal.max();

    if (maxVal == 0.0)
      maxVal = 1;

    int maxLog = log10(maxVal);
    maxLog = (maxLog > 0) ? floor(maxLog) + 1 : 1;
    const int padding = 4;
    convert.width(convert.precision() + maxLog + padding);
    printVal.raw_print(convert);
  }

  if (convert.fail())
  {
    PrefixIfNeeded();
    if (!ignoreInput)
    {
      destination << "Failed type conversion to string for output; output not "
          "shown." << std::endl;
      newlined = true;
    }
  }
  else
  {
    line = convert.str();

    // If the length of the casted thing was 0, it may have been a stream
    // manipulator, so send it directly to the stream and don't ask questions.
    if (line.length() == 0)
    {
      // The prefix cannot be necessary at this point.
      if (!ignoreInput) // Only if the user wants it.
        destination << val;

      return;
    }

    // Now, we need to check for newlines in the output and print it.
    size_t nl;
    size_t pos = 0;
    while ((nl = line.find('\n', pos)) != std::string::npos)
    {
      PrefixIfNeeded();

      // Only output if the user wants it.
      if (!ignoreInput)
      {
        destination << line.substr(pos, nl - pos);
        destination << std::endl;
      }

      newlined = true; // Ensure this is set for the fatal exception if needed.
      carriageReturned = true; // Regardless of whether or not we display it.

      pos = nl + 1;
    }

    if (pos != line.length()) // We need to display the rest.
    {
      PrefixIfNeeded();
      if (!ignoreInput)
        destination << line.substr(pos);
    }
  }

  // If we displayed a newline and we need to throw afterwards, do that.
  if (fatal && newlined)
  {
    if (!ignoreInput)
      destination << std::endl;

    // Print a backtrace, if we can.
#ifdef HAS_BFD_DL
    if (fatal && !ignoreInput)
    {
      size_t nl;
      size_t pos = 0;

      Backtrace bt;
      std::string btLine = bt.ToString();
      while ((nl = btLine.find('\n', pos)) != std::string::npos)
      {
        PrefixIfNeeded();

        if (!ignoreInput)
        {
          destination << btLine.substr(pos, nl - pos);
          destination << std::endl;
        }

        carriageReturned = true; // Regardless of whether or not we display it.

        pos = nl + 1;
      }
    }
#endif

    throw std::runtime_error("fatal error; see Log::Fatal output");
  }
}

// This is an inline function (that is why it is here and not in .cc).
void PrefixedOutStream::PrefixIfNeeded()
{
  // If we need to, output a prefix.
  if (carriageReturned)
  {
    if (!ignoreInput) // But only if we are allowed to.
      destination << prefix;

    carriageReturned = false; // Denote that the prefix has been displayed.
  }
}

} // namespace util
} // namespace mlpack

#endif // MLPACK_CORE_UTIL_PREFIXEDOUTSTREAM_IMPL_HPP
