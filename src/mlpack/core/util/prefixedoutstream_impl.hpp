/**
 * @file core/util/prefixedoutstream_impl.hpp
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

#ifdef MLPACK_HAS_BFD_DL
  #include "backtrace.hpp"
#endif

#include <iostream>
#include <sstream>

namespace mlpack {
namespace util {

template<typename T>
PrefixedOutStream& PrefixedOutStream::operator<<(const T& s)
{
  BaseLogic<T>(s);
  return *this;
}

/**
 * These are all necessary because gcc's template mechanism does not seem smart
 * enough to figure out what I want to pass into operator<< without these.  That
 * may not be the actual case, but it works when these is here.
 */

inline PrefixedOutStream& PrefixedOutStream::operator<<(bool val)
{
  BaseLogic<bool>(val);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(short val)
{
  BaseLogic<short>(val);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(unsigned short val)
{
  BaseLogic<unsigned short>(val);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(int val)
{
  BaseLogic<int>(val);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(unsigned int val)
{
  BaseLogic<unsigned int>(val);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(long val)
{
  BaseLogic<long>(val);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(unsigned long val)
{
  BaseLogic<unsigned long>(val);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(float val)
{
  BaseLogic<float>(val);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(double val)
{
  BaseLogic<double>(val);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(long double val)
{
  BaseLogic<long double>(val);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(void* val)
{
  BaseLogic<void*>(val);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(const char* str)
{
  BaseLogic<const char*>(str);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(std::string& str)
{
  BaseLogic<std::string>(str);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(std::streambuf* sb)
{
  BaseLogic<std::streambuf*>(sb);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(
    std::ostream& (*pf)(std::ostream&))
{
  BaseLogic<std::ostream& (*)(std::ostream&)>(pf);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(
    std::ios& (*pf)(std::ios&))
{
  BaseLogic<std::ios& (*)(std::ios&)>(pf);
  return *this;
}

inline PrefixedOutStream& PrefixedOutStream::operator<<(
    std::ios_base& (*pf) (std::ios_base&))
{
  BaseLogic<std::ios_base& (*)(std::ios_base&)>(pf);
  return *this;
}

// For non-Armadillo types.
template<typename T>
std::enable_if_t<!arma::is_arma_type<T>::value>
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
#ifdef MLPACK_HAS_BFD_DL
    if (fatal && !ignoreInput && backtrace)
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

// For Armadillo types.
template<typename T>
std::enable_if_t<arma::is_arma_type<T>::value>
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

    const int maxLog = int(log10(maxVal)) + 1;
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
#ifdef MLPACK_HAS_BFD_DL
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
