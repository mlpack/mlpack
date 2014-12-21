/**
 * @file prefixedoutstream.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Implementation of templated PrefixedOutStream member functions.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_UTIL_PREFIXEDOUTSTREAM_IMPL_HPP
#define __MLPACK_CORE_UTIL_PREFIXEDOUTSTREAM_IMPL_HPP

// Just in case it hasn't been included.
#include "prefixedoutstream.hpp"
#include <iostream>

namespace mlpack {
namespace util {

template<typename T>
PrefixedOutStream& PrefixedOutStream::operator<<(const T& s)
{
  CallBaseLogic<T>(s);
  return *this;
}

//! This handles forwarding all primitive types transparently
template<typename T>
void PrefixedOutStream::CallBaseLogic(const T& s,
    typename boost::disable_if<
        boost::is_class<T>
    >::type*)
{
  BaseLogic<T>(s);
}

// Forward all objects that do not implement a ToString() method transparently
template<typename T>
void PrefixedOutStream::CallBaseLogic(const T& s,
    typename boost::enable_if<
        boost::is_class<T>
    >::type*,
    typename boost::disable_if<
        HasToString<T, std::string(T::*)() const>
    >::type*)
{
  BaseLogic<T>(s);
}

// Call ToString() on all objects that implement ToString() before forwarding
template<typename T>
void PrefixedOutStream::CallBaseLogic(const T& s,
    typename boost::enable_if<
        boost::is_class<T>
    >::type*,
    typename boost::enable_if<
        HasToString<T, std::string(T::*)() const>
    >::type*)
{
  std::string result = s.ToString();
  BaseLogic<std::string>(result);
}

template<typename T>
void PrefixedOutStream::BaseLogic(const T& val)
{
  // We will use this to track whether or not we need to terminate at the end of
  // this call (only for streams which terminate after a newline).
  bool newlined = false;
  std::string line;

  // If we need to, output the prefix.
  PrefixIfNeeded();

  std::ostringstream convert;
  convert << val;

  if(convert.fail())
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

    // Now, we need to check for newlines in this line.  If we find one, output
    // up until the newline, then output the newline and the prefix and continue
    // looking.
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

  // If we displayed a newline and we need to terminate afterwards, do that.
  if (fatal && newlined)
    exit(1);
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

}; // namespace util
}; // namespace mlpack

#endif // MLPACK_CLI_PREFIXEDOUTSTREAM_IMPL_H
