/**
 * @file prefixedoutstream.hpp
 * @author Ryan Curtin
 * @author Matthew Amidon
 *
 * Implementation of templated PrefixedOutStream member functions.
 */
#ifndef __MLPACK_CORE_IO_PREFIXEDOUTSTREAM_IMPL_HPP
#define __MLPACK_CORE_IO_PREFIXEDOUTSTREAM_IMPL_HPP

// Just in case it hasn't been included.
#include "prefixedoutstream.hpp"

template<typename T>
PrefixedOutStream& PrefixedOutStream::operator<<(T s)
{
  BaseLogic<T>(s);

  return *this;
}

template<typename T>
void PrefixedOutStream::BaseLogic(T val)
{
  // We will use this to track whether or not we need to terminate at the end of
  // this call (only for streams which terminate after a newline).
  bool newlined = false;

  // If we need to, output the prefix.
  PrefixIfNeeded();

  // Now we try to output the T (whatever it is).
  try
  {
    std::string line = boost::lexical_cast<std::string>(val);

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
  catch (boost::bad_lexical_cast &e)
  {
    // Warn the user that there was a failure.
    PrefixIfNeeded();
    if (!ignoreInput)
    {
      destination << "Failed lexical_cast<std::string>(T) for output; output"
          " not shown." << std::endl;
      newlined = true;
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

#endif // MLPACK_CLI_PREFIXEDOUTSTREAM_IMPL_H
