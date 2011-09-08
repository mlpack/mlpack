#ifndef MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H
#define MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H

template<typename T>
PrefixedOutStream& PrefixedOutStream::operator<<(T s) {
  BaseLogic<T>(s);

  return *this;
}

template<typename T>
void PrefixedOutStream::BaseLogic(T val) {
  // Maintain the debug buffer.
  if (carriageReturned && ignoreInput) { 
    currentLine = "";
    carriageReturned = false;
  }

  try {
    size_t currentPos = currentLine.length();
    currentLine += boost::lexical_cast<std::string>(val);

    // Having added to our line, we need to check for any newlines.  If we find
    // one, output the line, then output the newline and the prefix and continue
    // looking.
    size_t newlinePos;
    while ((newlinePos = currentLine.find('\n', currentPos)) !=
        std::string::npos) {
      // Don't output if we are told to ignore input for debugging.
      if (!ignoreInput) {
        if (carriageReturned)
          destination << prefix;

        destination << currentLine.substr(currentPos, newlinePos - currentPos);

        destination << std::endl;
        carriageReturned = true;
      }
      currentPos = newlinePos + 1;
    }

    if (currentPos == currentLine.length()) {
      carriageReturned = true;
      currentLine = "";
    } else {
      // Display the rest of the output, if we want to.
      if (!ignoreInput) {
        if (carriageReturned)
          destination << prefix;

        destination << currentLine.substr(currentPos);
        carriageReturned = false;
      }
    }

  } catch (boost::bad_lexical_cast &e) {
    // Warn the user, if we are allowed to give output.
    if (!ignoreInput)
      destination << "Failed lexical_cast<std::string>(T) for output; output"
          " not shown." << std::endl;
  }
}

#endif //MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H
