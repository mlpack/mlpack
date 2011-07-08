#ifndef MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H
#define MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H

template<typename T>
PrefixedOutStream& PrefixedOutStream::operator<<(T s) {
  BaseLogic<T>(s);

  return *this;
}

template<typename T>
void PrefixedOutStream::BaseLogic(T val) {
  //Maintain the debug buffer.
  if (carriageReturned && ignoreInput) { 
    currentLine = "";
    carriageReturned = false;
  }

  try { 
    currentLine += boost::lexical_cast<std::string>(val);
  } catch(boost::bad_lexical_cast &e) {};

  //Cancel output operation?
  if (ignoreInput)
    return; 

  if (carriageReturned) {
    currentLine = "";
    destination << prefix << val;
    carriageReturned = false;
  }
  else
    destination << val;
}

#endif //MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H
