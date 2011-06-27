#ifndef MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H
#define MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H

template<typename T>
void PrefixedOutStream::BaseLogic(T val) {
  if (carriageReturned && !isNullStream) {
    destination << prefix << val;
    debugBuffer << prefix << val;
    carriageReturned = false;
  }
  else if (!isNullStream) {
    destination << val;
    debugBuffer << val;
  }
}

#endif //MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H
