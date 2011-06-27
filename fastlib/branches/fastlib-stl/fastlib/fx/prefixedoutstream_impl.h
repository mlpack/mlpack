#ifndef MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H
#define MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H

template<typename T>
void PrefixedOutStream::BaseLogic(T val) {
  if (carriageReturned) {
    destination << prefix << val;
    carriageReturned = false;
  }
  else
    destination << val;
}

#endif //MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H
