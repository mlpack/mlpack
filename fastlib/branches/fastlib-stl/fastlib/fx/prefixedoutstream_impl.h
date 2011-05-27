#ifndef MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H
#define MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H

template<typename T>
void PrefixedOutStream::BaseLogic(T val) {
  if (cariageReturned) {
    destination << prefix << val;
    cariageReturned = false;
  }
  else
    destination << val;
}

#endif //MLPACK_IO_PREFIXED_OUT_STREAM_IMPL_H
