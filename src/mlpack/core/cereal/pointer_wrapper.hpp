#ifndef CEREAL_PONTER_WRAPPER_HPP
#define CEREAL_PONTER_WRAPPER_HPP

/*
 * The objective of this class is to create a wrapper for
 * raw pointer by encapsulating them in a smart pointer unique_ptr
 * This will allow to serialize raw pointer in cereal as a smart pointer
 * because it will be difficult to change all pointer type in mlpack
 */

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/memory.hpp>

namespace cereal {

template<class T>
class pointer_wrapper
{
public:
  pointer_wrapper(T*& pointer)
    : localPointer(pointer)
  {}

  template<class Archive>
  void save(Archive& ar, const unsigned int /*version*/) const
  {
    std::unique_ptr<T> smartPointer = std::make_unique<T>(*this->localPointer);
    ar(CEREAL_NVP(smartPointer));
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int /*version*/)
  {
    std::unique_ptr<T> smartPointer;
    ar(CEREAL_NVP(smartPointer));
    localPointer = smartPointer.release();
  }

  T*& release() { return localPointer; }

private:
  T*& localPointer;
};

template<class T>
inline pointer_wrapper<T>
make_pointer(T* t)
{
  pointer_wrapper<T> a(t);
  return a;
}

} // end namespace cereal

#endif // CEREAL_POINTER_WRAPPER_HPP
