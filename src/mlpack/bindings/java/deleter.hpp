#ifndef MLPACK_BINDINGS_JAVA_DELETER_HPP
#define MLPACK_BINDINGS_JAVA_DELETER_HPP

#include <type_traits>

namespace mlpack {
namespace util {

template <typename T>
typename std::enable_if<!std::is_array<T>::value>::type Delete(void* p)
{
  delete static_cast<T*>(p);
}

template <typename T>
typename std::enable_if<std::is_array<T>::value>::type Delete(void* p)
{
  using U = typename std::remove_all_extents<T>::type;
  delete[] static_cast<U*>(p);
}

} // namespace util
} // namespace mlpack

#endif
