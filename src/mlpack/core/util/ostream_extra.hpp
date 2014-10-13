/**
 * @file ostream_extra.hpp
 * @author Ryan Curtin
 *
 * Allow ostream::operator<<(T&) for mlpack objects that have ToString
 * implemented.
 */
#ifndef __MLPACK_CORE_UTIL_OSTREAM_EXTRA_HPP
#define __MLPACK_CORE_UTIL_OSTREAM_EXTRA_HPP

#include "sfinae_utility.hpp"

// Hide in a namespace so we don't pollute the global namespace.

namespace mlpack {
namespace util {
namespace misc {

HAS_MEM_FUNC(ToString, HasToString);

} // namespace misc
} // namespace util
} // namespace mlpack

template<
    typename T,
    typename junk1 = typename boost::enable_if<boost::is_class<T> >::type,
    typename junk2 = typename boost::enable_if<
        mlpack::util::misc::HasToString<T, std::string(T::*)() const> >::type
>
std::ostream& operator<<(std::ostream& stream, const T& t)
{
  stream << t.ToString();
  return stream;
}

#endif
