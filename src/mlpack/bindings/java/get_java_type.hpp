/**
 * @file get_java_type.hpp
 * @author Vasyl Teliman
 *
 * Get the Java-named type of an mlpack C++ type.
 */
#ifndef MLPACK_BINDINGS_JAVA_GET_JAVA_TYPE_HPP
#define MLPACK_BINDINGS_JAVA_GET_JAVA_TYPE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/is_std_vector.hpp>
#include <mlpack/core/util/param_data.hpp>
#include "strip_type.hpp"

namespace mlpack {
namespace bindings {
namespace java {

/**
 * Fallback implementation
 */
template<typename T>
inline std::string GetJavaType(const util::ParamData&,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0)
{
  return "unknown_"; // This will cause an error most likely...
}

/**
 * Map bool -> Boolean
 */
template<>
inline std::string GetJavaType<bool>(const util::ParamData&,
    const typename std::enable_if<!util::IsStdVector<bool>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<bool>::value>::type*,
    const typename std::enable_if<!std::is_same<bool,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<bool>::value>::type*)
{
  return "Boolean";
}

/**
 * Map int -> Integer
 */
template<>
inline std::string GetJavaType<int>(const util::ParamData&,
    const typename std::enable_if<!util::IsStdVector<int>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<int>::value>::type*,
    const typename std::enable_if<!std::is_same<int,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<int>::value>::type*)
{
  return "Integer";
}

/**
 * Map size_t -> Long
 */
template<>
inline std::string GetJavaType<size_t>(const util::ParamData&,
    const typename std::enable_if<!util::IsStdVector<size_t>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<size_t>::value>::type*,
    const typename std::enable_if<!std::is_same<size_t,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<size_t>::value>::type*)
{
  return "Long";
}

/**
 * Map double -> Double
 */
template<>
inline std::string GetJavaType<double>(const util::ParamData&,
    const typename std::enable_if<!util::IsStdVector<double>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<double>::value>::type*,
    const typename std::enable_if<!std::is_same<double,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<double>::value>::type*)
{
  return "Double";
}

/**
 * Map std::string -> String
 */
template<>
inline std::string GetJavaType<std::string>(const util::ParamData&,
    const typename std::enable_if<
        !util::IsStdVector<std::string>::value>::type*,
    const typename std::enable_if<
        !arma::is_arma_type<std::string>::value>::type*,
    const typename std::enable_if<!std::is_same<std::string,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*,
    const typename std::enable_if<
        !data::HasSerialize<std::string>::value>::type*)
{
  return "String";
}

/**
 * Map std::vector<std::string> -> List<String>
 *     std::vector<int>         -> List<Integer>
 */
template<typename T>
inline std::string GetJavaType(const util::ParamData& data,
    const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0)
{
  return "List<" + GetJavaType<typename T::value_type>(data) + ">";
}

/**
 * Map arma::mat<*> -> INDArray
 */
template<typename T>
inline std::string GetJavaType(const util::ParamData&,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  return "INDArray";
}

/**
 * Map for categorical matrices
 */
template<typename T>
inline std::string GetJavaType(const util::ParamData&,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  return "MatrixWithInfo";
}

/**
 * Map for serializable types
 */
template<typename T>
inline std::string GetJavaType(const util::ParamData& d,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  return StripType(d.cppType) + "Type";
}

/**
 * Entry point to the implementation
 */
template <typename T>
void GetJavaType(const util::ParamData& d, const void*, void* out)
{
  *(std::string*)out = GetJavaType<typename std::remove_pointer<T>::type>(d);
}

} // namespace java
} // namespace bindings
} // namespace mlpack

#endif
