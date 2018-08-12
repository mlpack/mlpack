/**
 * @file print_class_defn.hpp
 * @author Yasmine Dumouchel
 *
 * Print the class definition and association functions
 * for generating a .go binding.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_CLASS_DEFN_HPP
#define MLPACK_BINDINGS_GO_PRINT_CLASS_DEFN_HPP

#include "strip_type.hpp"

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Non-serializable models don't require any special definitions, so this prints
 * nothing.
 */
template<typename T>
void PrintClassDefnCPP(
    const util::ParamData& /* d */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0)
{
  // Do nothing.
}

/**
 * Matrices don't require any special definitions, so this prints nothing.
 */
template<typename T>
void PrintClassDefnCPP(
    const util::ParamData& /* d */,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  // Do nothing.
}

/**
 * Serializable models require a special class definition.
 */
template<typename T>
void PrintClassDefnCPP(
    const util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  const std::string prefix = "  ";
  // First, we have to parse the type.  If we have something like, e.g.,
  // 'LogisticRegression<>', we must convert this to 'LogisticRegression[].'
  std::string strippedType, printedType, defaultsType;
  StripType(d.cppType, strippedType, printedType, defaultsType);


  /**
   * This will produce code like:
   */
   std::cout << "extern \"C\" void MLPACK_Set" << strippedType
              << "Ptr(const char* identifier, " << std::endl;
   std::cout << "               void* value)" << std::endl;
   std::cout << "{" << std::endl;
   std::cout << prefix << "SetParamPtr<" << printedType
              << ">(identifier," << std::endl;
   std::cout << prefix << prefix << prefix
              << "static_cast<" << printedType
              << "*>(value)," << std::endl;
   std::cout << prefix << prefix << prefix
              << "CLI::HasParam(\"copy_all_inputs\"));" << std::endl;
   std::cout << "}" << std::endl;
   std::cout << std::endl;
   std::cout << "extern \"C\" void *MLPACK_Get" << strippedType
              << "Ptr(const char* identifier)" << std::endl;
   std::cout << "{" << std::endl;
   std::cout << prefix <<printedType << " *modelptr = GetParamPtr<"
              << printedType << ">(identifier);" << std::endl;
   std::cout << prefix << "return modelptr;" << std::endl;
   std::cout << "}" << std::endl;
   std::cout << std::endl;
}

/**
 * Print the class definition to stdout.  Only serializable models require a
 * different class definition, so anything else does nothing.
 *
 * @param d Parameter data.
 * @param input Unused parameter.
 * @param output Unused parameter.
 */
template<typename T>
void PrintClassDefnCPP(const util::ParamData& d,
                    const void* /* input */,
                    void* /* output */)
{
  PrintClassDefnCPP<typename std::remove_pointer<T>::type>(d);
}


/**
 * Non-serializable models don't require any special definitions, so this prints
 * nothing.
 */
template<typename T>
void PrintClassDefnH(
    const util::ParamData& /* d */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0)
{
  // Do nothing.
}

/**
 * Matrices don't require any special definitions, so this prints nothing.
 */
template<typename T>
void PrintClassDefnH(
    const util::ParamData& /* d */,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  // Do nothing.
}

/**
 * Serializable models require a special class definition.
 */
template<typename T>
void PrintClassDefnH(
    const util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  // First, we have to parse the type.  If we have something like, e.g.,
  // 'LogisticRegression<>', we must convert this to 'LogisticRegression[].'
  std::string strippedType, printedType, defaultsType;
  StripType(d.cppType, strippedType, printedType, defaultsType);

  /**
   * This will produce code like:
   * extern void MLPACK_SetTypePtr(const char* identifier,
   *         MLPACK_Type* value);
   *
   * extern void *MLPACK_GetTypePtr(const char* identifier);
   *
   */
   std::cout << "extern void MLPACK_Set" << strippedType
              << "Ptr(const char* identifier," << std::endl;
   std::cout << "         void* value);"
              << std::endl;
   std::cout << std::endl;
   std::cout << "extern void *MLPACK_Get" << strippedType
              << "Ptr(const char* identifier);" << std::endl;
   std::cout << std::endl;
}

/**
 * Print the class definition to stdout.  Only serializable models require a
 * different class definition, so anything else does nothing.
 *
 * @param d Parameter data.
 * @param input Unused parameter.
 * @param output Unused parameter.
 */
template<typename T>
void PrintClassDefnH(const util::ParamData& d,
                    const void* /* input */,
                    void* /* output */)
{
  PrintClassDefnH<typename std::remove_pointer<T>::type>(d);
}

/**
 * Non-serializable models don't require any special definitions, so this prints
 * nothing.
 */
template<typename T>
void PrintClassDefnGo(
    const util::ParamData& /* d */,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0)
{
  // Do nothing.
}

/**
 * Matrices don't require any special definitions, so this prints nothing.
 */
template<typename T>
void PrintClassDefnGo(
    const util::ParamData& /* d */,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  // Do nothing.
}

/**
 * Serializable models require a special class definition.
 */
template<typename T>
void PrintClassDefnGo(
    const util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  // First, we have to parse the type.  If we have something like, e.g.,
  // 'LogisticRegression<>', we must convert this to 'LogisticRegression[].'
  std::string strippedType, printedType, defaultsType;
  StripType(d.cppType, strippedType, printedType, defaultsType);

  /**
   * This will produce code like:
   *
   * }
   */
   std::cout << "type " << strippedType << " struct {" << std::endl;
   std::cout << " mem unsafe.Pointer" << std::endl;
   std::cout << "}" << std::endl;
   std::cout << std::endl;
   std::cout << "func (m *" << strippedType << ") alloc"
              << strippedType << "(identifier string) {"
              << std::endl;
   std::cout << " m.mem = C.MLPACK_Get" << strippedType
              << "Ptr(C.CString(identifier))" << std::endl;
   std::cout << " runtime.KeepAlive(m)" << std::endl;
   std::cout << "}" << std::endl;

   std::cout << std::endl;
   std::cout << "func (m *" << strippedType << ") get"
              << strippedType << "(identifier string) {" << std::endl;
   std::cout << " m.alloc" << strippedType << "(identifier)" << std::endl;
   std::cout << " time.Sleep(time.Second)" << std::endl;
   std::cout << " runtime.GC()" << std::endl;
   std::cout << " time.Sleep(time.Second)" << std::endl;
   std::cout << "}" << std::endl;
   std::cout << std::endl;
   std::cout << "func set" << strippedType
              << "(identifier string, ptr *" << strippedType << ") {"
              << std::endl;
   std::cout << " C.MLPACK_Set" << strippedType
              << "Ptr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))"
              << std::endl;
   std::cout << "}" << std::endl;
   std::cout << std::endl;
}
/**
 * Print the class definition to stdout.  Only serializable models require a
 * different class definition, so anything else does nothing.
 *
 * @param d Parameter data.
 * @param input Unused parameter.
 * @param output Unused parameter.
 */
template<typename T>
void PrintClassDefnGo(const util::ParamData& d,
                    const void* /* input */,
                    void* /* output */)
{
  PrintClassDefnGo<typename std::remove_pointer<T>::type>(d);
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
