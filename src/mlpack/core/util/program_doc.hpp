/**
 * @file core/util/program_doc.hpp
 * @author Yashwant Singh Parihar
 * @author Matthew Amidon
 *
 * Implementation of mutiple classes that store information related to a binding.
 * The classes register themselves with IO when constructed.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_PROGRAM_DOC_HPP
#define MLPACK_CORE_UTIL_PROGRAM_DOC_HPP

namespace mlpack {
namespace util {

class BindingName
{
 public:
  /**
   * Construct a BindingName object.  When constructed, it will register itself
   * with IO.  A fatal error will be thrown if more than one is constructed for
   * a given bindingName.
   *
   * @param bindingName Name of the binding.
   * @param name Name displayed to user of the binding.
   */
  BindingName(const std::string& bindingName, const std::string& name);
};

class ShortDescription
{
 public:
  /**
   * Construct a ShortDescription object.  When constructed, it will register
   * itself with IO.  A fatal error will be thrown if more than one is
   * constructed.
   *
   * @param bindingName Name of the binding.
   * @param shortDescription A short two-sentence description of the binding,
   *     what it does, and what it is useful for.
   */
  ShortDescription(const std::string& bindingName,
                   const std::string& shortDescription);
};

class LongDescription
{
 public:
  /**
   * Construct a LongDescription object. When constructed, it will register
   * itself with IO.  A fatal error will be thrown if more than one is
   * constructed for a given `bindingName`.
   *
   * @param bindingName Name of the binding.
   * @param longDescription Long string containing documentation on
   *     what it is.  No newline characters are necessary; this is
   *     taken care of by IO later.
   */
  LongDescription(const std::string& bindingName,
                  const std::function<std::string()>& longDescription);
};

class Example
{
 public:
  /**
   * Construct a Example object.  When constructed, it will register itself
   * with IO for the given `bindingName`.
   *
   * @param bindingName Name of the binding.
   * @param example Documentation on how to use the binding.
   */
  Example(const std::string& bindingName,
          const std::function<std::string()>& example);
};

class SeeAlso
{
 public:
  /**
   * Construct a SeeAlso object.  When constructed, it will register itself
   * with IO for the given `bindingName`.
   *
   * @param bindingName Name of the binding.
   * @param description Description of SeeAlso.
   * @param link Link of SeeAlso.
   */
  SeeAlso(const std::string& bindingName,
          const std::string& description,
          const std::string& link);
};

} // namespace util
} // namespace mlpack

// Include implementation.
#include "program_doc_impl.hpp"

#endif
