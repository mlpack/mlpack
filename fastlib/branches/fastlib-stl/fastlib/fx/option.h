/***
 * @file option.h
 * @author Matthew Amidon
 * 
 * The Options class is used to facilitate compile-time instantiation
 * of parameters (or more specifically, at program start up).  This should
 * not be used outside of IO itself.  
 */

#ifndef MLPACK_IO_OPTION_H
#define MLPACK_IO_OPTION_H


template<typename N>
class Option {
  public:

 /***
 * Construct an Option object.  When constructed, it will register
 * itself with IO.
 *
 * @param ignoreTemplate Whether or not the template type matters for
 *       this option.  Essentially differs options with no value (flags)
 *       from those that do, and thus require a type.
 * @param defaultValue The default value this parameter will be initialized to.
 * @param identifier The name of the option (no dashes in front; for
 *       --help, we would pass "help").
 * @param description A short string describing the option.
 * @param parent Full pathname of the parent module that "owns" this option.
 *        The default is evaluates to the root node.
 * @param required Whether or not the option is required at runtime.
 */
  Option(bool ignoreTemplate,
            N defaultValue,
            const char* identifier,
            const char* description,
            const char* parent=NULL,
            bool required=false);

 /***
 * Construct an Option object without referencing the command line.  
 * When constructed, it will register itself with IO.
 * 
 * @param defaultValue The default value this parameter will be initialized to.
 * @param identifier The name of the option (no dashes in front; for
 *       --help, we would pass "help").
 * @param description A short string describing the option.
 * @param parent Parent module that "owns" this option.
 * @param required Whether or not the option is required at runtime.
 */
  Option(N defaultValue,
          const char* identifier,
          const char* description,
          const char* parent=NULL,
          bool required=false);
};

//For implementations of templated functions
#include "option_impl.h"

#endif 
