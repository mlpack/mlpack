/**
 * @file param.hpp
 * @author Matthew Amidon
 * @author Ryan Curtin
 *
 * Definition of PARAM_*_IN() and PARAM_*_OUT() macros, as well as the
 * PROGRAM_INFO() macro, which are used to define input and output parameters of
 * command-line programs and bindings to other languages.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_PARAM_HPP
#define MLPACK_CORE_UTIL_PARAM_HPP

/**
 * Document an executable.  Only one instance of this macro should be
 * present in your program!  Therefore, use it in the main.cpp
 * (or corresponding executable) in your program.
 *
 * @see mlpack::CLI, PARAM_FLAG(), PARAM_INT_IN(), PARAM_DOUBLE_IN(),
 * PARAM_STRING_IN(), PARAM_VECTOR_IN(), PARAM_INT_OUT(), PARAM_DOUBLE_OUT(),
 * PARAM_VECTOR_OUT(), PARAM_INT_IN_REQ(), PARAM_DOUBLE_IN_REQ(),
 * PARAM_STRING_IN_REQ(), PARAM_VECTOR_IN_REQ(), PARAM_INT_OUT_REQ(),
 * PARAM_DOUBLE_OUT_REQ(), PARAM_VECTOR_OUT_REQ(), PARAM_STRING_OUT_REQ().
 *
 * @param NAME Short string representing the name of the program.
 * @param DESC Long string describing what the program does and possibly a
 *     simple usage example.  Newlines should not be used here; this is taken
 *     care of by CLI (however, you can explicitly specify newlines to denote
 *     new paragraphs).
 */
#define PROGRAM_INFO(NAME, DESC) static mlpack::util::ProgramDoc \
    cli_programdoc_dummy_object = mlpack::util::ProgramDoc(NAME, DESC);

/**
 * Define a flag parameter.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 * @param ALIAS An alias for the parameter (one letter).
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_FLAG(ID, DESC, ALIAS) \
    PARAM_FLAG_INTERNAL(ID, DESC, ALIAS);

/**
 * Define an integer input parameter.
 *
 * The parameter can then be specified on the command line with
 * --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 * @param ALIAS An alias for the parameter (one letter).
 * @param DEF Default value of the parameter.
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_INT_IN(ID, DESC, ALIAS, DEF) \
    PARAM_IN(int, ID, DESC, ALIAS, DEF, false)

/**
 * Define an integer output parameter.  This parameter will be printed on stdout
 * at the end of the program; for instance, if the parameter name is "number"
 * and the value is 5, the output on stdout would be of the following form:
 *
 * @code
 * number: 5
 * @endcode
 *
 * If the parameter is not set by the end of the program, a fatal runtime error
 * will be issued.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_INT_OUT(ID, DESC) \
    PARAM_IN(int, ID, DESC, "", 0, false)

/**
 * Define a double input parameter.
 *
 * The parameter can then be specified on the command line with
 * --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 * @param ALIAS An alias for the parameter (one letter).
 * @param DEF Default value of the parameter.
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_DOUBLE_IN(ID, DESC, ALIAS, DEF) \
    PARAM_IN(double, ID, DESC, ALIAS, DEF, false)

/**
 * Define a double output parameter.  This parameter will be printed on stdout
 * at the end of the program; for instance, if the parameter name is "number"
 * and the value is 5.012, the output on stdout would be of the following form:
 *
 * @code
 * number: 5.012
 * @endcode
 *
 * If the parameter is not set by the end of the program, a fatal runtime error
 * will be issued.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_DOUBLE_OUT(ID, DESC) \
    PARAM_OUT(double, ID, DESC, "", 0.0, false)

/**
 * Define a string input parameter.
 *
 * The parameter can then be specified on the command line with
 * --ID=value. If ALIAS is equal to DEF_MOD (which is set using the
 * PROGRAM_INFO() macro), the parameter can be specified with just --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 * @param ALIAS An alias for the parameter (one letter).
 * @param DEF Default value of the parameter.
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_STRING_IN(ID, DESC, ALIAS, DEF) \
    PARAM_IN(std::string, ID, DESC, ALIAS, DEF, false)

/**
 * Define a string output parameter.
 *
 * If the parameter name does not end in "_file" (i.e. "output_file",
 * "predictions_file", etc.), then the string will be printed to stdout at the
 * end of the program.  For instance, if there was a string output parameter
 * called "something" with value "hello", at the end of the program the output
 * would be of the following form:
 *
 * @code
 * something: "hello"
 * @endcode
 *
 * If the parameter is not set by the end of the program, a fatal runtime error
 * will be issued.
 *
 * An alias is still allowed for string output parameters, because if the
 * parameter name ends in "_file", then the user must be able to specify it as
 * input.  The default value will always be the empty string.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 * @param ALIAS An alias for the parameter (one letter).
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_STRING_OUT(ID, DESC, ALIAS) \
    PARAM_OUT(std::string, ID, DESC, ALIAS, "", false)

/**
 * Define a vector input parameter.
 *
 * The parameter can then be specified on the command line with
 * --ID=value1,value2,value3.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 * @param ALIAS An alias for the parameter (one letter).
 * @param DEF Default value of the parameter.
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_VECTOR_IN(T, ID, DESC, ALIAS) \
    PARAM_IN(std::vector<T>, ID, DESC, ALIAS, std::vector<T>(), false)

/**
 * Define a vector output parameter.  This vector will be printed on stdout at
 * the end of the program; for instance, if the parameter name is "vector" and
 * the vector holds the array { 1, 2, 3, 4 }, the output on stdout would be of
 * the following form:
 *
 * @code
 * vector: 1, 2, 3, 4
 * @endcode
 *
 * If the parameter is not set by the end of the program, a fatal runtime error
 * will be issued.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_VECTOR_OUT(T, ID) \
    PARAM_OUT(std::vector<T>, ID, DESC, "", std::vector<T>(), false)

/**
 * Define a required integer input parameter.
 *
 * The parameter must then be specified on the command line with --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 * @param ALIAS An alias for the parameter (one letter).
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_INT_IN_REQ(ID, DESC, ALIAS) \
    PARAM_IN(int, ID, DESC, ALIAS, 0, true)

/**
 * Define a required double parameter.
 *
 * The parameter must then be specified on the command line with --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 * @param ALIAS An alias for the parameter (one letter).
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_DOUBLE_IN_REQ(ID, DESC, ALIAS) \
    PARAM_IN(double, ID, DESC, ALIAS, 0.0d, true)

/**
 * Define a required string parameter.
 *
 * The parameter must then be specified on the command line with --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 * @param ALIAS An alias for the parameter (one letter).
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_STRING_IN_REQ(ID, DESC, ALIAS) \
    PARAM_IN(std::string, ID, DESC, ALIAS, "", true)

/**
 * Define a required vector parameter.
 *
 * The parameter must then be specified on the command line with
 * --ID=value1,value2,value3.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).
 * @param ALIAS An alias for the parameter (one letter).
 *
 * @see mlpack::CLI, PROGRAM_INFO()
 *
 * @bug
 * The __COUNTER__ variable is used in most cases to guarantee a unique global
 * identifier for options declared using the PARAM_*() macros. However, not all
 * compilers have this support--most notably, gcc < 4.3. In that case, the
 * __LINE__ macro is used as an attempt to get a unique global identifier, but
 * collisions are still possible, and they produce bizarre error messages.  See
 * https://github.com/mlpack/mlpack/issues/100 for more information.
 */
#define PARAM_VECTOR_IN_REQ(T, ID, DESC, ALIAS) \
    PARAM_IN(std::vector<T>, ID, DESC, ALIAS, std::vector<T>(), true);

/**
 * @cond
 * Don't document internal macros.
 */

// These are ugly, but necessary utility functions we must use to generate a
// unique identifier inside of the PARAM() module.
#define JOIN(x, y) JOIN_AGAIN(x, y)
#define JOIN_AGAIN(x, y) x ## y
/** @endcond */

/**
 * Define an input parameter.  Don't use this function; use the other ones above
 * that call it.  Note that we are using the __LINE__ macro for naming these
 * actual parameters when __COUNTER__ does not exist, which is a bit of an ugly
 * hack... but this is the preprocessor, after all.  We don't have much choice
 * other than ugliness.
 *
 * @param T Type of the parameter.
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).
 * @param ALIAS Alias for this parameter (one letter).
 * @param DEF Default value of the parameter.
 * @param REQ Whether or not parameter is required (boolean value).
 */
#ifdef __COUNTER__
  #define PARAM_IN(T, ID, DESC, ALIAS, DEF, REQ) \
      static mlpack::util::Option<T> \
      JOIN(cli_option_dummy_object_in_, __COUNTER__) \
      (false, DEF, ID, DESC, ALIAS, REQ, true);

  #define PARAM_OUT(T, ID, DESC, ALIAS, DEF, REQ) \
      static mlpack::util::Option<T> \
      JOIN(cli_option_dummy_object_out_, __COUNTER__) \
      (false, DEF, ID, DESC, ALIAS, REQ, false);

  /** @cond Don't document internal macros. */
  #define PARAM_FLAG_INTERNAL(ID, DESC, ALIAS) static \
      mlpack::util::Option<bool> JOIN(__io_option_flag_object_, __COUNTER__) \
      (ID, DESC, ALIAS);
  /** @endcond */

#else
  // We have to do some really bizarre stuff since __COUNTER__ isn't defined.  I
  // don't think we can absolutely guarantee success, but it should be "good
  // enough".  We use the __LINE__ macro and the type of the parameter to try
  // and get a good guess at something unique.
  #define PARAM_IN(T, ID, DESC, ALIAS, DEF, REQ) \
      static mlpack::util::Option<T> \
      JOIN(JOIN(io_option_dummy_object_in_, __LINE__), opt) \
      (false, DEF, ID, DESC, ALIAS, REQ, true);

  #define PARAM_OUT(T, ID, DESC, ALIAS, DEF, REQ) \
      static mlpack::util::Option<T> \
      JOIN(JOIN(io_option_dummy_object_out_, __LINE__), opt) \
      (false, DEF, ID, DESC, ALIAS, REQ, false);

  /** @cond Don't document internal macros. */
  #define PARAM_FLAG_INTERNAL(ID, DESC, ALIAS) static \
      mlpack::util::Option<bool> JOIN(__io_option_flag_object_, __LINE__) \
      (ID, DESC, ALIAS);
  /** @endcond */

#endif

#endif
