/**
 * @file core/util/param.hpp
 * @author Matthew Amidon
 * @author Ryan Curtin
 *
 * Definition of PARAM_*_IN() and PARAM_*_OUT() macros, as well as the
 * Documentation related macro, which are used to define input and output
 * parameters of command-line programs and bindings to other languages.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_PARAM_HPP
#define MLPACK_CORE_UTIL_PARAM_HPP

#include "forward.hpp"

/**
 * @cond
 * Don't document internal macros.
 */

// These are ugly, but necessary utility functions we must use to generate a
// unique identifier inside of the PARAM() module.
#define JOIN(x, y) JOIN_AGAIN(x, y)
#define JOIN_AGAIN(x, y) x ## y
#define STRINGIFY(x) STRINGIFY_AGAIN(x)
#define STRINGIFY_AGAIN(x) #x

/** @endcond */

/**
 * Define the function to be called for a given binding.  BINDING_NAME should be
 * set before calling this.
 */
#define BINDING_FUNCTION(...) BINDING_NAME(__VA_ARGS__)

/**
 * Specify the user-friendly name of a binding.  Only one instance of this macro
 * should be present per binding.  BINDING_NAME should be set before calling
 * this.
 *
 * @param NAME User-friendly name.
 */
#ifdef __COUNTER__
  #define BINDING_USER_NAME(NAME) static \
      mlpack::util::BindingName \
      JOIN(io_bindingusername_dummy_object, __COUNTER__) = \
      mlpack::util::BindingName( \
          STRINGIFY(BINDING_NAME), NAME);
#else
  #define BINDING_USER_NAME(NAME) static \
      mlpack::util::BindingName \
      JOIN(JOIN(io_bindingusername_dummy_object, __LINE__), opt) = \
      mlpack::util::BindingName( \
          STRINGIFY(BINDING_NAME), NAME);
#endif

/**
 * Specify the short description of a binding.  Only one instance of this macro
 * should be present in your program!  Therefore, use it in the main.cpp
 * (or corresponding binding) in your program.
 *
 * @param SHORT_DESC Short two-sentence description of the program; it should
 *     describe what the program implements and does, and a quick overview of
 *     how it can be used and what it should be used for.
 */
#ifdef __COUNTER__
  #define BINDING_SHORT_DESC(SHORT_DESC) static \
      mlpack::util::ShortDescription \
      JOIN(io_programshort_desc_dummy_object, __COUNTER__) = \
      mlpack::util::ShortDescription( \
          STRINGIFY(BINDING_NAME), SHORT_DESC);
#else
  #define BINDING_SHORT_DESC(SHORT_DESC) static \
      mlpack::util::ShortDescription \
      JOIN(JOIN(io_programshort_desc_dummy_object, __LINE__), opt) = \
      mlpack::util::ShortDescription( \
          STRINGIFY(BINDING_NAME), SHORT_DESC);
#endif

/**
 * Specify the long description of a binding.  Only one instance of this macro
 * present in your program!  Therefore, use it in the main.cpp
 * (or corresponding binding) in your program.
 * If you wish to "revamp" some bindings, then use the BINDING_LONG_DESC()
 * of the method that you pass first into the group_bindings() macro. For all other
 * methods, it is fine if you keep the BINDING_LONG_DESC() empty.
 *
 * @param LONG_DESC Long string describing what the program does. Newlines
 *     should not be used here; this is taken care of by IO (however, you
 *     can explicitly specify newlines to denote new paragraphs).  You can
 *     also use printing macros like PRINT_PARAM_STRING(), PRINT_DATASET(),
 *     and others.
 */
#ifdef __COUNTER__
  #define BINDING_LONG_DESC(LONG_DESC) static \
      mlpack::util::LongDescription \
      JOIN(io_programlong_desc_dummy_object, __COUNTER__) = \
      mlpack::util::LongDescription( \
          STRINGIFY(BINDING_NAME), []() { return std::string(LONG_DESC); });
#else
  #define BINDING_LONG_DESC(LONG_DESC) static \
      mlpack::util::LongDescription \
      JOIN(JOIN(io_programlong_desc_dummy_object, __LINE__), opt) = \
      mlpack::util::LongDescription( \
          STRINGIFY(BINDING_NAME), []() { return std::string(LONG_DESC); });
#endif

/**
 * Specify the example of a binding.  Mutiple instance of this macro can be
 * present in your program!  Therefore, use it in the main.cpp
 * (or corresponding binding) in your program.
 *
 * @param EXAMPLE Long string describing a simple usage example.. Newlines
 *     should not be used here; this is taken care of by IO (however, you
 *     can explicitly specify newlines to denote new paragraphs).  You can
 *     also use printing macros like PRINT_CALL(), PRINT_DATASET(),
 *     and others.
 */
#ifdef __COUNTER__
  #define BINDING_EXAMPLE(EXAMPLE) static \
      mlpack::util::Example \
      JOIN(io_programexample_dummy_object_, __COUNTER__) = \
      mlpack::util::Example( \
          STRINGIFY(BINDING_NAME), []() { return(std::string(EXAMPLE)); });
#else
  #define BINDING_EXAMPLE(EXAMPLE) static \
      mlpack::util::Example \
      JOIN(JOIN(io_programexample_dummy_object_, __LINE__), opt) = \
      mlpack::util::Example( \
          STRINGIFY(BINDING_NAME), []() { return(std::string(EXAMPLE)); });
#endif

/**
 * Specify the see-also of a binding.  Mutiple instance of this macro can be
 * present in your program!  Therefore, use it in the main.cpp
 * (or corresponding binding) in your program.
 *
 * Provide a link for a binding's "see also" documentation section, which is
 * primarily (but not necessarily exclusively) used by the Markdown bindings
 * This link can be specified by calling SEE_ALSO("description", "link"), where
 * "description" is the description of the link and "link" may be one of the
 * following:
 *
 * - A direct URL, starting with http:// or https://.
 * - A page anchor for documentation, referencing another binding by its CMake
 *      binding name, i.e. "#knn".
 * - A link to a source file, using the source path after '@src', i.e.,
 *     "@src/mlpack/core/util/param.hpp"
 * - A link to a documentation file, using the path after '@doc', i.e.,
 *     "@doc/user/matrices.md"
 */
#ifdef __COUNTER__
  #define BINDING_SEE_ALSO(DESCRIPTION, LINK) static \
      mlpack::util::SeeAlso \
      JOIN(io_programsee_also_dummy_object_, __COUNTER__) = \
      mlpack::util::SeeAlso(STRINGIFY(BINDING_NAME), DESCRIPTION, LINK);
#else
  #define BINDING_SEE_ALSO(DESCRIPTION, LINK) static \
      mlpack::util::SeeAlso \
      JOIN(JOIN(io_programsee_also_dummy_object_, __LINE__), opt) = \
      mlpack::util::SeeAlso(STRINGIFY(BINDING_NAME), DESCRIPTION, LINK);
#endif

/**
 * Define a flag parameter.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_FLAG(ID, DESC, ALIAS) \
    PARAM_IN(bool, ID, DESC, ALIAS, false, false);

/**
 * Define an integer input parameter.
 *
 * The parameter can then be specified on the command line with
 * --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 * @param DEF Default value of the parameter.
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
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 */
#define PARAM_INT_OUT(ID, DESC) \
    PARAM_OUT(int, ID, DESC, "", 0, false)

/**
 * Define a double input parameter.
 *
 * The parameter can then be specified on the command line with
 * --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 * @param DEF Default value of the parameter.
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
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 */
#define PARAM_DOUBLE_OUT(ID, DESC) \
    PARAM_OUT(double, ID, DESC, "", 0.0, false)

/**
 * Define a string input parameter.
 *
 * The parameter can then be specified on the command line with
 * --ID=value. If ALIAS is equal to DEF_MOD (which is set using the
 * BINDING_LONG_DESC() macro), the parameter can be specified with just
 * --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 * @param DEF Default value of the parameter.
 */
#define PARAM_STRING_IN(ID, DESC, ALIAS, DEF) \
    PARAM_IN(std::string, ID, DESC, ALIAS, DEF, false)

/**
 * Define a string output parameter.
 *
 * The string will be printed to stdout at the end of the program.  For
 * instance, if there was a string output parameter called "something" with
 * value "hello", at the end of the program the output would be of the following
 * form:
 *
 * @code
 * something: "hello"
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_STRING_OUT(ID, DESC, ALIAS) \
    PARAM_OUT(std::string, ID, DESC, ALIAS, "", false)

/**
 * Define a matrix input parameter.  From the command line, the user can specify
 * the file that holds the matrix, using the name of the matrix parameter with
 * "_file" appended (and the same alias).  So for instance, if the name of the
 * matrix parameter was "mat", the user could specify that the "mat" matrix was
 * held in matrix.csv by giving the parameter
 *
 * @code
 * --mat_file matrix.csv
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_MATRIX_IN(ID, DESC, ALIAS) \
    PARAM_MATRIX(ID, DESC, ALIAS, false, true, true)

/**
 * Define a required matrix input parameter.  From the command line, the user
 * can specify the file that holds the matrix, using the name of the matrix
 * parameter with "_file" appended (and the same alias).  So for instance, if
 * the name of the matrix parameter was "mat", the user could specify that the
 * "mat" matrix was held in matrix.csv by giving the parameter
 *
 * @code
 * --mat_file matrix.csv
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_MATRIX_IN_REQ(ID, DESC, ALIAS) \
    PARAM_MATRIX(ID, DESC, ALIAS, true, true, true)

/**
 * Define a matrix output parameter.  When the program terminates, the matrix
 * will be saved to whatever it was set to by params.Get<arma::mat>(ID)
 * during the program.  From the command-line, the user may specify the file in
 * which to save the output matrix using a string option that is the name of the
 * matrix parameter with "_file" appended.  So, for instance, if the name of the
 * output matrix parameter was "mat", the user could speicfy that the "mat"
 * matrix should be saved in matrix.csv by giving the parameter
 *
 * @code
 * --mat_file matrix.csv
 * @endcode
 *
 * The output matrix will not be printed on stdout, like the other output option
 * types.
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_MATRIX_OUT(ID, DESC, ALIAS) \
    PARAM_MATRIX(ID, DESC, ALIAS, false, true, false)

/**
 * Define a transposed matrix input parameter.  This is useful when data is
 * desired in row-major form instead of the usual column-major form.  From the
 * command line, the user can specify the file that holds the matrix, using the
 * name of the matrix parameter with "_file" appended (and the same alias).  So
 * for instance, if the name of the matrix parameter was "mat", the user could
 * specify that the "mat" matrix was held in matrix.csv by giving the parameter
 *
 * @code
 * --mat_file matrix.csv
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_TMATRIX_IN(ID, DESC, ALIAS) \
    PARAM_MATRIX(ID, DESC, ALIAS, false, false, true)

/**
 * Define a required transposed matrix input parameter.  This is useful when
 * data is desired in row-major form instead of the usual column-major form.
 * From the command line, the user can specify the file that holds the matrix,
 * using the name of the matrix parameter with "_file" appended (and the same
 * alias).  So for instance, if the name of the matrix parameter was "mat", the
 * user could specify that the "mat" matrix was held in matrix.csv by giving the
 * parameter
 *
 * @code
 * --mat_file matrix.csv
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_TMATRIX_IN_REQ(ID, DESC, ALIAS) \
    PARAM_MATRIX(ID, DESC, ALIAS, true, false, true)

/**
 * Define a transposed matrix output parameter.  This is useful when data is
 * stored in a row-major form instead of the usual column-major form.  When the
 * program terminates, the matrix will be saved to whatever it was set to by
 * params.Get<arma::mat>(ID) during the program.  From the command-line, the
 * user may specify the file in which to save the output matrix using a string
 * option that is the name of the matrix parameter with "_file" appended.  So,
 * for instance, if the name of the output matrix parameter was "mat", the user
 * could speicfy that the "mat" matrix should be saved in matrix.csv by giving
 * the parameter
 *
 * @code
 * --mat_file matrix.csv
 * @endcode
 *
 * The output matrix will not be printed on stdout, like the other output option
 * types.
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_TMATRIX_OUT(ID, DESC, ALIAS) \
    PARAM_MATRIX(ID, DESC, ALIAS, false, false, false)

/**
 * Define an unsigned matrix input parameter (arma::Mat<size_t>).  From the
 * command line, the user can specify the file that holds the matrix, using the
 * name of the matrix parameter with "_file" appended (and the same alias).  So
 * for instance, if the name of the matrix parameter was "mat", the user could
 * specify that the "mat" matrix was held in matrix.csv by giving the parameter
 *
 * @code
 * --mat_file matrix.csv
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_UMATRIX_IN(ID, DESC, ALIAS) \
    PARAM_UMATRIX(ID, DESC, ALIAS, false, true, true)

/**
 * Define a required unsigned matrix input parameter (arma::Mat<size_t>).  From
 * the command line, the user can specify the file that holds the matrix, using
 * the name of the matrix parameter with "_file" appended (and the same alias).
 * So for instance, if the name of the matrix parameter was "mat", the user
 * could specify that the "mat" matrix was held in matrix.csv by giving the
 * parameter
 *
 * @code
 * --mat_file matrix.csv
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_UMATRIX_IN_REQ(ID, DESC, ALIAS) \
    PARAM_UMATRIX(ID, DESC, ALIAS, true, true, true)

/**
 * Define an unsigned matrix output parameter (arma::Mat<size_t>).  When the
 * program terminates, the matrix will be saved to whatever it was set to by
 * params.Get<arma::Mat<size_t>>(ID) during the program.  From the
 * command-line, the user may specify the file in which to save the output
 * matrix using a string option that is the name of the matrix parameter with
 * "_file" appended.  So, for instance, if the name of the output matrix
 * parameter was "mat", the user could speicfy that the "mat" matrix should be
 * saved in matrix.csv by giving the parameter
 *
 * @code
 * --mat_file matrix.csv
 * @endcode
 *
 * The output matrix will not be printed on stdout, like the other output option
 * types.
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_UMATRIX_OUT(ID, DESC, ALIAS) \
    PARAM_UMATRIX(ID, DESC, ALIAS, false, true, false)


/**
 * Define a vector input parameter (type arma::vec).  From the command line, the
 * user can specify the file that holds the vector, using the name of the vector
 * parameter with "_file" appended (and the same alias).  So for instance, if
 * the name of the vector parameter was "vec", the user could specify that the
 * "vec" vector was held in vec.csv by giving the parameter:
 *
 * @code
 * --vec_file vector.csv
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_COL_IN(ID, DESC, ALIAS) \
    PARAM_COL(ID, DESC, ALIAS, false, true, true)

/**
 * Define a required vector input parameter (type arma::vec).  From the command
 * line, the user can specify the file that holds the vector, using the name of
 * the vector parameter with "_file" appended (and the same alias).  So for
 * instance, if the name of the vector parameter was "vec", the user could
 * specify that the "vec" vector was held in vec.csv by giving the parameter:
 *
 * @code
 * --vec_file vector.csv
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_COL_IN_REQ(ID, DESC, ALIAS) \
    PARAM_COL(ID, DESC, ALIAS, true, true, true)

/**
 * Define a row vector input parameter (type arma::rowvec).  From the command
 * line, the user can specify the file that holds the vector, using the name of
 * the vector parameter with "_file" appended (and the same alias).  So for
 * instance, if the name of the vector parameter was "vec", the user could
 * specify that the "vec" vector was held in vec.csv by giving the parameter:
 *
 * @code
 * --vec_file vector.csv
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_ROW_IN(ID, DESC, ALIAS) \
    PARAM_ROW(ID, DESC, ALIAS, false, true, true)

/**
 * Define an unsigned vector input parameter (type arma::Col<size_t>).  From the
 * command line, the user can specify the file that holds the vector, using the
 * name of the vector parameter with "_file" appended (and the same alias).  So
 * for instance, if the name of the vector parameter was "vec", the user could
 * specify that the "vec" vector was held in vec.csv by giving the parameter:
 *
 * @code
 * --vec_file vector.csv
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_UCOL_IN(ID, DESC, ALIAS) \
    PARAM_UCOL(ID, DESC, ALIAS, false, true, true)

/**
 * Define an unsigned row vector input parameter (type arma::Row<size_t>).  From
 * the command line, the user can specify the file that holds the vector, using
 * the name of the vector parameter with "_file" appended (and the same alias).
 * So for instance, if the name of the vector parameter was "vec", the user
 * could specify that the "vec" vector was held in vec.csv by giving the
 * parameter:
 *
 * @code
 * --vec_file vector.csv
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_UROW_IN(ID, DESC, ALIAS) \
    PARAM_UROW(ID, DESC, ALIAS, false, true, true)

/**
 * Define a vector output parameter (type arma::vec).  When the program
 * terminates, the vector will be saved to whatever it was set to during the
 * program.  From the command-line, the user may specify the file in which to
 * save the output vector using a string option that is the name of the matrix
 * parameter with "_file" appended.  So, for instance, if the name of the output
 * vector parameter was "vec", the user could specify that the "vec" vector
 * should be saved in vector.csv by giving the parameter:
 *
 * @code
 * --vec_file vector.csv
 * @endcode
 *
 * The output vector will not be printed on stdout, like the other output option
 * types.
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_COL_OUT(ID, DESC, ALIAS) \
    PARAM_COL(ID, DESC, ALIAS, false, true, false)

/**
 * Define a row vector output parameter (type arma::rowvec).  When the program
 * terminates, the vector will be saved to whatever it was set to during the
 * program.  From the command-line, the user may specify the file in which to
 * save the output vector using a string option that is the name of the matrix
 * parameter with "_file" appended.  So, for instance, if the name of the output
 * vector parameter was "vec", the user could specify that the "vec" vector
 * should be saved in vector.csv by giving the parameter:
 *
 * @code
 * --vec_file vector.csv
 * @endcode
 *
 * The output vector will not be printed on stdout, like the other output option
 * types.
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_ROW_OUT(ID, DESC, ALIAS) \
    PARAM_ROW(ID, DESC, ALIAS, false, true, false)

/**
 * Define an unsigned vector output parameter (type arma::Col<size_t>).  When
 * the program terminates, the vector will be saved to whatever it was set to
 * during the program.  From the command-line, the user may specify the file in
 * which to save the output vector using a string option that is the name of the
 * matrix parameter with "_file" appended.  So, for instance, if the name of the
 * output vector parameter was "vec", the user could specify that the "vec"
 * vector should be saved in vector.csv by giving the parameter:
 *
 * @code
 * --vec_file vector.csv
 * @endcode
 *
 * The output vector will not be printed on stdout, like the other output option
 * types.
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_UCOL_OUT(ID, DESC, ALIAS) \
    PARAM_UCOL(ID, DESC, ALIAS, false, true, false)

/**
 * Define an unsigned row vector output parameter (type arma::Row<size_t>).
 * When the program terminates, the vector will be saved to whatever it was set
 * to during the program.  From the command-line, the user may specify the file
 * in which to save the output vector using a string option that is the name of
 * the matrix parameter with "_file" appended.  So, for instance, if the name of
 * the output vector parameter was "vec", the user could specify that the "vec"
 * vector should be saved in vector.csv by giving the parameter:
 *
 * @code
 * --vec_file vector.csv
 * @endcode
 *
 * The output vector will not be printed on stdout, like the other output option
 * types.
 *
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_UROW_OUT(ID, DESC, ALIAS) \
    PARAM_UROW(ID, DESC, ALIAS, false, true, false)

/**
 * Define a std::vector input parameter.
 *
 * The parameter can then be specified on the command line with
 * --ID=value1,value2,value3.
 *
 * @param T Type of the parameter.
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
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
 * @param T Type of the parameter.
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_VECTOR_OUT(T, ID, DESC, ALIAS) \
    PARAM_OUT(std::vector<T>, ID, DESC, ALIAS, std::vector<T>(), false)

/**
 * Define an input DatasetInfo/matrix parameter.  From the command line, the
 * user can specify the file that holds the matrix, using the name of the matrix
 * parameter with "_file" appended (and the same alias).  So for instance, if
 * the name of the matrix parameter was "matrix", the user could specify that
 * the "matrix" matrix was held in file.csv by giving the parameter
 *
 * @code
 * --matrix_file file.csv
 * @endcode
 *
 * Then the DatasetInfo and matrix type could be accessed with
 *
 * @code
 * DatasetInfo d = std::move(
 *     params.Get<std::tuple<DatasetInfo, arma::mat>>("matrix").get<0>());
 * arma::mat m = std::move(
 *     params.Get<std::tuple<DatasetInfo, arma::mat>>("matrix").get<1>());
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS One-character string representing the alias of the parameter.
 */
#define TUPLE_TYPE std::tuple<mlpack::data::DatasetInfo, arma::mat>
#define PARAM_MATRIX_AND_INFO_IN(ID, DESC, ALIAS) \
    PARAM(TUPLE_TYPE, ID, DESC, ALIAS, \
        "std::tuple<mlpack::data::DatasetInfo, arma::mat>", false, true, true, \
        TUPLE_TYPE())

/**
 * Define a required input DatasetInfo/matrix parameter.  From the command line,
 * the user can specify the file that holds the matrix, using the name of the
 * matrix parameter with "_file" appended (and the same alias).  So for
 * instance, if the name of the matrix parameter was "matrix", the user could
 * specify that the "matrix" matrix was held in file.csv by giving the parameter
 *
 * @code
 * --matrix_file file.csv
 * @endcode
 *
 * Then the DatasetInfo and matrix type could be accessed with
 *
 * @code
 * DatasetInfo d = std::move(
 *     params.Get<std::tuple<DatasetInfo, arma::mat>>("matrix").get<0>());
 * arma::mat m = std::move(
 *     params.Get<std::tuple<DatasetInfo, arma::mat>>("matrix").get<1>());
 * @endcode
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS One-character string representing the alias of the parameter.
 */
#define TUPLE_TYPE std::tuple<mlpack::data::DatasetInfo, arma::mat>
#define PARAM_MATRIX_AND_INFO_IN_REQ(ID, DESC, ALIAS) \
    PARAM(TUPLE_TYPE, ID, DESC, ALIAS, \
        "std::tuple<mlpack::data::DatasetInfo, arma::mat>", true, true, true, \
        TUPLE_TYPE())

/**
 * Define an input model.  From the command line, the user can specify the file
 * that holds the model, using the name of the model parameter with "_file"
 * appended (and the same alias).  So for instance, if the name of the model
 * parameter was "model", the user could specify that the "model" model was held
 * in model.bin by giving the parameter
 *
 * @code
 * --model_file model.bin
 * @endcode
 *
 * Note that the first parameter of this model is the type (the class name) of
 * the model to be loaded.  This model type must have a serialize() function; a
 * compilation error (a very long and complex one) will result if the model type
 * does not have the following function:
 *
 * @code
 * template<typename Archive>
 * void serialize(Archive& ar, const uint32_t version);
 * @endcode
 *
 * @param TYPE Type of the model to be loaded.
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter.  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_MODEL_IN(TYPE, ID, DESC, ALIAS) \
    PARAM_MODEL(TYPE, ID, DESC, ALIAS, false, true)

/**
 * Define a required input model.  From the command line, the user can specify
 * the file that holds the model, using the name of the model parameter with
 * "_file" appended (and the same alias).  So for instance, if the name of the
 * model parameter was "model", the user could specify that the "model" model
 * was held in model.bin by giving the parameter
 *
 * @code
 * --model_file model.bin
 * @endcode
 *
 * Note that the first parameter of this model is the type (the class name) of
 * the model to be loaded.  This model type must have a serialize() function; a
 * compilation error (a very long and complex one) will result if the model type
 * does not have the following function:
 *
 * @code
 * template<typename Archive>
 * void serialize(Archive& ar, const uint32_t version);
 * @endcode
 *
 * @param TYPE Type of the model to be loaded.
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter.  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_MODEL_IN_REQ(TYPE, ID, DESC, ALIAS) \
    PARAM_MODEL(TYPE, ID, DESC, ALIAS, true, true)

/**
 * Define an output model.  From the command line, the user can specify the file
 * that should hold the model, using the name of the model parameter with
 * "_file" appended (and the same alias).  So for instance, if the user desires
 * to save the model to model.bin and the parameter name is "model", they could
 * specify
 *
 * @code
 * --model_file model.bin
 * @endcode
 *
 * The model will be saved at the termination of the program.
 *
 * @param TYPE Type of the model to be saved.
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter.  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_MODEL_OUT(TYPE, ID, DESC, ALIAS) \
    PARAM_MODEL(TYPE, ID, DESC, ALIAS, false, false)

/**
 * Define a required integer input parameter.
 *
 * The parameter must then be specified on the command line with --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_INT_IN_REQ(ID, DESC, ALIAS) \
    PARAM_IN(int, ID, DESC, ALIAS, 0, true)

/**
 * Define a required double parameter.
 *
 * The parameter must then be specified on the command line with --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_DOUBLE_IN_REQ(ID, DESC, ALIAS) \
    PARAM_IN(double, ID, DESC, ALIAS, 0.0, true)

/**
 * Define a required string parameter.
 *
 * The parameter must then be specified on the command line with --ID=value.
 *
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_STRING_IN_REQ(ID, DESC, ALIAS) \
    PARAM_IN(std::string, ID, DESC, ALIAS, "", true)

/**
 * Define a required vector parameter.
 *
 * The parameter must then be specified on the command line with
 * --ID=value1,value2,value3.
 *
 * @param T Type of the parameter.
 * @param ID Name of the parameter.
 * @param DESC Quick description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS An alias for the parameter (one letter).
 */
#define PARAM_VECTOR_IN_REQ(T, ID, DESC, ALIAS) \
    PARAM_IN(std::vector<T>, ID, DESC, ALIAS, std::vector<T>(), true);

/**
 * Defining useful macros using PARAM macro defined later.
 */
#define PARAM_IN(T, ID, DESC, ALIAS, DEF, REQ) \
    PARAM(T, ID, DESC, ALIAS, #T, REQ, true, false, DEF);

#define PARAM_OUT(T, ID, DESC, ALIAS, DEF, REQ) \
    PARAM(T, ID, DESC, ALIAS, #T, REQ, false, false, DEF);

#define PARAM_MATRIX(ID, DESC, ALIAS, REQ, TRANS, IN) \
    PARAM(arma::mat, ID, DESC, ALIAS, "arma::mat", REQ, IN, \
        TRANS, arma::mat());

#define PARAM_UMATRIX(ID, DESC, ALIAS, REQ, TRANS, IN) \
    PARAM(arma::Mat<size_t>, ID, DESC, ALIAS, "arma::Mat<size_t>", \
        REQ, IN, TRANS, arma::Mat<size_t>());

#define PARAM_COL(ID, DESC, ALIAS, REQ, TRANS, IN) \
    PARAM(arma::vec, ID, DESC, ALIAS, "arma::vec", REQ, IN, TRANS, \
        arma::vec());

#define PARAM_UCOL(ID, DESC, ALIAS, REQ, TRANS, IN) \
    PARAM(arma::Col<size_t>, ID, DESC, ALIAS, "arma::Col<size_t>", \
        REQ, IN, TRANS, arma::Col<size_t>());

#define PARAM_ROW(ID, DESC, ALIAS, REQ, TRANS, IN) \
    PARAM(arma::rowvec, ID, DESC, ALIAS, "arma::rowvec", REQ, IN, \
    TRANS, arma::rowvec());

#define PARAM_UROW(ID, DESC, ALIAS, REQ, TRANS, IN) \
    PARAM(arma::Row<size_t>, ID, DESC, ALIAS, "arma::Row<size_t>", \
    REQ, IN, TRANS, arma::Row<size_t>());

/**
 * Define the PARAM(), PARAM_MODEL() macro. Don't use this function;
 * use the other ones above that call it.  Note that we are using the __LINE__
 * macro for naming these actual parameters when __COUNTER__ does not exist,
 * which is a bit of an ugly hack... but this is the preprocessor, after all.
 * We don't have much choice other than ugliness.
 *
 * @param T Type of the parameter.
 * @param ID Name of the parameter.
 * @param DESC Description of the parameter (1-2 sentences).  Don't use
 *      printing macros like PRINT_PARAM_STRING() or PRINT_DATASET() or others
 *      here---it will cause problems.
 * @param ALIAS Alias for this parameter (one letter).
 * @param DEF Default value of the parameter.
 * @param REQ Whether or not parameter is required (boolean value).
 */
#ifdef __COUNTER__
  #define PARAM(T, ID, DESC, ALIAS, NAME, REQ, IN, TRANS, DEF) \
      static mlpack::util::Option<T> \
      JOIN(io_option_dummy_object_in_, __COUNTER__) \
      (DEF, ID, DESC, ALIAS, NAME, REQ, IN, !TRANS, STRINGIFY(BINDING_NAME));

  #define PARAM_GLOBAL(T, ID, DESC, ALIAS, NAME, REQ, IN, TRANS, DEF) \
      static mlpack::util::Option<T> \
      JOIN(io_option_global_dummy_object_in_, __COUNTER__) \
      (DEF, ID, DESC, ALIAS, NAME, REQ, IN, !TRANS, "");

  // There are no uses of required models, so that is not an option to this
  // macro (it would be easy to add).
  #define PARAM_MODEL(TYPE, ID, DESC, ALIAS, REQ, IN) \
      static mlpack::util::Option<TYPE*> \
      JOIN(io_option_dummy_model_, __COUNTER__) \
      (nullptr, ID, DESC, ALIAS, #TYPE, REQ, IN, false, \
      STRINGIFY(BINDING_NAME));
#else
  // We have to do some really bizarre stuff since __COUNTER__ isn't defined. I
  // don't think we can absolutely guarantee success, but it should be "good
  // enough".  We use the __LINE__ macro and the type of the parameter to try
  // and get a good guess at something unique.
  #define PARAM(T, ID, DESC, ALIAS, NAME, REQ, IN, TRANS, DEF) \
      static mlpack::util::Option<T> \
      JOIN(JOIN(io_option_dummy_object_in_, __LINE__), opt) \
      (DEF, ID, DESC, ALIAS, NAME, REQ, IN, !TRANS, STRINGIFY(BINDING_NAME));

  #define PARAM_GLOBAL(T, ID, DESC,  ALIAS, NAME, REQ, IN, TRANS, DEF) \
      static mlpack::util::Option<T> \
      JOIN(JOIN(io_option_global_dummy_object_in_, __LINE__), opt) \
      (DEF, ID, DESC, ALIAS, NAME, REQ, IN, !TRANS, "");

  #define PARAM_MODEL(TYPE, ID, DESC, ALIAS, REQ, IN) \
      static mlpack::util::Option<TYPE*> \
      JOIN(JOIN(io_option_dummy_object_model_, __LINE__), opt) \
      (nullptr, ID, DESC, ALIAS, #TYPE, REQ, IN, false, \
      STRINGIFY(BINDING_NAME));
#endif

#endif
