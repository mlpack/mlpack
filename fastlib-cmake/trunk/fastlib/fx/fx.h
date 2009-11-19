/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file fx.h
 *
 * Integration support for the FASTexec experiment-running system.
 * This system automates the parsing of command line inputs,
 * management of timers, and reporting of results.  It simultaneously
 * makes it easy for you to extend your program's interface and to
 * document your program in a manner conveniently accessible by users.
 */

#ifndef FX_H
#define FX_H

#include "fastlib/fx/datanode.h"
#include "fastlib/fx/stopwatch.h"
//#include "datanode.h"
//#include "stopwatch.h"

#include "fastlib/base/common.h"

EXTERN_C_BEGIN

/** Somewhat clearer module typename for use with fx. */
typedef struct datanode fx_module;
/** Sonewhat clearer timer typename for use with fx. */
typedef struct stopwatch fx_timer;

/** The datanode under which all fx information is stored. */
extern fx_module *fx_root;

/** Whether to print messages about missing documentation. */
extern int fx_docs_nagging;

/*
 * TODO: Consider adding support for printing params, timers, and
 * results to stdout upon accessing/stopping/writing them.
 */

/**
 * The various kinds of data entries.
 *
 * This type is primarily used to make sure that entries are used
 * consistently.  Undocumented entries are created as FX_UNKNOWN and
 * their types set upon use.
 *
 * Note: the fx_module_is_type function is very sensative to the
 * integer values of the fx_mod_t enum and thus must be updated if any
 * changes are made to its values.
 *
 * @see struct fx_entry_doc, fx_module_is_type
 */
typedef enum {
  /** An undocumented entry. */
  FX_UNKNOWN = 0,
  /** An entry with children but no value. */
  FX_MODULE,
  /** Inputs from the command line or calling modules. */
  FX_PARAM,
  /** A kind of FX_PARAM; inputs that must be specified. */
  FX_REQUIRED,
  /** A kind of FX_PARAM; inputs that must not be specified. */
  FX_RESERVED,
  /** A kind of FX_PARAM; inputs from the command line. */
  FX_PROVIDED,
  /** A kind of FX_PARAM; indicates a default value. */
  FX_DEFAULT,
  /** Reported results. */
  FX_RESULT,
  /** Managed timers. */
  FX_TIMER
} fx_mod_t;

/** Markers for fx_mod_t values: UMPQVPDTR */
extern char fx_mod_marker[];
/** Human-readable names for fx_mod_t values. */
extern const char *fx_mod_name[];

/**
 * The type of the value stored at an entry.
 *
 * As with fx_mod_t, the purpose of this type is to make sure that
 * entries are used consistently.  Except for FX_CUSTOM, all values
 * are stored in string format and are implicitly converted upon use.
 *
 * @see fx_param_str, fx_param_double, fx_param_int, fx_param_bool
 */
typedef enum {
  /** An arbitrary string. */
  FX_STR = 0,
  /** Readable by scanf("%lf", ...). */
  FX_DOUBLE,
  /** Readable by scanf("%lld", ...). */
  FX_INT,
  /** In either set "1tTyY" or "0fFnN". */
  FX_BOOL,
  /** Comma-separated strings. */
  FX_STR_LIST,
  /** Comma-separated doubles. */
  FX_DOUBLE_LIST,
  /** Comma-separated ints. */
  FX_INT_LIST,
  /** Comma-separated bools. */
  FX_BOOL_LIST,
  /** Some other type, e.g. a timer. */
  FX_CUSTOM = -1
} fx_val_t;

/** Human-readable names for fx_val_t values. */
extern const char *fx_val_name[];

struct fx_module_doc;

/**
 * Documentation for an entry, for use with fx_module_doc.
 *
 * You should define a constant list of these at the top of any
 * file/class that makes use of an fx_module.  Such lists should be
 * terminated by FX_ENTRY_DOC_DONE.
 *
 * @see fx_module_doc, fx_submodule_doc, fx_init
 */
typedef struct fx_entry_doc {
  /** The entry's name; paths allowed. */
  const char *key;
  /** The entry's usage type, e.g. FX_REQUIRED. */
  fx_mod_t mod_type;
  /** The entry's data type, e.g. FX_INT. */
  fx_val_t val_type;
  /** Currently unused; constraints on val. */
  const char *meta;
  /** Information to print on --help. */
  const char *text;
} fx_entry_doc;

/**
 * Documentation for a submodule, for use with fx_module_doc.
 *
 * You should define a constant list of these at the top of any
 * file/class that makes use of an fx_module.  Such lists should be
 * terminated by FX_SUBMODULE_DOC_DONE.
 *
 * @see fx_module_doc, fx_entry_doc, fx_init
 */
typedef struct fx_submodule_doc {
  /** The submodule's name; paths allowed. */
  const char *key;
  /** The submodule's documentation. */
  const struct fx_module_doc *doc;
  /** Information to print on --help. */
  const char *text;
} fx_submodule_doc;

/**
 * Documentation for an fx_module, used triply to provide usage
 * information, to check input variables for correctness, and to
 * enforce that programmers indeed do document their parameters.
 *
 * In addition to lists of fx_entry_docs and fx_submodule_docs, you
 * should define a constant of type fx_module_doc at the top of any
 * file/class that makes use of an fx_module.  This constant is then
 * used with either fx_init or other fx_submodule_docs.
 *
 * Example (a hypothetical manifold method):
 * @code
 *   const fx_entry_doc manifold_entries[] = {
 *     {"maxiter", FX_PARAM, FX_INT, NULL,
 *      "  Maximum number of refinement iterations (def 50).\n"},
 *
 *     {"thresh", FX_REQUIRED, FX_DOUBLE, NULL,
 *      "  Threshold parameter of the algorithm.\n"
 *      "  Smaller is more accurate but slower.\n"},
 *
 *     {"emit_iter_times", FX_PARAM, FX_BOOL, NULL,
 *      "  Whether to emit running times for each iteration.\n"}
 *
 *     {"time_i", FX_TIMER, FX_DOUBLE, NULL,
 *      "  The running time of iteration i.\n"},
 *
 *     {"total_time", FX_TIMER, FX_DOUBLE, NULL,
 *      "  The running time of all iterations.\n"},
 *
 *     {"sq_error", FX_RESULT, FX_DOUBLE, NULL,
 *      "  The squared error of the final result.\n"},
 *
 *     {"knn_i/k", FX_RESERVED, FX_INT, NULL,
 *      "  Reserved (selected by algorithm).\n"},
 *
 *     FX_ENTRY_DOC_DONE
 *   };
 *
 *  const fx_submodule_doc manifold_submodules[] = {
 *    {"knn_i", &knn_doc,
 *     "  The k-nearest-neighbors step used in each iteration.\n"},
 *
 *    FX_SUBMODULE_DOC_DONE
 *  };
 *
 *  const fx_moduld_doc manifold_doc = {
 *    manifold_entries, manifold_submodules,
 *    "An implementation of a hypothetical manifold method.\n"
 *  };
 * @endcode
 *
 * In the above, note that "knn_i/k" is a reserved parameter.  This
 * fx_entry_doc will path into the knn_i submodule and override its
 * documented type (normally an FX_PARAM).  The manifold module is
 * then free to assume that "knn_i/k" has not been set on the command
 * line, which would be mistaken because the algorithm sets this
 * parameter itself.
 *
 * Also observe documentation for timer "time_i".  This is not
 * literally the name of the timer entries that will be emitted, but
 * instead serves as a stub to be copied into new entries "time_1",
 * "time_2", and so forth.  The same copying will occur for the knn_i
 * submodule for each iteration.
 *
 * The above is incomplete, however.  It is the documentation for the
 * manifold module, but not the documentation for the entire program.
 * In the same file that defines main, there should be an additional
 * fx_module_doc that references manifold_doc as well as parameters
 * listing which files to open, etc.  This documentation would include
 * &manifold_doc as one of its submodules.
 *
 * @see struct fx_entry_doc, struct fx_submodule_doc, fx_init
 */
typedef struct fx_module_doc {
  /** All entries (params, timers, results) used by the module. */
  const struct fx_entry_doc *entries;
  /** All submodules used by the module. */
  const struct fx_submodule_doc *submodules;
  /** Information to print on --help. */
  const char *text;
} fx_module_doc;

/** A terminator for lists of fx_entry_docs. */
#define FX_ENTRY_DOC_DONE {NULL, (fx_mod_t)0, (fx_val_t)0, NULL}

/** A terminator for lists of fx_submodule_docs. */
#define FX_SUBMODULE_DOC_DONE {NULL, NULL, NULL}

/**
 * Tests whether an entry could be considered to be of a given type.
 *
 * Types have hierarchical relationships given as follows:
 * @code
 *   FX_UNKNOWN
 *     FX_MODULE
 *       FX_PARAM
 *         FX_REQUIRED
 *           FX_RESERVED
 *           FX_PROVIDED
 *           FX_DEFAULT
 *       FX_TIMER
 *       FX_RESULT
 * @endcode
 *
 * Thus, FX_REQUIRED may also serve as FX_PARAM, but not the other way
 * around; additionally FX_RESERVED, FX_PROVIDED, and FX_DEFAULT
 * satisfy FX_REQUIRED.  Note that any kind of entry other than an
 * FX_UNKNOWN may serve as an FX_MODULE, meaning that every kind of
 * entry is permitted to have child entries and, equivalently, that it
 * is possible for entries with children (modules) to have values.
 *
 * @param entry the entry to be tested
 * @param type the type to look for, i.e. is mod of this type?
 *
 * @see fx_mod_t, struct fx_module
 */
int fx_module_is_type(fx_module *entry, fx_mod_t type);

/**
 * Obtains a string parameter, failing if unspecified.
 *
 * @see fx_param_str
 */
const char *fx_param_str_req(fx_module *mod, const char *key);
/**
 * Obtains a floating-point parameter, failing if unspecified.
 *
 * @see fx_param_double
 */
double fx_param_double_req(fx_module *mod, const char *key);
/**
 * Obtains an integral parameter, failing if unspecified.
 *
 * @see fx_param_int
 */
long long fx_param_int_req(fx_module *mod, const char *key);
/**
 * Obtains a boolean parameter, failing if unspecified.
 *
 * @see fx_param_bool
 */
int fx_param_bool_req(fx_module *mod, const char *key);

/**
 * Obtains a list of strings from a parameter, failing if unspecified.
 *
 * @see fx_param_str_list
 */
const char **fx_param_str_list_req(fx_module *mod, const char *key,
				   size_t *size_ptr);
/**
 * Obtains a list of floating-point values from a parameter, failing
 * if unspecified.
 *
 * @see fx_param_double_list
 */
double *fx_param_double_list_req(fx_module *mod, const char *key,
				 size_t *size_ptr);
/**
 * Obtains a list of integral values from a parameter, failing if
 * unspecified.
 *
 * @see fx_param_int_list
 */
long long *fx_param_list_int_req(fx_module *mod, const char *key,
				 size_t *size_ptr);
/**
 * Obtains a list of boolean values from a parameter, failing if
 * unspecified.
 *
 * @see fx_param_int_list
 */
int *fx_param_bool_list_req(fx_module *mod, const char *key,
			    size_t *size_ptr);

/**
 * Obtains a string parameter, using the provided default if
 * unspecified.
 *
 * @param mod the parameter's containing module
 * @param key the name of the parameter; paths allowed
 * @param def value used if parameter not given
 * @returns a const char *; do not free
 *
 * @see fx_param_double, fx_param_int, fx_param_bool,
 *      fx_param_str_list, fx_param_str_req, fx_set_param_str,
 *      fx_default_param
 */
const char *fx_param_str(fx_module *mod, const char *key, const char *def);
/**
 * Obtains a floating-point parameter, using the provided default if
 * unspecified.
 *
 * @param mod the parameter's containing module
 * @param key the name of the parameter; paths allowed
 * @param def value used if parameter not given
 * @returns a double
 *
 * @see fx_param_str, fx_param_double_list, fx_param_double_req,
 *      fx_set_param_double
 */
double fx_param_double(fx_module *mod, const char *key, double def);
/**
 * Obtains an integral parameter, using the provided default if
 * unspecified.
 *
 * @param mod the parameter's containing module
 * @param key the name of the parameter; paths allowed
 * @param def value used if parameter not given
 * @returns a long long; type may be converted implicitly
 *
 * @see fx_param_str, fx_param_int_list, fx_param_int_req
 *      fx_set_param_int
 */
long long fx_param_int(fx_module *mod, const char *key, long long def);
/**
 * Obtains a boolean parameter, using the provided default if
 * unspecified.
 *
 * Booleans should be given with leading character in either set
 * "1tTyY" for true or "0fFnN" for false.  The empty string is also
 * considered true, and thus booleans set on the command line with
 * "--some/bool" (omitting the '=t') are true.
 *
 * @param mod the parameter's containing module
 * @param key the name of the parameter; paths allowed
 * @param def value used if parameter not given
 * @returns an int with value 1 or 0
 *
 * @see fx_param_str, fx_param_bool_list, fx_param_bool_req
 *      fx_set_param_bool
 */
int fx_param_bool(fx_module *mod, const char *key, int def);

/**
 * Obtains a list of comma-separated strings from a parameter, using
 * the trailing arguments as defaults if unspecified.
 *
 * Element strings may contain escaped commas given by "%2c".  Because
 * '%' is itself be escaped, this is "%252c" from the command line.
 *
 * Input *size_ptr must be initialized.  If nonzero, it establishes a
 * required list length.  For example, you may set *size_ptr = 0 but
 * then use it for multiple lists to ensure they are the same length.
 *
 * Do not free the returned list.
 *
 * @param mod the parameter's containing module
 * @param key the name of the parameter; paths allowed
 * @param size_ptr receives the length of the list or establishes the
 *        required length of the list if nonzero
 * @param def_size the number of trailing arguments
 * @param ... values to be used if parameter not given
 * @returns a const char **; do not free
 *
 * @see fx_param_double_list, fx_param_int_list, fx_param_bool_list,
 *      fx_param_str, fx_param_str_array, fx_param_str_list_req,
 *      fx_set_param_str_list, fx_default_param_list
 */
const char **fx_param_str_list(fx_module *mod, const char *key,
			       size_t *size_ptr, size_t def_size, ...);
/**
 * Obtains a list of comma-separated floating-point values from a
 * parameter, using the trailing arguments as defaults if unspecified.
 *
 * Input *size_ptr must be initialized.  If nonzero, it establishes a
 * required list length.  For example, you may set *size_ptr = 0 but
 * then use it for multiple lists to ensure they are the same length.
 *
 * Do not free the returned list.
 *
 * @param mod the parameter's containing module
 * @param key the name of the parameter; paths allowed
 * @param size_ptr receives the length of the list or establishes the
 *        required length of the list if nonzero
 * @param def_size the number of trailing arguments
 * @param ... values to be used if parameter not given
 * @returns a double *; do not free
 *
 * @see fx_param_str_list, fx_param_double, fx_param_double_array,
 *      fx_param_double_list_req, fx_set_param_double_list
 */
double *fx_param_double_list(fx_module *mod, const char *key,
			     size_t *size_ptr, size_t def_size, ...);
/**
 * Obtains a list of comma-separated integral values from a parameter,
 * using the trailing arguments as defaults if unspecified.
 *
 * Input *size_ptr must be initialized.  If nonzero, it establishes a
 * required list length.  For example, you may set *size_ptr = 0 but
 * then use it for multiple lists to ensure they are the same length.
 *
 * Do not free the returned list.
 *
 * @param mod the parameter's containing module
 * @param key the name of the parameter; paths allowed
 * @param size_ptr receives the length of the list or establishes the
 *        required length of the list if nonzero
 * @param def_size the number of trailing arguments
 * @param ... values to be used if parameter not given
 * @returns a long long *; no implicit type conversion; do not free
 *
 * @see fx_param_str_list, fx_param_int, fx_param_int_array,
 *      fx_param_int_list_req, fx_set_param_int_list
 */
long long *fx_param_int_list(fx_module *mod, const char *key,
			     size_t *size_ptr, size_t def_size, ...);
/**
 * Obtains a list of comma-separated boolean values from a parameter,
 * using the trailing arguments as defaults if unspecified.
 *
 * Input *size_ptr must be initialized.  If nonzero, it establishes a
 * required list length.  For example, you may set *size_ptr = 0 but
 * then use it for multiple lists to ensure they are the same length.
 *
 * Do not free the returned list.
 *
 * @param mod the parameter's containing module
 * @param key the name of the parameter; paths allowed
 * @param size_ptr receives the length of the list or establishes the
 *        required length of the list if nonzero
 * @param def_size the number of trailing arguments
 * @param ... values to be used if parameter not given
 * @returns an int * with values 1 or 0; do not free
 *
 * @see fx_param_str_list, fx_param_bool, fx_param_bool_array,
 *      fx_param_bool_list_req, fx_set_param_bool_list
 */
int *fx_param_bool_list(fx_module *mod, const char *key,
			size_t *size_ptr, size_t def_size, ...);

/**
 * Obtains a list of comma-separated strings from a parameter, using
 * the provided array as defaults if unspecified.
 *
 * @see fx_param_str_list
 */
const char **fx_param_str_array(fx_module *mod, const char *key,
				size_t *size_ptr, size_t def_size,
				const char *const *def_array);
/**
 * Obtains a list of comma-separated floating-point values from a
 * parameter, using the provided array as defaults if unspecified.
 *
 * @see fx_param_double_list
 */
double *fx_param_double_array(fx_module *mod, const char *key,
			      size_t *size_ptr, size_t def_size,
			      const double *def_array);
/**
 * Obtains a list of comma-separated integral values from a parameter,
 * using the provided array as defaults if unspecified.
 *
 * @see fx_param_int_list
 */
long long *fx_param_int_array(fx_module *mod, const char *key,
			      size_t *size_ptr, size_t def_size,
			      const long long *def_array);
/**
 * Obtains a list of comma-separated boolean values from a parameter,
 * using the provided array as defaults if unspecified.
 *
 * @see fx_param_bool_list
 */
int *fx_param_bool_array(fx_module *mod, const char *key,
			 size_t *size_ptr, size_t def_size,
			 const int *def_array);

/**
 * Sets a reserved parameter to a given string.
 *
 * @see fx_param_str, fx_set_param_str_list, fx_result_str,
 *      fx_format_param
 */
void fx_set_param_str(fx_module *mod, const char *key, const char *val);
/**
 * Sets a reserved parameter to a given double.
 *
 * @see fx_param_double, fx_set_param_double_list, fx_result_double
 */
void fx_set_param_double(fx_module *mod, const char *key, double val);
/**
 * Sets a reserved parameter to a given int.
 *
 * @see fx_param_int, fx_set_param_int_list, fx_result_int
 */
void fx_set_param_int(fx_module *mod, const char *key, long long val);
/**
 * Sets a reserved parameter to a given bool.
 *
 * @see fx_param_bool, fx_set_param_bool_list, fx_result_bool
 */
void fx_set_param_bool(fx_module *mod, const char *key, int val);

/**
 * Sets a reserved parameter to a given list of strings.
 *
 * @see fx_param_str_list, fx_set_param_str, fx_set_param_str_array,
 *      fx_result_str_list, fx_format_param_list
 */
void fx_set_param_str_list(fx_module *mod, const char *key,
			   size_t size, ...);
/**
 * Sets a reserved parameter to a given list of doubles.
 *
 * @see fx_param_double_list, fx_set_param_double,
 *      fx_set_param_double_array, fx_result_double_list
 */
void fx_set_param_double_list(fx_module *mod, const char *key,
			      size_t size, ...);
/**
 * Sets a reserved parameter to a given list of ints.
 *
 * @see fx_param_int_list, fx_set_param_int, fx_set_param_int_array,
 *      fx_result_int_list
 */
void fx_set_param_int_list(fx_module *mod, const char *key,
			   size_t size, ...);
/**
 * Sets a reserved parameter to a given list of bools.
 *
 * @see fx_param_bool_list, fx_set_param_bool,
 *      fx_set_param_bool_array, fx_result_bool_list
 */
void fx_set_param_bool_list(fx_module *mod, const char *key,
			    size_t size, ...);

/**
 * Sets a reserved parameter to a given array of strings.
 *
 * @see fx_param_str_array, fx_set_param_str_list
 */
void fx_set_param_str_array(fx_module *mod, const char *key,
			    size_t size, const char *const *array);
/**
 * Sets a reserved parameter to a given array of doubles.
 *
 * @see fx_param_double_array, fx_set_param_double_list
 */
void fx_set_param_double_array(fx_module *mod, const char *key,
			       size_t size, const double *array);
/**
 * Sets a reserved parameter to a given array of ints.
 *
 * @see fx_param_int_array, fx_set_param_int_list
 */
void fx_set_param_int_array(fx_module *mod, const char *key,
			    size_t size, const long long *array);
/**
 * Sets a reserved parameter to a given array of booleans.
 *
 * @see fx_param_bool_array, fx_set_param_bool_list
 */
void fx_set_param_bool_array(fx_module *mod, const char *key,
			     size_t size, const int *array);

/**
 * Defaults a parameter to a string formatted as in printf.
 *
 * @see fx_param_str, fx_default_param_list, fx_format_param
 */
COMPILER_PRINTF(3, 4)
void fx_default_param(fx_module *mod, const char *key,
		      const char *def_format, ...);
/**
 * Defaults a parameter to a list of strings formatted as in printf.
 *
 * Values should be provided in batches composed of element format
 * strings followed by their arguments.
 *
 * Example (the cheese stands alone):
 * @code
 *   fx_default_param_list(root, "foo", 3,
 *       "%s chases %s", dog_name, cat_name,
 *       "%d mice observed", mouse_count,
 *       "%g%% cheese remaining", 100 * cheese);
 * @endcode
 *
 * @see fx_param_str_list, fx_default_param, fx_format_param_list
 */
void fx_default_param_list(fx_module *mod, const char *key,
			   size_t *size_ptr, size_t def_size, ...);
/**
 * Sets a reserved parameter to a string formatted as in printf.
 *
 * @see fx_set_param_str, fx_default_param, fx_format_param_list
 */
COMPILER_PRINTF(3, 4)
void fx_format_param(fx_module *mod, const char *key,
		     const char *format, ...);
/**
 * Sets a reserved parameter to a list of strings formatted as in
 * printf.
 *
 * Values should be provided in batches composed of element format
 * strings followed by their arguments.
 *
 * @see fx_set_param_str_list, fx_default_param_list, fx_format_param
 */
void fx_format_param_list(fx_module *mod, const char *key,
			  size_t size, ...);

/**
 * Unsets a parameter's value, effectively removing the entry.
 *
 * @param mod the parameter's containing module
 * @param key the name of the parameter; paths allowed
 *
 * @param fx_param_exists, fx_param_str
 */
void fx_clear_param(fx_module *mod, const char *key);
/**
 * Tests whether a parameter has been specified.
 *
 * @param mod the parameter's containing module
 * @param key the name of the parameter; paths allowed
 * @returns whether the parameter has a value
 *
 * @see fx_param_clear, fx_param_str
 */
int fx_param_exists(fx_module *mod, const char *key);

/**
 * Obtains a string result, failing if not found.
 *
 * @param mod the result's containing module
 * @param key the name of the result; paths allowed
 * @returns a const char *; do not free
 *
 * @see fx_result_str, fx_get_result_str_list, fx_param_str
 */
const char *fx_get_result_str(fx_module *mod, const char *key);
/**
 * Obtains a floating-point result, failing if not found.
 *
 * @param mod the result's containing module
 * @param key the name of the result; paths allowed
 * @returns a double
 *
 * @see fx_result_double, fx_get_result_double_list, fx_param_double
 */
double fx_get_result_double(fx_module *mod, const char *key);
/**
 * Obtains an integral result, failing if not found.
 *
 * @param mod the result's containing module
 * @param key the name of the result; paths allowed
 * @returns a long long; type may be converted implicitly
 *
 * @see fx_result_int, fx_get_result_int_list, fx_param_int
 */
long long fx_get_result_int(fx_module *mod, const char *key);
/**
 * Obtains a boolean result, failing if not found.
 *
 * @param mod the result's containing module
 * @param key the name of the result; paths allowed
 * @returns an int with value 1 or 0
 *
 * @see fx_result_bool, fx_get_result_bool_list, fx_param_bool
 */
int fx_get_result_bool(fx_module *mod, const char *key);

/**
 * Obtains a list of strings from a result, failing if not found.
 *
 * @see fx_get_result_str, fx_result_str_list, fx_param_str_list
 */
const char **fx_get_result_str_list(fx_module *mod, const char *key,
				    size_t *size_ptr);
/**
 * Obtains a list of floating-point values from a result, failing if
 * not found.
 *
 * @see fx_get_result_double, fx_result_double_list,
 *      fx_param_double_list
 */
double *fx_get_result_double_list(fx_module *mod, const char *key,
				  size_t *size_ptr);
/**
 * Obtains a list of integral values from a result, failing if not
 * found.
 *
 * @see fx_param_int_list, fx_get_result_int, fx_result_int_list
 */
long long *fx_get_result_list_int(fx_module *mod, const char *key,
				  size_t *size_ptr);
/**
 * Obtains a list of boolean values from a result, failing if not
 * found.
 *
 * @see fx_get_result_bool, fx_result_bool_list, fx_param_bool_list
 */
int *fx_get_result_bool_list(fx_module *mod, const char *key,
			     size_t *size_ptr);

/**
 * Sets a result to a given string.
 *
 * @see fx_result_dobule, fx_result_int, fx_result_bool,
 *      fx_result_str_list, fx_get_result_str, fx_param_str,
 *      fx_format_result
 */
void fx_result_str(fx_module *mod, const char *key, const char *val);
/**
 * Sets a result to a given double.
 *
 * @see fx_result_str, fx_result_double_list, fx_get_result_double,
 *      fx_param_double
 */
void fx_result_double(fx_module *mod, const char *key, double val);
/**
 * Sets a result to a given int.
 *
 * @see fx_result_str, fx_result_int_list, fx_get_result_int,
 *      fx_param_int
 */
void fx_result_int(fx_module *mod, const char *key, long long val);
/**
 * Sets a result to a given bool.
 *
 * @see fx_result_str, fx_result_bool_list, fx_get_result_bool,
 *      fx_param_bool
 */
void fx_result_bool(fx_module *mod, const char *key, int val);

/**
 * Sets a result to a given list of strings.
 *
 * @see fx_get_result_str_list, fx_result_str, fx_result_str_array,
 *      fx_param_str_list, fx_format_result_list
 */
void fx_result_str_list(fx_module *mod, const char *key,
			size_t size, ...);
/**
 * Sets a result to a given list of doubles.
 *
 * @see fx_get_result_double_list, fx_result_double,
 *      fx_result_double_array, fx_param_double_list
 */
void fx_result_double_list(fx_module *mod, const char *key,
			   size_t size, ...);
/**
 * Sets a result to a given list of ints.
 *
 * @see fx_get_result_int_list, fx_result_int, fx_result_int_array,
 *      fx_param_int_list
 */
void fx_result_int_list(fx_module *mod, const char *key,
			size_t size, ...);
/**
 * Sets a result to a given list of bools.
 *
 * @see fx_get_result_bool_list, fx_result_bool, fx_result_bool_array,
 *      fx_param_bool_list
 */
void fx_result_bool_list(fx_module *mod, const char *key,
			 size_t size, ...);

/**
 * Sets a result to a given array of strings.
 *
 * @see fx_result_str_list, fx_param_str_list
 */
void fx_result_str_array(fx_module *mod, const char *key,
			 size_t size, const char *const *array);
/**
 * Sets a result to a given array of doubles.
 *
 * @see fx_result_double_list, fx_param_double_list
 */
void fx_result_double_array(fx_module *mod, const char *key,
			    size_t size, const double *array);
/**
 * Sets a result to a given array of ints.
 *
 * @see fx_result_int_list, fx_param_int_list
 */
void fx_result_int_array(fx_module *mod, const char *key,
			 size_t size, const long long *array);
/**
 * Sets a result to a given array of bools.
 *
 * @see fx_result_bool_list, fx_param_bool_list
 */
void fx_result_bool_array(fx_module *mod, const char *key,
			  size_t size, const int *array);

/**
 * Sets a result to a string formatted as in printf.
 *
 * @see fx_result_str, fx_format_result_list, fx_format_param
 */
COMPILER_PRINTF(3, 4)
void fx_format_result(fx_module *mod, const char *key,
		      const char *format, ...);
/**
 * Sets a result to a list of strings formatted as in printf.
 *
 * Values should be provided in batches composed of element format
 * strings followed by their arguments.
 *
 * @see fx_result_str_list, fx_format_result, fx_format_param_list
 */
void fx_format_result_list(fx_module *mod, const char *key,
			   size_t size, ...);

/**
 * Unsets a result's value, effectively removing the entry.
 *
 * @param mod the result's containing module
 * @param key the name of the result; paths allowed
 *
 * @param fx_result_exists, fx_param_exists, fx_result_str
 */
void fx_clear_result(fx_module *mod, const char *key);
/**
 * Tests whether a result has been specified.
 *
 * @param mod the result's containing module
 * @param key the name of the result; paths allowed
 * @returns whether the result has a value
 *
 * @see fx_result_clear, fx_param_clear, fx_result_str
 */
int fx_result_exists(fx_module *mod, const char *key);

/**
 * Gets a timer for the purpose of reading its time.
 *
 * @param mod the timer's containing module
 * @param key the name of the timer; paths allowed
 * @returns an fx_timer * (struct stopwatch *); do not free
 *
 * @see struct stopwatch, fx_timer_start, fx_timer_stop
 */
fx_timer *fx_get_timer(fx_module *mod, const char *key);
/**
 * Starts or continues a named timer, or creates one if it does not
 * exist.
 *
 * @param mod the timer's containing module
 * @param key the name of the timer; paths allowed
 *
 * @see fx_timer_stop, fx_reset_timer, fx_get_timer
 */
void fx_timer_start(fx_module *mod, const char *key);
/**
 * Pauses a named timer.
 *
 * @param mod the timer's containing module
 * @param key the name of the timer; paths allowed
 *
 * @see fx_timer_start, fx_reset_timer, fx_get_timer
 */
void fx_timer_stop(fx_module *mod, const char *key);
/**
 * Resets a named timer's total time to zero.
 *
 * @param mod the timer's containing module
 * @param key the name of the timer; paths allowed
 *
 * @see fx_timer_start, fx_timer_stop, fx_get_timer
 */
void fx_reset_timer(fx_module *mod, const char *key);

/**
 * Paths into a submodule of a FASTexec module, permitting access of
 * contained parameters without explicitly pathing to them.
 *
 * If the submodule does not exist, it is created and is initialy
 * empty.  Parameter access will then populate it with defaults.
 *
 * Submodules serve to compartmentalize modular components of a
 * program, e.g. so that their parameters and results may be managed
 * separately from those of other components.  Working within a
 * submodule should be understood as changing the working directory in
 * a UNIX file system.  Like the terms "directory" and "subdirectory",
 * there is no real distinction between "module" and "submodule"
 * except relative to one another.
 *
 * Example (tree building command line input):
 * @code
 *   ./myprog --r=data.txt --r/leaf_size=30 --r/split_median
 * @endcode
 *
 * Example (tree building parameter parsing):
 * @code
 *   const char *r_file = fx_param_str_req(root, "r")
 *   ...
 *   fx_module *r_mod = fx_submodule(root, "r");
 *   int leaf_size = fx_param_int(r_mod, "leaf_size", 30);
 *   int split_median = fx_param_bool(r_mod, "split_median", 0);
 * @endcode
 *
 * Note that submodules may also have values.  It is possible to
 * obtain the value of "r" from r_mod with @c fx_param_str(r_mod, ".")
 *
 * @param mod the containing module of the submodule
 * @param key the name of the submodule; paths allowed
 * @returns an fx_module * (struct datanode *); do not free
 *
 * @see fx_copy_module, fx_param_str, fx_init, fx_done
 */
fx_module *fx_submodule(fx_module *mod, const char *key);
/**
 * Copies a (sub)module, its contents, and documentation to a new
 * location.
 *
 * This function is primarily useful when iterative steps of an
 * algorithm deserve their own submodules for parameters and results.
 *
 * Example (hypothetical manifold method's iteration):
 * @code
 *   fx_param_int(manifold_mod, "knn_i/leaf_size", 5);
 *   while (...) {
 *     fx_module *knn_i =
 *         fx_copy_module(manifold_mod, "knn_i", "knn_%d", iter);
 *     fx_set_param_int(knn_i, "k", cur_k);
 *     ...
 *     AllKNN allknn;
 *     allknn.Init(knn_i);
 *     ...
 *   }
 * @endcode
 *
 * This code first provides an alternate default for parameter
 * "leaf_size" of knn_i, which may have been specified on the command
 * line and is presumably different from the default that class AllKNN
 * uses.  It then copies the submodule (replicating any command line
 * arguments and the default for "leaf_size") for each iteration of a
 * loop, setting a specific value for "k".
 *
 * Example (forwarding documentation):
 * @code
 *   while(...) {
 *     fx_module *timer_mod =
 *         fx_copy_module(manifold_mod, "time_i", "time_%d", iter);
 *     ...
 *     fx_timer_start(timer_mod, ".");
 *     ...
 *   }
 * @endcode
 *
 * FASTexec is rather pedantic about documentation, but tricks like
 * the above allow you to have run-time generated entries that borrow
 * documentation from entries specified in the fx_entry_docs.  Also,
 * the printf-like formatting of destination names makes it easier for
 * you to give your generated entries unique names.
 *
 * @param mod the containing module of the source and destination
 * @param src_key the name of the source submodule; paths allowed
 * @param dest_format a format string for the destination name
 * @param ... format arguments, as in printf
 * @returns the dest fx_module * (struct datanode *); do not free
 *
 * @see fx_submodule, fx_param_str, fx_init, fx_done
 */
COMPILER_PRINTF(3, 4)
fx_module *fx_copy_module(fx_module *mod, const char *src_key,
			  const char *dest_format, ...);

/**
 * Prints help information from a FASTexec documentation structure for
 * a given submodule.
 *
 * This function is somewhat distinct from other FASTexec routines
 * because it works directly on the fx_module_doc structure rather
 * than an fx_module.  Pathing works similarly, though ".." does not
 * work because documentation structures do not necessarily have
 * unique parents.  Use "" to obtain the structure's root details.
 *
 * All possible matches to the queried key are printed to screen.
 *
 * @param doc the documentation structure containing the submodule
 * @param key the name of the submodule; paths allowed
 *
 * @see fx_module_doc, fx_init
 */
success_t fx_help(const fx_module_doc *doc, const char *key);

/**
 * Creates and prepares an fx_module for use given the command line
 * and a documentation structure.
 *
 * You should always use fx_init to create fx_modules and fx_done when
 * finished with them.
 *
 * This function performs numerous additional tasks such as loading
 * files specified with "--fx/load", reporting system information, and
 * initializing debug-mode parameters.
 *
 * You may use fx_init in the absense of command line arguments by
 * setting argc = 0 and argv = NULL.  You may use fx_init in the
 * absense of documentation by setting doc = NULL, but this will
 * elicit many warnings about undocumented entries and submodules.
 *
 * You may use fx_init more than once in a program, but note that each
 * use that parses the command line (or a command-line-like structure)
 * will reset globals associated with base/debug.h.
 *
 * @param argc as in main
 * @param argv as in main
 * @param doc a FASTexec documentation structure
 * @returns an fx_module * (struct datanode *); free with fx_done
 *
 * @see fx_done, fx_param_int, fx_module_doc
 */
fx_module *fx_init(int argc, char **argv, const fx_module_doc *doc);
/**
 * Finalizes, emits, and destructs an fx_module.
 *
 * You should always use fx_init to create fx_modules and fx_done when
 * finished with them.
 *
 * Parameter settings, measured times, and results are printed to
 * screen unless boolean fx/silent has been set (from the command line
 * or otherwise).  This information is also written to the file given
 * by string fx/store, if provided.
 *
 * You may also optionally emit rusage information by setting boolean
 * fx/rusage.
 *
 * @see fx_param_init
 */
void fx_done(fx_module *root);

EXTERN_C_END

#endif
