// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file fx.h
 *
 * Support for integration with the experiment-running system, FASTexec.
 *
 * The primary advantage of using these functions is all of the reporting
 * that is automatically done for you, and how easy it is to report new
 * things.
 *
 * This has several main parts:
 *
 * - Parsing command lines
 * - Arbitrary timers
 * - Automatic recording of resource statistics
 * - Returning data back to FASTexec
 */

#ifndef FX_H
#define FX_H

#include "datastore.h"
#include "timer.h"

#include "base/compiler.h"
#include "base/common.h"

#include <stdio.h>

EXTERN_C_START

/** Datanode under which all fx information is stored. */
extern struct datanode *fx_root;

/**
 * Initialize FASTexec with data from the command line.
 *
 * This also records some standard system info and starts the default
 * timer.
 */
void fx_init(int argc, char **argv);
/**
 * Call this at the end of your program.
 */
void fx_done(void);

/**
 * Check if a particular parameter was specified.
 */
int fx_param_exists(struct datanode *module, const char *name);
/**
 * Obtain a string parameter or use provided default.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter (paths allowed)
 * @param def value used if param not given, or NULL if required
 */
const char *fx_param_str(struct datanode *module, const char *name,
			 const char *def);
/**
 * Obtain a string parameter, failing if it is not specified.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter (paths allowed)
 */
const char *fx_param_str_req(struct datanode *module, const char *name);
/**
 * Obtain a floating-point parameter or use provided default.
 *
 * If the parameter is not specified by the user and the default is not
 * DBL_NAN, then the default value will be stored explicitly in the tree.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter (paths allowed)
 * @param def value used if param not given
 */
double fx_param_double(struct datanode *module, const char *name, double def);
/**
 * Obtain a floating-point parameter, failing if it is not specified.
 *
 * If the parameter is not specified by the user and the default is not
 * DBL_NAN, then the default value will be stored explicitly in the tree.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter (paths allowed)
 */
double fx_param_double_req(struct datanode *module, const char *name);
/**
 * Obtain an integral parameter or use provided default.
 *
 * If the parameter is not specified by the user and the default is not -1,
 * then the default value will be stored explicitly in the tree.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter (paths allowed)
 * @param def value used if param not given
 */
long long int fx_param_int(struct datanode *module, const char *name, long long int def);
/**
 * Obtain an integral parameter, failing if it is not specified.
 *
 * If the parameter is not specified by the user and the default is not -1,
 * then the default value will be stored explicitly in the tree.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter (paths allowed)
 */
long long int fx_param_int_req(struct datanode *module, const char *name);
/**
 * Obtain a boolean parameter or use provided default.
 *
 * Values starting with f, F, n, N, or 0 are false; all others are
 * true.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter (paths allowed)
 * @param def value used if param not given
 */
int fx_param_bool(struct datanode *module, const char *name, int def);
/**
 * Obtain a boolean parameter, failing if it is not specified.
 *
 * Values starting with f, F, n, N, or 0 are false; all others are
 * true.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter (paths allowed)
 */
int fx_param_bool_req(struct datanode *module, const char *name);
/**
 * Obtain a segment of the datastore corresponding to a parameter.
 *
 * Useful for parameters with sub-options.  Note that no default value
 * may be specified; missing sub-parameters will be filled with
 * defaults when they are queried.
 *
 * Sub-parameters names must be given with preceding ./, as follows:
 *
 * @code
 * struct datanode *l_prop = fx_param_node(NULL, "leaf_properties");
 * int l_size = fx_param_int(l_prop, "./leaf_size", "30");
 * @code
 *
 * The above may alternately have been accomplished with:
 *
 * @code
 * int l_size = fx_param_int(NULL, "leaf_properties/leaf_size", "30");
 * @endcode
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter node (paths allowed)
 */
struct datanode *fx_param_node(struct datanode *module, const char *name);

/**
 * Set a parameter to a default value if it does not already exist.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter (paths allowed)
 * @param def the value the parameter assumes if unspecified
 */
void fx_default_param(struct datanode *module, const char *name,
		      const char *def);
/**
 * Set a parameter to a given value.
 *
 * This raises a warning if the parameter already exists.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter (paths allowed)
 * @param val the string to set the parameter to
 */
void fx_set_param(struct datanode *module, const char *name,
		  const char *val);
/**
 * Set a parameter to a formatted string.
 *
 * This raises a warning if the parameter already exists.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter (paths allowed)
 * @param format a format string for the parameter, as in printf
 */
COMPILER_PRINTF(3, 4)
void fx_format_param(struct datanode *module, const char *name,
		     const char *format, ...);
/**
 * Remove a parameter or parameter node if it exists.
 *
 * This raises a warning if the parameter existed.
 *
 * @param module the param's containing module, or NULL for global
 * @param name the name of the parameter to clear (paths allowed)
 */
void fx_clear_param(struct datanode *module, const char *name);

/**
 * Copy parameters from one module to another as defaults.
 *
 * This is especially useful if a group of parameters must be copied
 * between various modules.
 *
 * Copies dest_module+"/params/"+destname to src_module+"/params/"+srcname.
 *
 * @param dest_module the module to copy it to
 * @param destname the name under the module's params to copy it under, you
 *        will likely want to use an empty string
 * @param src_module the module to copy it from
 * @param srcname the name under the module's params to copy it from
 */
void fx_default_param_node(struct datanode *dest_module, const char *destname,
			   struct datanode *src_module, const char *srcname);
/**
 * Copy parameters from one module to another, overwriting.
 *
 * This raises warnings if any parameters are overwitten.  Parameters
 * that exist in the destination node but not the source node are
 * unchanged.
 *
 * Copies dest_module+"/params/"+destname to src_module+"/params/"+srcname.
 *
 * @param dest_module the module to copy it to
 * @param destname the name under the module's params to copy it under, you
 *        will likely want to use an empty string
 * @param src_module the module to copy it from
 * @param srcname the name under the module's params to copy it from
 */
void fx_set_param_node(struct datanode *dest_module, const char *destname,
		       struct datanode *src_module, const char *srcname);

/**
 * Record a result.
 *
 * Note that datastores are not intended to store large results such
 * as matrices.  These should be written to their own files and the
 * file names should be referenced in the datastore.
 *
 * @param module the result's containing module, or NULL for global
 * @param name the name of the result (paths allowed)
 * @param val the value to set it to
 */
void fx_set_result(struct datanode *module, const char *name, const char *val);
/**
 * Record a result with a formatted string.
 *
 * Note that datastores are not intended to store large results such
 * as matrices.  These should be written to their own files and the
 * file names should be referenced in the datastore.
 *
 * @param module the result's containing module, or NULL for global
 * @param name the name of the result (paths allowed)
 * @param format a format string for the result, as in printf
 */
COMPILER_PRINTF(3, 4)
void fx_format_result(struct datanode *module, const char *name,
		      const char *format, ...);
/**
 * Remove a result or result node if it exists.
 *
 * @param module the result's containing module, or NULL for global
 * @param name the name of the result to clear (paths allowed)
 */
void fx_clear_result(struct datanode *module, const char *name);

/**
 * Starts or continues a named timer.
 *
 * If the named timer does not exist, it is created and initialized to
 * zero.
 *
 * @param module the timer's containing module, or NULL for global
 * @param name the timer name, or NULL for the default timer
 */
void fx_timer_start(struct datanode *module, const char *name);
/**
 * Stops or pauses a named timer.
 *
 * Only pause the default timer sparingly.
 *
 * @param module the timer's containing module, or NULL for global
 * @param name the timer name, or NULL for the default timer
 */
void fx_timer_stop(struct datanode *module, const char *name);
/**
 * Gets the particular timer so you can read it's information.
 *
 * @param module the timer's containing module, or NULL for global
 * @param name the timer name, or NULL for the default timer
 */
struct timer *fx_timer(struct datanode *module, const char *name);


/**
 * Obtain a sub-module, optionally copying parameters into it.
 *
 * This is used to prepare a section of the datastore for use by a
 * modular component of the program.  Parameters specified in the
 * command line that pertain to the submodule must be forwarded from
 * the calling module into the submodule's new parameter space.  For
 * instance, we may call a program with:
 *
 * @code
 * --tree_building/leaf_size 20 --tree_building/split_on midpoint
 * @endcode
 *
 * These are stored in global params/tree_building, but should be
 * moved into the tree-building module's parameters, tree_bldg/params,
 * before they are used.
 *
 * @code
 * struct datanode *tb_mod = fx_module(NULL, "tree_building", "tree_bldg");
 * @endcode
 *
 * Any existing parameters (e.g. if the submodule is not fresh) are
 * overwritten, raising warnings.  Specifying NULL for param disables
 * forwarding.
 *
 * @param module the containing module of the submodule
 * @param param the param node to forward (paths allowed) or NULL
 * @param name_format the name of the submodule (paths allowed);
 *        formatted as in printf
 */
COMPILER_PRINTF(3, 4)
struct datanode *fx_submodule(struct datanode *module, const char *param,
			      const char *name_format, ...);

/**
 * Scopes all FASTexec output under an overarching module.
 *
 * This is used in MPI runs, so that each machine writes to a different
 * part of FASTexec.
 *
 * @param scope_name the overarching scope name
 */
void fx_scope(const char *scope_name);

/**
 * Cause the current process not to output any fastexec results.
 */
void fx_silence();

EXTERN_C_END

#endif
