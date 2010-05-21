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
 * @file fx.c
 *
 * Definitions for integration with the experiment-running system.
 */

#include "fx.h"
/*#include "fx.h"*/

#include "../base/debug.h"

#include <stdarg.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/utsname.h>
#include <pthread.h>



/* TODO: Use this mutex where appropriate */
// static pthread_mutex_t fx__mutex;

fx_module *fx_root = NULL;
int fx_docs_nagging = 1;



char fx_mod_marker[] = "UMPQVPDRT";

const char *fx_mod_name[] = {
  "unknown",
  "submodule",
  "parameter",
  "required parameter",
  "reserved parameter",
  "provided parameter",
  "default parameter",
  "result",
  "timer"
};

const char *fx_val_name[] = {
  "string",
  "double",
  "int",
  "bool",
  "string list",
  "double list",
  "int list",
  "bool list"
};



/* Long enough to store a double, long long, or bool as a string
 * - doubles have 16 sig figs, as in: -X.XXXXXXXXXXXXXXXe-XXX
 * - ints have at most 3 sig figs per byte; they need 2 more chars for
 *   sign and null, but have enough spare even if just 4 bytes
 * - bools are strings "true" or "false"
 */
#define FX__BUF_SIZE (3 * (sizeof(long long) > 8 ? sizeof(long long) : 8))



const fx_entry_doc fx__fx_entries[] = {
  {"load", FX_PARAM, FX_STR_LIST, NULL,
   "   Load files containing additional input parameters.\n"},
  {"output", FX_PARAM, FX_STR, NULL,
   "   Output results to this file.\n"},
  {"timing", FX_PARAM, FX_BOOL, NULL,
   "   Whether to attempt speed up for a timing run.\n"},
  {"rusage", FX_PARAM, FX_BOOL, NULL,
   "   Whether to emit rusage data in \"info\".\n"},
  {"silent", FX_PARAM, FX_BOOL, NULL,
   "   Whether to skip the printing of results to screen.\n"},
  {"no_output_types", FX_PARAM, FX_BOOL, NULL,
   "   Whether to skip the printing of entry types (e.g. \"/param:P\").\n"},
  {"no_docs_nagging", FX_PARAM, FX_BOOL, NULL,
   "   Whether to suppress messages about missing documentation.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc fx__fx_doc = {
  fx__fx_entries, NULL,
  "Options for the FASTexec experiment-running system.\n"
};

const fx_entry_doc fx__debug_entries[] = {
  {"verbosity_level", FX_PARAM, FX_DOUBLE, NULL,
   "   Controls the amount of debug info to print.\n"},
  {"print_got_heres", FX_PARAM, FX_BOOL, NULL,
   "   Whether to print \"got here\" notices.\n"},
  {"print_warnings", FX_PARAM, FX_BOOL, NULL,
   "   Whether to print warning messages.\n"},
  {"abort_on_nonfatal", FX_PARAM, FX_BOOL, NULL,
   "   Whether to treat nonfatal errors as fatal.\n"},
  {"pause_on_nonfatal", FX_PARAM, FX_BOOL, NULL,
   "   Whether to wait for user feedback on nonfatal errors.\n"},
  {"print_notify_locs", FX_PARAM, FX_BOOL, NULL,
   "   Whether to print \"file:function:line:\" before notifications.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc fx__debug_doc = {
  fx__debug_entries, NULL,
  "Options for FASTlib's debugging features.\n"
};

const fx_entry_doc fx__rusage_entries[] = {
  {"utime/sec", FX_RESULT, FX_INT, NULL,
   "   Seconds of computation in user-time.\n"},
  {"utime/usec", FX_RESULT, FX_INT, NULL,
   "   Additional microseconds spent in user-time.\n"},
  {"stime/sec", FX_RESULT, FX_INT, NULL,
   "   Seconds of computation in system-time.\n"},
  {"stime/usec", FX_RESULT, FX_INT, NULL,
   "   Additional microseconds spent in system-time.\n"},
  {"minflt", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"majflt", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"maxrss", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"ixrss", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"idrss", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"isrss", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"nswap", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"inblock", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"oublock", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"msgsnd", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"msgrcv", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"nsignals", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"nvcsw", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  {"nivcsw", FX_RESULT, FX_INT, NULL,
   "   (See getrusage documentation.)\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc fx__rusage_doc = {
  fx__rusage_entries, NULL,
  "Resource usage information generated by getrusage.\n"
};

const fx_submodule_doc fx__info_submods[] = {
  {"rusage/self", &fx__rusage_doc,
   "   Resources consumed by the experiment's main thread.\n"},
  {"rusage/children", &fx__rusage_doc,
   "   Resources consumed by all child threads.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_entry_doc fx__info_entries[] = {
  {"sys/node/name", FX_RESULT, FX_STR, NULL,
   "   The host computer of the experiment.\n"},
  {"sys/arch/name", FX_RESULT, FX_STR, NULL,
   "   The host computer's architecture, i.e. 64-bit x86.\n"},
  {"sys/kernel/name", FX_RESULT, FX_STR, NULL,
   "   The name of operating system.\n"},
  {"sys/kernel/release", FX_RESULT, FX_STR, NULL,
   "   The version of the operating system.\n"},
  {"sys/kernel/build", FX_RESULT, FX_STR, NULL,
   "   Further OS version details.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc fx__info_doc = {
  fx__info_entries, fx__info_submods,
  "Extraneous system details pertaining to an experiment.\n"
};

const fx_submodule_doc fx__std_submods[] = {
  {"fx", &fx__fx_doc,
   "   Options for the FASTexec experiment-running system.\n"},
  {"debug", &fx__debug_doc,
   "   Options for FASTlib's debugging feautres.\n"},
  {"info", &fx__info_doc,
   "   Extraneous system details pertaining to an experiment.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_entry_doc fx__std_entries[] = {
  {"help", FX_PARAM, FX_STR, NULL,
   "   Prints this information.  Permits --help=path/to/submod.\n"},
  {"total_time", FX_TIMER, FX_CUSTOM, NULL,
   "   The measured running time of the program.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc fx__std_doc = {
  fx__std_entries, fx__std_submods,
  "Standard FASTlib options:\n"
};



const fx_entry_doc fx__timer_entries[] = {
  {"cycles", FX_TIMER, FX_INT, NULL, "text"},
  {"real", FX_TIMER, FX_DOUBLE, NULL, "text"},
  {"user", FX_TIMER, FX_DOUBLE, NULL, "text"},
  {"sys", FX_TIMER, FX_DOUBLE, NULL, "text"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc fx__timer_doc = {
  fx__timer_entries, NULL, "text"
};



static void fx__write_path(fx_module *entry, FILE *stream)
{
  if (entry->parent) {
    fx__write_path(entry->parent, stream);
    putc('/', stream);
  }
  fprintf(stream, "%s", entry->key);
}

__attribute((noreturn))
static void fx__print_fatal_msg(const char *file, const char *func, int line,
                                const char *prefix, fx_module *entry,
                                const char *suffix, ...)
{
  va_list vl;

  fl_print_msg_header(fl_msg_marker[FL_MSG_FATAL],
                      fl_msg_color[FL_MSG_FATAL]);
  fl_print_msg_loc(file, func, line);

  va_start(vl, suffix);
  vfprintf(stderr, prefix, vl);
  fx__write_path(entry, stderr);
  vfprintf(stderr, suffix, vl);
  va_end(vl);

  fprintf(stderr, "\n");

  fl_abort();
}

static void fx__print_msg(const char *file, const char *func, int line,
                          fl_msg_t msg_type, const char *prefix,
                          fx_module *entry, const char *suffix, ...)
{
  va_list vl;

  fl_print_msg_header(fl_msg_marker[msg_type], fl_msg_color[msg_type]);
  if (msg_type < FL_MSG_NOTIFY_STAR || print_notify_locs) {
    fl_print_msg_loc(file, func, line);
  }

  va_start(vl, suffix);
  vfprintf(stderr, prefix, vl);
  fx__write_path(entry, stderr);
  vfprintf(stderr, suffix, vl);
  va_end(vl);

  fprintf(stderr, "\n");

  if (msg_type < FL_MSG_NOTIFY_STAR) {
    if (msg_type == FL_MSG_FATAL || abort_on_nonfatal) {
      fl_abort();
    } else if (pause_on_nonfatal) {
      fl_pause();
    }
  }
}

#define FX__FATAL(prefix, entry, msg_params...) \
    fx__print_fatal_msg(__FILE__, __FUNCTION__, __LINE__, \
        prefix, entry, msg_params)

#define FX__NONFATAL(prefix, entry, msg_params...) \
    fx__print_msg(__FILE__, __FUNCTION__, __LINE__, \
        FL_MSG_NONFATAL, prefix, entry, msg_params)

#define FX__SEMIFATAL(success, prefix, entry, msg_params...) \
    if (!success) { \
      FX__FATAL(prefix, entry, msg_params); \
    } else if (!FAILED(*success)) { \
      *success = SUCCESS_FAIL; \
      FX__NONFATAL(prefix, entry, msg_params); \
    } else NOP



int fx_module_is_type(fx_module *entry, fx_mod_t mod_type)
{
  if (mod_type >= FX_RESERVED) { /* Types without subtypes */
    return entry->mod_type == mod_type;
  } else if (mod_type >= FX_PARAM) { /* (Required) parameters */
    return entry->mod_type >= mod_type && entry->mod_type < FX_TIMER;
  } else { /* Modules and unknowns */
    return entry->mod_type >= mod_type;
  }
}

static success_t fx__check_mod_type(const char *header, fx_mod_t mod_type,
                                    fx_module *entry)
{
  if (unlikely(!fx_module_is_type(entry, mod_type))) {
    if (entry->mod_type < FX_PARAM) {
      if (fx_docs_nagging) {
	FX__NONFATAL("%s %s \"", entry, "\" is not documented.",
	    header, fx_mod_name[mod_type]);
      }
      return SUCCESS_WARN;
    } else {
      FX__NONFATAL("%s %s \"", entry, "\" is documented as a %s.",
          header, fx_mod_name[mod_type], fx_mod_name[entry->mod_type]);
      return SUCCESS_FAIL;
    }
  }
  return SUCCESS_PASS;
}

static success_t fx__check_val_type(const char *header, fx_val_t val_type,
                                    fx_module *entry)
{
  if (unlikely(entry->val_type != val_type)) {
    if (entry->val_type < 0) {
      FX__NONFATAL("%s %s \"", entry, "\" is of custom type.",
          header, fx_val_name[val_type]);
      return SUCCESS_FAIL;
    } else {
      FX__NONFATAL("%s %s \"", entry, "\" is of type %s.",
          header, fx_val_name[val_type], fx_val_name[entry->val_type]);
      return SUCCESS_WARN;
    }
  }
  return SUCCESS_PASS;
}



static double fx__scan_double_impl(fx_module *entry, const char *val,
                                   success_t *success_ptr, int list)
{
  double retval = 0.0;

  if (unlikely(sscanf(val, "%lf", &retval) != 1)) {
    FX__SEMIFATAL(success_ptr,
        "%c%s \"", entry, "\" is not a double%s: \"%s\".",
        toupper(*fx_mod_name[entry->mod_type]),
        fx_mod_name[entry->mod_type] + 1,
        list ? " list" : "", entry->val);
  }

  return retval;
}

static long long fx__scan_int_impl(fx_module *entry, const char *val,
                                   success_t *success_ptr, int list)
{
  long long retval = 0;

  if (sscanf(val, "%lld", &retval) != 1) {
    FX__SEMIFATAL(success_ptr,
        "%c%s \"", entry, "\" is not an int%s: \"%s\".",
        toupper(*fx_mod_name[entry->mod_type]),
        fx_mod_name[entry->mod_type] + 1,
        list ? " list" : "", entry->val);
  }

  return retval;
}

static int fx__scan_bool_impl(fx_module *entry, const char *val,
                              success_t *success_ptr, int list)
{
  int retval = 0;

  if (strchr("1tTyY", val[0])) {
    retval = 1;
  } else if (!strchr("0fFnN", val[0])) {
    FX__SEMIFATAL(success_ptr,
        "%c%s \"", entry, "\" is not a bool%s: \"%s\".",
        toupper(*fx_mod_name[entry->mod_type]),
        fx_mod_name[entry->mod_type] + 1,
        list ? " list" : "", entry->val);
  }

  return retval;
}



static double fx__scan_double(fx_module *entry)
{
  return fx__scan_double_impl(entry, entry->val, NULL, 0);
}

static long long fx__scan_int(fx_module *entry)
{
  return fx__scan_int_impl(entry, entry->val, NULL, 0);
}

static int fx__scan_bool(fx_module *entry)
{
  return fx__scan_bool_impl(entry, entry->val, NULL, 0);
}



static size_t fx__list_size(fx_module *entry, size_t req_size,
                            success_t *success_ptr)
{
  char *val = entry->val;
  size_t count = 1;

  while ((val = strchr(val, ','))) {
    ++count;
    ++val;
  }

  if (req_size && count != req_size) {
    FX__SEMIFATAL(success_ptr,
        "%c%s \"", entry, "\" is not of length %d: \"%s\".",
        toupper(*fx_mod_name[entry->mod_type]),
        fx_mod_name[entry->mod_type] + 1,
        req_size, entry->val);
  }

  return count;
}

static const char **fx__scan_str_list(fx_module *mod, size_t *size_ptr,
                                      success_t *success_ptr)
{
  char **retval;
  char **elem;
  char *str;

  size_t len = strlen(mod->val);
  size_t count = fx__list_size(mod, *size_ptr, success_ptr);

  /* Allocate space for string, elements, and array */
  mod->val = realloc(mod->val,
      stride_align(2 * len + 2, char *) + count * sizeof(char *));

  retval = (char **)(mod->val + stride_align(2 * len + 2, char *));
  *size_ptr = count;

  /* Copy string to form elements; simulate "leading comma" */
  str = mod->val + len;
  strcpy(str + 1, mod->val);

  /* Demark elements and add to array */
  elem = retval;
  do {
    *str++ = '\0';
    *elem++ = str;
  } while ((str = strchr(str, ',')));

  /* Parse escaped characters */
  while (elem-- > retval) {
    unhex_in_place(*elem);
  }

  return (const char **)retval;
}

static double *fx__scan_double_list(fx_module *mod, size_t *size_ptr,
                                    success_t *success_ptr)
{
  double *retval;
  double *elem;
  const char *str;

  size_t len = strlen(mod->val);
  size_t count = fx__list_size(mod, *size_ptr, success_ptr);

  /* Allocate space for string and array */
  mod->val = realloc(mod->val,
      stride_align(len + 1, double) + count * sizeof(double));

  retval = (double *)(mod->val + stride_align(len + 1, double));
  *size_ptr = count;

  /* Backstep once to simulate "leading comma" */
  str = mod->val - 1;

  /* Fill the array */
  elem = retval;
  do {
    *elem++ = fx__scan_double_impl(mod, ++str, success_ptr, 1);
  } while ((str = strchr(str, ',')));

  return retval;
}

static long long *fx__scan_int_list(fx_module *mod, size_t *size_ptr,
                                    success_t *success_ptr)
{
  long long *retval;
  long long *elem;
  const char *str;

  size_t len = strlen(mod->val);
  size_t count = fx__list_size(mod, *size_ptr, success_ptr);

  /* Allocate space for string and array */
  mod->val = realloc(mod->val,
      stride_align(len + 1, long long) + count * sizeof(long long));

  retval = (long long *)(mod->val + stride_align(len + 1, long long));
  *size_ptr = count;

  /* Backstep once to simulate "leading comma" */
  str = mod->val - 1;

  /* Fill the array */
  elem = retval;
  do {
    *elem++ = fx__scan_int_impl(mod, ++str, success_ptr, 1);
  } while ((str = strchr(str, ',')));

  return retval;
}

static int *fx__scan_bool_list(fx_module *mod, size_t *size_ptr,
                               success_t *success_ptr)
{
  int *retval;
  int *elem;
  const char *str;

  size_t len = strlen(mod->val);
  size_t count = fx__list_size(mod, *size_ptr, success_ptr);

  /* Allocate space for string and array */
  mod->val = realloc(mod->val,
      stride_align(len + 1, int) + count * sizeof(int));

  retval = (int *)(mod->val + stride_align(len + 1, int));
  *size_ptr = count;

  /* Backstep once to simulate "leading comma" */
  str = mod->val - 1;

  /* Fill the array */
  elem = retval;
  do {
    *elem++ = fx__scan_bool_impl(mod, ++str, success_ptr, 1);
  } while ((str = strchr(str, ',')));

  return retval;
}



static char *fx__alloc_double(double val)
{
  char *retval = malloc(FX__BUF_SIZE * sizeof(char));

  sprintf(retval, "%.16g", val);

  return retval;
}

static char *fx__alloc_int(long long val)
{
  char *retval = malloc(FX__BUF_SIZE * sizeof(char));

  sprintf(retval, "%lld", val);

  return retval;
}

static char *fx__alloc_bool(int val)
{
  return strdup(val ? "t" : "f");
}



static char *fx__alloc_str_list(size_t size, va_list vl)
{
  if (size == 0) {
    return strdup("");
  } else {
    va_list vl_temp;
    char *retval;
    char *str;
    size_t len;
    size_t i;

    va_copy(vl_temp, vl);
    len = 0;
    for (i = 0; i < size; ++i) {
      len += strlen(va_arg(vl_temp, const char *));
    }
    va_end(vl_temp);

    retval = malloc((3 * len + size) * sizeof(char));

    str = hex_to_string(retval, va_arg(vl, const char *), "_.-+");
    while (--size) {
      *str++ = ',';
      str = hex_to_string(str, va_arg(vl, const char *), "_.-+");
    }

    return retval;
  }
}

static char *fx__alloc_double_list(size_t size, va_list vl)
{
  if (size == 0) {
    return strdup("");
  } else {
    char *retval = malloc(size * FX__BUF_SIZE * sizeof(char));
    char *str = retval;

    str += sprintf(str, "%.16g", va_arg(vl, double));
    while (--size) {
      str += sprintf(str, ",%.16g", va_arg(vl, double));
    }

    return retval;
  }
}

static char *fx__alloc_int_list(size_t size, va_list vl)
{
  if (size == 0) {
    return strdup("");
  } else {
    char *retval = malloc(size * FX__BUF_SIZE * sizeof(char));
    char *str = retval;

    str += sprintf(str, "%lld", va_arg(vl, long long));
    while (--size) {
      str += sprintf(str, ",%lld", va_arg(vl, long long));
    }

    return retval;
  }
}

static char *fx__alloc_bool_list(size_t size, va_list vl)
{
  if (size == 0) {
    return strdup("");
  } else {
    char *retval = malloc(size * 2 * sizeof(char));
    char *str = retval;

    *str++ = va_arg(vl, int) ? 't' : 'f';
    while (--size) {
      *str++ = ',';
      *str++ = va_arg(vl, int) ? 't' : 'f';
    }
    *str = '\0';

    return retval;
  }
}



static char *fx__alloc_str_array(size_t size, const char *const *list)
{
  if (size == 0) {
    return strdup("");
  } else {
    char *retval;
    char *str;
    size_t len;
    size_t i;

    len = 0;
    for (i = 0; i < size; ++i) {
      len += strlen(list[i]);
    }

    retval = malloc((3 * len + size) * sizeof(char));

    str = hex_to_string(retval, *list++, "_.-+");
    while (--size) {
      *str++ = ',';
      str = hex_to_string(str, *list++, "_.-+");
    }

    return retval;
  }
}

static char *fx__alloc_double_array(size_t size, const double *list)
{
  if (size == 0) {
    return strdup("");
  } else {
    char *retval = malloc(size * FX__BUF_SIZE * sizeof(char));
    char *str = retval;

    str += sprintf(str, "%.16g", *list++);
    while (--size) {
      str += sprintf(str, ",%.16g", *list++);
    }

    return retval;
  }
}

static char *fx__alloc_int_array(size_t size, const long long *list)
{
  if (size == 0) {
    return strdup("");
  } else {
    char *retval = malloc(size * FX__BUF_SIZE * sizeof(char));
    char *str = retval;

    str += sprintf(str, "%lld", *list++);
    while (--size) {
      str += sprintf(str, ",%lld", *list++);
    }

    return retval;
  }
}

static char *fx__alloc_bool_array(size_t size, const int *list)
{
  if (size == 0) {
    return strdup("");
  } else {
    char *retval = malloc(size * 2 * sizeof(char));
    char *str = retval;

    *str++ = *list++ ? 't' : 'f';
    while (--size) {
      *str++ = ',';
      *str++ = *list++ ? 't' : 'f';
    }
    *str = '\0';

    return retval;
  }
}



static char *fx__alloc_format(const char *format, va_list vl)
{
  va_list vl_temp;
  char *retval;
  size_t len;

  va_copy(vl_temp, vl);
  len = vsnprintf(NULL, 0, format, vl_temp);
  va_end(vl_temp);

  retval = malloc((len + 1) * sizeof(char));
  vsprintf(retval, format, vl);

  return retval;
}

static char *fx__alloc_format_list(size_t size, va_list vl)
{
  if (size == 0) {
    return strdup("");
  } else {
    va_list vl_temp;
    char *retval;
    char *str;
    char *temp;
    size_t len;
    size_t i;

    va_copy(vl_temp, vl);
    len = 0;
    for (i = 0; i < size; ++i) {
      len += vsnprintf(NULL, 0, va_arg(vl_temp, const char *), vl_temp);
    }
    va_end(vl_temp);

    retval = malloc((4 * len + size + 1) * sizeof(char));
    temp = retval + 3 * len + size;

    vsprintf(temp, va_arg(vl, const char *), vl);
    str = hex_to_string(retval, temp, "_.-+");
    while (--size) {
      *str++ = ',';
      vsprintf(temp, va_arg(vl, const char *), vl);
      str = hex_to_string(retval, temp, "_.-+");
    }

    return retval;
  }
}



static fx_module *fx__lookup(fx_module *mod, const char *key)
{
  return datanode_lookup(mod ? mod : fx_root, key, 1);
}

static void fx__check_lookup(fx_mod_t mod_type, fx_val_t val_type,
                             fx_module *entry)
{
  if (PASSED(fx__check_mod_type("Requested", mod_type, entry))) {
    fx__check_val_type("Requested", val_type, entry);
  }
}

static fx_module *fx__get_entry(fx_module *mod, const char *key,
                                fx_mod_t mod_type, fx_val_t val_type)
{
  fx_module *retval = fx__lookup(mod, key);
  DEBUG_ONLY(fx__check_lookup(mod_type, val_type, retval));
  return retval;
}



static fx_module *fx__param(fx_module *mod, const char *key,
                            fx_val_t val_type)
{
  fx_module *retval = fx__get_entry(mod, key, FX_PARAM, val_type);

  if (retval->mod_type == FX_PARAM) {
    if (retval->val) {
      retval->mod_type = FX_PROVIDED;
    } else {
      retval->mod_type = FX_DEFAULT;
    }
  }

  return retval;
}

static fx_module *fx__param_req(fx_module *mod, const char *key,
                                fx_val_t val_type)
{
  fx_module *retval = fx__get_entry(mod, key, FX_REQUIRED, val_type);

  if (!retval->val) {
    FX__FATAL("Required parameter \"", retval, "\" is unspecified.");
  }

  return retval;
}



const char *fx_param_str_req(fx_module *mod, const char *key)
{
  return fx__param_req(mod, key, FX_STR)->val;
}

double fx_param_double_req(fx_module *mod, const char *key)
{
  return fx__scan_double(fx__param_req(mod, key, FX_DOUBLE));
}

long long fx_param_int_req(fx_module *mod, const char *key)
{
  return fx__scan_int(fx__param_req(mod, key, FX_INT));
}

int fx_param_bool_req(fx_module *mod, const char *key)
{
  return fx__scan_bool(fx__param_req(mod, key, FX_BOOL));
}



const char **fx_param_str_list_req(fx_module *mod, const char *key,
                                   size_t *size_ptr)
{
  fx_module *param = fx__param_req(mod, key, FX_STR_LIST);
  return fx__scan_str_list(param, size_ptr, NULL);
}

double *fx_param_double_list_req(fx_module *mod, const char *key,
                                 size_t *size_ptr)
{
  fx_module *param = fx__param_req(mod, key, FX_DOUBLE_LIST);
  return fx__scan_double_list(param, size_ptr, NULL);
}

long long *fx_param_int_list_req(fx_module *mod, const char *key,
                                 size_t *size_ptr)
{
  fx_module *param = fx__param_req(mod, key, FX_INT_LIST);
  return fx__scan_int_list(param, size_ptr, NULL);
}

int *fx_param_bool_list_req(fx_module *mod, const char *key,
                            size_t *size_ptr)
{
  fx_module *param = fx__param_req(mod, key, FX_BOOL_LIST);
  return fx__scan_bool_list(param, size_ptr, NULL);
}



const char *fx_param_str(fx_module *mod, const char *key, const char *def)
{
  fx_module *param = fx__param(mod, key, FX_STR);

  if (!param->val) {
    param->val = strdup(def);
  }

  return param->val;
}

double fx_param_double(fx_module *mod, const char *key, double def)
{
  fx_module *param = fx__param(mod, key, FX_DOUBLE);

  if (!param->val) {
    param->val = fx__alloc_double(def);
  } else {
    def = fx__scan_double(param);
  }

  return def;
}

long long fx_param_int(fx_module *mod, const char *key, long long def)
{
  fx_module *param = fx__param(mod, key, FX_INT);

  if (!param->val) {
    param->val = fx__alloc_int(def);
  } else {
    def = fx__scan_int(param);
  }

  return def;
}

int fx_param_bool(fx_module *mod, const char *key, int def)
{
  fx_module *param = fx__param(mod, key, FX_BOOL);

  if (!param->val) {
    param->val = fx__alloc_bool(def);
  } else {
    def = fx__scan_bool(param);
  }

  return def;
}



const char **fx_param_str_list(fx_module *mod, const char *key,
                               size_t *size_ptr, size_t def_size, ...)
{
  fx_module *param = fx__param(mod, key, FX_STR_LIST);

  if (!param->val) {
    va_list vl;

    va_start(vl, def_size);
    param->val = fx__alloc_str_list(def_size, vl);
    va_end(vl);
  }

  return fx__scan_str_list(param, size_ptr, NULL);
}

double *fx_param_double_list(fx_module *mod, const char *key,
                             size_t *size_ptr, size_t def_size, ...)
{
  fx_module *param = fx__param(mod, key, FX_DOUBLE_LIST);

  if (!param->val) {
    va_list vl;

    va_start(vl, def_size);
    param->val = fx__alloc_double_list(def_size, vl);
    va_end(vl);
  }

  return fx__scan_double_list(param, size_ptr, NULL);
}

long long *fx_param_int_list(fx_module *mod, const char *key,
                             size_t *size_ptr, size_t def_size, ...)
{
  fx_module *param = fx__param(mod, key, FX_INT_LIST);

  if (!param->val) {
    va_list vl;

    va_start(vl, def_size);
    param->val = fx__alloc_int_list(def_size, vl);
    va_end(vl);
  }

  return fx__scan_int_list(param, size_ptr, NULL);
}

int *fx_param_bool_list(fx_module *mod, const char *key,
                        size_t *size_ptr, size_t def_size, ...)
{
  fx_module *param = fx__param(mod, key, FX_BOOL_LIST);

  if (!param->val) {
    va_list vl;

    va_start(vl, def_size);
    param->val = fx__alloc_bool_list(def_size, vl);
    va_end(vl);
  }

  return fx__scan_bool_list(param, size_ptr, NULL);
}



const char **fx_param_str_array(fx_module *mod, const char *key,
                                size_t *size_ptr, size_t def_size,
                                const char *const *def_array)
{
  fx_module *param = fx__param(mod, key, FX_STR_LIST);

  if (!param->val) {
    param->val = fx__alloc_str_array(def_size, def_array);
  }

  return fx__scan_str_list(param, size_ptr, NULL);
}

double *fx_param_double_array(fx_module *mod, const char *key,
                              size_t *size_ptr, size_t def_size,
                              const double *def_array)
{
  fx_module *param = fx__param(mod, key, FX_DOUBLE_LIST);

  if (!param->val) {
    param->val = fx__alloc_double_array(def_size, def_array);
  }

  return fx__scan_double_list(param, size_ptr, NULL);
}

long long *fx_param_int_array(fx_module *mod, const char *key,
                              size_t *size_ptr, size_t def_size,
                              const long long *def_array)
{
  fx_module *param = fx__param(mod, key, FX_INT_LIST);

  if (!param->val) {
    param->val = fx__alloc_int_array(def_size, def_array);
  }

  return fx__scan_int_list(param, size_ptr, NULL);
}

int *fx_param_bool_array(fx_module *mod, const char *key,
                         size_t *size_ptr, size_t def_size,
                         const int *def_array)
{
  fx_module *param = fx__param(mod, key, FX_BOOL_LIST);

  if (!param->val) {
    param->val = fx__alloc_bool_array(def_size, def_array);
  }

  return fx__scan_bool_list(param, size_ptr, NULL);
}



void fx_set_param_str(fx_module *mod, const char *key, const char *val)
{
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_STR);

  free(param->val);
  param->val = strdup(val);
}

void fx_set_param_double(fx_module *mod, const char *key, double val)
{
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_DOUBLE);

  free(param->val);
  param->val = fx__alloc_double(val);
}

void fx_set_param_int(fx_module *mod, const char *key, long long val)
{
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_INT);

  free(param->val);
  param->val = fx__alloc_int(val);
}

void fx_set_param_bool(fx_module *mod, const char *key, int val)
{
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_BOOL);

  free(param->val);
  param->val = fx__alloc_bool(val);
}



void fx_set_param_str_list(fx_module *mod, const char *key,
                           size_t size, ...)
{
  va_list vl;
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_STR_LIST);

  va_start(vl, size);
  free(param->val);
  param->val = fx__alloc_str_list(size, vl);
  va_end(vl);
}

void fx_set_param_double_list(fx_module *mod, const char *key,
                              size_t size, ...)
{
  va_list vl;
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_DOUBLE_LIST);

  va_start(vl, size);
  free(param->val);
  param->val = fx__alloc_double_list(size, vl);
  va_end(vl);
}

void fx_set_param_int_list(fx_module *mod, const char *key,
                           size_t size, ...)
{
  va_list vl;
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_INT_LIST);

  va_start(vl, size);
  free(param->val);
  param->val = fx__alloc_int_list(size, vl);
  va_end(vl);
}

void fx_set_param_bool_list(fx_module *mod, const char *key,
                            size_t size, ...)
{
  va_list vl;
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_BOOL_LIST);

  va_start(vl, size);
  free(param->val);
  param->val = fx__alloc_bool_list(size, vl);
  va_end(vl);
}



void fx_set_param_str_array(fx_module *mod, const char *key,
                            size_t size, const char *const *array)
{
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_STR_LIST);

  free(param->val);
  param->val = fx__alloc_str_array(size, array);
}

void fx_set_param_double_array(fx_module *mod, const char *key,
                               size_t size, const double *array)
{
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_DOUBLE_LIST);

  free(param->val);
  param->val = fx__alloc_double_array(size, array);
}

void fx_set_param_int_array(fx_module *mod, const char *key,
                            size_t size, const long long *array)
{
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_INT_LIST);

  free(param->val);
  param->val = fx__alloc_int_array(size, array);
}

void fx_set_param_bool_array(fx_module *mod, const char *key,
                             size_t size, const int *array)
{
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_BOOL_LIST);

  free(param->val);
  param->val = fx__alloc_bool_array(size, array);
}



void fx_default_param(fx_module *mod, const char *key,
                      const char *def_format, ...)
{
  fx_module *param = fx__param(mod, key, FX_STR);

  if (!param->val) {
    va_list vl;

    va_start(vl, def_format);
    param->val = fx__alloc_format(def_format, vl);
    va_end(vl);
  }
}

void fx_default_param_list(fx_module *mod, const char *key,
                           size_t *size_ptr, size_t def_size, ...)
{
  fx_module *param = fx__param(mod, key, FX_STR_LIST);

  if (!param->val) {
    va_list vl;

    va_start(vl, def_size);
    param->val = fx__alloc_format_list(def_size, vl);
    va_end(vl);
  }
}

void fx_format_param(fx_module *mod, const char *key,
                     const char *format, ...)
{
  va_list vl;
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_STR);

  va_start(vl, format);
  free(param->val);
  param->val = fx__alloc_format(format, vl);
  va_end(vl);
}

void fx_format_param_list(fx_module *mod, const char *key,
                          size_t size, ...)
{
  va_list vl;
  fx_module *param = fx__get_entry(mod, key, FX_RESERVED, FX_STR_LIST);

  va_start(vl, size);
  free(param->val);
  param->val = fx__alloc_format_list(size, vl);
  va_end(vl);
}



void fx_clear_param(fx_module *mod, const char *key)
{
  fx_module *param = fx__lookup(mod, key);

  DEBUG_ONLY(fx__check_mod_type("Requested", FX_RESERVED, param));

  free(param->val);
  param->val = NULL;
}

int fx_param_exists(fx_module *mod, const char *key)
{
  fx_module *param = fx__lookup(mod, key);

  DEBUG_ONLY(fx__check_mod_type("Requested", FX_PARAM, param));

  if (param->mod_type == FX_PARAM) {
    if (param->val) {
      param->mod_type = FX_PROVIDED;
    }
  }

  return param->val != 0;
}



static fx_module *fx__get_result(fx_module *mod, const char *key,
                                 fx_val_t val_type)
{
  fx_module *retval = fx__get_entry(mod, key, FX_RESULT, val_type);

  if (!retval->val) {
    FX__FATAL("Accessed result \"", retval, "\" is unspecified.");
  }

  return retval;
}



const char *fx_get_result_str(fx_module *mod, const char *key)
{
  return fx__get_result(mod, key, FX_STR)->val;
}

double fx_get_result_double(fx_module *mod, const char *key)
{
  return fx__scan_double(fx__get_result(mod, key, FX_DOUBLE));
}

long long fx_get_result_int(fx_module *mod, const char *key)
{
  return fx__scan_int(fx__get_result(mod, key, FX_INT));
}

int fx_get_result_bool(fx_module *mod, const char *key)
{
  return fx__scan_bool(fx__get_result(mod, key, FX_BOOL));
}



const char **fx_get_result_str_list(fx_module *mod, const char *key,
                                    size_t *size_ptr)
{
  fx_module *result = fx__get_result(mod, key, FX_STR_LIST);
  return fx__scan_str_list(result, size_ptr, NULL);
}

double *fx_get_result_double_list(fx_module *mod, const char *key,
                                  size_t *size_ptr)
{
  fx_module *result = fx__get_result(mod, key, FX_DOUBLE_LIST);
  return fx__scan_double_list(result, size_ptr, NULL);
}

long long *fx_get_result_int_list(fx_module *mod, const char *key,
                                  size_t *size_ptr)
{
  fx_module *result = fx__get_result(mod, key, FX_INT_LIST);
  return fx__scan_int_list(result, size_ptr, NULL);
}

int *fx_get_result_bool_list(fx_module *mod, const char *key,
                             size_t *size_ptr)
{
  fx_module *result = fx__get_result(mod, key, FX_BOOL_LIST);
  return fx__scan_bool_list(result, size_ptr, NULL);
}



void fx_result_str(fx_module *mod, const char *key, const char *val)
{
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_STR);

  free(result->val);
  result->val = strdup(val);
}

void fx_result_double(fx_module *mod, const char *key, double val)
{
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_DOUBLE);

  free(result->val);
  result->val = fx__alloc_double(val);
}

void fx_result_int(fx_module *mod, const char *key, long long val)
{
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_INT);

  free(result->val);
  result->val = fx__alloc_int(val);
}

void fx_result_bool(fx_module *mod, const char *key, int val)
{
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_BOOL);

  free(result->val);
  result->val = fx__alloc_bool(val);
}



void fx_result_str_list(fx_module *mod, const char *key,
                        size_t size, ...)
{
  va_list vl;
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_STR_LIST);

  va_start(vl, size);
  free(result->val);
  result->val = fx__alloc_str_list(size, vl);
  va_end(vl);
}

void fx_result_double_list(fx_module *mod, const char *key,
                           size_t size, ...)
{
  va_list vl;
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_DOUBLE_LIST);

  va_start(vl, size);
  free(result->val);
  result->val = fx__alloc_double_list(size, vl);
  va_end(vl);
}

void fx_result_int_list(fx_module *mod, const char *key,
                        size_t size, ...)
{
  va_list vl;
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_INT_LIST);

  va_start(vl, size);
  free(result->val);
  result->val = fx__alloc_int_list(size, vl);
  va_end(vl);
}

void fx_result_bool_list(fx_module *mod, const char *key,
                         size_t size, ...)
{
  va_list vl;
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_BOOL_LIST);

  va_start(vl, size);
  free(result->val);
  result->val = fx__alloc_bool_list(size, vl);
  va_end(vl);
}



void fx_result_str_array(fx_module *mod, const char *key,
                         size_t size, const char *const *array)
{
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_STR_LIST);

  free(result->val);
  result->val = fx__alloc_str_array(size, array);
}

void fx_result_double_array(fx_module *mod, const char *key,
                            size_t size, const double *array)
{
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_DOUBLE_LIST);

  free(result->val);
  result->val = fx__alloc_double_array(size, array);
}

void fx_result_int_array(fx_module *mod, const char *key,
                         size_t size, const long long *array)
{
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_INT_LIST);

  free(result->val);
  result->val = fx__alloc_int_array(size, array);
}

void fx_result_bool_array(fx_module *mod, const char *key,
                          size_t size, const int *array)
{
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_BOOL_LIST);

  free(result->val);
  result->val = fx__alloc_bool_array(size, array);
}



void fx_format_result(fx_module *mod, const char *key,
                      const char *format, ...)
{
  va_list vl;
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_STR);

  va_start(vl, format);
  free(result->val);
  result->val = fx__alloc_format(format, vl);
  va_end(vl);
}

void fx_format_result_list(fx_module *mod, const char *key,
                           size_t size, ...)
{
  va_list vl;
  fx_module *result = fx__get_entry(mod, key, FX_RESULT, FX_STR_LIST);

  va_start(vl, size);
  free(result->val);
  result->val = fx__alloc_format_list(size, vl);
  va_end(vl);
}



void fx_clear_result(fx_module *mod, const char *key)
{
  fx_module *result = fx__lookup(mod, key);

  DEBUG_ONLY(fx__check_mod_type("Requested", FX_RESULT, result));

  free(result->val);
  result->val = NULL;
}

int fx_result_exists(fx_module *mod, const char *key)
{
  fx_module *result = fx__lookup(mod, key);

  DEBUG_ONLY(fx__check_mod_type("Requested", FX_RESULT, result));

  return result->val != NULL;
}



fx_timer *fx_get_timer(fx_module *mod, const char *key) {
  fx_module *timer = fx__lookup(mod, key);

  DEBUG_ONLY(fx__check_mod_type("Requested", FX_TIMER, timer));

  if (!timer->val) {
    timer->mod_type = FX_TIMER;
    timer->val_type = FX_CUSTOM;
    timer->val = malloc(sizeof(fx_timer));
    stopwatch_init( (fx_timer *)(timer->val) );
  }

  return (fx_timer *)(timer->val);
}

void fx_timer_start(fx_module *mod, const char *key)
{
  stopwatch_start(fx_get_timer(mod, key));
}

void fx_timer_stop(fx_module *mod, const char *key)
{
  struct timestamp now;
  timestamp_now(&now);

  stopwatch_stop(fx_get_timer(mod, key), &now);
}

void fx_reset_timer(fx_module *mod, const char *key)
{
  stopwatch_init(fx_get_timer(mod, key));
}



fx_module *fx_submodule(fx_module *mod, const char *key)
{
  fx_module *submod = fx__lookup(mod, key);

  DEBUG_ONLY(fx__check_mod_type("Requested", FX_MODULE, submod));

  return submod;
}

fx_module *fx_copy_module(fx_module *mod, const char *src_key,
                          const char *dest_format, ...)
{
  va_list vl;
  char *dest_key;
  fx_module *dest_mod;

  va_start(vl, dest_format);
  dest_key = fx__alloc_format(dest_format, vl);
  va_end(vl);

  dest_mod = fx__lookup(mod, dest_key);
  if (src_key) {
    datanode_copy(dest_mod, fx__lookup(mod, src_key), 1);

    if (fx_module_is_type(dest_mod, FX_MODULE)) {
      fx_module *parent = dest_mod;

      while ((parent = parent->parent) && parent->mod_type == FX_UNKNOWN) {
        parent->mod_type = FX_MODULE;
      }
    }
  }

  DEBUG_ONLY(fx__check_mod_type("Requested", FX_MODULE, dest_mod));

  free(dest_key);
  return dest_mod;
}



static const char *fx__match_prefix(const char *key, const char *prefix)
{
  for (; *key == '/'; ++key);  
  for (; *prefix == '/'; ++prefix);  

  while (*prefix) {
    const char *slash = strchr(prefix, '/');
    size_t len = slash ? slash - prefix : strlen(prefix);

    if (strncmp(key, prefix, len) == 0
        && (key[len] == '\0' || key[len] == '/')) {
      for (key += len; *key == '/'; ++key);
      for (prefix += len; *prefix == '/'; ++prefix);
    } else {
      return NULL;
    }
  }

  return key;
}

success_t fx_help(const fx_module_doc *doc, const char *key, const char *origkey)
{
  success_t retval = SUCCESS_WARN;
//  printf("key is '%s'\n", origkey);

  for (; *key == '/'; ++key);

  if (*key == '\0') {
    retval = SUCCESS_PASS;
    printf("%s\n", doc->text);
  }

  if (doc->entries) {
    const fx_entry_doc *entry_doc;

    /* we'll loop through each type of entry */
    /* first, parameters to the program */
    if (*key == '\0') {
      printf("Parameters:\n");
    }

    for (entry_doc = doc->entries; entry_doc->key; ++entry_doc) {
      if (entry_doc->text) {
        if (fx__match_prefix(entry_doc->key, key)) {
          retval = SUCCESS_PASS;
          if(entry_doc->mod_type == FX_PARAM || entry_doc->mod_type == FX_REQUIRED) {
            if (entry_doc->val_type < 0) {
              if(*origkey != '\0') {
                printf("--%s/%s, %s:\n", origkey, entry_doc->key,
                       fx_mod_name[entry_doc->mod_type]);
              } else {
                printf("--%s, %s:\n", entry_doc->key,
                       fx_mod_name[entry_doc->mod_type]);
              }
            } else {
              if(*origkey != '\0') {
                printf("--%s/%s, %s:\n", origkey, entry_doc->key,
                     fx_val_name[entry_doc->val_type],
                     fx_mod_name[entry_doc->mod_type]);
              } else {
                printf("--%s=(%s), %s:\n", entry_doc->key,
                     fx_val_name[entry_doc->val_type],
                     fx_mod_name[entry_doc->mod_type]);
              }
            }
          }
          printf("%s\n", entry_doc->text);
        }
      }
    }

    
  }

  if (doc->submodules) {
    const fx_submodule_doc *submod_doc;

    if (*key == '\0') {
      printf("Submodules:\n");
    }

    for (submod_doc = doc->submodules; submod_doc->key; ++submod_doc) {
      const char *match = fx__match_prefix(key, submod_doc->key);

      if (match) {
        retval |= fx_help(submod_doc->doc, match, origkey);
      } else if (submod_doc->text) {
        if (fx__match_prefix(submod_doc->key, key)) {
          retval = SUCCESS_PASS;
          printf("\"%s\":\n%s\n", submod_doc->key, submod_doc->text);
        }
      }
    }
  }

  return retval;
}

__attribute__((noreturn))
static void fx__std_help(const char *prog, const char *help,
                         const fx_module_doc *doc)
{
  success_t success;

  if (doc) {
    success = fx_help(doc, help, help);
  } else {
    NONFATAL("Program \"%s\" is not documented.\n", prog);
    success = SUCCESS_WARN;
  }
  success |= fx_help(&fx__std_doc, help, help);

  if (!PASSED(success)) {
    NONFATAL("No documentation available for \"%s\".\n", help);
  }

  exit(1);
}



static void fx__fill_docs(fx_module *mod, const fx_module_doc *doc)
{
  if (mod->mod_type == FX_UNKNOWN) {
    mod->mod_type = FX_MODULE;
  }

  if (doc->submodules) {
    const fx_submodule_doc *submod_doc;

    for (submod_doc = doc->submodules; submod_doc->key; ++submod_doc) {
      fx_module *submod = datanode_lookup(mod, submod_doc->key, 1);

      fx__fill_docs(submod, submod_doc->doc);

      while ((submod = submod->parent) && submod->mod_type == FX_UNKNOWN) {
        submod->mod_type = FX_MODULE;
      }
    }
  }

  if (doc->entries) {
    const fx_entry_doc *entry_doc;

    for (entry_doc = doc->entries; entry_doc->key; ++entry_doc) {
      fx_module *entry = datanode_lookup(mod, entry_doc->key, 1);

      DEBUG_ASSERT(entry_doc->mod_type != FX_UNKNOWN);
      DEBUG_ASSERT(entry_doc->mod_type != FX_PROVIDED);
      DEBUG_ASSERT(entry_doc->mod_type != FX_DEFAULT);

      DEBUG_ASSERT(entry_doc->mod_type != FX_PARAM
                   || entry_doc->val_type != FX_CUSTOM);

      entry->mod_type = entry_doc->mod_type;
      entry->val_type = entry_doc->val_type;
      entry->meta = entry_doc->meta;

      while ((entry = entry->parent) && entry->mod_type == FX_UNKNOWN) {
        entry->mod_type = FX_MODULE;
      }
    }
  }
}

static void fx__parse_cmd_line(fx_module *root, int argc, char *argv[])
{
  int i;

  for (i = 0; i < argc; ++i) {
    if (argv[i][0] != '-' && argv[i][1] != '-') {
      NONFATAL("Ignoring argument missing \"--\": \"%s\".", argv[i]);
    } else {
      char *arg = strdup(argv[i] + 2);
      char *val = strchr(arg, '=');
      fx_module *entry;

      if (val) {
        *val++ = '\0';
        unhex_in_place(val);
      } else {
        val = "";
      }
      unhex_in_place(arg);

      entry = datanode_lookup_expert(root, arg, 1);
      if (entry->val) {
        FX__NONFATAL("Repeated \"--", entry, "=%s\" overwriting \"%s\".",
            val, entry->val);
        free(entry->val);
      }
      entry->val = strdup(val);

      free(arg);
    }
  }
}

static void fx__load_param_files(fx_module *root)
{
  size_t size = 0;
  const char **load = fx_param_str_list_req(root, "fx/load", &size);

  while (size--) {
    FILE *stream = fopen(*load++, "r");
    if (likely(stream)) {
      datanode_read(root, stream, NULL, 0);
    } else {
      FATAL("Cannot open file for \"--fx/load=%s\".", *(load - 1));
    }
    fclose(stream);
  }
}

static success_t fx__check_param(fx_module *param)
{
  success_t success = SUCCESS_PASS;
  size_t size = 0;

  /* Note: Params can't be FX_CUSTOM */
  switch (param->val_type) {
  case FX_STR:
    break;
  case FX_DOUBLE:
    fx__scan_double_impl(param, param->val, &success, 0);
    break;
  case FX_INT:
    fx__scan_int_impl(param, param->val, &success, 0);
    break;
  case FX_BOOL:
    fx__scan_bool_impl(param, param->val, &success, 0);
    break;
  case FX_STR_LIST:
    break;
  case FX_DOUBLE_LIST:
    fx__scan_double_list(param, &size, &success);
    break;
  case FX_INT_LIST:
    fx__scan_int_list(param, &size, &success);
    break;
  case FX_BOOL_LIST:
    fx__scan_bool_list(param, &size, &success);
    break;
  }

  return success;
}

static success_t fx__check_inputs(fx_module *mod)
{
  fx_module *child;
  success_t success = SUCCESS_PASS;

  if (mod->mod_type == FX_REQUIRED && !mod->val) {
    FX__NONFATAL("Required parameter \"", mod, "\" is unspecified.");
    success = SUCCESS_FAIL;
  } else if (mod->mod_type == FX_RESERVED && mod->val) {
    FX__NONFATAL("Reserved parameter \"", mod, "\" must not be specified.");
    success = SUCCESS_FAIL;
  } else if (mod->val) {
    success = fx__check_mod_type("Input", FX_PARAM, mod);
    if (PASSED(success)) {
      success &= fx__check_param(mod);
    }
  }

  for (child = mod->first_child; child; child = child->next) {
    success &= fx__check_inputs(child);
  }

  return success;
}

static void fx__report_sys(fx_module *sys)
{
  struct utsname info;

  uname(&info);

  fx_result_str(sys, "node/name", info.nodename);
  fx_result_str(sys, "arch/name", info.machine);
  fx_result_str(sys, "kernel/name", info.sysname);
  fx_result_str(sys, "kernel/release", info.release);
  fx_result_str(sys, "kernel/build", info.version);
}

static void fx__read_debug_params(fx_module *debug)
{
  /* TODO: Default to current settings for customizability? */
  verbosity_level = fx_param_double(debug, "verbosity_level", 1.0);
  print_got_heres = fx_param_bool(debug, "print_got_heres", 1);
  print_warnings = fx_param_bool(debug, "print_warnings", 1);
  abort_on_nonfatal = fx_param_bool(debug, "abort_on_nonfatal", 0);
  pause_on_nonfatal = fx_param_bool(debug, "pause_on_nonfatal", 0);
  print_notify_locs = fx_param_bool(debug, "print_notify_locs", 0);
}

static void fx__attempt_speedup(fx_module *root)
{
  /* Try to minimize disk access */
  sync();
  sleep(3);

  /* Start and stop default timer to bring code into cache */
  fx_timer_start(root, "total_time");
  fx_timer_stop(root, "total_time");
  fx_reset_timer(root, "total_time");
}

fx_module *fx_init(int argc, char *argv[], const fx_module_doc *doc)
{
  fx_module *root = malloc(sizeof(fx_module));

  /* Check a couple cases to see if the user is asking for help.  This is an
   * ugly, stupid hack, and I hate that I have to do THIS to make it work.  This
   * system will be gone soon, and then we can all rejoice in sensible code. */
  if((argc == 2 &&
      (strcmp(argv[0], "-h") == 0 ||
       strcmp(argv[0], "-help") == 0)) ||
      (argc == 1)) {
    argc = 2;
    free(argv[1]);
    argv[1] = strcpy((char *) malloc(sizeof(char) * 7), "--help");
  }

  /* First fx_init sets true root but can call to create more trees */
  if (!fx_root) {
    fx_root = root;
  }

  datanode_init(root, "");
  fx__fill_docs(root, &fx__std_doc);
  if (doc) {
    fx__fill_docs(root, doc);
  }

  /* Set argc = 0 to omit command line parsing */
  if (argc > 0) {
    fx__parse_cmd_line(root, argc - 1, argv + 1);

    if (fx_param_exists(root, "help")) {
      fx__std_help(argv[0], fx_param_str_req(root, "help"), doc);
    }

    if (fx_param_exists(root, "fx/load")) {
      fx__load_param_files(root);
    }

    if (fx_param_bool(root, "fx/no_docs_nagging", 0)) {
      fx_docs_nagging = 0;
    }
    MUST_NOT_FAIL_MSG(fx__check_inputs(root),
        "There were problems with input parameters; try --help.");

    fx__report_sys(fx_submodule(root, "info/sys"));
    fx__read_debug_params(fx_submodule(root, "debug"));

    if (fx_param_bool(root, "fx/timing", 0)) {
      fx__attempt_speedup(root);
    }
  } else {
    NONFATAL("No arguments supplied!");
    fx__std_help(argv[0], fx_param_str_req(root, "help"), doc);
  }

  fx_timer_start(root, "total_time");

  return root;
}



static void fx__timer_double(fx_module *mod, const char *key, double val)
{
  fx_module *timer = fx__lookup(mod, key);

  DEBUG_ONLY(fx__check_lookup(FX_TIMER, FX_DOUBLE, timer));

  free(timer->val);
  timer->val = fx__alloc_double(val);
}

/* #ifdef because gcc whines about not using this function */
#ifdef HAVE_RDTSC
static void fx__timer_int(fx_module *mod, const char *key, long long val)
{
  fx_module *timer = fx__lookup(mod, key);

  DEBUG_ONLY(fx__check_lookup(FX_TIMER, FX_INT, timer));

  free(timer->val);
  timer->val = fx__alloc_int(val);
}
#endif

static void fx__stop_timers(fx_module *mod, struct timestamp *now)
{
  fx_module *child;

  for (child = mod->first_child; child; child = child->next) {
    fx__stop_timers(child, now);
  }

  if (mod->mod_type == FX_TIMER && mod->val_type == FX_CUSTOM && mod->val) {
    fx_timer *timer = (fx_timer *)mod->val;
    if (STOPWATCH_ACTIVE(timer)) {
      stopwatch_stop(timer, now);
    }

    /* Squelch "undocumented" nagging with bonus timer docs */
    fx__fill_docs(mod, &fx__timer_doc);

#ifdef HAVE_RDTSC
    fx__timer_int(mod, "cycles", timer->total.cycles);
#endif
    fx__timer_double(mod, "real", timer->total.micros / 1e6);
    fx__timer_double(mod, "user",
        (double) timer->total.cpu.tms_utime / sysconf(_SC_CLK_TCK));
    fx__timer_double(mod, "sys",
        (double) timer->total.cpu.tms_stime / sysconf(_SC_CLK_TCK));
  }
}

static void fx__report_rusage(fx_module *mod, int usage_type)
{
  struct rusage usage;

  getrusage(usage_type, &usage);

  /* Note: Many of these are unsupported by various OSes */
  fx_result_int(mod, "utime/sec", usage.ru_utime.tv_sec);
  fx_result_int(mod, "utime/usec", usage.ru_utime.tv_usec);
  fx_result_int(mod, "stime/sec", usage.ru_stime.tv_sec);
  fx_result_int(mod, "stime/usec", usage.ru_stime.tv_usec);
  fx_result_int(mod, "minflt", usage.ru_minflt);
  fx_result_int(mod, "majflt", usage.ru_majflt);
  fx_result_int(mod, "maxrss", usage.ru_maxrss);
  fx_result_int(mod, "ixrss", usage.ru_ixrss);
  fx_result_int(mod, "idrss", usage.ru_idrss);
  fx_result_int(mod, "isrss", usage.ru_isrss);
  fx_result_int(mod, "nswap", usage.ru_nswap);
  fx_result_int(mod, "inblock", usage.ru_inblock);
  fx_result_int(mod, "oublock", usage.ru_oublock);
  fx_result_int(mod, "msgsnd", usage.ru_msgsnd);
  fx_result_int(mod, "msgrcv", usage.ru_msgrcv);
  fx_result_int(mod, "nsignals", usage.ru_nsignals);
  fx_result_int(mod, "nvcsw", usage.ru_nvcsw);
  fx_result_int(mod, "nivcsw", usage.ru_nivcsw);
}

static void fx__output_results(fx_module *root)
{
  char *type_char = fx_mod_marker;

  /* Disable the printing of type characters if appropriate */
  if (fx_param_bool(root, "fx/no_output_types", 0)) {
    type_char = NULL;
  }

  /* Pipe a transcript of the params if output specified */
  if (fx_param_exists(root, "fx/output")) {
    FILE *stream = fopen(fx_param_str_req(root, "fx/output"), "w");
    datanode_write(root, stream, type_char);
    fclose(stream);
  }

  /* Still pipe to stdout unless explicitly silenced */
  if (!fx_param_bool(root, "fx/silent", 0)) {
    datanode_write(root, stdout, type_char);
  }
}

void fx_done(fx_module *root)
{
  struct timestamp now;
  timestamp_now(&now);

  /* Can input NULL, but also additionally created trees */
  if (!root) {
    root = fx_root;
    fx_root = NULL;
  }

  DEBUG_ASSERT_MSG(root != NULL,
      "Cannot call fx_done without first calling fx_init.");

  fx__stop_timers(root, &now);

  if (fx_param_bool(root, "fx/rusage", 0)) {
    fx__report_rusage(
        fx_submodule(root, "info/rusage/self"), RUSAGE_SELF);
    fx__report_rusage(
        fx_submodule(root, "info/rusage/children"), RUSAGE_CHILDREN);
  }

  fx__output_results(root);

  datanode_destroy(root);
  free(root);
}
