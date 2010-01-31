/**
 * @file fx.c
 *
 * Definitions for the experiment-running system integration hooks.
 */

#include "fx.h"
#include "datastore.h"
#include "timer.h"
#include "base/debug.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/utsname.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#if defined(VERBOSE) || defined(DEBUG)
#define FX_SHOW_RESULTS_TIMERS
#endif

static int fx__show_results_timers = 0;

static struct datanode *fx__real_root;
static int fx__silent = 0;

struct datanode *fx_root;

/* TODO: Use this mutex everywhere */
static pthread_mutex_t fx__mutex;

static const char *fx__module_name(const struct datanode *node) {
  if (node) {
    return node->key;
  } else {
    return fx_root->key;
  }
}

static void fx__report_system(struct datanode *node)
{
  struct utsname info;
  
  uname(&info);

  fx_set_result(node, "./node/name", info.nodename);
  fx_set_result(node, "./arch/name", info.machine);
  fx_set_result(node, "./kernel/name", info.sysname);
  fx_set_result(node, "./kernel/release", info.release);
  fx_set_result(node, "./kernel/build", info.version);
}

static void fx__parse_cmd_line(struct datanode *node, int argc, char *argv[])
{
  int i;
  
  for (i = 0; i < argc; i++) {
    if (argv[i][0] == '-' && argv[i][1] == '-') {
      char *s = strdup(argv[i]);
      char *path = s + 2;
      char *val = strchr(path, '=');

      if (!val) {
        val = "1";
      } else {
        *val++ = '\0';
      }

      if (strcmp(path, "fx/load") == 0) {
        FILE *stream = fopen(val, "r");
        if (stream == NULL) {
          FATAL("File not found (in --fx/load): [%s].", val);
        } else {
          datanode_read(node, NODETYPE_PARAM, stream);
          (void)fclose(stream);
        }
      } else {
        datanode_get_path(node, path, NODETYPE_PARAM)->val = strdup(val);
      }

      free(s);
    } else {
      NOTIFY("Ignoring parameter without \"--\": \"%s\"", argv[i]);
    }
  }
}

static void fx__read_debug_params(struct datanode *node)
{
#ifdef VERBOSE
  verbosity_level = fx_param_double(node, "./verbosity_level", 1.0);
  print_got_heres = fx_param_bool(node, "./print_got_heres", 1);
#endif
  print_warnings = fx_param_bool(node, "./print_warnings", 1);
  abort_on_nonfatal = fx_param_bool(node, "./abort_on_nonfatal", 0);
  pause_on_nonfatal = fx_param_bool(node, "./pause_on_nonfatal", 0);
  print_notify_locs = fx_param_bool(node, "./print_notify_locs", 0);
  fx__show_results_timers = fx_param_bool(node, "./noisy", 0);
}

static void fx__attempt_speedup()
{
  /* Try to minimize disk access. */
  sync();
  sleep(3);

  /* Stop and restart default timer to ensure timer code is dynamically
   * loaded and hopefully in L2 cache. */
  fx_timer_start(NULL, "default");
  fx_timer_stop(NULL, "default");
}

void fx_init(int argc, char **argv) {
  DEBUG_ASSERT_MSG(fx_root == NULL, "Cannot call fx_init twice.");
  
  pthread_mutex_init(&fx__mutex, NULL);

  fx_root = malloc(sizeof(struct datanode));
  fx__real_root = fx_root;
  datanode_init(fx_root, "FX_ROOT", NODETYPE_MODULE);
  fx__report_system(
      datanode_get_path(fx_root, "info/system", NODETYPE_RESULT));
  fx__parse_cmd_line(
      datanode_get_node(fx_root, "params", NODETYPE_PARAM),
      argc - 1, argv + 1);
  fx__read_debug_params(fx_param_node(NULL, "debug"));

  if (fx_param_bool(NULL, "fx/timing", 0)) {
    fx__attempt_speedup();
  }

  fx_timer_start(NULL, "default");
}

static void fx__finalize_timers(struct datanode *node,
                                struct timestamp *now)
{
  struct datanode *child;

  DEBUG_WARN_MSG_IF(node->type != NODETYPE_TIMER,
                    "Finalizing non-timer \"%s\" as timer",
                    node->key);

  for (child = node->children; child; child = child->next) {
    fx__finalize_timers(child, now);
  }

  if (node->val) {
    if (TIMER_IS_ACTIVE((struct timer *)node->val)) {
      timer_stop(node->val, now);
    }
    timer_emit_results(node->val, node);

    free(node->val);
    node->val = NULL;
  }
}

static void fx__finalize_module_timers(struct datanode *node,
                                       struct timestamp *now)
{
  struct datanode *child;

  for (child = node->children; child; child = child->next) {
    if (strcmp("timers", child->key) == 0) {
      fx__finalize_timers(child, now);
    } else if (strcmp("info", child->key) != 0
               && strcmp("params", child->key) != 0
               && strcmp("results", child->key) != 0) {
      fx__finalize_module_timers(child, now);
    }
  }
}

static void fx__report_usage(int type, struct datanode *node)
{
  struct rusage usage;
  
  getrusage(type, &usage);
  
  /* TODO: It might be nice to disable this in debug mode. */
  
  /* Store all getrusage values, even if they are bogus. */
  fx_set_result(node, "./WARNING", "your_OS_might_not_support_all_of_these");
  fx_format_result(node, "./utime", 
      "%ld.%06ld", (long)usage.ru_utime.tv_sec, (long)usage.ru_utime.tv_usec);
  fx_format_result(node, "./stime",
      "%ld.%06ld", (long)usage.ru_stime.tv_sec, (long)usage.ru_stime.tv_usec);
  fx_format_result(node, "./minflt", "%ld", usage.ru_minflt);
  fx_format_result(node, "./majflt", "%ld", usage.ru_majflt);
  fx_format_result(node, "./maxrss", "%ld", usage.ru_maxrss);
  fx_format_result(node, "./ixrss", "%ld", usage.ru_ixrss);
  fx_format_result(node, "./idrss", "%ld", usage.ru_idrss);
  fx_format_result(node, "./isrss", "%ld", usage.ru_isrss);
  fx_format_result(node, "./nswap", "%ld", usage.ru_nswap);
  fx_format_result(node, "./inblock", "%ld", usage.ru_inblock);
  fx_format_result(node, "./oublock", "%ld", usage.ru_oublock);
  fx_format_result(node, "./msgsnd", "%ld", usage.ru_msgsnd);
  fx_format_result(node, "./msgrcv", "%ld", usage.ru_msgrcv);
  fx_format_result(node, "./nsignals", "%ld", usage.ru_nsignals);
  fx_format_result(node, "./nvcsw", "%ld", usage.ru_nvcsw);
  fx_format_result(node, "./nivcsw", "%ld", usage.ru_nivcsw);
}

static void fx__write(struct datanode *node)
{
  if (fx_param_exists(node, "fx/output")) {
    FILE *f = fopen(fx_param_str(node, "fx/output", NULL), "w");
    datanode_write(node, f);
    fclose(f);
  } else if (!fx__silent) {
    datanode_write(node, stdout);
  }
}

void fx_done(void) {
  struct timestamp now;

  DEBUG_ASSERT_MSG(fx_root != NULL,
      "fx_done called twice, or fx_init was not called.");

  /* TODO: Report compile flags. */
  /* TODO: Report the command line verbatim.  (Avoid doing this.) */
  timestamp_now(&now);
  fx__finalize_module_timers(fx_root, &now);
  fx__report_usage(RUSAGE_SELF,
      datanode_get_path(fx_root, "info/rusage/self", NODETYPE_RESULT));
  fx__report_usage(RUSAGE_CHILDREN,
      datanode_get_path(fx_root, "info/rusage/children", NODETYPE_RESULT));

  fx__write(fx__real_root);

  datanode_destroy(fx__real_root);
  free(fx__real_root);
  fx_root = NULL;
  fx__real_root = NULL;

  pthread_mutex_destroy(&fx__mutex);
}

static struct datanode *fx__param(struct datanode *module, const char *name,
                                  int create)
{
  nodetype_t create_type = create ? NODETYPE_PARAM : NODETYPE_NO_CREATE;
  struct datanode *node;

  if (!module) {
    module = fx_root;
  }

  DEBUG_ERR_MSG_IF(name == NULL/* || name[0] == '\0'*/,
                   "Empty path; maybe you want \".\"");

  if (name[0] == '.' && (name[1] == '/' || name[1] == '\0')) {
    node = datanode_get_path(module, name + 1, create_type);
  } else {
    DEBUG_WARN_MSG_IF(module->type != NODETYPE_MODULE,
                      "Accessing \"params\" of non-module \"%s\".",
                      module->key);
    node = datanode_get_paths(module, create_type, "params", name, NULL);
  }

  DEBUG_WARN_MSG_IF(node && node->type != NODETYPE_PARAM,
                    "Datastore entry \"%s\" is not a parameter.", name);

  return node;
}

static void fx__show_param(struct datanode *node) {
/*
  Consider doing something with this code

  const char *value = node->val;
  char buffer[1024];
  struct datanode *cur = node;
  char *pos = buffer + sizeof(buffer);

  while (cur != NULL && cur->parent != fx_root) {
    int len;

    if (cur->source) {
      cur = cur->source;
    }

    len = strlen(cur->key);
    pos -= len + 1;
    strcpy(pos, cur->key);
    pos[len] = '/';
    cur = cur->parent;
  }

  buffer[sizeof(buffer) - 1] = '\0';

  fprintf(stderr, "%s %s\n", pos, value);
*/
}

int fx_param_exists(struct datanode *module, const char *name)
{
  struct datanode *node;
  
  node = fx__param(module, name, 0);

  return node && (node->val || node->children);
}

const char *fx_param_str(struct datanode *module, const char *name,
                         const char *def)
{
  struct datanode *node;
  
  node = fx__param(module, name, 1);

  if (!node->val) {
    if (!def) {
      FATAL("Required parameter \"%s\" unspecified in module \"%s\".",
            name, fx__module_name(module));
    } else {
      node->val = strdup(def);
    }
  }

  fx__show_param(node);

  return node->val;
}

const char *fx_param_str_req(struct datanode *module, const char *name)
{
  return fx_param_str(module, name, NULL);
}

double fx_param_double(struct datanode *module, const char *name, double def)
{
  struct datanode *node;

  node = fx__param(module, name, 1);

  if (!node->val) {
    char buf[32];
    sprintf(buf, "%.16g", def);
    node->val = strdup(buf);
  } else {
    if (sscanf(node->val, "%lf", &def) != 1) {
      FATAL("Parameter \"%s\" in module \"%s\" is not a double: \"%s\".",
            name, fx__module_name(module), (char*)node->val);
    }
  }

  fx__show_param(node);

  return def;
}

double fx_param_double_req(struct datanode *module, const char *name)
{
  const char *val = fx_param_str_req(module, name);
  double res;

  if (sscanf(val, "%lf", &res) != 1) {
    FATAL("Parameter \"%s\" in module \"%s\" is not a double: \"%s\".",
          name, fx__module_name(module), (char*)val);
  }

  return res;
}

long long int fx_param_int(struct datanode *module, const char *name, long long int def)
{
  struct datanode *node = fx__param(module, name, 1);

  if (!node->val) {
    char buf[3 * sizeof(long long int) + 2];
    sprintf(buf, "%lli", def);
    node->val = strdup(buf);
  } else {
    if (sscanf(node->val, "%lli", &def) != 1) {
      FATAL("Parameter \"%s\" in module \"%s\" is not an int: \"%s\".",
            name, fx__module_name(module), (char*)node->val);
    }
  }

  fx__show_param(node);

  return def;
}

long long int fx_param_int_req(struct datanode *module, const char *name)
{
  const char *val = fx_param_str_req(module, name);
  long long int res;

  if (sscanf(val, "%lli", &res) != 1) {
    FATAL("Parameter \"%s\" in module \"%s\" is not an int: \"%s\".",
          name, fx__module_name(module), val);
  }

  return res;
}

int fx_param_bool(struct datanode *module, const char *name, int def)
{
  struct datanode *node = fx__param(module, name, 1);

  if (!node->val) {
    node->val = strdup(def ? "1" : "0");
    return def;
  } else {
    if (strchr("1tTyY", ((char *)node->val)[0]) != NULL) {
      return 1;
    } else if (strchr("0fFnN", ((char *)node->val)[0]) != NULL) {
      return 0;
    } else {
      FATAL("Parameter \"%s\" in module \"%s\" is not a bool: \"%s\".",
            name, fx__module_name(module), (char *)node->val);
    }
  }

  fx__show_param(node);

  return -1;
}

int fx_param_bool_req(struct datanode *module, const char *name)
{
  const char *val = fx_param_str_req(module, name);

  if (strchr("1tTyY", val[0]) != NULL) {
    return 1;
  } else if (strchr("0fFnN", val[0]) != NULL) {
    return 0;
  } else {
    FATAL("Parameter \"%s\" in module \"%s\" is not a bool: \"%s\".",
          name, fx__module_name(module), val);
  }
  return -1;
}

struct datanode *fx_param_node(struct datanode *module, const char *name)
{
  return fx__param(module, name, 1);
}

static void fx__set_param_val(struct datanode *node, const char *val,
                              int overwrite)
{
  if (!node->val) {
    node->val = strdup(val);
  } else if (overwrite) {
    NONFATAL("Parameter \"%s\" existed before being set.", node->key);
    free(node->val);
    node->val = strdup(val);
  }
}

void fx_default_param(struct datanode *module, const char *name,
                      const char *def)
{
  fx__set_param_val(fx__param(module, name, 1), def, 0);
}

void fx_set_param(struct datanode *module, const char *name,
                  const char *val)
{
  fx__set_param_val(fx__param(module, name, 1), val, 1);
}

void fx_format_param(struct datanode *module, const char *name,
                     const char *format, ...)
{
  char buf[1024];
  va_list vl;

  va_start(vl, format);
  vsnprintf(buf, 1024, format, vl);
  va_end(vl);

  fx_set_param(module, name, buf);
}

void fx_clear_param(struct datanode *module, const char *name)
{
  struct datanode *node = fx__param(module, name, 0);

  if (node && (node->val || node->children)) {
    NONFATAL("Parameter \"%s\" existed before being cleared.", node->key);
    datanode_clear(node);
  }
}

static void fx__copy_params(struct datanode *node,
                            struct datanode *source, int overwrite)
{
  struct datanode *child;

  DEBUG_WARN_MSG_IF(node->type != NODETYPE_PARAM,
                    "Destination \"%s\" is not a parameter.", node->key);
  DEBUG_WARN_MSG_IF(source->type != NODETYPE_PARAM,
                    "Source \"%s\" is not a parameter.", source->key);

  if (source->val) {
    fx__set_param_val(node, source->val, overwrite);
  }

  for (child = source->children; child; child = child->next) {
    fx__copy_params(
        datanode_get_node(node, child->key, NODETYPE_PARAM), child,
        overwrite);
  }
}

void fx_default_param_node(struct datanode *dest_module, const char *destname,
                           struct datanode *src_module, const char *srcname)
{
  struct datanode *source_node = fx__param(src_module, srcname, 0);

  if (source_node) {
    dest_module->source = source_node;
    fx__copy_params(fx__param(dest_module, destname, 1), source_node, 0);
  }
}

void fx_set_param_node(struct datanode *dest_module, const char *destname,
                       struct datanode *src_module, const char *srcname)
{
  struct datanode *source_node = fx__param(src_module, srcname, 0);

  if (source_node) {
    dest_module->source = source_node;
    fx__copy_params(fx__param(dest_module, destname, 1), source_node, 1);
  }
}

static struct datanode *fx__result(struct datanode *module, const char *name,
                                   int create)
{
  nodetype_t create_type = create ? NODETYPE_RESULT : NODETYPE_NO_CREATE;
  struct datanode *node;

  if (!module) {
    module = fx_root;
  }

  DEBUG_ERR_MSG_IF(name == NULL/* || name[0] == '\0'*/,
                   "Empty path; maybe you want \".\"");

  if (name[0] == '.' && (name[1] == '/' || name[1] == '\0')) {
    node = datanode_get_path(module, name + 1, create_type);
  } else {
    DEBUG_WARN_MSG_IF(module->type != NODETYPE_MODULE,
                      "Accessing \"results\" of non-module \"%s\".",
                      module->key);
    node = datanode_get_paths(module, create_type, "results", name, NULL);
  }

  DEBUG_WARN_MSG_IF(node && node->type != NODETYPE_RESULT,
                    "Datastore entry \"%s\" is not a result.", name);

  return node;
}

void fx_set_result(struct datanode *module, const char *name, const char *val)
{
  struct datanode *node = fx__result(module, name, 1);

  node->val = strdup(val);

#ifdef FX_SHOW_RESULTS_TIMERS
  if (unlikely(fx__show_results_timers))
    fprintf(stderr, ANSI_HBLACK"Result [%s] in [%s] set to [%s]."ANSI_CLEAR"\n",
        name, module->key, val);
#endif
}

void fx_format_result(struct datanode *module, const char *name,
                      const char *format, ...)
{
  char buf[1024];
  va_list vl;

  va_start(vl, format);
  vsnprintf(buf, 1024, format, vl);
  va_end(vl);

  fx_set_result(module, name, buf);
}

void fx_clear_result(struct datanode *module, const char *name)
{
  struct datanode *node = fx__result(module, name, 0);

  if (node) {
    datanode_clear(node);
  }
}

const char *fx_get_result_str(struct datanode *module, const char *name)
{
  struct datanode *node;
  
  node = fx__result(module, name, 0);

  if (!node->val) {
    FATAL("Result \"%s\" unspecified in module \"%s\".",
          name, fx__module_name(module));
  }

  return node->val;
}

double fx_get_result_double(struct datanode *module, const char *name)
{
  const char *val = fx_get_result_str(module, name);
  double res;

  if (sscanf(val, "%lf", &res) != 1) {
    FATAL("Result \"%s\" in module \"%s\" is not a double: \"%s\".",
          name, fx__module_name(module), (char*)val);
  }

  return res;
}

long long int fx_get_result_int(struct datanode *module, const char *name)
{
  const char *val = fx_get_result_str(module, name);
  long long int res;

  if (sscanf(val, "%lli", &res) != 1) {
    FATAL("Result \"%s\" in module \"%s\" is not an int: \"%s\".",
          name, fx__module_name(module), val);
  }

  return res;
}

int fx_get_result_bool(struct datanode *module, const char *name)
{
  const char *val = fx_get_result_str(module, name);

  if (strchr("1tTyY", val[0]) != NULL) {
    return 1;
  } else if (strchr("0fFnN", val[0]) != NULL) {
    return 0;
  } else {
    FATAL("Result \"%s\" in module \"%s\" is not a bool: \"%s\".",
          name, fx__module_name(module), val);
  }
  return -1;
}

static struct datanode *fx__timer(struct datanode *module, const char *name,
                                  int create)
{
  nodetype_t create_type = create ? NODETYPE_TIMER : NODETYPE_NO_CREATE;
  struct datanode *node;

  if (!module) {
    module = fx_root;
  }
  /*if (!name) {
    name = "default";
  }*/

  DEBUG_ERR_MSG_IF(name == NULL/* || name[0] == '\0'*/,
                   "Empty path; maybe you want \".\"");

  if (name[0] == '.' && (name[1] == '/' || name[1] == '\0')) {
    node = datanode_get_path(module, name + 1, create_type);
  } else {
    DEBUG_WARN_MSG_IF(module->type != NODETYPE_MODULE,
                      "Accessing \"timers\" of non-module \"%s\".",
                      module->key);
    node = datanode_get_paths(module, create_type, "timers", name, NULL);
  }

  DEBUG_WARN_MSG_IF(node && node->type != NODETYPE_TIMER,
                    "Datastore entry \"%s\" is not a timer.", name);

  return node;
}

struct timer *fx_timer(struct datanode *module, const char *name) {
  struct datanode *node = fx__timer(module, name, 1);

  if (!node->val) {
    node->val = malloc(sizeof(struct timer));
    timer_init(node->val);
  }
  
  return (struct timer *)node->val;
}

void fx_timer_start(struct datanode *module, const char *name)
{
  struct datanode *node = fx__timer(module, name, 1);

#ifdef FX_SHOW_RESULTS_TIMERS
  if (unlikely(fx__show_results_timers))
    fprintf(stderr, ANSI_HBLACK"Timer [%s] in [%s] started."ANSI_CLEAR"\n", name,
        fx__module_name(module));
#endif

  /*VERBOSE_MSG(0.0, "Timer \"%s\" in module \"%s\" started.",
            name ? name : "default", module ? module->key : fx_root->key);*/

  if (!node->val) {
    node->val = malloc(sizeof(struct timer));
    timer_init(node->val);
  }

  timer_start(node->val);
}

void fx_timer_stop(struct datanode *module, const char *name)
{
  struct timestamp now;
  struct datanode *node;

  timestamp_now(&now);

  node = fx__timer(module, name, 0);

  if (node) {
    timer_stop(node->val, &now);
#ifdef FX_SHOW_RESULTS_TIMERS
    if (unlikely(fx__show_results_timers))
      fprintf(stderr, ANSI_HBLACK"Timer [%s] in [%s] stopped, totalling %.3f wall-secs."ANSI_CLEAR"\n",
          name, module->key,
          (int)((struct timer *)node->val)->total.micros / 1.0e6);
#endif
  } else {
    NONFATAL("No timer named \"%s\" in module \"%s\".",
             name, fx__module_name(module));
  }

}

struct datanode *fx_submodule(struct datanode *module, const char *params,
                              const char *name_format, ...)
{
  struct datanode *node;
  char buf[1024];
  va_list vl;

  if (!module) {
    module = fx_root;
  }

  DEBUG_ERR_MSG_IF(name_format == NULL, "Empty submodule path.");

  va_start(vl, name_format);
  vsnprintf(buf, 1024, name_format, vl);
  va_end(vl);

  DEBUG_ERR_MSG_IF(buf[0] == '\0', "Empty submodule path.");

  node = datanode_get_path(module, buf, NODETYPE_MODULE);

  DEBUG_WARN_MSG_IF(node->type != NODETYPE_MODULE,
                    "Datastore entry \"%s\" is not a module.", buf);

  if (params) {
    struct datanode *param_node =
        datanode_get_node(node, "params", NODETYPE_PARAM);

    param_node->source = fx_param_node(module, params);
    fx__copy_params(param_node, param_node->source, 1);
  }

  return node;
}

void fx_scope(const char *scope_name) {
  struct datanode *old_root = fx__real_root;

  free(fx__real_root->key);
  fx__real_root->key = strdup(scope_name);

  fx__real_root = malloc(sizeof(struct datanode));
  datanode_init(fx__real_root, "FX_ROOT", NODETYPE_MODULE);
  datanode_add_child(fx__real_root, old_root);
}

void fx_silence() {
  fx__silent = 1;
}
