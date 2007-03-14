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

struct datanode *fx_root;

/* TODO: Use this mutex everywhere */
static pthread_mutex_t fx__mutex;

static void fx__report_system(struct datanode *node)
{
  struct utsname info;
  
  uname(&info);

  fx_set_result(node, "./node/name", info.nodename);
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

      datanode_get_path(node, path, NODETYPE_PARAM)->val = strdup(val);

      free(s);
    } else {
      NOTIFY("Ignoring parameter without \"--\": \"%s\"", argv[i]);
    }
  }
}

static void fx__read_debug_params(struct datanode *node)
{
  debug_verbosity = fx_param_double(node, "./verbosity_level", 1.0);
  print_got_heres = fx_param_bool(node, "./print_got_heres", "1");
  print_warnings = fx_param_bool(node, "./print_warnings", "1");
  abort_on_nonfatal = fx_param_bool(node, "./abort_on_nonfatal", "0");
  pause_on_nonfatal = fx_param_bool(node, "./pause_on_nonfatal", "0");
  print_notify_headers = fx_param_bool(node, "./print_notify_headers", "1");
}

static void fx__attempt_speedup()
{
  /* Try to minimize disk access. */
  sync();
  sleep(3);

  /* Stop and restart default timer to cache the timer code. */
  fx_timer_start(NULL, NULL);
  fx_timer_stop(NULL, NULL);
}

void fx_init(int argc, char **argv) {
  pthread_mutex_init(&fx__mutex, NULL);

  fx_root = malloc(sizeof(struct datanode));
  datanode_init(fx_root, "FX_ROOT", NODETYPE_MODULE);
  fx__report_system(
      datanode_get_path(fx_root, "info/system", NODETYPE_RESULT));
  fx__parse_cmd_line(
      datanode_get_node(fx_root, "params", NODETYPE_PARAM),
      argc - 1, argv + 1);
  fx__read_debug_params(fx_param_node(NULL, "debug"));

  if (fx_param_bool(NULL, "fx/timing", "0")) {
    fx__attempt_speedup();
  }

  fx_timer_start(NULL, NULL);
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
  
  /* Store all getrusage values, even if they are bogus. */
  fx_set_result(node, "./WARNING", "your_OS_might_not_support_all_of_these");
  fx_format_result(node, "./utime", 
      "%ld.%06ld", usage.ru_utime.tv_sec, usage.ru_utime.tv_usec);
  fx_format_result(node, "./stime",
      "%ld.%06ld", usage.ru_stime.tv_sec, usage.ru_stime.tv_usec);
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
  FILE *f = stdout;

  if (fx_param_exists(node, "fx/output")) {
    f = fopen(fx_param_str(node, "fx/output", NULL), "w");
  }

  datanode_write(node, f);

  if (f != stdout) {
    fclose(f);
  }
}

void fx_done(void) {
  struct timestamp now;

  /* TODO: Report compile flags. */
  /* TODO: Report the command line verbatim.  (Avoid doing this.) */
  timestamp_now(&now);
  fx__finalize_module_timers(fx_root, &now);
  fx__report_usage(RUSAGE_SELF,
      datanode_get_path(fx_root, "info/rusage/self", NODETYPE_RESULT));
  fx__report_usage(RUSAGE_CHILDREN,
      datanode_get_path(fx_root, "info/rusage/children", NODETYPE_RESULT));
  fx__write(fx_root);

  datanode_destroy(fx_root);
  free(fx_root);

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

  if (name[0] == '.' && name[1] == '/') {
    node = datanode_get_path(module, name + 2, create_type);
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

int fx_param_exists(struct datanode *module, const char *name)
{
  struct datanode *node = fx__param(module, name, 0);

  return node && (node->val || node->children);
}

const char *fx_param_str(struct datanode *module, const char *name,
			 const char *def)
{
  struct datanode *node = fx__param(module, name, 1);

  if (!node->val) {
    if (!def) {
      FATAL("Required parameter \"%s\" unspecified in module \"%s\".",
	    name, module ? module->key : fx_root->key);
    } else {
      node->val = strdup(def);
    }
  }

  return node->val;
}

double fx_param_double(struct datanode *module, const char *name,
		       double def)
{
  struct datanode *node = fx__param(module, name, 1);
  
  if (!node->val) {
    if (isnan(def)) {
      FATAL("Required parameter \"%s\" unspecified in module \"%s\".",
          name, module ? module->key : fx_root->key);
    } else {
      char buf[100];
      sprintf(buf, "%.15g", def);
      node->val = strdup(buf);
      return def;
    }
  } else {
    if (sscanf(node->val, "%lf", &def) != 1) {
      FATAL("Parameter \"%s\" in \"%s\" is not a valid floating-point number: \"%s\".",
          name, module ? module->key : fx_root->key, (char*)node->val);
    }
    return def;
  }
}

int fx_param_int(struct datanode *module, const char *name, int def)
{
  struct datanode *node = fx__param(module, name, 1);
  
  if (!node->val) {
    if (def == -1) {
      FATAL("Required parameter \"%s\" unspecified in module \"%s\".",
          name, module ? module->key : fx_root->key);
    } else {
      char buf[3 * sizeof(int) + 2];
      sprintf(buf, "%d", def);
      node->val = strdup(buf);
      return def;
    }
  } else {
    if (sscanf(node->val, "%d", &def) != 1) {
      FATAL("Parameter \"%s\" in \"%s\" is not a valid integer: \"%s\".",
          name, module ? module->key : fx_root->key, (char*)node->val);
    }
    return def;
  }
}

int fx_param_bool(struct datanode *module, const char *name,
		  const char *def)
{
  return strchr("0fFnN", fx_param_str(module, name, def)[0]) == NULL;
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

void fx_def_param(struct datanode *module, const char *name, const char *def)
{
  fx__set_param_val(fx__param(module, name, 1), def, 0);
}

void fx_set_param(struct datanode *module, const char *name, const char *val)
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

void fx_def_param_node(struct datanode *dest_module, const char *destname,
		       struct datanode *src_module, const char *srcname)
{
  struct datanode *source_node = fx__param(src_module, srcname, 0);
  if (source_node) {
    fx__copy_params(fx__param(dest_module, destname, 1), source_node, 0);
  }
}

void fx_set_param_node(struct datanode *dest_module, const char *destname,
		       struct datanode *src_module, const char *srcname)
{
  struct datanode *source_node = fx__param(src_module, srcname, 0);
  if (source_node) {
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

  if (name[0] == '.' && name[1] == '/') {
    node = datanode_get_path(module, name + 2, create_type);
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

static struct datanode *fx__timer(struct datanode *module, const char *name,
				  int create)
{
  nodetype_t create_type = create ? NODETYPE_TIMER : NODETYPE_NO_CREATE;
  struct datanode *node;

  if (!module) {
    module = fx_root;
  }
  if (!name) {
    name = "default";
  }

  if (name[0] == '.' && name[1] == '/') {
    node = datanode_get_path(module, name + 2, create_type);
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

void fx_timer_start(struct datanode *module, const char *name)
{
  struct datanode *node = fx__timer(module, name, 1);

  DEBUG_MSG(0.0, "Timer \"%s\" in module \"%s\" started.",
            name ? name : "default", module ? module->key : fx_root->key);

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
    DEBUG_MSG(0.0, "Timer \"%s\" in module \"%s\" stopped at %5.3f wall-secs.",
	      name ? name : "default", module ? module->key : fx_root->key,
	      ((struct timer *)node->val)->total.micros / 1.0e6);
  } else {
    NONFATAL("No timer named \"%s\" in module \"%s\".",
	     name ? name : "default", module ? module->key : fx_root->key);
  }
}

struct datanode *fx_submodule(struct datanode *module, const char *name,
			      const char *params_path_template, ...)
{
  struct datanode *node;

  if (!module) {
    module = fx_root;
  }

  node = datanode_get_path(module, name, NODETYPE_MODULE);

  DEBUG_WARN_MSG_IF(node->type != NODETYPE_MODULE,
		    "Datastore entry \"%s\" is not a module.", name);

  if (params_path_template) {
    char params_path[256];
    va_list vl;
    va_start(vl, params_path_template);
    snprintf(params_path, sizeof(params_path), params_path_template, vl);
    va_end(vl);
    fx__copy_params(datanode_get_node(node, "params", NODETYPE_PARAM), 
		    fx_param_node(module, params_path), 1);
  }

  return node;
}
