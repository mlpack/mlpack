/**
 * @file datastore.c
 *
 * Implementation for the arbitrary path-based datastore, similar to the
 * corresponding python implementation.
 */

#include "datastore.h"
#include "base/debug.h"

#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <ctype.h>
#include <limits.h>

void datanode_init_len(struct datanode *node, const char *key, int len,
		       nodetype_t type)
{
  DEBUG_ERR_MSG_IF(type == NODETYPE_NO_CREATE,
		   "Datanode created with type NODETYPE_NO_CREATE");

  node->key = malloc(len + 1);
  strncpy(node->key, key, len);
  node->key[len] = 0;

  node->val = NULL;
  node->children = NULL;
  node->next = NULL;
  node->parent = NULL;
  node->source = NULL;

  DEBUG_ONLY(node->type = type);
}

void datanode_init(struct datanode *node, const char *key,
		   nodetype_t type)
{
  datanode_init_len(node, key, strlen(key), type);
}

void datanode_clear(struct datanode *node)
{
  struct datanode *child;

  if (node->val) {
    free(node->val);
    node->val = NULL;
  }

  child = node->children;
  while (child) {
    struct datanode *cur = child;

    datanode_destroy(cur);
    child = cur->next;

    free(cur);
  }
  node->children = NULL;
}

void datanode_destroy(struct datanode *node)
{
  datanode_clear(node);

  if (node->key) {
    free(node->key);
    node->key = NULL;
  }
}

#if DEBUG
static int datanode__valid_path_len(const char *path, int len)
{
  char *slash;

  if (path[0] == '.'
      && (path[1] == '\0' || (path[1] == '.' && path[2] == '\0'))) {
    return 0;
  }

  slash = strchr(path, '/');
  return !slash || slash - path >= len;
}

static int datanode__valid_path(const char *path)
{
  return datanode__valid_path_len(path, INT_MAX);
}
#endif

void datanode_add_child(struct datanode *node, struct datanode *child) {
  child->next = node->children;
  node->children = child;
  child->parent = node;
}

struct datanode *datanode_get_node(struct datanode *node, const char *name,
				   nodetype_t create)
{
  struct datanode *child;

  DEBUG_ASSERT_MSG(name != NULL, "NULL path");
  DEBUG_ASSERT_MSG(datanode__valid_path(name), "Invalid path %s", name);

  /* Seek query in node's children */
  for (child = node->children; child; child = child->next) {
    if (strcmp(name, child->key) == 0) {
      break;
    }
  }

  /* Create child if query not found */
  if (!child && create != NODETYPE_NO_CREATE) {
    child = malloc(sizeof(struct datanode));
    datanode_init(child, name, create);

    datanode_add_child(node, child);
  }

  return child;
}

struct datanode *datanode_get_path(struct datanode *node, const char *path,
				   nodetype_t create)
{
#if DEBUG
  const char *orig_path = path;
#endif

  DEBUG_ASSERT_MSG(path != NULL, "NULL path");

  while (node) {
    struct datanode *child;
    const char *slash;
    ssize_t len;

    while (*path == '/') {
      path++;
    }
    if (*path == '\0') {
      break;
    }

    for (slash = path; *slash != '\0' && *slash != '/'; slash++) {}

    /* Obtain length of current path query */
    len = slash - path;

    DEBUG_ASSERT_MSG(datanode__valid_path_len(path, len),
		     "Invalid path %s", orig_path);

    /* Seek query in node's children */
    for (child = node->children; child; child = child->next) {
      if (strncmp(path, child->key, len) == 0
          && child->key[len] == '\0') {
	break;
      }
    }

    /* Create child if query not found */
    if (!child && create != NODETYPE_NO_CREATE) {
      child = malloc(sizeof(struct datanode));
      datanode_init_len(child, path, len, create);

      datanode_add_child(node, child);
    }

    node = child;
    path = slash;
  }

  return node;
}

struct datanode *datanode_get_paths(struct datanode *node,
				    nodetype_t create, ...)
{
  const char *arg;
  va_list vl;

  va_start(vl, create);
  while ((arg = va_arg(vl, char *)) && node) {
    node = datanode_get_path(node, arg, create);
  }
  va_end(vl);

  return node;
}

static int datanode__unhex_char(char c) {
  c = (c & ~0x20);
  if (c <= '9') {
    return c - '0';
  } else {
    return c - 'A' + 10;
  }
}

static char *datanode__unhex(char *s) {
  char *d = s;
  while (*s) {
    if (*s == '%' && isxdigit(s[1]) && isxdigit(s[2])) {
      *d = datanode__unhex_char(s[1]) * 16 + datanode__unhex_char(s[2]);
      s += 2;
    } else {
      *d = *s;
    }
    s++;
    d++;
  }
  *d = '\0';
  return d;
}

void datanode_read(struct datanode *node, nodetype_t type, FILE *stream) {
  char buf[4096];

  while (fgets(buf, sizeof(buf), stream) != NULL) {
    int len;
    char *key;
    char *value;
    struct datanode *result;
    
    // strip whitespace
    for (len = strlen(buf); len >= 0 && isspace(buf[len-1]); len--) {}
    buf[len] = '\0';
    key = buf;

    if (len == 0 || buf[0] == '#') {
      // ignore blank lines and #comments
      continue;
    }

    value = strchr(key, ' ');
    if (!value) {
      NONFATAL("Unrecognized line: %s", buf);
      break;
    }

    *value = '\0';
    while (isspace(*++value)) {}

    datanode__unhex(key);
    datanode__unhex(value);

    result = datanode_get_path(node, key, type);
    if (result->val) {
      NONFATAL("Overwriting [%s] - old value [%s], new [%s].",
          key, (char*)result->val, value);
      free(result->val);
    }
    result->val = strdup(value);
  }
}

static char *datanode__hex(char *dest, const char *src, int remaining)
{
  int c;

  while ((c = *src++) && remaining > 1) {
    if (isalnum(c) || strchr(".-_+", c)) {
      *dest++ = c;
      remaining--;
    } else {
      int len = snprintf(dest, remaining, "%%%02X", (unsigned)c);

      if (len > remaining - 1) {
	len = remaining - 1;
      }

      dest += len;
      remaining -= len;
    }
  }

  *dest = '\0';

  return dest;
}

static void datanode__write_buf(struct datanode *node, FILE *f,
				char *prefix, char *buf);

static void datanode__write_buf_backwards(struct datanode *node, FILE *f,
    char *prefix, char *buf) {
  if (node) {
    datanode__write_buf_backwards(node->next, f, prefix, buf);
    char *child_buf =
          datanode__hex(buf + 1, node->key, 4096 - (buf - prefix) -1);
    datanode__write_buf(node, f, prefix, child_buf);
  }
}

static void datanode__write_buf(struct datanode *node, FILE *f,
				char *prefix, char *buf)
{
  if (node->val) {
    buf[0] = ' ';
    datanode__hex(buf + 1, node->val, 4096 - (buf - prefix) - 1);
    fprintf(f, "%s\n", prefix);
  }

  buf[0] = '/';
  datanode__write_buf_backwards(node->children, f, prefix, buf);
}

void datanode_write(struct datanode *node, FILE *f)
{
  char buf[4096];

  /* TODO: Avoid buffer size limitations. */

  buf[0] = '\0';
  datanode__write_buf(node, f, buf, buf);
}
