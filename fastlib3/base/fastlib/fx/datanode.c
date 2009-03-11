// Copyright 2007 Georgia Institute of Technology. All rights reserved.
/**
 * @file datanode.c
 *
 * Implementation for path-based string data storage.
 */

#include "datanode.h"

#ifndef LINE_MAX
#define LINE_MAX 2048
#endif

static void datanode__reset(struct datanode *node)
{
  node->mod_type = 0;
  node->val_type = 0;
  node->meta = NULL;

  node->key = NULL;
  node->val = NULL;

  node->first_child = NULL;
  node->last_child = NULL;
  node->next = NULL;
  node->parent = NULL;
}

void datanode_init(struct datanode *node, const char *key)
{
  datanode__reset(node);
  node->key = strdup(key);
}

void datanode_destroy(struct datanode *node)
{
  struct datanode *child = node->first_child;

  while (child) {
    struct datanode *next = child->next;

    datanode_destroy(child);
    free(child);

    child = next;
  }

  free(node->key);
  free(node->val);

  datanode__reset(node);
}



struct datanode *datanode_lookup_expert(struct datanode *node, char *path,
					int create)
{
  char *end = path + strlen(path);
  *end = '/'; /* Note: not null-terminated! */

  while (path < end) {
    char *slash = strchr(path, '/');
    *slash = '\0'; /* Demark end of current query */

    /* Match special paths: "", ".", ".." */
    if (unlikely(*path == '\0' || strcmp(path, ".") == 0)) {
      path = slash + 1;
      continue;
    } else if (unlikely(strcmp(path, "..") == 0)) {
      if (node->parent) {
	node = node->parent;
      }
      path = slash + 1;
      continue;
    }

    /* Seek query in node's children, moving it to the end
     * - moving things to end gets them out of the way
     * - but check end first for fast repeat lookups
     */   
    if (!node->last_child
	|| likely(strcmp(path, node->last_child->key) != 0)) {
      struct datanode *child;
      struct datanode **prev;

      /* Note: last child can't match, already checked it
       * - important because use of node->last_child would fail
       */
      for (prev = &node->first_child, child = node->first_child; child;
	   prev = &child->next, child = child->next) {
	if (unlikely(strcmp(path, child->key) == 0)) {
	  *prev = child->next;
	  child->next = NULL;
	  node->last_child->next = child;
	  node->last_child = child;
	  break;
	}
      }

      /* Create query if not found */
      if (unlikely(!child)) {
	if (!create) {
	  return NULL;
	}

	child = malloc(sizeof(struct datanode));
	datanode__reset(child);

	child->key = malloc((slash - path + 1) * sizeof(char));
	strcpy(child->key, path);
	child->parent = node;
	*prev = child; /* Note: handles empty lists, too */
	node->last_child = child;
      }
    }

    node = node->last_child;
    path = slash + 1;
  }

  return node;
}

struct datanode *datanode_lookup(struct datanode *node, const char *path,
				 int create)
{
  struct datanode *retval;
  char *buf = strdup(path);

  retval = datanode_lookup_expert(node, buf, create);

  free(buf);
  return retval;
}

static int datanode__exists_impl(struct datanode *node)
{
  struct datanode *child;

  if (node->val) {
    return 1;
  }

  for (child = node->first_child; child; child = child->next) {
    if (datanode__exists_impl(child)) {
      return 1;
    }
  }

  return 0;
}

int datanode_exists(struct datanode *node, const char *path)
{
  node = datanode_lookup(node, path, 0);

  return node && datanode__exists_impl(node);
}

void datanode_copy(struct datanode *dest, struct datanode *src,
		   int overwrite)
{
  struct datanode *child;

  if (!dest->val || overwrite) {
    dest->mod_type = src->mod_type;
    dest->val_type = src->val_type;
    dest->meta = src->meta;

    /* TODO: consider clearing val if val_type < 0 */
    if (src->val && src->val_type >= 0) {
      free(dest->val);
      dest->val = strdup(src->val);
    }
  }

  for (child = src->first_child; child; child = child->next) {
    datanode_copy(datanode_lookup_expert(dest, child->key, 1), child,
		  overwrite);
  }
}



static void datanode__write_path(struct datanode *node, FILE *stream)
{
  if (node->parent) {
    datanode__write_path(node->parent, stream);
    putc('/', stream);
  }
  hex_to_stream(stream, node->key, "_.-+");
}

void datanode_write(struct datanode *node, FILE *stream,
		    const char *type_char)
{
  struct datanode *child;

  if (node->val && node->val_type >= 0) {
    datanode__write_path(node, stream);
    if (type_char) {
      fprintf(stream, ":%c", type_char[node->mod_type]);
    }
    putc(' ', stream);
    hex_to_stream(stream, node->val, "_.-+");
    putc('\n', stream);
  }

  for (child = node->first_child; child; child = child->next) {
    datanode_write(child, stream, type_char);
  }
}

void datanode_read(struct datanode *node, FILE *stream,
		   const char *type_char, int overwrite)
{
  char buf[LINE_MAX];

  while (fgets(buf, sizeof(buf), stream) != NULL) {
    struct datanode *entry;
    char *key;
    char *val;
    int type;
    int len;

    /* Strip whitespace, skipping blanks and comments */
    for (key = buf; isspace(*key); ++key) {}
    if (*key == '\0' || *key == '#') {
      continue;
    }
    for (len = strlen(key); isspace(key[len-1]); --len) {}
    key[len] = '\n'; /* Note: not null-terminated! */

    /* Demark the end of key and beginning of value */
    for (val = key; !isspace(*val); ++val) {}
    key[len] = '\0'; /* Restore null-termination */
    if (val - key == len) {
      val = "";
    } else {
      len = val - key;
      *val = '\0';
      while (isspace(*++val)) {}
    }

    /* Identify the node type, if provided */
    type = 0;
    if (len > 2 && key[len-2] == ':') {
      if (type_char) {
	type = strchr(type_char, key[len-1]) - type_char;
      }
      key[len-2] = '\0';
    } else if (key[len-1] == ':') {
      key[len-1] = '\0';
    }

    unhex_in_place(key);
    unhex_in_place(val);

    entry = datanode_lookup_expert(node, key, 1);
    if (!entry->val || overwrite) {
      if (type_char) {
	entry->mod_type = type;
      }
      free(entry->val);
      entry->val = strdup(val);
    }
  }
}
