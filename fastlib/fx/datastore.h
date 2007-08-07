// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file datastore.h
 *
 * Path-based arbitray (but mostly string) data storage.
 *
 * Provides tools for navigating paths and serializing results.
 */

#ifndef FX_DATSTORE_H
#define FX_DATSTORE_H

#include <stdio.h>
#include "base/compiler.h"

EXTERN_C_START

/**
 * The various kinds of nodes supported by the datastore.
 *
 * We use nodetype_t to indicate what kind of node should be created
 * if a nonexistent node is queried.  The value NODETYPE_NO_CREATE
 * disables the creation of nonexistent queries.
 */
typedef enum {
  NODETYPE_NO_CREATE,
  NODETYPE_PARAM,
  NODETYPE_RESULT,
  NODETYPE_TIMER,
  NODETYPE_MODULE
} nodetype_t;

/**
 * A node in the datastore.
 *
 * Nodes are identified with key and store an arbitrary value (often a
 * string); val is freed if non-NULL, but not in any secure manner.
 * The two recursive pointers serve different purposes: one helps form
 * the hierarchy and the other a list of nodes for the present section
 * of the tree.  In debug mode, type is filled such that it is
 * possible to determine whether the node is being used property
 * (e.g., to make sure we don't read a timer as a string).
 *
 * At current, only one of val or children should be non-NULL.
 */
struct datanode {
  char *key;
  void *val;
  struct datanode *children;
  struct datanode *next;
#ifdef DEBUG
  nodetype_t type;
#endif
};

/**
 * Initializes a datanode with the beginning of given key.
 */
void datanode_init_len(struct datanode *node, const char *key, int len,
		       nodetype_t type);
/**
 * Initializes a datanode with given key.
 */
void datanode_init(struct datanode *node, const char *key,
		   nodetype_t type);
/**
 * Frees a datanode's key, val, and children.
 */
void datanode_destroy(struct datanode *node);
/**
 * Frees a datanode's val, and children.
 */
void datanode_clear(struct datanode *node);

/**
 * Finds and optionally creates a subnode with a given name.
 */
struct datanode *datanode_get_node(struct datanode *node, const char *name,
				   nodetype_t create);
/**
 * Finds and optionally creates a subnode with a given path (with slashes).
 */
struct datanode *datanode_get_path(struct datanode *node, const char *path,
				   nodetype_t create);
/**
 * Finds a subnode, concatenating given paths.
 *
 * Terminate series of paths with a NULL.
 */
struct datanode *datanode_get_paths(struct datanode *node,
				    nodetype_t create, ...);

/**
 * Write the datanode to a file.
 */
void datanode_write(struct datanode *node, FILE *f);

/**
 * Populates a datanode from a file.
 */
void datanode_read(struct datanode *node, nodetype_t type, FILE *stream);

EXTERN_C_END

#endif
