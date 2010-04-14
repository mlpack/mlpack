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
 * @file datanode.h
 *
 * File-system-like storage of string data.  Useful, for instance, for
 * managing parameters, timers, and results between a hierarchy of
 * program components (modules).
 */

#ifndef FX_DATANODE_H
#define FX_DATANODE_H

#include "../base/common.h"

EXTERN_C_BEGIN

/**
 * A node composed of a key-value pair, metadata, and child nodes.
 *
 * Use datanode_init and datanode_lookup to create and obtain nodes.
 *
 * Nodes are always stored in linked lists, hence the pointer to the
 * next element.  Strings key and val are assumed to be "owned" by the
 * datanode and will be freed when it is destroyed; never set these to
 * string constants.  Other pointers specify the heirachy of nodes and
 * are maintained internally.
 *
 * Fields mod_type and val_type primarily serve as conveniences to
 * external code.  The datanode_read and datanode_write functions
 * optionally map mod_type to and from a character, and all functions
 * ignore the value stored at a node if val_type is negative (needed,
 * for instance, if val is not a string).  Both fields default to 0.
 *
 * Field meta allows external code to denote, e.g., constraints for
 * val.  It is copied by reference and never freed, and thus may be
 * set to constant or externally managed strings.  Its intended use is
 * for lists of nominal values, numeric bounds, and regexp matching.
 *
 * @see datanode_init, datanode_lookup, datanode_read
 */
struct datanode {
  /** The intended use of the node; managed externally. */
  int mod_type;
  /** The type of data stored at the node; managed externally. */
  int val_type;
  /** Additional information about the node; managed extermally. */
  const char *meta;
  /** String to match during lookup and pathing. */
  char *key;
  /** The node's data; assumed string unless val_type negative. */
  char *val;
  /** Beginning of a linked list of child nodes or NULL. */
  struct datanode *first_child;
  /** End of linked list; NULL iff first_child is NULL. */
  struct datanode *last_child;
  /** Pointer to a sibling node in containing linked list. */
  struct datanode *next;
  /** The node's parent; used in pathing and printing. */
  struct datanode *parent;
};

/**
 * Initializes a blank datanode.
 *
 * @param node a freshly constructed datanode
 * @param key the name of the datanode; e.g. "" for root
 *
 * @see datanode_destruct, datanode_lookup, struct datanode
 */
void datanode_init(struct datanode *node, const char *key);
/**
 * Frees memory allocated for all node beneath a given node.
 *
 * @param node the node to destruct
 *
 * @see datanode_init
 */
void datanode_destroy(struct datanode *node);

/**
 * The same as datanode_lookup, except that the input path is
 * destructively modified (slashes replaced with '\0').
 *
 * @param node the containing node of the node to find
 * @param path the path to the node to find; destructively modified
 * @param create whether to create the node if it does not exist
 * @returns the found/created node, or NULL if not found
 *
 * @see datanode_lookup
 */
struct datanode *datanode_lookup_expert(struct datanode *node, char *path,
					int create);
/**
 * Obtains a node beneath a given node, pathing through child nodes as
 * appropriate.
 *
 * The key passed to this function may be a slash-delimited path as in
 * UNIX.  Child nodes are (optionally) created if they do not exist,
 * and both found and created entries are moved to the ends of their
 * containing lists.  This causes nodes to be printed in LRU order.
 *
 * It is never your responsibility to allocate or free memory for
 * entire nodes, though when modifying a node's value, you must free
 * the old value and provide a new one that may later be freed;
 * i.e. do not use string constants, but instead strdup them first.
 *
 * @param node the containing node of the node to find
 * @param path the path of the node to find
 * @param create whether to create the node if it does not exist
 * @returns the found/created node, or NULL if not found
 *
 * @see datanode_lookup_expert, datanode_exists, datanode_init
 */
struct datanode *datanode_lookup(struct datanode *node, const char *path,
				 int create);
/**
 * Tests whether any values exists at or beneath a node.
 *
 * This is a somewhat awkward definition of "exists", but reflects the
 * fact that valueless node may be created for metadata purposes only.
 * It is equivalent to asking whether a node's key will occur with a
 * value or in a path when printing.
 *
 * @param node the containing node of the node to check for
 * @param path the path of the node to chedk for
 * @returns whether the node was found
 *
 * @see datanode_lookup
 */
int datanode_exists(struct datanode *node, const char *path);
/**
 * Copies values at and beneath a source node to a destination,
 * optionally overwriting.
 *
 * The destination must already have been initialized via
 * datanode_init or obtained via datanode_lookup.
 *
 * This function also copies mod_type, val_type, and meta even from
 * source nodes that do not have values, but only if overwiting or if
 * the destination has no value.  Further, this function does not copy
 * val if val_type is negative (assumed to be non-string).
 *
 * @param dest the node to receive copied values
 * @param src the node to be copied
 * @param overwrite whether to overwite existing values
 *
 * @see datanode_lookup, datanode_init
 */
void datanode_copy(struct datanode *dest, struct datanode *src,
		   int overwrite);

/**
 * Prints all value-containing nodes beneath a given node in LRU
 * (least recently used) order.
 *
 * Node are printed in the format:
 * @code
 *   /path/from/root/key:type value
 * @endcode
 *
 * Type is given by character @c type_char[node->type] if type_char is
 * non-NULL; otherwise, it and its preceding colon are omitted.  Nodes
 * with negative val_type are not printed.
 *
 * Due to conflicts with certain valid characters in keys and values
 * (e.g. colons, spaces, and newlines), non-alphanumeric characters
 * other than "_+,-.[]" are converted to '%XX', where XX is the
 * hexadecimal ASCII value.
 *
 * @param node the node to print
 * @param stream the output stream
 * @param type_char a string of characters indexed by node->type, or
 *        NULL if types should not be printed
 *
 * @see fx_module_read, struct datanode
 */
void datanode_write(struct datanode *node, FILE *stream,
		    const char *type_char);
/**
 * Reads nodes from a file stream into a given node.
 *
 * Input format is identical to the output format for datanode_write.
 * The ":type" segment is optional, with type set to 0 if emitted and
 * left unchanged if type_chars is NULL (it still defaults to 0 for
 * newly created nodes).  Type is set via @c strchr(type_char, c) .
 *
 * @param node the node to be filled
 * @param stream the input stream
 * @param type_char a string of characters indexed by node->type, or
 *        NULL if types should be ignored
 * @param overwrite whether to overwerite existing values
 *
 * @see fx_module_write, struct fx_module, fx_type_marker
 */
void datanode_read(struct datanode *node, FILE *stream,
		   const char *type_char, int overwrite);

EXTERN_C_END

#endif /* FX_DATANODE_H */
