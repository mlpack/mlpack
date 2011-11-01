/***
 * @file optionshierarchy.hpp
 * @author Matthew Amidon
 *
 * This file holds the OptionsHierarchy class, which is a tree
 * of parameters.  Each node can have an arbitrary number of children.
 */
#ifndef __MLPACK_CORE_IO_OPTIONSHIERARCHY_HPP
#define __MLPACK_CORE_IO_OPTIONSHIERARCHY_HPP

#include <map>
#include <string>
#include <vector>

namespace mlpack
{
namespace io
{

/**
 * Aids in the extensibility of OptionsHierarchy by focusing the potential
 * changes into one structure.
 */
struct OptionsData
{
  //! Name of this node.
  std::string node;
  //! Description of this node, if any.
  std::string desc;
  //! Type information of this node.
  std::string tname;
};

/**
 * A node in the hierarchy of parameters used by CLI.  Each node holds
 * information about itself and can have any number of children, each with
 * unique names.
 */
class OptionsHierarchy
{
 private:
  //! Holds all node specific data.
  OptionsData nodeData;

  //! Convenience typedef.
  typedef std::map<std::string, OptionsHierarchy> ChildMap;

  //! Map of this node's children, which should each have a corresponding
  //! OptionsHierarchy object.
  ChildMap children;

  /**
   * Returns the name foo in the pathname foo/bar/fizz.
   *
   * @param pathname The full pathname of the parameter,
   *   eg foo/bar in foo/bar.
   *
   * @return The name of the next node in the path
   *   eg foo in foo/bar.
   */
  std::string GetName(std::string& pathname);

  /* Returns the path bar/fizz in the pathname foo/bar/fizz.
   *
   * @param pathname The full pathname of the parameter,
   *   eg foo/bar in foo/bar.
   *
   * @return The identifiers of all nodes after the next node in the path,
   *   eg fizz/bar in foo/fizz/bar.
   */
  std::string GetPath(std::string& pathname);

 public:
  /**
   * Constructs an empty OptionsHierarchy node.
   */
  OptionsHierarchy();

  /**
   * Constructs an empty OptionsHierarchy node with the given name.
   *
   * @param name The name of the node to be created.
   */
  OptionsHierarchy(const char* name);

  /**
   * Copies the given node.
   *
   * @param other The node to be copied.
   */
  OptionsHierarchy(const OptionsHierarchy& other);

  /**
   * Destroys the node.
   */
  ~OptionsHierarchy();

  /**
   * Add a node as a child of this node.
   *
   * Will never fail, as given paths are relative to current node
   * and will be generated if not found.
   *
   * @param pathname The full pathname of the given node, eg /foo/bar.
   * @param tname A string unique to the type of the node.
   */
  void AppendNode(std::string& pathname, std::string& tname);

  /**
   * Add a node as a child of this node.
   *
   * Will never fail, as given paths are relative to current node
   * and will be generated if not found.
   *
   * @param pathname The full pathname of the given node, eg /foo/bar.
   * @param tname A string unique to the type of the node.
   * @param description String description of the node.
   */
  void AppendNode(std::string& pathname, std::string& tname,
                  std::string& description);

  /**
   * Add a node as a child of this node.
   *
   * Will never fail, as given paths are relative to current node
   * and will be generated if not found.
   *
   * @param pathname The full pathname of the given node, eg /foo/bar.
   * @param tname A string unique to the type of the node.
   * @param description String description of the node.
   * @param data Specifies all fields of the new node.
   */
  void AppendNode(std::string& pathname, std::string& tname,
                  std::string& description, OptionsData& data);

  /**
   * Returns the various data associated with a node.  Passed by copy,
   * since this is only for unit testing.
   *
   * @return The data associated with the node,
   * eg it's name, description, and value.
   */
  OptionsData GetNodeData();

  /**
   * Returns a vector containing the relative pathnames of nodes subordinant
   * to the one specified.
   *
   * @param pathname The node to start the traversal at.
   *
   * @return Vector containing relative pathnames of all subordinant nodes.
   */
  std::vector<std::string> GetRelativePaths(std::string& pathname);
  std::vector<std::string> GetRelativePathsHelper(OptionsHierarchy& node);

  /**
   * Will return the node associated with a pathname.
   *
   * @param pathname The full pathname of the node, e.g. foo/bar in foo/bar.
   *
   * @return Pointer to the node with that pathname, null if not found.
   */
  OptionsHierarchy* FindNode(std::string& pathname);
  OptionsHierarchy* FindNodeHelper(std::string& pathname, std::string& target);

  /* Print functions */
  //! Prints a single node, and outlines relations.
  void Print();

  //! Prints all nodes, plus their data.
  void PrintAll();
  //! Prints a node and its data.
  void PrintNode();

  //! Prints all nodes, plus their description.
  void PrintAllHelp();
  //! Prints a node and its description.
  void PrintNodeHelp();

  //! Prints the leaves of a node.
  void PrintLeaves();
  //! Prints the timers of a node.
  void PrintTimers();
  //! Prints the branches of a node.
  void PrintBranches();

  /**
   * Hyphenate a string or split it onto multiple 80-character lines, with some
   * amount of padding on each line.  This is used for option output.
   *
   * @param str String to hyphenate (splits are on ' ').
   * @param padding Amount of padding on the left for each new line.
   *
   * @return Hyphenated string with newlines.
   */
  static std::string HyphenateString(std::string str, int padding);
};

}; // namespace io
}; // namespace mlpack

#endif
