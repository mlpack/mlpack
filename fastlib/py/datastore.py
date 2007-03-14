import util

import unittest
import os
import StringIO
import operator
import copy
import sys

class SexpressionStream:
  """Stream for tokenizing an s-expressions."""
  def __init__(self, file):
    self.file = file
    self.nextchar()
  def nextchar(self):
    self.peek = self.file.read(1)
    return self.peek
  def skip(self):
    while True:
      if self.peek == ';':
        while self.peek != "" and self.peek != "\n" and self.peek != "\r":
          self.nextchar()
      elif self.peek.isspace():
        self.nextchar()
      else:
        break
  def readstr(self):
    self.skip()
    name = ""
    while self.peek.isalnum() or self.peek == "_":
      name += self.peek
      self.nextchar()
    return util.unescape_sexpression(name)
  def is_done(self):
    return self.peek == ""

class UnknownField:
  def __init__(self, message):
    self.message = message
  def __repr__(self):
    return message

class DataNode:
  """Abstract hierarchial key-value data stucture.
  
  This is intended more for small data sets, like the output
  of a single C program, where the overhead of any more complex
  outut mechanism would be a ridiculous annoyance.
  
  Thinking of the Windows registry should convince you not to use this
  for large files.
  """
  def __init__(self, val = None):
    """Create an empty data store.
    """
    self.subnodes = {}
    self.val = val

  def get_node(self, path_elements):
    """Gets an node via a particular path sequence.
    """
    if len(path_elements) == 0:
      return self
    else:
      return self.subnodes[path_elements[0]].get_node(path_elements[1:])

  def get_node_create(self, path_elements):
    """Gets an node via a particular path sequence.
    """
    if len(path_elements) != 0:
      return self.get_subnode_create(path_elements[0]) \
          .get_node_create(path_elements[1:])
    else:
      return self
  
  def get_subnode_create(self, first):
    """Gets a sub-node, creating one if it doesn't exist.
    """
    if not first in self.subnodes.keys():
      node = DataNode()
      self.subnodes[first] = node
    else:
      node = self.subnodes[first]
    return node

  def get_val(self, path_elements):
    """Get a string leaf node by its path sequence.
    
    Returns None if no such element was found.
    """
    try:
      return self.get_node(path_elements).val
    except:
      return None

  def get_val_path(self, path):
    """Get a string leaf node by its escaped-and-slash-separated path.
    """
    try:
      return self.get_node(util.split_path(path)).val
    except:
      return None

  def set_val(self, path_elements, val):
    """Sets a leaf string by a particular sequence of path elements.
    """
    self.get_node_create(path_elements).val = val

  def set_val_path(self, path, val):
    """Sets a leaf string by a particular escaped-and-slash-separated
    path.
    """
    self.get_node_create(util.split_path(path)).val = val

  def __write_pathstyle_impl(self, file, prefix):
    """Implementation method for writing in path-style output format.
    """
    if self.val != None:
      file.write(prefix + " " + util.escape_dspath(self.val) + "\n")
    for k, v in self.subnodes.iteritems():
      v.__write_pathstyle_impl(file, prefix + "/" + util.escape_dspath(k))

  def __write_sexpression_impl(self, file, prefix):
    """Implementation method for writing in sexpression-style output format.
    """
    if self.val != None:
      file.write(util.escape_sexpression(self.val))
    for k, v in self.subnodes.iteritems():
      file.write("\n" + prefix + "(" + util.escape_sexpression(k) + " ")
      v.__write_sexpression_impl(file, prefix + "  ")
      file.write(")")

  def write_pathstyle(self, file, prefix = ""):
    """Writes the data structure to the output in path-style format,
    where every line is a data node, where the left side of the line
    is an escaped and slash-separated path and the right side is a
    string.
    """
    self.__write_pathstyle_impl(file, prefix)

  def write_sexpression(self, file, prefix = ""):
    """Writes the data structure to the output in sexpression format.
    """
    file.write("; Fx s-expression writer version 1.0")
    self.__write_sexpression_impl(file, prefix)
    file.write("\n")

  def read_pathstyle(self, file):
    """Read a flat-file in key-value format.
    """
    for line in file:
      try:
        parts = line.strip().split(" ")
        self.set_val_path(parts[0], util.unescape_dspath(parts[1]))
      except:
        print "Warning: Line ignored: %s" % (line.strip())

  def read_sexpression(self, file):
    """Read s-expression from a file.
    
    This only supports a subset of s-expressions that correspond to nodes.
    There are two examples that are unsupported:
    
    (foo bar baz) is invalid because 'bar' and 'baz' cannot both be associated
    with the same node.
    
    (foo (bar a) (bar b)) also cannot work, because there cannot be two 'bar'
    nodes associated with foo.
    """
    self.__read_sexpression_impl(SexpressionStream(file))

  def __read_sexpression_impl(self, stream):
    try:
      more_subexpressions = True
      while more_subexpressions:
        stream.skip()
        if stream.peek.isalnum() or stream.peek == '_':
          if self.val != None:
            print "Warning: Value reassigned from [%s]." % self.val
          self.val = stream.readstr()
        elif stream.peek == '(':
          # A sub exprsesion
          stream.nextchar()
          name = stream.readstr()
          if name in self.subnodes.keys():
            print "Warning: Node [%s] is specified twice." % name
          self.get_subnode_create(name).__read_sexpression_impl(stream)
          stream.skip()
          if stream.peek != ')':
            raise util.ParseError
          stream.nextchar()
        elif stream.peek == ')':
          more_subexpressions = False
          # do not gobble
        elif stream.peek == '':
          more_subexpressions = False
        else:
          raise util.ParseError()
    except:
      raise util.ParseError()
  
  def select(self, pathspecs):
    """Selects a natural join of several path specifications.
    
    See util.select() for more information.
    """
    return select(self, self.SELECTFUNCTIONS, pathspecs)
  
  def select_to_csv_file(self, file, pathspecs, header = True):
    """Selects a natural join of several path specifications, and write
    to a CSV file.
    
    See util.select() for more information on how the path specifications
    work.
    """
    elements = self.select(pathspecs)
    if header:
      all = [pathspecs] + elements
    else:
      all = elements
    util.write_csv_file(file, all)

  def select_tree(self, matchtree):
    """Selects based on a match tree.
    """
    return matchtree_walk(self, self.SELECTFUNCTIONS, matchtree)

  def __repr__(self):
    """Converts into debugging s-exp representation.
    """
    stream = StringIO.StringIO()
    self.write_sexpression(stream)
    return stream.getvalue()
  
  def __get_typedval(self):
    """Converts into a typed value by guessing its type.
    """
    try:
      return int(self.val)
    except:
      try:
        return float(self.val)
      except:
        return self.val
  
  typedval=property(__get_typedval)
  
  SELECTFUNCTIONS=(lambda x:x.val,
        lambda x:x.subnodes.iterkeys(),
        lambda x:x!=None, lambda x,y:x.subnodes.get(y, None), None)

def sortify(str):
  """Makes the string suitable for sorting by returning a tuple, where
  the left element is typed and the right element is the string.
  """
  val = str
  try:
    val = float(str)
  except:
    pass
  return (val, str)

def subcolumns(col_nums, rows):
  """Extract certain columns.
  
  example:
  
    col_nums = [0, 2, 1]
    rows = [ ["ted", "likes", "shrimp", "always"],
             ["bill", "eats", "pasta", "sometimes"] ]
  
  returns:
  
    [ ["ted", "shrimp", "likes"],
      ["bill", "pasta", "eats] ]
  """
  return [ [ row[i] for i in col_nums ] for row in rows ]

def group(groupcols, rows, remove_key = False):
  """Groups a matrix by particular columns.
  
  Example:
  
     groupcols = [ 0, 1 ]
     
     rows = [ [ 1, 1, 1, 1 ],
              [ 1, 1, 1, 2 ],
              [ 1, 1, 2, 1 ],
              [ 1, 1, 2, 2 ],
              [ 2, 1, 1, 1 ],
              [ 2, 1, 1, 2 ],
              [ 1, 3, 2, 1 ],
              [ 1, 3, 2, 2 ] ]
  
  returns:
  
     groups={(1, 1): [ [ 1, 1, 1, 1 ],
                       [ 1, 1, 1, 2 ],
                       [ 1, 1, 2, 1 ],
                       [ 1, 1, 2, 2 ] ],
             (2, 1): [ [ 2, 1, 1, 1 ],
                       [ 2, 1, 1, 2 ] ],
             (1, 3): [ [ 1, 3, 2, 1 ],
                       [ 1, 3, 2, 2 ] ] }
  
  Note this should work for dictionaries too; if rows is a collection of
  dictionaries, and groupcols corresponds to dictionary keys.
  """
  groups = {}
  
  for row in rows:
    key = tuple([row[x] for x in groupcols])
    if not key in groups.keys():
      groups[key] = []
    if remove_key:
      newrow = [row[i] for i in util.keys(row) if i not in groupcols]
    else:
      newrow = row
    groups[key].append(newrow)
  
  return groups

def groupreduce(groups, reductionpairs):
  """Takes the result of group, and reduces every specified element.
  
  Example:
  
     groups= { (1, 1): [ [ 1, 1, 1, 1 ],
                         [ 1, 1, 1, 2 ],
                         [ 1, 1, 2, 1 ],
                         [ 1, 1, 2, 2 ] ],
               (2, 1): [ [ 2, 1, 1, 1 ],
                         [ 2, 1, 1, 2 ] ],
               (1, 3): [ [ 1, 3, 2, 1 ],
                         [ 1, 3, 2, 2 ] ] }
     
     reductionpairs = [ (2, operator.add), (3, operator.mul) ]
  
  Returns:
  
     reduced = { (1, 1) : [1, 1, 6, 4],
                 (2, 1) : [2, 1, 2, 2],
                 (1, 3) : [1, 3, 4, 2] }
  
  In the "reduced" array, if a column lacks a reduction operator, an
  its value from an arbitrary row will be chosen.
  """
  reduced = {}
  
  for (key, rows) in groups.items():
    newrow = copy.copy(rows[0]) # copy the row
    for row in rows[1:]:
      for (index, reduction) in reductionpairs:
        newrow[index] = reduction(newrow[index], row[index])
    reduced[key] = newrow
  
  return reduced  

def groupsum(groups, indices = [0]):
  """Sums all specified columns of a grouping.
  """
  reductionpairs = []
  for index in indices:
    reductionpairs.append((index, operator.sum))
  return groupreduce(groups, reductionpairs)

def select(place, (handlefn, listfn, existfn, divefn, bridgefn), pathspecs):
  """Yields the desired data on a natural join of pathspecs, sorted.
  
  The pathspecs are a slash-separated list of pathnames as you would expect
  to be stored in a DataNode tree.  Path elements may contain a "*" in
  which case the sub-elements are listed and each traversed.
  
  If two different pathspecs contain a "*", the rules are interesting.
  If both pathspecs are rooted at the same place up to the "*", the two
  pathspecs are traversed together.  Otherwise, a cartesian product is
  done between the two.  Example:
  
  Say our structure looks like:
  
     /runs/1/x = 1x
     /runs/1/y = 1y
     /runs/2/x = 1x
     /runs/2/y = 1y
     /bill/a = a
     /bill/b = b
     /constant = q
  
  ['/q', '/runs/*/x'] repeats the uninteresting one:
    ['q','1x], ['q','2x']
  
  ['/runs/*/x', '/runs/*/y'] yields related pairs:
    ['1x','1y'] , ['2x','2y']
  
  ['/runs/*/x', '/bill/*'] yields all combinations:
    ['1x','a'], ['1x','b'], ['1y','a'], ['1y','b']
  
  ['/runs/*/x', '/runs/*/y', '/bill/*'] yields:
    ['1x','1y','a'], ['1x','1y','b'], ['2x','2y','a'], ['2x','2y','b']
  
  The resulting data is then sorted first by the first column, then the
  second, and so on.  Values that look like numbers are sorted in numerical
  order.
  
  The long tuple of functions are the information necessary for iteration
  (you likely won't have to specify these yourself):
  
  - place: the starting 'place', such as a directory name
  - listfn: a path to list sub-'place's of a place, such as os.listdir
  - existfn: to test the existence of a sub-field like os.path.exists
  - divefn: to take a sub-path and add on a new element, such as os.path.join
  - bridgefn: more complicated; when a ':' appears at the end of an element,
  the divefn is called to get the new 'place' to go to and a new tuple of
  relevant functions.  for example, in select_fields, the ':' operator
  parses the specified file as a DataNode pathstyle file, and plunges into
  the DataNode hierarchy
  """
  matchtree = DataNode()
  for element in pathspecs:
    matchtree.set_val_path(element, element)
  results = matchtree_walk(place,
    (handlefn, listfn, existfn, divefn, bridgefn), matchtree)
  def tolist(map):
    return [ sortify(map.get(x, "")) for x in pathspecs ]
  lines = [ tolist(x) for x in results ]
  lines.sort()
  def fix(line):
    return [ x[1] for x in line ]
  return [ fix(line) for line in lines ]

def matchtree_walk(place, (handlefn, listfn, existfn, divefn, bridgefn), matchtree):
  """Implements the natural join over an arbitrary hierarchial data backend.
  
  See util.select() for information on the behavior.
  """
  functions = (handlefn, listfn, existfn, divefn, bridgefn)
  def combine(subanswers):
    newanswers = []
    for answer1 in answers:
      for answer2 in subanswers:
        elem = {}
        elem.update(answer1)
        elem.update(answer2)
        newanswers.append(elem)
    return newanswers
  myanswer = {}
  if matchtree.val != None:
    myanswer[matchtree.val] = handlefn(place)
  answers = [myanswer]
  for (matchname, submatchtree) in matchtree.subnodes.iteritems():
    if matchname[-1] == ':':
      processfn = bridgefn
      matchname = matchname[:-1]
    else:
      processfn = lambda x:(x, functions)
    def subwalkfn((subplace, subfunctions), submatchtree):
      return matchtree_walk(subplace, subfunctions, submatchtree)
    if matchname == util.STAR:
      subanswers = []
      for subname in listfn(place):
        subplace = divefn(place, subname)
        returned = subwalkfn(processfn(subplace), submatchtree)
        if returned != [{}]: # special case: directory that completely failed
          subanswers += returned
      answers = combine(subanswers)
    else:
      subplace = divefn(place, matchname)
      if existfn(subplace):
        answers = combine(subwalkfn(processfn(subplace), submatchtree))
      else:
        pass # unknown directories are ignored
  return answers

def select_fields(rootpath, pathspecs):
  """Selects first from files, and then from their datanode-pathstyle
  contents.
  
  Pathspecs are in the format:
  
    runs/*/FILENAME:/info/params/x
    runs/*/FILENAME:/info/timers/*/cycles
  
  Be careful: Both "FILENAME/:/" and "FILENAME/:info" will choke horribly.
  Be very careful.
  """
  treefunctions = DataNode.SELECTFUNCTIONS
  def bridge(filename):
    f = open(filename, "r")
    try:
      result = DataNode()
      result.read_pathstyle(f)
      return (result, treefunctions)
    finally:
      f.close()
  pathfunctions = (lambda x:x, os.listdir,
    os.path.exists, os.path.join, bridge)
  return select(rootpath, pathfunctions, pathspecs)

def select_fields_to_table_file(openfile, rootpath, pathspecs, writefn = util.write_csv_file):
  """Selects fields first from files and then from the pathstyle contents,
  writing results to an OPEN csv or other type of text file file.
  
  See util.select_fields().
  """
  headers = [x.strip("/").split("/")[-1] for x in pathspecs]
  writefn(openfile, [headers] + select_fields(rootpath, pathspecs))

def select_fields_to_csv_file(csvfile, rootpath, pathspecs):
  """Selects fields first from files and then from the pathstyle contents,
  writing results to an OPEN csv file.
  
  See util.select_fields().
  """
  select_fields_to_table_file(csvfile, rootpath, pathspecs, util.write_csv_file)

def select_fields_to_csv(csvfname, rootpath, pathspecs):
  """Selects fields first from files and then from the pathstyle contents,
  writing results to a to-be-created csv file name.
  
  See util.select_fields().
  """
  f = open(csvfname, "w")
  try:
    select_fields_to_csv_file(f, rootpath, pathspecs)
  finally:
    f.close()

def select_files(place, pathspecs):
  """Selects filenames in the natural-join style.
  
  See util.select() for info.
  """
  functions = (lambda x:x, os.listdir, os.path.exists, os.path.join, None)
  return select(place, functions, pathspecs)

def match_files(place, pathspec):
  """Match files based on a single path specification.
  
  For instance runs/*/SUMMARY matches files named SUMMARY in any
  directory called runs.
  """
  return [ x[0] for x in select_files(place, [pathspec]) ]

#----------------------------------------------------------------------------
# Unit tests

class IoTest(unittest.TestCase):
  def setUp(self):
    (tmpdir, tmpfile) = os.path.split(os.tempnam())
    self.tmpdir = "%s/fxsys-util-ut-%s" % (tmpdir, tmpfile)
    self.rootpath = self.tmpdir
    os.makedirs(self.rootpath)
  
  def tearDown(self):
    util.remove_dir_recursive(self.tmpdir)
  
  def test_match_files(self):
    def fix(x):
      return os.path.join(self.rootpath, x)
    matchdirs = ["foo/a/bar/a/baz", "foo/a/bar/b/baz", "foo/a/bar/c/baz",
            "foo/b/bar/a/baz", "foo/b/bar/c/baz",
            "foo/c/bar/b/baz"]
    nonmatchdirs = ["foon/a/bar/a/baz", "foo/g/bar/b/bazz", "foo/a/x", "subsume"]
    for dir in matchdirs + nonmatchdirs:
      os.makedirs(os.path.join(self.rootpath, dir))
    found = match_files(self.rootpath, "foo/*/bar/*/baz")
    found.sort()
    matchdirs.sort()
    matchdirs = [ fix(x) for x in found ]
    self.assertEqual(matchdirs, found)
    
  def test_select_files(self):
    def fix(x):
      return os.path.join(self.rootpath, x)
    leftdirs = ["foo/a/bar/a/baz", "foo/a/bar/b/baz", "foo/a/bar/c/baz",
            "foo/b/bar/a/baz", "foo/b/bar/c/baz",
            "foo/c/bar/b/baz"]
    middirs = ["foo/a/bar/a/bon", "foo/a/bar/b/bon", "foo/a/bar/c/bon",
            "foo/b/bar/a/bon", "foo/b/bar/c/bon",
            "foo/c/bar/b/bon"]
    rightdirs = ["param/a/foo", "param/b/foo", "param/c/foo"]
    nondirs = ["foon/a/bar/a/baz", "foo/g/bar/b/bazz", "foo/a/x", "subsume"]
    for dir in leftdirs + middirs + rightdirs + nondirs:
      os.makedirs(os.path.join(self.rootpath, dir))
    found = select_files(self.rootpath,
        ["foo/*/bar/*/baz", "foo/*/bar/*/bon", "param/*/foo"])
    found.sort()
    expect = []
    for (leftdir, middir) in zip(leftdirs, middirs):
      for rightdir in rightdirs:
        expect.append([fix(leftdir), fix(middir), fix(rightdir)])
    expect.sort()
    found.sort()
    self.assertEqual(expect, found)
    
    

class DatastoreTest(unittest.TestCase):
  def setUp(self):
    self.f = DataNode()
    self.f.get_node_create(["walks", "10", "param", "speed"]).val = "SLOW"
    self.f.get_node_create(["runs", "10", "param"]).val = "lots of params"
    self.f.get_node_create(["runs", "10", "param", "x"]).val = "2"
    self.f.set_val(["runs", "10", "param", "y"], "3")
    self.f.set_val_path("runs/20/param/x", "4")
    self.f.set_val_path("/runs/20/param/y", "5")
    self.f.set_val(["runs", "10", "metric", "total performance"], "very good")
    self.f.set_val_path("/runs/30/param/x", "8");
    self.f.set_val_path("/runs/30/param/y", "9");
    self.f.set_val_path("/runs/40/param/x", "11");
  
  def test_get_val(self):
    self.assertEqual(self.f.get_val(["runs", "10", "param", "x"]), "2");
    self.assertEqual(self.f.get_val(["runs", "10", "param", "y"]), "3");
    self.assertEqual(self.f.get_val(["runs", "20", "param", "x"]), "4");
    self.assertEqual(self.f.get_val(["runs", "20", "param", "y"]), "5");

  def test_get_val_path(self):
    self.assertEqual(self.f.get_val_path("runs/20/param/y"), "5");
    self.assertEqual(self.f.get_val_path("/runs/20/param/y"), "5");
    self.assertEqual(self.f.get_val_path("/runs/10/metric/total%20performance"), "very good");
  
  def help_test_save_get(self, readfun, writefun):
    print "(START)"
    first_outstream = StringIO.StringIO()
    writefun(self.f, first_outstream)
    print first_outstream.getvalue()
    print "(REREAD)"
    f2 = DataNode()
    readfun(f2, StringIO.StringIO(first_outstream.getvalue()))
    second_outstream = StringIO.StringIO()
    writefun(f2, second_outstream)
    print second_outstream.getvalue()
    print "(END)"
    self.assertEqual(first_outstream.getvalue(), second_outstream.getvalue());
  
  def test_pathstyle_save(self):
    self.help_test_save_get(DataNode.read_pathstyle, DataNode.write_pathstyle)
  
  def test_sexpression(self):
    self.help_test_save_get(DataNode.read_sexpression, DataNode.write_sexpression)
  
  def test_select(self):
    selected = self.f.select(["runs/*/param/x", "runs/*/param/y"])
    expected = [
        ['2', '3'],
        ['4', '5'],
        ['8', '9'],
        ['11', '']]
    selected.sort()
    expected.sort()
    self.assertEqual(expected, selected)
  
  def test_group(self):
    groupcols = [ 0, 1 ]
    rows = [ [ 1, 1, 1, 1 ],
             [ 1, 1, 1, 2 ],
             [ 1, 1, 2, 1 ],
             [ 1, 1, 2, 2 ],
             [ 2, 1, 1, 1 ],
             [ 2, 1, 1, 2 ],
             [ 1, 3, 2, 1 ],
             [ 1, 3, 2, 2 ] ]
    expect = { (1, 1): [ [ 1, 1, 1, 1 ],
                      [ 1, 1, 1, 2 ],
                      [ 1, 1, 2, 1 ],
                      [ 1, 1, 2, 2 ] ],
            (2, 1): [ [ 2, 1, 1, 1 ],
                      [ 2, 1, 1, 2 ] ],
            (1, 3): [ [ 1, 3, 2, 1 ],
                      [ 1, 3, 2, 2 ] ] }
    self.assertEqual(expect, group(groupcols, rows))
  
  def test_group_nonint(self):
    k_a = "a"
    k_b = (3.2, "food")
    k_c = "hamburger"
    groupcols = [ k_a, k_b ]
    rows = [ { k_a:1, k_b:1, k_c:1, "d":1 },
             { k_a:1, k_b:1, k_c:1, "d":2 },
             { k_a:1, k_b:1, k_c:2, "d":1 },
             { k_a:1, k_b:1, k_c:2, "d":2 },
             { k_a:2, k_b:1, k_c:1, "d":1 },
             { k_a:2, k_b:1, k_c:1, "d":2 },
             { k_a:1, k_b:3, k_c:2, "d":1 },
             { k_a:1, k_b:3, k_c:2, "d":2 } ]
    expect = { (1, 1): [ { k_a:1, k_b:1, k_c:1, "d":1 },
                         { k_a:1, k_b:1, k_c:1, "d":2 },
                         { k_a:1, k_b:1, k_c:2, "d":1 },
                         { k_a:1, k_b:1, k_c:2, "d":2 } ],
               (2, 1): [ { k_a:2, k_b:1, k_c:1, "d":1 },
                         { k_a:2, k_b:1, k_c:1, "d":2 } ],
               (1, 3): [ { k_a:1, k_b:3, k_c:2, "d":1 },
                         { k_a:1, k_b:3, k_c:2, "d":2 } ] }
    self.assertEqual(expect, group(groupcols, rows))
  
  def test_subcolumns(self):
    col_nums = [0, 2, 1]
    rows = [ ["ted", "likes", "shrimp", "always"],
             ["bill", "eats", "pasta", "sometimes"] ]
    expect = [ ["ted", "shrimp", "likes"],
      ["bill", "pasta", "eats"] ]
    self.assertEqual(expect, subcolumns(col_nums, rows))
  
  def test_groupreduce(self):
    groups= { (1, 1): [ [ 1, 1, 1, 1 ],
                        [ 1, 1, 1, 2 ],
                        [ 1, 1, 2, 1 ],
                        [ 1, 1, 2, 2 ] ],
              (2, 1): [ [ 2, 1, 1, 1 ],
                        [ 2, 1, 1, 2 ] ],
              (1, 3): [ [ 1, 3, 2, 1 ],
                        [ 1, 3, 2, 2 ] ] }
    reductionpairs = [ (2, operator.add), (3, operator.mul) ]
    reduced = { (1, 1) : [1, 1, 6, 4],
                (2, 1) : [2, 1, 2, 2],
                (1, 3) : [1, 3, 4, 2] }
    self.assertEqual(reduced, groupreduce(groups, reductionpairs))
  
  def test_csv_file_head(self):
    stream = StringIO.StringIO()
    self.f.select_to_csv_file(stream, ["runs/*/param/x", "runs/*/param/y"])
    expected = "runs/*/param/x, runs/*/param/y\n2, 3\n8, 9\n4, 5\n11, \n";
    self.assertEqual(util.sorted(expected.split("\n")), util.sorted(stream.getvalue().split("\n")))
  
  def test_csv_file_nohead(self):
    stream = StringIO.StringIO()
    self.f.select_to_csv_file(stream, ["runs/*/param/x", "runs/*/param/y"], False)
    expected = "2, 3\n8, 9\n4, 5\n11, \n";
    self.assertEqual(util.sorted(expected.split("\n")), util.sorted(stream.getvalue().split("\n")))

if __name__ == "__main__":
  unittest.main()
  test_matching()
