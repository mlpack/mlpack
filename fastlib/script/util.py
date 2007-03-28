import random
import re
import os
import sys
import StringIO
import unittest

if sys.hexversion < 0x020200F0:
  print "*" * 70
  print "*" * 70
  print "   Your Python version is too old.  2.2 is required at minimum."
  print "   Proceeding, but it will probably not work!"
  print "*" * 70
  print "*" * 70

# TODO: this isn't our own work, I don't remember where it was pulled from
def natsort_key(item):
  chunks = re.split('(\d+(?:\.\d+)?)', item)
  for ii in range(len(chunks)):
    if chunks[ii] and chunks[ii][0] in '0123456789':
      if '.' in chunks[ii]:
        numtype = float
      else:
        numtype = int
        chunks[ii] = (0, numtype(chunks[ii]))
    else:
      chunks[ii] = (1, chunks[ii])
  return (chunks, item)

def natsort(seq):
  l = list(seq)
  l.sort(key=natsort_key)
  return l

def sorted(l):
  """Returns the sorted version of the list.
  
  (Remove once everyone is running Python 2.4+)
  """
  l_copy = list(l)
  l_copy.sort()
  return l_copy

def map_values(f, d):
  """Returns a copy of the dictionary but with all the values mapped."""
  return dict([(k, f(v)) for (k, v) in d.items()])

def map_keys(f, d):
  """Returns a copy of the dictionary but with all the values mapped."""
  return dict([(f(k), v) for (k, v) in d.items()])

def dicthash(d):
  """Turns a dictionary into something hashable."""
  return tuple(sorted(d.items()))

def collapse_once(collection_of_collection):
  """Turns a list of lists into just the items."""
  result = []
  for collection in collection_of_collection:
    result += collection
  return result

def remove_dir_recursive(dirname):
  """Removes a directory like rm -rf."""
  for subname in os.listdir(dirname):
    name = os.path.join(dirname, subname)
    if os.path.isdir(name):
      remove_dir_recursive(name)
    else:
      os.remove(name)
  os.rmdir(dirname)


def testfile(filename):
  """Tests if a file exists.
  """
  return os.access(filename, os.F_OK)

def createlock(filename):
  """Tries to lock a file for writing.
  
  Currently this is a stub and just tests for existence, but there is no
  real locking semantics.
  """
  # TODO: Mode operation
  if testfile(filename):
    return False
  try:
    fname = os.open(filename, os.O_CREAT|os.O_EXCL, 0660)
    os.close(fname)
    return True
  except OSError:
    return False
    

def writefile(filename, text):
  """Writes the text to a file by name.
  """
  f = open(filename, "w")
  try:
    f.write(text)
  finally:
    f.close()

def readfile(filename):
  """Reads the text from a file.
  """
  f = open(filename, "r")
  try:
    text = f.read()
  finally:
    f.close()
  return text

def writelines(filename, lines):
  """Writes each line to the specified file.
  
  The Unix newline character will be appended to each line.
  """
  # TODO: Unix versus Dos CR/LV
  f = open(filename, "w")
  try:
    f.writelines(["%s\n" % line for line in lines])
  finally:
    f.close()
  return lines

def readlines(filename):
  """Reads each line from a file to a list, with all whitespace
  stripped from the end of each line.
  """
  f = open(filename, "r")
  try:
    lines = f.readlines()
  finally:
    f.close()
  lines = [ l.rstrip() for l in lines ]
  return lines

def read_csv(fname):
  """Reads a comma-separated-value file as a matrix.
  """
  f = open(fname, "r")
  try:
    return read_csv_file(f)
  finally:
    f.close()

def read_csv_file(f):
  """Reads an open comma-separated-value file as a matrix.
  """
  return [[s.strip() for s in l.split(",")] for l in f.readlines()]

def write_csv(fname, lines):
  """Writes the specified matrix as a comma-separated-value file.
  """
  f = open(fname, "w")
  try:
    write_csv_file(f, lines)
  finally:
    f.close()

def write_csv_file(f, lines):
  """Writes the specified matrix as a comma-separated-value open file.
  """
  for line in lines:
    sanitized = [ str(field).replace(",", ";") for field in line ]
    f.write(", ".join(sanitized) + "\n")

def escape_latex(str):
  result = ""
  for c in str:
    if c == '\\':
      result += "$\\backslash$"
    elif c in "#%&~$_^{}":
      result += "\\" + c
    else:
      result += c
  return result

def write_latex_table_file(f, lines, align = "r"):
  """Writes specified text to a file as a latex table.
  
  The first line is assumed to be the column headings.
  """
  def formatline(line):
    return " & ".join([escape_latex(field) for field in line]) + " \\\\\n"
  
  max_width = max([len(line) for line in lines])
  
  # ensure len(align) equals max_width
  while len(align) < max_width:
    align += align[-1]
  align = align[0:max_width]
  
  f.write("\\documentclass[letter]{article}\n")
  f.write("\\begin{document}\n")
  f.write("\\begin{tabular}{|%s|}\n" % ("|".join(align)))
  f.write("\\hline\n")
  f.write(formatline(lines[0]))
  f.write("\\hline\n")
  for line in lines[1:]:
    f.write(formatline(line))
  f.write("\\hline\n")
  f.write("\\end{tabular}\n\n")
  f.write("\\end{document}\n")

def write_random_ints(filename, count):
  """Writes a sequence of random integers to a file.
  """
  nums = [ "%d" % random.randint(0, 99999999) for i in range(count) ]
  writelines(filename, nums)

def shellquote(s):
  """Quotes shell parameters.
  
  Note that things like newlines and special characters are included literally
  in the string, compliant with BASH.
  
  Perhaps another shellquote function should be written, which uses the
  $"string" format, that allows C-like escaping.
  """
  result = ""
  map = {"$":"\\$", "\"":"\\\"", "`":"\\`", "!":"\\!"}
  changed = False
  for c in str(s):
    if c in map.keys():
      result += map[c]
      changed = True
    elif c.isalnum() or c == '/':
      result += c
    else:
      result += c
      changed = True
  if changed:
    return "\"%s\"" % result
  else:
    return result

def sanitize_basename(s, replacechar = "_"):
  """Sanitizes the base of a filename.
  
  This is used in the naming of runs.
  
  replacechar: string, the character to replace bad characters with
  """
  result = ""
  allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_0123456789."
  replaceable = "./="
  for c in str(s):
    if c in allowed:
      result += c
    elif c in replaceable:
      result += replacechar
  return result

def ensuredir(dirpath):
  """Ensure that a directory exists.
  """
  if (not os.access(dirpath, os.R_OK|os.W_OK)):
    os.makedirs(dirpath)

def spawn_redirect(rundir, args, infile = None, outfile = None, errfile = None):
  """Spawn a new process, optionally redirecting one of standard in, standard
  out, or standard error to the specified filenames.
  
  - rundir: the directory to change to when running the program
  - args: the arguments, where args[0] is the file to run
  - infile: filename of standard in or None
  - outfile: filename of standard out or None
  - errfile: filename of standard error or None
  """
  pid = os.fork()
  if not pid:
    try:
      if rundir != None:
        os.chdir(rundir)
      if not infile:
        infile = "/dev/null"
      if infile != None:
        infd = os.open(infile, os.O_RDONLY)
        os.close(0)
        os.dup2(infd, 0)
        os.close(infd)
      else:
        os.close(0)
      if outfile != None:
        outfd = os.open(outfile, os.O_WRONLY|os.O_CREAT|os.O_TRUNC)
        os.close(1)
        os.dup2(outfd, 1)
        os.close(outfd)
      else:
        outfd = None
        #os.close(1)
      if errfile != None:
        # TODO: Mode 755?
        errfd = os.open(errfile, os.O_WRONLY|os.O_CREAT|os.O_APPEND)
        os.close(2)
        os.dup2(errfd, 2)
        os.close(errfd)
      else:
        errfd = None
        #os.close(2) -- Show standard error
      os.execvp(args[0], args)
    finally:
      # WALDO
      try:
        if infile:
          os.close(0)
      except:
        pass
      try:
        if outfile:
          os.close(1)
      except:
        pass
      try:
        if errfile:
          os.close(2)
      except:
        pass
      os._exit(1)
  (p, status) = os.waitpid(pid, 0)
  # Status is a 16-bit number.
  # The right 8 bits of status is the signal killed (always zero in case
  # of Windows).
  # The left eight bits are set to the exit value.
  if status & 0xFF != 0:
    return -1 # It was killed -- the right eight bits are non-zero
  else:
    return status >> 8

def shell(args, infile = None, outfile = None, errfile = None):
  """Executes either a list of arguments (argv[0] being the command)
  or a string command.
  """
  if isinstance(args, str):
    args = args.split(" ")
  result = spawn_redirect(None, args, infile, outfile, errfile)
  if result != 0:
    raise OSError("error executing command: " + " ".join(args))

class ParseError:
  """Parse error when unqouting.
  """
  pass


def __dspath_allowed_compute():
  global __dspath_allowed
  __dspath_allowed = {}
  l = range(ord('A'), ord('Z'))
  l += range(ord('a'), ord('z'))
  l += range(ord('0'), ord('9'))
  l += [ord('.'), ord('-'), ord('_')]
  for c in l:
    __dspath_allowed[chr(c)] = True

__dspath_allowed_compute()

def escape_dspath(str):
  """Quote method used for DataNode path elements, encoded in HTML-like
  format.
  
  Only alphanumeric characters, the period, plus sign, minus sign, and
  underscore are preserved.  Everything else uses a percent (%) sign and
  a 2-digit hex code.  This is not due to necessity as much as it
  is convenient that these characters won't need to be escaped for instance
  if you type them from a shell, or in a regular expression.
  
  (TODO: Decide if the plus sign should also be escaped, for HTML
  purposes.)
  """
  global __dspath_allowed
  result = ""
  for c in str:
    if c in __dspath_allowed.keys():
      result += c
    else:
      result += "%%%02X" % ord(c)
  return result

def unescape_dspath(s):
  """Decodes DataNode path elements, by interpreting the special
  percent character.
  """
  try:
    result = ""
    hexdigits = "0123456789ABCDEF"
    i = 0
    while True:
      f = s.find("%", i)
      if f == -1:
        result += s[i:]
        break
      result += s[i:f]
      result += chr(hexdigits.index(s[f+1]) * 16 + hexdigits.index(s[f+2]))
      i = f + 3
    return result
  except ParseError:
    raise ParseError()

def combine_path(elements):
  """Combines and escapes path elements.
  """
  result = ""
  for e in elements:
    result += "/" + escape_dspath(e)
  return result

def split_path(path):
  """Splits and unescapes path elements.
  """
  return [unescape_dspath(x) for x in path.strip("/").split("/")]

def escape_sexpression(str):
  """s-expression quoting hack."""
  result = ""
  for c in str:
    if c.isalnum():
      result += c
    else:
      result += "_%02X" % ord(c)
  return result

def unescape_sexpression(s):
  """Decodes sexpression-escaped elements, by
  interpreting the special underscore character.
  """
  try:
    result = ""
    hexdigits = "0123456789ABCDEF"
    i = 0
    while i < len(s):
      c = s[i]
      if c == '_':
        result += chr(hexdigits.index(s[i+1]) * 16 + hexdigits.index(s[i+2]))
        i += 3
      else:
        result += c
        i += 1
    return result
  except:
    raise ParseError()

def keys(collection):
  """Returns the keys of a collection, whether it is an array or dict.
  
  If it is a list, it returns range(len(x)), otherwise x.keys().
  """
  if isinstance(collection, list):
    return range(len(collection))
  else:
    return collection.keys()

def typeconvert_list(items, *type_func_pairs):
  return [typeconvert(item, *type_func_pairs) for item in items]

def typeconvert(item, *type_func_pairs):
  for (type, func) in type_func_pairs:
    if isinstance(item, type):
      item = func(item)
      break
  return item

## Not sure why this code is here.  Will be deleted.
# class Table:
#   def __init__(self, headings, matrix = []):
#     self.headings = headings
#     self.matrix = matrix
#   def lookup(self, column):
#     if not isinstance(column, int):
#       column = self.headings.index(column)
#     return column
#   def restricted_fn(self, column, testfn):
#     column = self.lookup(column)
#     newmatrix = [line for line in newmatrix if testfn(line[column])]
#     return Table(self.headings, newmatrix)
#   def restricted(self, column, allowed):
#     if not isinstance(allowed, list):
#       allowed = [allowed]
#     column = self.lookup(column)
#     newmatrix = [line for line in self.matrix if line[column] in allowed]
#     return Table(self.headings, newmatrix)
# 
# def read_table(fname):
#   """Reads the text from a file.
#   """
#   f = open(fname, "r")
#   try:
#     return read_table_file(f)
#   finally:
#     f.close()
# 
# def read_table_file(f, headings = None):
#   return table_from_matrix(read_csv_file(f), headings)
# 
# def table_from_matrix(matrix, headings = None):
#   if not headings:
#     headings = matrix[0]
#     matrix = matrix[1:]
#   table = Table(headings, matrix)
#   return table


# TODO: Unit tests

class UtilTest(unittest.TestCase):
  def test_typeconvert(self):
    self.assertEqual("1", typeconvert(1, (int, str)))
    self.assertEqual(1, typeconvert(1, (str, int)))
    self.assertEqual([1], typeconvert([1], (int, str)))
  def test_typeconvert_list(self):
    self.assertEqual(["1", "2", "3", "4"],
        typeconvert_list([1, 2, 3, "4"], (int, str)))

if __name__ == "__main__":
  unittest.main()
  test_matching()
