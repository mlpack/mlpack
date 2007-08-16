#!/usr/bin/python

import util
import fx

import os
import sys

source_dir = fx.param_str("source_dir")
genfiles_dir = fx.param_str("genfiles_dir")

base_in = os.path.join(source_dir, "base")
base_out = os.path.join(genfiles_dir, "base")

if not os.access(base_out, os.R_OK+os.W_OK):
  os.makedirs(base_out)

print "Starting config!"
print "... determining architecture"

(kernel, nodename, kernelver, kernelbuild, arch) = os.uname()

if arch == "x86_64":
  print "*** found 64-bit extensions (Opteron style)"
elif arch == "i686":
  print "*** found 32-bit Intel"
else:
  print "!!! We haven't tested this on '%s' systems.  It might not work." % (arch)

print "... determining sizes of basic types"

util.shell(
    ["gcc",
     os.path.join(base_in, "config/template_types.c"),
     "-o",
     os.path.join(base_out, "_template_types")])

util.shell(
    [os.path.join(base_out, "_template_types")],
    outfile = os.path.join(base_out, "basic_types.h"))

print "*** created basic_types.h"

# NOTE: Removed "shares" feature
# highest priority to lowest
#checked_shares = [
#        "/cygdrive/d/nick/Fast_code/share",
#        "/net/hu11/garryb/fastlab/%s" % (arch),
#        "/usr/local"]
#found_shares = [share for share in checked_shares if os.access(share, os.R_OK)]

#if len(found_shares) == 0:
#  print "XXX No suitable share found."
#  print "    Looked in: ", checked_shares
#  sys.exit(1)
#    "LIB=%s" % " ".join(["-L%s/lib" for share in found_shares]),
#    "INC=%s" % " ".join(["-I%s/include" for share in found_shares])

print "... outputting Makefile.inc"

makefile_inc_lines = [
    "FASTLIB=%s" % source_dir,
]

util.writelines(os.path.join(base_in, "Makefile.inc"), makefile_inc_lines)

print "*** done"
