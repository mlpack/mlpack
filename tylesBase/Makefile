LIB_PKGS=extlib #findlib package name
LIB_NAMES=extLib
LIBS=$(patsubst %,%.cma,$(LIB_NAMES))
DIRS=src

# Given a library package name, return the directory in which the library exists
lib_inc = -I $(shell ocamlfind query -format "%d" $(1))

# Given a directory, return list of cmo files in order in which they should be loaded
get_deps = $(patsubst $(1)/%,%,$(shell ocamldsort -I $(1) -byte $(1)/*.ml))

INCLUDE=$(foreach lib,$(LIB_PKGS),$(call lib_inc,$(lib))) $(patsubst %,-I _build/%,$(DIRS))


.PHONY: all doc clean

all:
	ocamlbuild src/tylesBase.cmo
	ocamlbuild src/tylesBase.cma

doc: src/*.mli
	rm -rf html
	mkdir html
	ocamldoc $(INCLUDE) -I _build/src -d html -no-stop -html $^

clean:
	ocamlbuild -clean
	rm -rf html
