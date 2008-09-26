LIB_PKGS=extlib #findlib package name
LIB_NAMES=extLib
LIBS=$(patsubst %,%.cma,$(LIB_NAMES))
DIRS=src

# Given a library package name, return the directory in which the library exists
lib_inc = -I $(shell ocamlfind query -format "%d" $(1))

# Given a directory, return list of cmo files in order in which they should be loaded
get_deps = $(patsubst $(1)/%,%,$(shell ocamldsort -I $(1) -byte $(1)/*.ml))

INCLUDE=$(foreach lib,$(LIB_PKGS),$(call lib_inc,$(lib))) $(patsubst %,-I _build/%,$(DIRS)) -I _build

.PHONY: all doc clean dist

all:
	ocamlbuild src/tylesBase.cma
	ocamlbuild src/tylesBase.cmxa

doc/html: src/*.mli
	rm -rf $@
	mkdir -p $@
	ocamldoc $(INCLUDE) -d $@ -no-stop -html $^

doc: doc/html

tylesBase.zip: clean
	cd ..; zip -yr tylesBase/$@ tylesBase

dist: tylesBase.zip

clean:
	ocamlbuild -clean
	rm -rf doc/html
	rm -f tylesBase.zip
