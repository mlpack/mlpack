(** Intended usage is to do "open TylesBase", which replaces several third-party modules with modified versions, and provides some new modules. *)

module Array = Array2
module Char = Char2
module DynArray = DynArray2
module List = List2
module Map = Map2
module PMap = PMap2
module Set = Set2
module Stream = Stream2
module String = String2

module Msg = Msg
module Pos = Pos
module Tuple = Tuple

include Pervasives2
