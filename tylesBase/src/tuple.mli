(** Tuples. Functions for making, extracting elements from, mapping, and (un)currying Pairs, Triples, and Quadruples. Function names are consistent across modules.

    Function documentation:
    - [make] construct a tuple
    - [prjn] project [n]th item from a tuple
    - [prjmn..] project [m]th, [n]th, ... items from a tuple
    - [map] apply functions to every item of a tuple
    - [mapn] apply a function to the [n]th item of a tuple
    - [curry] convert a function taking a tuple into curried form
    - [uncurry] convert a curried function into one taking a tuple
*)

(** Pairs *)
module Pr : sig
  val make : 'a -> 'b -> ('a * 'b)
  val prj1 : ('a * 'b) -> 'a
  val prj2 : ('a * 'b) -> 'b
  val map : ('a -> 'c) -> ('b -> 'd) -> ('a * 'b) -> ('c * 'd)
  val map1 : ('a -> 'c) -> ('a * 'b) -> ('c * 'b)
  val map2 : ('b -> 'c) -> ('a * 'b) -> ('a * 'c)
  val curry : ('a * 'b -> 'c) -> 'a -> 'b -> 'c
  val uncurry : ('a -> 'b -> 'c) -> 'a * 'b -> 'c
end
  
(** Triples *)
module Tr : sig
  val make : 'a -> 'b -> 'c -> ('a * 'b * 'c)
  val prj1 : ('a * 'b * 'c) -> 'a
  val prj2 : ('a * 'b * 'c) -> 'b
  val prj3 : ('a * 'b * 'c) -> 'c
  val prj12 : ('a * 'b * 'c) -> ('a * 'b)
  val prj13 : ('a * 'b * 'c) -> ('a * 'c)
  val prj23 : ('a * 'b * 'c) -> ('b * 'c)
  val map : ('a -> 'd) -> ('b -> 'e) -> ('c -> 'f) -> ('a * 'b * 'c) -> ('d * 'e * 'f)
  val map1 : ('a -> 'd) -> ('a * 'b * 'c) -> ('d * 'b * 'c)
  val map2 : ('b -> 'd) -> ('a * 'b * 'c) -> ('a * 'd * 'c)
  val map3 : ('c -> 'd) -> ('a * 'b * 'c) -> ('a * 'b * 'd)
  val curry : ('a * 'b * 'c -> 'd) -> 'a -> 'b -> 'c -> 'd
  val uncurry : ('a -> 'b -> 'c -> 'd) -> 'a * 'b * 'c -> 'd
end

(** Quadruples *)
module Fr : sig
  val make : 'a -> 'b -> 'c -> 'd -> ('a * 'b * 'c * 'd)
  val prj1 : ('a * 'b * 'c * 'd) -> 'a
  val prj2 : ('a * 'b * 'c * 'd) -> 'b
  val prj3 : ('a * 'b * 'c * 'd) -> 'c
  val prj4 : ('a * 'b * 'c * 'd) -> 'd
  val prj12 : ('a * 'b * 'c * 'd) -> ('a * 'b)
  val prj13 : ('a * 'b * 'c * 'd) -> ('a * 'c)
  val prj14 : ('a * 'b * 'c * 'd) -> ('a * 'd)
  val prj23 : ('a * 'b * 'c * 'd) -> ('b * 'c)
  val prj24 : ('a * 'b * 'c * 'd) -> ('b * 'd)
  val prj34 : ('a * 'b * 'c * 'd) -> ('c * 'd)
  val prj123 : ('a * 'b * 'c * 'd) -> ('a * 'b * 'c)
  val prj124 : ('a * 'b * 'c * 'd) -> ('a * 'b * 'd)
  val prj234 : ('a * 'b * 'c * 'd) -> ('b * 'c * 'd)
  val map : ('a -> 'e) -> ('b -> 'f) -> ('c -> 'g) -> ('d -> 'h) -> ('a * 'b * 'c * 'd) -> ('e * 'f * 'g * 'h)
  val map1 : ('a -> 'e) -> ('a * 'b * 'c * 'd) -> ('e * 'b * 'c * 'd)
  val map2 : ('b -> 'e) -> ('a * 'b * 'c * 'd) -> ('a * 'e * 'c * 'd)
  val map3 : ('c -> 'e) -> ('a * 'b * 'c * 'd) -> ('a * 'b * 'e * 'd)
  val map4 : ('d -> 'e) -> ('a * 'b * 'c * 'd) -> ('a * 'b * 'c * 'e)
  val curry : ('a * 'b * 'c * 'd -> 'e) -> 'a -> 'b -> 'c -> 'd -> 'e
  val uncurry : ('a -> 'b -> 'c -> 'd -> 'e) -> 'a * 'b * 'c * 'd -> 'e
end
