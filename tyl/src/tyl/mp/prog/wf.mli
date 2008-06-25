(* Section 4.3 *)

open Ast

val isType : typ -> bool
val isOfType : Context.t -> expr -> typ -> bool
val isProp : Context.t -> prop -> bool
val isMP : prog -> bool

