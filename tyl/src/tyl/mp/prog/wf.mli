(* Section 4.3 *)

open Ast

val isType : typ -> bool
val isOfType : Ctxt.t -> expr -> typ -> bool
val isProp : Ctxt.t -> prop -> bool
val isMP : prog -> bool
