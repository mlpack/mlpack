(* Section 4.3 *)

open Ast

val isType : typ -> bool
val isOfType : Ctxt.t -> expr -> typ -> bool
val isProp : Ctxt.t -> prop -> bool
val isMP : prog -> bool

val isLiteral : expr -> bool
val isDLF : expr -> bool
val isCNF : expr -> bool
val isConj : expr -> bool
val toCNF : expr -> expr

val bounded : typ -> bool
val existVarsBounded : prop -> bool
val disjVarsBounded : Ctxt.t -> prop -> bool
