open Ast

val isType : typ -> bool
val isOfType : Context.t -> expr -> typ -> bool
val isProp : Context.t -> prop -> bool
val isMP : prog -> bool

val bounded : typ -> bool
val existVarsBounded : prop -> bool
val disjVarsBounded : Context.t -> prop -> bool
