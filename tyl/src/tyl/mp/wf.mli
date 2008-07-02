open Ast

val isType : typ -> bool
val isOfType : context -> expr -> typ -> bool
val isProp : context -> prop -> bool
val isMP : prog -> bool

val bounded : typ -> bool
val existVarsBounded : prop -> bool
val disjVarsBounded : context -> prop -> bool
