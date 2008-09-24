open Ast

val isType : typ -> bool
val isOfType : typ -> context -> expr -> bool
val isProp : context -> prop -> bool
val isMP : prog -> bool

val bounded : typ -> bool
val existVarsBounded : prop -> bool
val disjVarsBounded : context -> prop -> bool
