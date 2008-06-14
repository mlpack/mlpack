(* Section 4.3 *)

open Ast

val wfType : typ -> bool
val isOfType : context -> expr -> typ -> bool
val wfProp : context -> prop -> bool
val wfProg : prog -> bool
