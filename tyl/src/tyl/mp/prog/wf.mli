(* Section 4.3 *)

open Ast

val wfType : typ -> bool
val isOfType : Ctxt.t -> expr -> typ -> bool
val wfProp : Ctxt.t -> prop -> bool
val wfProg : prog -> bool
