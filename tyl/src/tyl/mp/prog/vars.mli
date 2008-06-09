open Ast
open Util

val freeVarsp : prog -> Id.Set.t

(* val freeVars : category -> syntax -> Id.Set.t *)
(* val freeVarsa : syntax -> Id.Set.t *)
  
(* val isClosed : category -> syntax -> bool *)
(* val isCloseda : syntax -> bool *)

(* val sub : syntax -> Id.t -> syntax -> syntax *)
(* val subee : expr -> Id.t -> expr -> expr *)
(* val subec : expr -> Id.t -> prop -> prop *)
(* (\* val subeit : I.expr -> Id.t -> typ -> typ *\) *)
(* (\* val subeie : I.expr -> Id.t -> expr -> expr *\) *)
(* (\* val subeiz : I.expr -> Id.t -> prop_typ -> prop_typ *\) *)
(* (\* val subeic : I.expr -> Id.t -> prop -> prop *\) *)
(* val subse : syntax -> Id.t -> expr -> expr *)
(* val subst : syntax -> Id.t -> typ -> typ *)
(* val subsc : syntax -> Id.t -> prop -> prop *)
(* val subsp : syntax -> Id.t -> prog -> prog *)
(* val subss : syntax -> Id.t -> syntax -> syntax *)
(* (\* val subeiCtxt : I.expr -> Id.t -> context -> context *\) *)

(* val simSub : syntax Id.Map.t -> syntax -> syntax *)

(* val alphaConverte : category -> (Id.Set.t * expr) -> expr *)
(* val alphaConvertt : category -> (Id.Set.t * typ) -> typ *)
(* val alphaConvertc : category -> (Id.Set.t * prop) -> prop *)
(* val alphaConvertz : category -> (Id.Set.t * prop_typ) -> prop_typ *)

(* val matchtt : category -> (typ * typ) -> (typ * typ) *)
(* val matchzz : category -> (prop_typ * prop_typ) -> (prop_typ * prop_typ) *)
(* val matchet : category -> (expr * typ) -> (expr * typ) *)
(* val matchcz : category -> (prop * prop_typ) -> (prop * prop_typ) *)
