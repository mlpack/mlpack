open Util 
open List

type nullOp = Real of float | Bool of bool
type unaryOp = Neg | Not 
type binaryOp = Plus | Minus | Mult | Or | And
type numRel = Equal | Lte | Gte
type propOp = Disj | Conj
type quant = Exists
type direction = Min | Max

(* type 'a interval = Bounded of 'a * 'a | Lower of 'a | Upper of 'a | Unbounded *)
(* type refinedReal = RInt of int interval | RReal of float interval *)
(* type refinedBool = RSingleton of bool | RBool *)

type typ = 
  | TReal (* of refinedReal *)
  | TBool (* of refinedBool *)
      
and expr = 
  | EVar of Id.t
  | EConst of nullOp
  | EUnaryOp of unaryOp * expr
  | EBinaryOp of binaryOp * expr * expr
      
and prop_typ = 
  | ZProp

and prop = 
  | CBoolVal of bool
  | CIsTrue of expr
  | CNumRel of numRel * expr * expr
  | CPropOp of propOp * prop list
  | CQuant of quant * Id.t * typ * prop
      
and prog = 
  | PMain of direction * ((Id.t * typ) list) * expr * prop

(* Aliases for the unrefined types *)
(* let real' = TReal (RReal Unbounded) *)
(* let bool' = TBool RBool *)

module Ctxt = struct
  type t = (Id.t * typ) list
  let add ctxt x t = (x,t)::ctxt
  let fromList xts = xts
  let contains ctxt x t = exists (fun (x',t') -> Id.equal x x' && t == t') ctxt
  let lookup ctxt x = snd (find (fun (x',_) -> x=x') ctxt)
end
