open Util 
open List

type nullOp = Real of float | Bool of bool
type unaryOp = Neg | Not 
type binaryOp = Plus | Minus | Mult | Or | And
type numRel = Equal | Lte | Gte
type propOp = Disj | Conj
type quant = Exists
type direction = Min | Max


type 'a interval = 'a option * 'a option
type refinedReal = IntInterval of int interval | RealInterval of float interval 
type refinedBool = bool option

type typ = 
  | TReal of refinedReal
  | TBool of refinedBool
      
type prop_typ = 
  | ZProp

type expr = 
  | EVar of Id.t
  | EConst of nullOp
  | EUnaryOp of unaryOp * expr
  | EBinaryOp of binaryOp * expr * expr

type prop = 
  | CBoolVal of bool
  | CIsTrue of expr
  | CNumRel of numRel * expr * expr
  | CPropOp of propOp * prop list
  | CQuant of quant * Id.t * typ * prop
      
type prog = 
  | PMain of direction * ((Id.t * typ) list) * expr * prop

module Ctxt = struct
  type t = (Id.t * typ) list
  let add ctxt x t = (x,t)::ctxt
  let fromList xts = xts
  let coarseContains ctxt x t = 
    let coarseEqual (x',t') = Id.equal x x' &&
      match t,t' with
        | TReal _ , TReal _ -> true
        | TBool _ , TBool _ -> true
        | _ -> false
    in 
      exists coarseEqual ctxt
  let lookup ctxt x = snd (find (fun (x',_) -> Id.equal x x') ctxt)
end
