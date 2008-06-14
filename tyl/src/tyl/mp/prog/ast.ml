open Util 
open List

type nullOp = Real of float | Bool of bool
type unaryOp = Neg | Not 
type binaryOp = Plus | Minus | Mult | Or | And
type numRel = Equal | Lte | Gte
type propOp = Disj | Conj
type quant = Exists
type direction = Min | Max
    
type typ = 
  | TReal
  | TBool
      
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

module Ctxt = struct
  type t = (Id.t * typ) list
  let add ctxt x t = (x,t)::ctxt
  let fromList xx = xx
  let contains ctxt x t = mem (x,t) ctxt
end
