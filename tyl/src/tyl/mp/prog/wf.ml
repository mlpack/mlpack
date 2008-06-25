open Ast 
open List

module C = Context
module E = Edsl

(* sanity checking on intervals *)
let isType t = match t with
  | TReal (Discrete (Some lo, Some hi)) -> lo <= hi
  | TReal (Continuous (Some lo, Some hi)) -> lo <= hi
  | _ -> true

let rec isOfType ctxt e t = match e,t with 
  | EVar x                  , _       -> C.coarseContains ctxt x t
  | EConst (Bool _)         , TBool _ -> true
  | EConst (Int _)          , TReal _ -> true
  | EConst (Real _)         , TReal _ -> true
  | EUnaryOp (Neg,e')       , TReal _ -> isOfType ctxt e' E.real
  | EUnaryOp (Not,e')       , TBool _ -> isOfType ctxt e' E.bool
  | EBinaryOp (Plus,e1,e2)  , TReal _ -> isOfType ctxt e1 E.real && isOfType ctxt e2 E.real
  | EBinaryOp (Minus,e1,e2) , TReal _ -> isOfType ctxt e1 E.real && isOfType ctxt e2 E.real
  | EBinaryOp (Mult,e1,e2)  , TReal _ -> isOfType ctxt e1 E.real && isOfType ctxt e2 E.real
  | EBinaryOp (Or,e1,e2)    , TBool _ -> isOfType ctxt e1 E.real && isOfType ctxt e2 E.bool
  | EBinaryOp (And,e1,e2)   , TBool _ -> isOfType ctxt e1 E.bool && isOfType ctxt e2 E.bool
  | _ -> false

let rec isProp ctxt c = match c with 
  | CBoolVal _        -> true
  | CIsTrue e         -> isOfType ctxt e E.bool
  | CNumRel (_,e1,e2) -> isOfType ctxt e1 E.real && isOfType ctxt e2 E.real
  | CPropOp (_,cs)    -> for_all (isProp ctxt) cs
  | CQuant (_,x,t,c') -> isType t && isProp (C.add ctxt x t) c'

let isMP p = match p with 
  | PMain (_,xts,e,c) -> 
      let ctxt = C.fromList xts in 
        for_all (fun (_,t) -> isType t) xts && isOfType ctxt e E.real && isProp ctxt c
