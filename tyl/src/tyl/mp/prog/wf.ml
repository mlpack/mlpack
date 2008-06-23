open Ast 
open List

module C = Ctxt
module S = Util.Id.Set

let isType t = true

let rec isOfType ctxt e t = match e,t with 
  | EVar x                  , _     -> C.contains ctxt x t
  | EConst (Bool _)         , TBool -> true
  | EConst (Real _)         , TReal -> true
  | EUnaryOp (Neg,e')       , TReal -> isOfType ctxt e' TReal
  | EUnaryOp (Not,e')       , TBool -> isOfType ctxt e' TBool
  | EBinaryOp (Plus,e1,e2)  , TReal -> isOfType ctxt e1 TReal && isOfType ctxt e2 TReal
  | EBinaryOp (Minus,e1,e2) , TReal -> isOfType ctxt e1 TReal && isOfType ctxt e2 TReal
  | EBinaryOp (Mult,e1,e2)  , TReal -> isOfType ctxt e1 TReal && isOfType ctxt e2 TReal
  | EBinaryOp (Or,e1,e2)    , TBool -> isOfType ctxt e1 TBool && isOfType ctxt e2 TBool
  | EBinaryOp (And,e1,e2)   , TBool -> isOfType ctxt e1 TBool && isOfType ctxt e2 TBool
  | _ -> false

let rec isProp ctxt c = match c with 
  | CBoolVal _        -> true
  | CIsTrue e         -> isOfType ctxt e TBool
  | CNumRel (_,e1,e2) -> isOfType ctxt e1 TReal && isOfType ctxt e2 TReal
  | CPropOp (_,cs)    -> for_all (isProp ctxt) cs
  | CQuant (_,x,t,c') -> isType t && isProp (C.add ctxt x t) c'

let isMP p = match p with 
  | PMain (_,xts,e,c) -> 
      let ctxt = C.fromList xts in 
        for_all (fun (_,t) -> isType t) xts && isOfType ctxt e TReal && isProp ctxt c
