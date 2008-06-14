open Ast 
open List

let wfType t = true

let rec isOfType ctxt e t = match e,t with 
  | EVar x                  , t'    -> failwith "FIXME"
  | EConst (Bool _)         , TBool -> true
  | EConst (Rational _)     , TReal -> true
  | EUnaryOp (Neg,e')       , TReal -> isOfType ctxt e' TReal
  | EUnaryOp (Not,e')       , TBool -> isOfType ctxt e' TBool
  | EBinaryOp (Plus,e1,e2)  , TReal -> isOfType ctxt e1 TReal && isOfType ctxt e2 TReal
  | EBinaryOp (Minus,e1,e2) , TReal -> isOfType ctxt e1 TReal && isOfType ctxt e2 TReal
  | EBinaryOp (Mult,e1,e2)  , TReal -> isOfType ctxt e1 TReal && isOfType ctxt e2 TReal
  | EBinaryOp (Or,e1,e2)    , TBool -> isOfType ctxt e1 TBool && isOfType ctxt e2 TBool
  | EBinaryOp (And,e1,e2)   , TBool -> isOfType ctxt e1 TBool && isOfType ctxt e2 TBool
  | _ -> false

let rec wfProp ctxt c = match c with 
  | CBoolVal _ -> true
  | _ -> failwith "FIXME"

let wfProg p = match p with 
  | PMain (_,xx,e,c) -> 
      let ctxt = "" in 
      let (_,tt) = split xx in 
        for_all wfType tt && isOfType ctxt e TReal && wfProp ctxt c 

