open Ast

let all = List.for_all
let fromList xx : context = ""

let isType t = true

let (%) a b = (a,b)

let rec (|-) ctxt = function
  | EVar x                  , t'    -> failWith "FIXME"
  | EConst (Bool _)         , TBool -> true
  | EConst (Rational _)     , TReal -> true
  | EConst (Int _)          , TReal -> true
  | EUnaryOp (Neg,e)        , TReal -> (ctxt |- e  % TReal)
  | EUnaryOp (Not,e)        , TBool -> (ctxt |- e  % TBool)
  | EBinaryOp (Plus,e1,e2)  , TReal -> (ctxt |- e1 % TReal) && (ctxt |- e2 % TReal)
  | EBinaryOp (Minus,e1,e2) , TReal -> (ctxt |- e1 % TReal) && (ctxt |- e2 % TReal)
  | EBinaryOp (Mult,e1,e2)  , TReal -> (ctxt |- e1 % TReal) && (ctxt |- e2 % TReal)
  | EBinaryOp (Or,e1,e2)    , TBool -> (ctxt |- e1 % TBool) && (ctxt |- e2 % TBool)
  | EBinaryOp (And,e1,e2)   , TBool -> (ctxt |- e1 % TBool) && (ctxt |- e2 % TBool)
  | _ -> false

let rec (|=) ctxt = function
  | CBoolVal _ -> true
  | 

let rec isProg = function 
  | PLet _ -> failWith "FIXME"
  | PMarked (_,p) -> isProg p
  | PMain (_,xx,e,c) -> 
      let ctxt = fromList xx in 
      let (_,tt) = split xx in 
        all isType tt && (ctxt |- e % TReal) && (ctxt |= c)
