open Ast 
open List
open Vars

module E = Edsl
module S = Id.Set

(* sanity checking on intervals *)
let isType t = match t with
  | TReal (Discrete (Some lo, Some hi)) -> lo <= hi
  | TReal (Continuous (Some lo, Some hi)) -> lo <= hi
  | _ -> true

let rec isOfType ctxt e t = match e,t with 
  | EVar x                  , _       -> coarseContains ctxt x t
  | EConst (Bool _)         , TBool _ -> true
  | EConst (Int _)          , TReal _ -> true
  | EConst (Real _)         , TReal _ -> true
  | EUnaryOp (Neg,e')       , TReal _ -> isOfType ctxt e' E.real
  | EUnaryOp (Not,e')       , TBool _ -> isOfType ctxt e' E.bool
  | EBinaryOp (Plus,e1,e2)  , TReal _ -> isOfType ctxt e1 E.real && isOfType ctxt e2 E.real
  | EBinaryOp (Minus,e1,e2) , TReal _ -> isOfType ctxt e1 E.real && isOfType ctxt e2 E.real
  | EBinaryOp (Mult,e1,e2)  , TReal _ -> isOfType ctxt e1 E.real && isOfType ctxt e2 E.real
  | EBinaryOp (Or,e1,e2)    , TBool _ -> isOfType ctxt e1 E.bool && isOfType ctxt e2 E.bool
  | EBinaryOp (And,e1,e2)   , TBool _ -> isOfType ctxt e1 E.bool && isOfType ctxt e2 E.bool
  | _ -> false

let rec isProp ctxt c = match c with 
  | CBoolVal _        -> true
  | CIsTrue e         -> isOfType ctxt e E.bool
  | CNumRel (_,e1,e2) -> isOfType ctxt e1 E.real && isOfType ctxt e2 E.real
  | CPropOp (_,cs)    -> for_all (isProp ctxt) cs
  | CQuant (_,x,t,c') -> isType t && isProp ((x,t)::ctxt) c'

let isMP p = match p with 
  | PMain (_,ctxt,e,c) -> 
      for_all (fun (_,t) -> isType t) ctxt && isOfType ctxt e E.real && isProp ctxt c

(* misc type operations *)

let bounded t = assert (isType t) ; 
  match t with 
  | TReal (Discrete (Some _, Some _))
  | TReal (Continuous (Some _, Some _))
  | TBool _ -> true
  | _       -> false

let rec existVarsBounded c = match c with 
  | CBoolVal _ 
  | CIsTrue _ 
  | CNumRel _              -> true
  | CPropOp (_,cs)         -> for_all existVarsBounded cs
  | CQuant (Exists,_,t,c') -> bounded t && existVarsBounded c'

let rec disjVarsBounded ctxt c = match c with
  | CBoolVal _
  | CIsTrue _
  | CNumRel _              -> true
  | CPropOp (Disj,cs)      -> 
      let xs = S.elements (S.union' (map freeVarsc cs)) in
      let ts = map (lookup ctxt) xs in
        for_all bounded ts && for_all existVarsBounded cs 
  | CPropOp (Conj,cs)      -> for_all (disjVarsBounded ctxt) cs 
  | CQuant (Exists,x,t,c') -> disjVarsBounded ((x,t)::ctxt) c'
