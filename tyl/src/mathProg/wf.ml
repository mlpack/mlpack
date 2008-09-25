open TylesBase
open Ast 
open Vars
module E = Edsl
module S = Id.Set

(* sanity checking on intervals *)
let isType t = match t with
  | TReal (Discrete (Some lo, Some hi)) -> lo <= hi
  | TReal (Continuous (Some lo, Some hi)) -> lo <= hi
  | _ -> true

let rec isOfType t ctxt e = match e,t with 
  | EVar x                  , _       -> coarseContains ctxt x t
  | EConst (Bool _)         , TBool _ -> true
  | EConst (Int _)          , TReal _ -> true
  | EConst (Real _)         , TReal _ -> true
  | EUnaryOp (Neg,e')       , TReal _ -> isOfType E.real ctxt e'
  | EUnaryOp (Not,e')       , TBool _ -> isOfType E.bool ctxt e'
  | EBinaryOp (Plus,e1,e2)  , TReal _ -> List.for_all (isOfType E.real ctxt) [e1;e2]
  | EBinaryOp (Minus,e1,e2) , TReal _ -> List.for_all (isOfType E.real ctxt) [e1;e2]
  | EBinaryOp (Mult,e1,e2)  , TReal _ -> List.for_all (isOfType E.real ctxt) [e1;e2]
  | EBinaryOp (Or,e1,e2)    , TBool _ -> List.for_all (isOfType E.bool ctxt) [e1;e2]
  | EBinaryOp (And,e1,e2)   , TBool _ -> List.for_all (isOfType E.bool ctxt) [e1;e2]
  | _ -> false

let rec isProp ctxt c = match c with 
  | CBoolVal _        -> true
  | CIsTrue e         -> isOfType E.bool ctxt e
  | CNumRel (_,e1,e2) -> List.for_all (isOfType E.real ctxt) [e1;e2]
  | CPropOp (_,cs)    -> List.for_all (isProp ctxt) cs
  | CQuant (_,x,t,c') -> isType t && isProp ((x,t)::ctxt) c'

let isMP p = match p with PMain (_,ctxt,e,c) ->
  List.for_all (isType <<- snd) ctxt && isOfType E.real ctxt e && isProp ctxt c

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
  | CPropOp (_,cs)         -> List.for_all existVarsBounded cs
  | CQuant (Exists,_,t,c') -> bounded t && existVarsBounded c'

let rec disjVarsBounded ctxt c = match c with
  | CBoolVal _
  | CIsTrue _
  | CNumRel _              -> true
  | CPropOp (Disj,cs)      -> 
      let xs = S.elements <<- S.unions & List.map freeVarsc cs in
      let ts = List.map (lookup ctxt) xs in
        List.for_all bounded ts && List.for_all existVarsBounded cs 
  | CPropOp (Conj,cs)      -> List.for_all (disjVarsBounded ctxt) cs 
  | CQuant (Exists,x,t,c') -> disjVarsBounded ((x,t)::ctxt) c'
