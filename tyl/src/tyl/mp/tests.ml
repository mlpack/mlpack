open Smallcheck
open Ast
open Util

let nullOps = pure _Bool %% bools ++ pure _Int %% ints ++ pure _Real %% floats
let unaryOps = pure Neg ++ pure Not
let binaryOps = pure Plus ++ pure Minus ++ pure Mult ++ pure Or ++ pure And
let numRels = pure Equal ++ pure Lte ++ pure Gte
let propOps = pure Disj ++ pure Conj
let quants = pure Exists
let directions = pure Min
let intervals a = pairs (options a) (options a)
let refinedReals = pure _Discrete %% intervals ints ++ pure _Continuous %% intervals floats
let refinedBools = options bools
let ids = pure (Id.make "a") ++ pure (Id.make "b") ++ pure (Id.make "c") 

let typs = pure _TReal %% refinedReals ++ pure _TBool %% refinedBools

let rec exprs = {fold = fun n f b -> (
                   pure _EVar %% ids 
                   ++ pure _EConst %% nullOps 
                   ++ pure _EUnaryOp %% unaryOps %% exprs
                   ++ pure _EBinaryOp %% binaryOps %% exprs %% exprs
                 ).fold n f b}

let rec props = {fold = fun n f b -> (
                   pure _CBoolVal %% bools 
                   ++ pure _CIsTrue %% exprs 
                   ++ pure _CNumRel %% numRels %% exprs %% exprs
                   ++ pure _CPropOp %% propOps %% lists props 
                   ++ pure _CQuant %% quants %% ids %% typs %% props
                 ).fold n f b}

let progs = pure _PMain %% directions %% lists (pairs ids typs) %% exprs %% props

let contexts = lists (pairs ids typs)
