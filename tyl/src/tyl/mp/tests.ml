open SmallCheck
open Ast
open Util

let boolNullOps = pure _Bool %% bools 
let realNullOps = pure _Int %% ints ++ pure _Real %% floats
let nullOps = boolNullOps ++ realNullOps 
let boolUnaryOps = pure Not
let realUnaryOps = pure Neg
let unaryOps = boolUnaryOps ++ realUnaryOps
let boolBinaryOps = pure Or ++ pure And
let realBinaryOps = pure Plus ++ pure Minus ++ pure Mult
let binaryOps = boolBinaryOps ++ realBinaryOps
let numRels = pure Equal ++ pure Lte ++ pure Gte
let propOps = pure Disj ++ pure Conj
let quants = pure Exists
let directions = pure Min
let intervals a = pairs (options a) (options a)
let refinedReals = pure _Discrete %% intervals ints ++ pure _Continuous %% intervals floats
let refinedBools = options bools
let ids = pure (Id.make "a") ++ pure (Id.make "b") ++ pure (Id.make "c") 

let boolTyps = pure _TBool %% refinedBools
let realTyps = pure _TReal %% refinedReals
let typs = boolTyps ++ realTyps

let rec boolExprs = {fold = fun n f b -> (
                   pure _EVar %% ids 
                   ++ pure _EConst %% boolNullOps 
                   ++ pure _EUnaryOp %% boolUnaryOps %% boolExprs
                   ++ pure _EBinaryOp %% boolBinaryOps %% boolExprs %% boolExprs
                     ).fold n f b}

let rec realExprs = {fold = fun n f b -> (
                   pure _EVar %% ids 
                   ++ pure _EConst %% realNullOps 
                   ++ pure _EUnaryOp %% realUnaryOps %% realExprs
                   ++ pure _EBinaryOp %% realBinaryOps %% realExprs %% realExprs
                     ).fold n f b}

let exprs = boolExprs ++ realExprs 

let rec props = {fold = fun n f b -> (
                   pure _CBoolVal %% bools 
                   ++ pure _CIsTrue %% exprs 
                   ++ pure _CNumRel %% numRels %% exprs %% exprs
                   ++ pure _CPropOp %% propOps %% lists props 
                   ++ pure _CQuant %% quants %% ids %% typs %% props
                 ).fold n f b}

let progs = pure _PMain %% directions %% lists (pairs ids typs) %% exprs %% props

let contexts = lists (pairs ids typs)
