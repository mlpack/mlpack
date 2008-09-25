open Ast
module E = Edsl

(* lift functions into options *)
let liftO f a = match a with None -> None | Some x -> Some (f x)
let liftO2 f a b = match a with None -> None | Some x -> liftO (f x) b

(* pre: isNumExpr e 
   this impl keeps things 'int' as long as possible

let intervalOf ctxt e = 
  match e with 
    | EVar x -> lookup ctxt x
    | EConst (Int r) -> Disc (Some r, Some r)
    | EConst (Real r) -> Cont (Some r, Some r)
    | EUnaryOp (Neg e') -> 
        ( match intervalOf e' with 
            | Disc(a,b) -> Disc(liftO (~-) a, liftO (~-) b)
            | Cont(a,b) -> Disc(liftO (~-.) a, liftO (~-.) b) )
    | EBinaryOp (Plus,e1,e2) ->
        let add x y = liftO2 (+) in
        let add' x y = liftO2 (+.) in 
          ( match intervalOf e1, intervalOf e2 with
              | Disc(a,b),Disc(c,d) -> Disc(add a c, add b d)
              | Cont(a,b),Cont(c,d) -> Cont(add' a c, add' b d) )
    | EBinaryOp (Minus,e1,e2) -> 
        let sub x y = liftO2 (-) in 
        let sub' x y = liftO2 (-.) in 
          ( match intervalOf e1, intervalOf e2 with
              | Disc(a,b),Disc(c,d) -> Disc(sub a d, sub b c)
              | Cont(a,b),Cont(c,d) -> Cont(sub' a d, sub' b c) )
    | EBinaryOp (Mult,e1,e2)
        let mul x y = liftO2 ( * ) in 
        let mul' x y = liftO2 ( *. ) in 
          ( match intervalOf e1, intervalOf e2 with
              | Disc(a,b),Disc(c,d) -> Disc(???)
              | Cont(a,b),Cont(c,d) -> Cont(???)
*)
