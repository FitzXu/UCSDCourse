(*
 * expr.ml
 * cse130
 * by Guanghao Chen
 *)

(* Please do not modify the names or types of any of the following
 * type constructors, or predefined functions, unless EXPLICITLY
 * asked to. You will loose points if you do.
 *)


(* REMEMBER TO DOCUMENT ALL FUNCTIONS THAT YOU WRITE OR COMPLETE *)

(* 
  This file is to implement several mathematical expressions
  we first define the types for each operation
*)
type expr = 
    VarX
  | VarY
  | Sine     of expr
  | Cosine   of expr
  | Average  of expr * expr
  | Times    of expr * expr
  | Thresh   of expr * expr * expr * expr	
  | Sigmoid  of expr
  | SineTripple  of  expr * expr * expr

(* exprToString
 *  In this part, we define a recursive function which can convert the type expr into a corresponding string
 *  Expect the operations that teacher give us, I  defined two new opearatins
 *)
let rec exprToString e =
  match e with
    | VarX -> "x"
    | VarY -> "y"
    | Sine (e) -> "sin(pi*" ^ exprToString e ^ ")"
    | Cosine (e) -> "cos(pi*" ^exprToString e ^")"
    | Average (e1,e2) -> "((" ^ exprToString e1 ^ "+" ^ exprToString e2 ^ ")/2)"
    | Times (e1,e2) -> (exprToString e1) ^ "*" ^ (exprToString e2)
    | Thresh (e1,e2,e3,e4) -> "("^(exprToString e1)^"<"^(exprToString e2)^"?"^(exprToString e3)^":" ^(exprToString e4)^")"
    | Sigmoid (e) -> "(1/(1+exp(-("^exprToString e^"))))"
    | SineTripple (e1,e2,e3) -> "sin("^(exprToString e1)^"+"^(exprToString e2)^"+"^(exprToString e3)^")"

(* build functions:
     Use these helper functions to generate elements of the expr
     datatype rather than using the constructors directly.  This
     provides a little more modularity in the design of your program *)

let buildX()                       = VarX
let buildY()                       = VarY
let buildSine(e)                   = Sine(e)
let buildCosine(e)                 = Cosine(e)
let buildAverage(e1,e2)            = Average(e1,e2)
let buildTimes(e1,e2)              = Times(e1,e2)
let buildThresh(a,b,a_less,b_less) = Thresh(a,b,a_less,b_less)
let buildSigmoid(e)                = Sigmoid(e)
let buildSineTripple(e1,e2,e3)     = SineTripple(e1,e2,e3)


let pi = 4.0 *. atan 1.0
let exp = 2.71

(* eval
 * This function recursively calculate the value of expression
 *) 
let rec eval (e,x,y) = 
  match e with
  | VarX -> x
  | VarY -> y
  | Sine (e) -> sin (pi*.(eval (e,x,y)))
  | Cosine (e) -> cos (pi*.(eval (e,x,y)))
  | Average (e1,e2) -> ((eval (e1,x,y))+.(eval (e2,x,y)))/.2.0 
  | Times (e1,e2) -> (eval (e1,x,y))*.(eval (e2,x,y))
  | Thresh (e1,e2,e3,e4) -> if (eval (e1,x,y))<(eval (e2,x,y)) then (eval (e3,x,y)) else (eval (e4,x,y))
  | Sigmoid (e) -> 1.0/.(1.0+.(exp**(0.0-.eval (e,x,y))))
  | SineTripple (e1,e2,e3) -> sin (eval (e1,x,y)+.eval (e2,x,y)+.eval (e3,x,y))


(* (eval_fn e (x,y)) evaluates the expression e at the point (x,y) and then
 * verifies that the result is between -1 and 1.  If it is, the result is returned.  
 * Otherwise, an exception is raised.
 *)
let eval_fn e (x,y) = 
  let rv = eval (e,x,y) in
  assert (-1.0 <= rv && rv <= 1.0);
  rv

let sampleExpr =
      buildCosine(buildSine(buildTimes(buildCosine(buildAverage(buildCosine(
      buildX()),buildTimes(buildCosine (buildCosine (buildAverage
      (buildTimes (buildY(),buildY()),buildCosine (buildX())))),
      buildCosine (buildTimes (buildSine (buildCosine
      (buildY())),buildAverage (buildSine (buildX()), buildTimes
      (buildX(),buildX()))))))),buildY())))

let sampleExpr2 =
  buildThresh(buildX(),buildY(),buildSine(buildX()),buildCosine(buildY()))


(************** Add Testing Code Here ***************)
