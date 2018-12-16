(* CSE 130: Programming Assignment 2
 * misc.ml
 * by Guanghao Chen
 * A53276390
 *)

(* Assoc Function
 * Logic: for each time, we take out the head element in the list. Then we parse the head into two parts. Key and value.
 *  If the key is the target, we return the value,else we recursive to search in the tail list.
 *)
let rec assoc (d,k,l) =
    match l with
    | [] -> d
    | head::tail -> let (first,second) = head in
      if first= k then second else assoc(d,k,tail);;


(* removeDuplicates
 * Logic: use a base empty list to store the elem that we have already dealed with.
 * For each time, if the element is dealed before. We will overlook it else we will insert it into the base list.
 * Lastly, we can reverse the base list as the result.
 *)
let removeDuplicates l = 
  let rec helper (ok,remain) = 
      match remain with 
      | [] -> ok
      | h::t -> 
        let updateOk = if List.mem h ok then ok else h::ok in
        let updateRemain = t in 
	       helper (updateOk,updateRemain) 
  in
      List.rev (helper ([],l))

(* wwhile
 * Logic: the input is a function that require a bool and return the new value and the bool under an expression
 * What the wwhile do is to recursively use the function f until the bool elem becomes true.
 * Therefore, for each time, we check whether the boolean elem of the result of f b is true,
 * if it is true, we return the first elem as result, else we continue calling the function wwhle with the new value
 *)
let rec wwhile (f,b) =
  let (first,second) = f b in 
    if second == true then wwhile (f,first) else first;;

(* fixpoint
 *  Logic: I define a helper function as the f function in wwhile, which returns whether the f b equals b.
 *)
let fixpoint (f,b) = wwhile (( fun helper -> let b = (f helper) in (b, b != helper)), b);;


(* ffor: int * int * (int -> unit) -> unit
   Applies the function f to all the integers between low and high
   inclusive; the results get thrown away.
 *)
let rec ffor (low,high,f) = 
  if low>high 
  then () 
  else let _ = f low in ffor (low+1,high,f)
      
(************** Add Testing Code Here ***************)
