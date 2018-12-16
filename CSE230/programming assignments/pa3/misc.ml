(* CSE 130: Programming Assignment 3
 * misc.ml
 * by Guanghao Chen
 *)

(* For this assignment, you may use the following library functions:

   List.map
   List.fold_left
   List.fold_right
   List.split
   List.combine
   List.length
   List.append
   List.rev

   See http://caml.inria.fr/pub/docs/manual-ocaml/libref/List.html for
   documentation.
*)



(* Do not change the skeleton code! The point of this assignment is to figure
 * out how the functions can be written this way (using fold). You may only
 * replace the   failwith "to be implemented"   part. *)



(*****************************************************************)
(******************* 1. Warm Up   ********************************)
(*****************************************************************)

(* sqsum
 * Logic: This function uses List.fold_left to calculate the summation of the integer list.
 * We set 0 as the accumulater(base). Each time, we add one element to the accumulator.
 *)
let sqsum xs = 
  let f a x = a + x*x in
  let base = 0 in
    List.fold_left f base xs
(* pipe 
 * Logic: This function input a list of functions and a number. Then iteratively using the functions onto the value.
 * In order to use the List.fold_left. We need to define a base and a fold function.
 * In this case, the definiition of the function doesn't include the parameter for the number.
 * Therefore, inside the implementation, we need to define a variable to receive this variable.
 * So the base should be a function transport the value samely.And each time, we need to apply the function to the base to produce the new function.
 *)
let pipe fs = 
  let f a x = fun y -> x (a y) in
  let base = fun y -> y in
    List.fold_left f base fs

(* sepConcat
 * This function concatenate the strings inside the list.
 * The base should be the first elem. And for each time in fold function, we need to append the seperator to the base string.
 * And then, appeding the current x to the string.
 *)
let rec sepConcat sep sl = match sl with 
  | [] -> ""
  | h :: t -> 
      let f a x = a^sep^x in
      let base = h in
      let l = t in
        List.fold_left f base l
(* stringOfList
 * This function wants to convert the whole list to a string.
 * For each elem, we first need to apply the to_string functions(given) to convert each elem.
 * This part we can use List.map function to apply every function to the elements.
 * Then we can utilize the above sepConcat function and define the seperator as "; "
 * Besides, we need to include the while string into a pair of "[" and "]"
 *)
let stringOfList f l =
    "["^(sepConcat "; " (List.map f l))^"]"



(*****************************************************************)
(******************* 2. Big Numbers ******************************)
(*****************************************************************)

(* clone
 * Logic: call a recursive function for  times
 * for each time we can insert a elem x to the head of the list
 *)
let rec clone x n = 
    match n<=0 with
    | true  -> []
    | false -> x::(clone x (n-1))

let rec padZero l1 l2 = 
    let len1 = List.length l1 in 
    let len2 = List.length l2 in
    if len1<len2 then ((clone 0 (len2-len1))@l1,l2)
    else if len1>len2 then (l1,(clone 0 (len1-len2))@l2)
    else (l1,l2)

let rec removeZero l =
	match l with
	| [] -> []
	| h::t -> if h == 0 then removeZero t else l


(* Bigadd
 * Logic: the function should mock the way that how we do the summation with hands
 * for example, [1;2;3;4]+[5;0;0;0], we need to align the two intergers and sum from the last one to the first one
 * besides, we need to set a temperary variable to store the carry for each step in case the summation will be over 10
 * Therefore, 
 * First of all, we should first reverse the two list so that we can calculate from the rear 
 * to the front by using List.
 * Secondly, the accumulator should be included both carry and the summation_list
 * Thirdly, the 
 *)
let bigAdd l1 l2 = 
  let add (l1, l2) = 
    let f a x = (
        let (carry,summation) = a in
        let (elem1,elem2) = x in
        ((elem1+elem2+carry)/10,((elem1+elem2+carry)mod 10)::summation) 
    ) in
    let base = (0,[]) in
    let args = List.combine (List.rev l1) (List.rev l2) in
    let (carry, res) = List.fold_left f base args in
      if carry = 1 then 1::res else res
  in 
    removeZero (add (padZero l1 l2))

(* mulByDigit
 * Logic: we can convert multiplication into addtion
 * For example 8 * [9,9,9], we can convert it into 8 times addition for [9,9,9]
 *)
let rec mulByDigit i l = 
    if i = 0 then [0] 
    else bigAdd l (mulByDigit (i-1) l)

(* bigMul
 * Logic: For each time, we can create the exact addition using the corresponding digit times 10^n times the first bigInt
 * Therefore, we need to clone the first bigInt for times of lenth(l2) because List.fold_left will pick up one elem each time
 *)
let bigMul l1 l2 = 
  let f a x = (
    let (digit,temp) = a in
    let (left,right) = x in
    (digit*10,(bigAdd (mulByDigit (digit*right) left) temp))
  ) in
  let base = (1,[]) in
  let args = List.combine (clone l1 (List.length l2)) (List.rev l2) in
  let (digit, res) = List.fold_left f base args in
    res
