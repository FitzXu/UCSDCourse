# CSE231 Compilers

+ Compilers
  + Parsers + Optimizers+ Code Generator
+ Optimizer Goal
  + Save running time
  + Memory Usage
+ What Optimizers do?
  + Compute information about a program
    + amount of variables 
    + jumps made
    + e.g. branches
  + Use that information to perform program transformation
+ What to learn in this class?
  + Representations of programs
  + Analyze and transformations
  + Applications
+ Course Work Schedule
  + First Mid-term 30%
  + Second Mid-term 30%
  + Course Project 40%
+ Analyzers Issues (Discussion)
  + Input: Code (Parsed Version e.g. assembly language)
  + Output: Information useful
  + Issues
    + Correctness
+ Transformer
  + Input: Output of Analyzer + Code + Additional Information: Optimizer Flags, Architecture, Profile Information 
  + Output: Representation
  + Relation
    + sometimes the ouput is a better version of input
    + Same behavior? Do we have to replicate the bugs of the original input?
    + Does the order of transforming matter? 
  + How to metric how profitable is the transformer ?
+ Common Optimization
  + Constant Folding
  + Constant Propogation
  + Partial Redundency Elimination: Some Paths of the code have redundancy
  + Dead code elimination ( unreachable code elimination)