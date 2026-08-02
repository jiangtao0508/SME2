module {
  func.func @simple_stream(%arg0: memref<?xf32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      %value = memref.load %arg0[%i] : memref<?xf32>
    }
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root
      : (!transform.any_op) -> !transform.op<"func.func">
    %updated = transform.apply_registered_pass "prefetch-materialize" to %funcs
      {options = "argument-index=0 distance=4 locality=3"}
      : (!transform.op<"func.func">) -> !transform.op<"func.func">
    transform.yield
  }
}
