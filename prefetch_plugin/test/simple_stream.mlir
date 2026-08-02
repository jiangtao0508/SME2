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
