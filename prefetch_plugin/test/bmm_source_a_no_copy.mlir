// Simulates the onsite Triton bufferized form where the A panel view is
// consumed directly by a vector transfer (no memref.copy -> private alloc).
module {
  func.func @bmm_kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %pad = arith.constant 0.000000e+00 : bf16
    %0 = scf.for %i = %c0 to %c4 step %c1 iter_args(%off = %c0) -> (index) {
      %a_view = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [4, 4], strides: [%c8, 1] : memref<*xbf16> to memref<4x4xbf16, strided<[?, 1], offset: ?>>
      %t = vector.transfer_read %a_view[%c0, %c0], %pad : memref<4x4xbf16, strided<[?, 1], offset: ?>>, vector<4x4xbf16>
      %n = arith.addi %off, %c4 : index
      scf.yield %n : index
    }
    return
  }
}
