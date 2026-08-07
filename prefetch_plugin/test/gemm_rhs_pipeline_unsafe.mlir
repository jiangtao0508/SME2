module {
  module {
    func.func @kernel(%input: memref<64x256xf32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %pad = arith.constant 0.0 : f32
      %rhs = memref.alloc() : memref<64x256xf32>
      scf.for %k = %c0 to %c64 step %c1 {
        %loaded = memref.load %input[%k, %c0] : memref<64x256xf32>
        memref.store %loaded, %rhs[%k, %c0] : memref<64x256xf32>
        %panel = memref.subview %rhs[%k, %c0] [1, 16] [1, 1]
          : memref<64x256xf32> to memref<1x16xf32, strided<[256, 1], offset: ?>>
        %value = vector.transfer_read %panel[%c0, %c0], %pad {in_bounds = [true]}
          : memref<1x16xf32, strided<[256, 1], offset: ?>>, vector<16xf32>
      }
      return
    }
  }
}
