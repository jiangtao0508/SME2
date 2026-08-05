module {
  module {
    func.func @kernel(%input: memref<64x256x__ELEMENT_TYPE__>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %pad = arith.constant 0.0 : __ELEMENT_TYPE__
      %rhs = memref.alloc() : memref<64x256x__ELEMENT_TYPE__>
      memref.copy %input, %rhs : memref<64x256x__ELEMENT_TYPE__> to memref<64x256x__ELEMENT_TYPE__>
      scf.for %k = %c0 to %c64 step %c1 {
        %panel = memref.subview %rhs[%k, %c0] [1, 16] [1, 1]
          : memref<64x256x__ELEMENT_TYPE__> to memref<1x16x__ELEMENT_TYPE__, strided<[256, 1], offset: ?>>
        %value = vector.transfer_read %panel[%c0, %c0], %pad {in_bounds = [true]}
          : memref<1x16x__ELEMENT_TYPE__, strided<[256, 1], offset: ?>>, vector<16x__ELEMENT_TYPE__>
      }
      return
    }
  }
}
