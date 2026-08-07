module {
  module {
    func.func @kernel(%a: memref<64x16xf32>, %b: memref<64x16xf32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %pad = arith.constant 0.0 : f32
      %a_pack = memref.alloc() : memref<64x16xf32>
      %b_pack = memref.alloc() : memref<64x16xf32>
      %result = memref.alloc() : memref<64x16xf32>

      // Packing completes before the consuming loop.  Hoisting the first
      // reads into a software-pipeline prologue is therefore safe.
      scf.for %k = %c0 to %c64 step %c1 {
        scf.for %j = %c0 to %c16 step %c1 {
          %av = memref.load %a[%k, %j] : memref<64x16xf32>
          %bv = memref.load %b[%k, %j] : memref<64x16xf32>
          memref.store %av, %a_pack[%k, %j] : memref<64x16xf32>
          memref.store %bv, %b_pack[%k, %j] : memref<64x16xf32>
        }
      }

      scf.for %k = %c0 to %c64 step %c1 {
        %a_panel = memref.subview %a_pack[%k, %c0] [1, 16] [1, 1]
          : memref<64x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
        %a_value = vector.transfer_read %a_panel[%c0, %c0], %pad {in_bounds = [true]}
          : memref<1x16xf32, strided<[16, 1], offset: ?>>, vector<16xf32>
        %b_panel = memref.subview %b_pack[%k, %c0] [1, 16] [1, 1]
          : memref<64x16xf32> to memref<1x16xf32, strided<[16, 1], offset: ?>>
        %b_value = vector.transfer_read %b_panel[%c0, %c0], %pad {in_bounds = [true]}
          : memref<1x16xf32, strided<[16, 1], offset: ?>>, vector<16xf32>
        %sum = arith.addf %a_value, %b_value : vector<16xf32>
        vector.transfer_write %sum, %result[%k, %c0] {in_bounds = [true]}
          : vector<16xf32>, memref<64x16xf32>
      }
      return
    }
  }
}
