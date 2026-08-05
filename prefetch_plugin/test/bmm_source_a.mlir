module {
  module {
    func.func @bmm_source_fixture(
        %arg0: memref<*xbf16>, %arg1: memref<*xbf16>,
        %a_base: index, %b_base: index, %k_tiles: i32,
        %k_stride: index, %n_stride: index) {
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      scf.for %iter = %c0_i32 to %k_tiles step %c1_i32
          iter_args(%a_offset = %a_base, %b_offset = %b_base) -> (index, index) : i32 {
        %a_view = memref.reinterpret_cast %arg0 to
          offset: [%a_offset], sizes: [4, 4], strides: [%k_stride, 1]
          : memref<*xbf16> to memref<4x4xbf16, strided<[?, 1], offset: ?>>
        %b_view = memref.reinterpret_cast %arg1 to
          offset: [%b_offset], sizes: [4, 4], strides: [%n_stride, 1]
          : memref<*xbf16> to memref<4x4xbf16, strided<[?, 1], offset: ?>>
        %a_private = memref.alloc() : memref<4x4xbf16>
        %b_private = memref.alloc() : memref<4x4xbf16>
        memref.copy %a_view, %a_private
          : memref<4x4xbf16, strided<[?, 1], offset: ?>> to memref<4x4xbf16>
        memref.copy %b_view, %b_private
          : memref<4x4xbf16, strided<[?, 1], offset: ?>> to memref<4x4xbf16>
        %next_a = arith.addi %a_offset, %c4 : index
        %b_step = arith.muli %n_stride, %c4 : index
        %next_b = arith.addi %b_offset, %b_step : index
        memref.dealloc %a_private : memref<4x4xbf16>
        memref.dealloc %b_private : memref<4x4xbf16>
        scf.yield %next_a, %next_b : index, index
      }
      return
    }
  }
}
