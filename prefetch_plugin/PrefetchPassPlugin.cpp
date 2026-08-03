#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct PrefetchSnapshotPass
    : public PassWrapper<PrefetchSnapshotPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefetchSnapshotPass)

  PrefetchSnapshotPass() = default;
  PrefetchSnapshotPass(const PrefetchSnapshotPass &pass) : PassWrapper(pass) {}

  Option<std::string> outputPath{
      *this, "output-path",
      llvm::cl::desc("Path for the bufferized function snapshot"),
      llvm::cl::init("")};

  StringRef getArgument() const final { return "prefetch-snapshot"; }
  StringRef getDescription() const final {
    return "No-op marker used to print the bufferized payload before SME";
  }

  void runOnOperation() override {
    if (outputPath.empty()) {
      getOperation().emitError("prefetch-snapshot requires output-path");
      return signalPassFailure();
    }

    std::error_code error;
    llvm::raw_fd_ostream output(outputPath, error, llvm::sys::fs::OF_Text);
    if (error) {
      getOperation().emitError() << "cannot open snapshot output '"
                                 << outputPath << "': " << error.message();
      return signalPassFailure();
    }

    output << "module {\n";
    getOperation().print(output);
    output << "\n}\n";
  }
};

struct PrefetchMaterializePass
    : public PassWrapper<PrefetchMaterializePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefetchMaterializePass)

  PrefetchMaterializePass() = default;
  PrefetchMaterializePass(const PrefetchMaterializePass &pass)
      : PassWrapper(pass) {}

  Option<unsigned> argumentIndex{
      *this, "argument-index",
      llvm::cl::desc("Function memref argument to prefetch"),
      llvm::cl::init(0)};
  Option<int64_t> distance{
      *this, "distance",
      llvm::cl::desc("Prefetch distance in scf.for iterations"),
      llvm::cl::init(1)};
  Option<unsigned> locality{
      *this, "locality",
      llvm::cl::desc("LLVM/MLIR locality hint in the range 0..3"),
      llvm::cl::init(3)};

  StringRef getArgument() const final { return "prefetch-materialize"; }
  StringRef getDescription() const final {
    return "Insert a guarded memref.prefetch for a simple SCF load stream";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (distance <= 0) {
      function.emitError("prefetch distance must be positive");
      return signalPassFailure();
    }
    if (locality > 3) {
      function.emitError("prefetch locality must be in the range 0..3");
      return signalPassFailure();
    }
    if (argumentIndex >= function.getNumArguments()) {
      function.emitError("prefetch argument index is out of range");
      return signalPassFailure();
    }

    Value target = function.getArgument(argumentIndex);
    if (!isa<MemRefType>(target.getType())) {
      function.emitError("selected function argument is not a ranked memref");
      return signalPassFailure();
    }

    memref::LoadOp selectedLoad;
    scf::ForOp selectedLoop;
    function.walk([&](memref::LoadOp load) {
      if (selectedLoad || load.getMemRef() != target)
        return;
      auto loop = load->getParentOfType<scf::ForOp>();
      if (!loop)
        return;
      if (!llvm::is_contained(load.getIndices(), loop.getInductionVar()))
        return;
      selectedLoad = load;
      selectedLoop = loop;
    });

    if (!selectedLoad) {
      function.emitError(
          "could not find a load from the selected memref indexed by an "
          "enclosing scf.for induction variable");
      return signalPassFailure();
    }

    OpBuilder builder(selectedLoad);
    Location loc = selectedLoad.getLoc();
    Value distanceValue = builder.create<arith::ConstantIndexOp>(loc, distance);
    Value delta = builder.create<arith::MulIOp>(loc, selectedLoop.getStep(),
                                                distanceValue);
    Value future = builder.create<arith::AddIOp>(
        loc, selectedLoop.getInductionVar(), delta);
    Value inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, future, selectedLoop.getUpperBound());

    SmallVector<Value> futureIndices(selectedLoad.getIndices());
    for (Value &index : futureIndices) {
      if (index == selectedLoop.getInductionVar())
        index = future;
    }

    builder.create<scf::IfOp>(
        loc, inBounds, [&](OpBuilder &nestedBuilder, Location nestedLoc) {
          nestedBuilder.create<memref::PrefetchOp>(
              nestedLoc, target, futureIndices, /*isWrite=*/false,
              static_cast<uint32_t>(locality), /*isDataCache=*/true);
          nestedBuilder.create<scf::YieldOp>(nestedLoc);
        });
  }
};

struct PrefetchGemmRhsPass
    : public PassWrapper<PrefetchGemmRhsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefetchGemmRhsPass)

  PrefetchGemmRhsPass() = default;
  PrefetchGemmRhsPass(const PrefetchGemmRhsPass &pass) : PassWrapper(pass) {}

  Option<int64_t> distance{
      *this, "distance",
      llvm::cl::desc("Prefetch distance in K-loop iterations"),
      llvm::cl::init(4)};
  Option<unsigned> locality{
      *this, "locality",
      llvm::cl::desc("LLVM/MLIR locality hint in the range 0..3"),
      llvm::cl::init(3)};
  Option<unsigned> coverageLines{
      *this, "coverage-lines",
      llvm::cl::desc(
          "Number of consecutive cache lines to prefetch from the RHS panel"),
      llvm::cl::init(1)};
  Option<unsigned> issueEvery{
      *this, "issue-every",
      llvm::cl::desc("Issue prefetches every N K-loop iterations"),
      llvm::cl::init(1)};
  Option<unsigned> cacheLineBytes{
      *this, "cache-line-bytes",
      llvm::cl::desc(
          "Cache-line size used to space consecutive RHS prefetches"),
      llvm::cl::init(64)};

  StringRef getArgument() const final { return "prefetch-gemm-rhs"; }
  StringRef getDescription() const final {
    return "Insert a guarded prefetch for the RHS panel of tiled GEMM";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (distance <= 0) {
      function.emitError("prefetch distance must be positive");
      return signalPassFailure();
    }
    if (locality > 3) {
      function.emitError("prefetch locality must be in the range 0..3");
      return signalPassFailure();
    }
    if (coverageLines == 0) {
      function.emitError("prefetch coverage-lines must be positive");
      return signalPassFailure();
    }
    if (issueEvery == 0) {
      function.emitError("prefetch issue-every must be positive");
      return signalPassFailure();
    }
    if (cacheLineBytes == 0) {
      function.emitError("prefetch cache-line-bytes must be positive");
      return signalPassFailure();
    }

    struct Candidate {
      scf::ForOp loop;
      memref::SubViewOp subview;
      int64_t panelColumns;
      int64_t elementsPerCacheLine;
    };
    SmallVector<Candidate> candidates;

    function.walk([&](scf::ForOp loop) {
      Candidate best;
      for (Operation &operation : loop.getBody()->getOperations()) {
        auto subview = dyn_cast<memref::SubViewOp>(operation);
        if (!subview)
          continue;
        auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
        if (!sourceType || sourceType.getRank() != 2 ||
            !subview.getSource().getDefiningOp<memref::AllocOp>())
          continue;

        SmallVector<OpFoldResult> offsets = subview.getMixedOffsets();
        if (offsets.size() != 2)
          continue;
        Value reductionOffset = offsets[0].dyn_cast<Value>();
        Value columnOffset = offsets[1].dyn_cast<Value>();
        if (reductionOffset != loop.getInductionVar() || !columnOffset)
          continue;

        bool feedsVectorRead =
            llvm::any_of(subview->getUsers(), [](Operation *user) {
              return user->getName().getStringRef() == "vector.transfer_read";
            });
        if (!feedsVectorRead)
          continue;

        int64_t columns = sourceType.getShape()[1];
        if (ShapedType::isDynamic(columns))
          continue;
        Type elementType = sourceType.getElementType();
        unsigned bitWidth = 0;
        if (auto floatType = dyn_cast<FloatType>(elementType))
          bitWidth = floatType.getWidth();
        else if (auto integerType = dyn_cast<IntegerType>(elementType))
          bitWidth = integerType.getWidth();
        if (bitWidth == 0 || bitWidth % 8 != 0)
          continue;
        int64_t elementBytes = bitWidth / 8;
        if (cacheLineBytes % elementBytes != 0)
          continue;
        int64_t elementsPerCacheLine = cacheLineBytes / elementBytes;
        if (!best.subview || columns > best.panelColumns)
          best = {loop, subview, columns, elementsPerCacheLine};
      }
      if (best.subview)
        candidates.push_back(best);
    });

    if (candidates.empty()) {
      function.emitError(
          "could not find a tiled GEMM RHS subview in an scf.for K loop");
      return signalPassFailure();
    }

    for (Candidate candidate : candidates) {
      OpBuilder builder(candidate.subview);
      Location loc = candidate.subview.getLoc();
      Value distanceValue =
          builder.create<arith::ConstantIndexOp>(loc, distance);
      Value delta = builder.create<arith::MulIOp>(loc, candidate.loop.getStep(),
                                                  distanceValue);
      Value future = builder.create<arith::AddIOp>(
          loc, candidate.loop.getInductionVar(), delta);
      Value inBounds =
          builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, future,
                                        candidate.loop.getUpperBound());
      Value shouldIssue = inBounds;
      if (issueEvery > 1) {
        Value elapsed =
            builder.create<arith::SubIOp>(loc, candidate.loop.getInductionVar(),
                                          candidate.loop.getLowerBound());
        Value ordinal = builder.create<arith::DivUIOp>(
            loc, elapsed, candidate.loop.getStep());
        Value frequency =
            builder.create<arith::ConstantIndexOp>(loc, issueEvery);
        Value remainder =
            builder.create<arith::RemUIOp>(loc, ordinal, frequency);
        Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value onFrequency = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, remainder, zero);
        shouldIssue = builder.create<arith::AndIOp>(loc, inBounds, onFrequency);
      }
      Value column = candidate.subview.getMixedOffsets()[1].dyn_cast<Value>();
      Value source = candidate.subview.getSource();
      Value columnLimit =
          builder.create<arith::ConstantIndexOp>(loc, candidate.panelColumns);

      for (unsigned line = 0; line < coverageLines; ++line) {
        Value columnDelta = builder.create<arith::ConstantIndexOp>(
            loc, line * candidate.elementsPerCacheLine);
        Value futureColumn =
            builder.create<arith::AddIOp>(loc, column, columnDelta);
        Value columnInBounds = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ult, futureColumn, columnLimit);
        Value lineCondition =
            builder.create<arith::AndIOp>(loc, shouldIssue, columnInBounds);
        builder.create<scf::IfOp>(
            loc, lineCondition,
            [&](OpBuilder &nestedBuilder, Location nestedLoc) {
              nestedBuilder.create<memref::PrefetchOp>(
                  nestedLoc, source, ValueRange{future, futureColumn},
                  /*isWrite=*/false, static_cast<uint32_t>(locality),
                  /*isDataCache=*/true);
              nestedBuilder.create<scf::YieldOp>(nestedLoc);
            });
      }
    }
  }
};

void registerPrefetchPasses() {
  PassRegistration<PrefetchSnapshotPass>();
  PassRegistration<PrefetchMaterializePass>();
  PassRegistration<PrefetchGemmRhsPass>();
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "PrefetchPassPlugin", LLVM_VERSION_STRING,
          []() { registerPrefetchPasses(); }};
}
