#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <set>

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

static std::optional<int64_t> constantIndex(Value value) {
  APInt result;
  if (matchPattern(value, m_ConstantInt(&result)))
    return result.getSExtValue();
  return std::nullopt;
}

static int64_t elementBitWidth(Type elementType) {
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return floatType.getWidth();
  if (auto integerType = dyn_cast<IntegerType>(elementType))
    return integerType.getWidth();
  return 0;
}

static llvm::json::Value optionalInteger(std::optional<int64_t> value) {
  if (value)
    return *value;
  return nullptr;
}

static Value stripKnownMemrefViews(Value value) {
  while (Operation *defining = value.getDefiningOp()) {
    StringRef name = defining->getName().getStringRef();
    if (name != "memref.subview" && name != "memref.cast" &&
        name != "memref.reinterpret_cast" && name != "memref.view")
      break;
    if (defining->getNumOperands() == 0)
      break;
    value = defining->getOperand(0);
  }
  return value;
}

static void appendValuePredecessors(Value value,
                                    SmallVectorImpl<Value> &worklist) {
  if (auto blockArgument = dyn_cast<BlockArgument>(value)) {
    Operation *parent = blockArgument.getOwner()->getParentOp();
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent)) {
      unsigned argumentNumber = blockArgument.getArgNumber();
      if (argumentNumber > 0) {
        unsigned iterIndex = argumentNumber - 1;
        if (iterIndex < forOp.getInitArgs().size())
          worklist.push_back(forOp.getInitArgs()[iterIndex]);
        auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        if (yield && iterIndex < yield.getNumOperands())
          worklist.push_back(yield.getOperand(iterIndex));
      }
    }
  }
  if (Operation *defining = value.getDefiningOp())
    llvm::append_range(worklist, defining->getOperands());
}

static void collectFunctionArgumentIndices(Value root, func::FuncOp function,
                                           std::set<int64_t> &indices) {
  SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visited;
  unsigned traversed = 0;
  while (!worklist.empty() && traversed++ < 256) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (unsigned index = 0; index < function.getNumArguments(); ++index) {
      if (value == function.getArgument(index)) {
        indices.insert(index);
        break;
      }
    }
    appendValuePredecessors(value, worklist);
  }
}

struct UpstreamLoadSummary {
  int64_t memrefLoads = 0;
  int64_t vectorTransferReads = 0;
  int64_t tptrLoads = 0;
  int64_t otherLoads = 0;
  int64_t externalLoads = 0;
  std::set<int64_t> externalArgumentIndices;
};

static UpstreamLoadSummary traceUpstreamLoads(Value root,
                                              func::FuncOp function) {
  UpstreamLoadSummary summary;
  SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visitedValues;
  llvm::DenseSet<Operation *> visitedLoads;
  unsigned traversed = 0;
  while (!worklist.empty() && traversed++ < 512) {
    Value value = worklist.pop_back_val();
    if (!visitedValues.insert(value).second)
      continue;
    Operation *defining = value.getDefiningOp();
    if (!defining) {
      appendValuePredecessors(value, worklist);
      continue;
    }
    StringRef name = defining->getName().getStringRef();
    bool isLoad = false;
    if (name == "memref.load") {
      ++summary.memrefLoads;
      isLoad = true;
    } else if (name == "vector.transfer_read") {
      ++summary.vectorTransferReads;
      isLoad = true;
    } else if (name.starts_with("tptr.") && name.contains("load")) {
      ++summary.tptrLoads;
      isLoad = true;
    } else if (name.ends_with(".load") || name.contains("load")) {
      ++summary.otherLoads;
      isLoad = true;
    }
    if (isLoad && visitedLoads.insert(defining).second) {
      std::set<int64_t> loadArguments;
      for (Value operand : defining->getOperands())
        collectFunctionArgumentIndices(operand, function, loadArguments);
      if (!loadArguments.empty()) {
        ++summary.externalLoads;
        summary.externalArgumentIndices.insert(loadArguments.begin(),
                                               loadArguments.end());
      }
    }
    appendValuePredecessors(value, worklist);
  }
  return summary;
}

struct AllocationLineage {
  int64_t copyWriters = 0;
  int64_t vectorWriters = 0;
  int64_t storeWriters = 0;
  int64_t linalgWriters = 0;
  std::set<int64_t> sourceArgumentIndices;
  UpstreamLoadSummary upstreamLoads;
};

static AllocationLineage traceAllocationLineage(Value allocation,
                                                func::FuncOp function) {
  AllocationLineage lineage;
  function.walk([&](Operation *operation) {
    StringRef name = operation->getName().getStringRef();
    std::optional<unsigned> destination;
    if (name == "memref.copy" && operation->getNumOperands() >= 2) {
      destination = 1;
      ++lineage.copyWriters;
    } else if (name == "vector.transfer_write" &&
               operation->getNumOperands() >= 2) {
      destination = 1;
      ++lineage.vectorWriters;
    } else if (name == "memref.store" && operation->getNumOperands() >= 2) {
      destination = 1;
      ++lineage.storeWriters;
    } else if (name.starts_with("linalg.") &&
               operation->getNumOperands() >= 1) {
      destination = operation->getNumOperands() - 1;
      ++lineage.linalgWriters;
    } else {
      return;
    }

    if (stripKnownMemrefViews(operation->getOperand(*destination)) !=
        allocation) {
      if (name == "memref.copy")
        --lineage.copyWriters;
      else if (name == "vector.transfer_write")
        --lineage.vectorWriters;
      else if (name == "memref.store")
        --lineage.storeWriters;
      else
        --lineage.linalgWriters;
      return;
    }
    SmallVector<unsigned> sourceOperands;
    if (name == "memref.copy" || name == "vector.transfer_write" ||
        name == "memref.store") {
      sourceOperands.push_back(0);
    } else {
      for (unsigned index = 0; index < *destination; ++index)
        sourceOperands.push_back(index);
    }
    for (unsigned index : sourceOperands) {
      Value source = operation->getOperand(index);
      collectFunctionArgumentIndices(source, function,
                                     lineage.sourceArgumentIndices);
      UpstreamLoadSummary traced = traceUpstreamLoads(source, function);
      lineage.upstreamLoads.memrefLoads += traced.memrefLoads;
      lineage.upstreamLoads.vectorTransferReads += traced.vectorTransferReads;
      lineage.upstreamLoads.tptrLoads += traced.tptrLoads;
      lineage.upstreamLoads.otherLoads += traced.otherLoads;
      lineage.upstreamLoads.externalLoads += traced.externalLoads;
      lineage.upstreamLoads.externalArgumentIndices.insert(
          traced.externalArgumentIndices.begin(),
          traced.externalArgumentIndices.end());
    }
  });
  return lineage;
}

struct AnalyzeGemmRhsPass
    : public PassWrapper<AnalyzeGemmRhsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeGemmRhsPass)

  AnalyzeGemmRhsPass() = default;
  AnalyzeGemmRhsPass(const AnalyzeGemmRhsPass &pass) : PassWrapper(pass) {}

  Option<std::string> outputPath{
      *this, "output-path",
      llvm::cl::desc("Path for numeric packed GEMM RHS features"),
      llvm::cl::init("")};

  StringRef getArgument() const final { return "prefetch-analyze-gemm-rhs"; }
  StringRef getDescription() const final {
    return "Extract numeric packed GEMM RHS prefetch features without changing "
           "IR";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (outputPath.empty()) {
      module.emitError("prefetch-analyze-gemm-rhs requires output-path");
      return signalPassFailure();
    }

    llvm::json::Array candidates;
    int64_t candidateId = 0;
    module.walk([&](scf::ForOp loop) {
      std::optional<llvm::json::Object> bestFeature;
      Value bestAllocation;
      int64_t bestColumns = -1;
      for (Operation &operation : loop.getBody()->getOperations()) {
        auto subview = dyn_cast<memref::SubViewOp>(operation);
        if (!subview)
          continue;
        auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
        if (!sourceType || sourceType.getRank() != 2 ||
            !subview.getSource().getDefiningOp<memref::AllocOp>())
          continue;

        SmallVector<OpFoldResult> offsets = subview.getMixedOffsets();
        if (offsets.size() != 2 ||
            offsets[0].dyn_cast<Value>() != loop.getInductionVar() ||
            !offsets[1].dyn_cast<Value>())
          continue;

        int64_t vectorReadBytes = 0;
        int64_t bitWidth = elementBitWidth(sourceType.getElementType());
        if (bitWidth <= 0 || bitWidth % 8 != 0)
          continue;
        int64_t elementBytes = bitWidth / 8;
        for (Operation *user : subview->getUsers()) {
          if (user->getName().getStringRef() != "vector.transfer_read" ||
              user->getNumResults() == 0)
            continue;
          auto vectorType = dyn_cast<VectorType>(user->getResult(0).getType());
          if (vectorType)
            vectorReadBytes = std::max<int64_t>(
                vectorReadBytes, vectorType.getNumElements() * elementBytes);
        }
        if (vectorReadBytes == 0)
          continue;

        int64_t rows = sourceType.getShape()[0];
        int64_t columns = sourceType.getShape()[1];
        if (ShapedType::isDynamic(rows) || ShapedType::isDynamic(columns))
          continue;
        if (columns <= bestColumns)
          continue;
        auto lower = constantIndex(loop.getLowerBound());
        auto upper = constantIndex(loop.getUpperBound());
        auto step = constantIndex(loop.getStep());
        std::optional<int64_t> tripCount;
        if (lower && upper && step && *step > 0 && *upper >= *lower)
          tripCount = (*upper - *lower + *step - 1) / *step;

        ArrayRef<int64_t> staticSizes = subview.getStaticSizes();
        auto staticOrNull = [](int64_t value) -> llvm::json::Value {
          if (ShapedType::isDynamic(value))
            return nullptr;
          return value;
        };
        llvm::json::Object feature{
            {"candidate_id", 0},
            {"source_rows", rows},
            {"source_columns", columns},
            {"element_bits", bitWidth},
            {"element_bytes", elementBytes},
            {"source_allocation_bytes", rows * columns * elementBytes},
            {"rhs_row_bytes", columns * elementBytes},
            {"vector_read_bytes", vectorReadBytes},
            {"loop_lower", optionalInteger(lower)},
            {"loop_upper", optionalInteger(upper)},
            {"loop_step", optionalInteger(step)},
            {"loop_trip_count", optionalInteger(tripCount)},
            {"subview_reduction_extent", staticSizes.empty()
                                             ? llvm::json::Value(nullptr)
                                             : staticOrNull(staticSizes[0])},
            {"subview_column_extent", staticSizes.size() < 2
                                          ? llvm::json::Value(nullptr)
                                          : staticOrNull(staticSizes[1])},
            {"bytes_advanced_per_loop_iteration",
             step ? llvm::json::Value(*step * columns * elementBytes)
                  : llvm::json::Value(nullptr)},
        };
        bestColumns = columns;
        bestAllocation = subview.getSource();
        bestFeature = std::move(feature);
      }
      if (bestFeature) {
        func::FuncOp function = loop->getParentOfType<func::FuncOp>();
        AllocationLineage lineage =
            traceAllocationLineage(bestAllocation, function);
        llvm::json::Array argumentIndices;
        for (int64_t index : lineage.sourceArgumentIndices)
          argumentIndices.push_back(index);
        llvm::json::Array externalLoadArguments;
        for (int64_t index : lineage.upstreamLoads.externalArgumentIndices)
          externalLoadArguments.push_back(index);
        int64_t writerCount = lineage.copyWriters + lineage.vectorWriters +
                              lineage.storeWriters + lineage.linalgWriters;
        int64_t upstreamLoadCount = lineage.upstreamLoads.memrefLoads +
                                    lineage.upstreamLoads.vectorTransferReads +
                                    lineage.upstreamLoads.tptrLoads +
                                    lineage.upstreamLoads.otherLoads;
        llvm::json::Object lineageJson{
            {"writer_operation_count", writerCount},
            {"memref_copy_writer_count", lineage.copyWriters},
            {"vector_transfer_write_count", lineage.vectorWriters},
            {"memref_store_writer_count", lineage.storeWriters},
            {"linalg_writer_count", lineage.linalgWriters},
            {"source_argument_indices", std::move(argumentIndices)},
            {"source_argument_count",
             static_cast<int64_t>(lineage.sourceArgumentIndices.size())},
            {"upstream_load_count", upstreamLoadCount},
            {"upstream_memref_load_count", lineage.upstreamLoads.memrefLoads},
            {"upstream_vector_transfer_read_count",
             lineage.upstreamLoads.vectorTransferReads},
            {"upstream_tptr_load_count", lineage.upstreamLoads.tptrLoads},
            {"upstream_other_load_count", lineage.upstreamLoads.otherLoads},
            {"external_load_count", lineage.upstreamLoads.externalLoads},
            {"external_load_argument_indices",
             std::move(externalLoadArguments)},
        };
        (*bestFeature)["lineage"] = std::move(lineageJson);
        (*bestFeature)["candidate_id"] = candidateId++;
        candidates.push_back(std::move(*bestFeature));
      }
    });

    if (candidates.empty()) {
      module.emitError(
          "could not find a tiled GEMM RHS subview in an scf.for K loop");
      return signalPassFailure();
    }
    llvm::json::Object root{
        {"schema_version", "1.0"},
        {"source_stage", "bufferized_before_sme"},
        {"matcher", "allocated_rank2_rhs_subview_feeding_vector_read"},
        {"candidate_count", static_cast<int64_t>(candidates.size())},
        {"candidates", std::move(candidates)},
    };

    std::error_code error;
    llvm::raw_fd_ostream output(outputPath, error, llvm::sys::fs::OF_Text);
    if (error) {
      module.emitError() << "cannot open GEMM feature output '" << outputPath
                         << "': " << error.message();
      return signalPassFailure();
    }
    output << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
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

/// Software-pipeline explicit vector reads from packed GEMM operand
/// allocations.  The first vector is loaded in a prologue, carried as a new
/// scf.for iter_arg, and the next row is loaded before the current compute.
/// This is the CPU/SME analogue of TritonGPU's local-load prefetch pipeline;
/// it is intentionally separate from memref.prefetch/PRFM materialization.
struct PipelineGemmRhsLoadPass
    : public PassWrapper<PipelineGemmRhsLoadPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipelineGemmRhsLoadPass)

  PipelineGemmRhsLoadPass() = default;
  PipelineGemmRhsLoadPass(const PipelineGemmRhsLoadPass &pass)
      : PassWrapper(pass) {}

  StringRef getArgument() const final { return "pipeline-gemm-rhs-load"; }
  StringRef getDescription() const final {
    return "Software-pipeline packed GEMM operand reads across an scf.for";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, scf::SCFDialect>();
  }

  struct Candidate {
    scf::ForOp loop;
    memref::SubViewOp subview;
    Operation *read = nullptr;
  };

  static bool availableOutsideLoop(Value value, scf::ForOp loop,
                                   Operation *allowedSubview = nullptr) {
    if (value == loop.getInductionVar())
      return true;
    if (allowedSubview && value.getDefiningOp() == allowedSubview)
      return true;
    if (auto argument = dyn_cast<BlockArgument>(value))
      return argument.getOwner() != loop.getBody();
    Operation *defining = value.getDefiningOp();
    return !defining || !loop->isAncestor(defining);
  }

  static bool allocationWrittenInLoop(Value allocation, scf::ForOp loop) {
    bool written = false;
    loop.walk([&](Operation *operation) {
      StringRef name = operation->getName().getStringRef();
      std::optional<unsigned> destination;
      if (name == "memref.copy" && operation->getNumOperands() >= 2)
        destination = 1;
      else if (name == "vector.transfer_write" &&
               operation->getNumOperands() >= 2)
        destination = 1;
      else if (name == "memref.store" && operation->getNumOperands() >= 2)
        destination = 1;
      if (destination && stripKnownMemrefViews(
                             operation->getOperand(*destination)) == allocation)
        written = true;
    });
    return written;
  }

  static std::optional<Candidate> findCandidate(func::FuncOp function) {
    std::optional<Candidate> result;
    function.walk([&](scf::ForOp loop) -> WalkResult {
      auto lower = constantIndex(loop.getLowerBound());
      auto upper = constantIndex(loop.getUpperBound());
      auto step = constantIndex(loop.getStep());
      if (!lower || !upper || !step || *step <= 0 || *upper <= *lower)
        return WalkResult::advance();

      for (Operation &operation : loop.getBody()->without_terminator()) {
        auto subview = dyn_cast<memref::SubViewOp>(operation);
        if (!subview || !subview->hasOneUse())
          continue;
        auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
        if (!sourceType || sourceType.getRank() != 2 ||
            !subview.getSource().getDefiningOp<memref::AllocOp>())
          continue;
        if (allocationWrittenInLoop(subview.getSource(), loop))
          continue;
        SmallVector<OpFoldResult> offsets = subview.getMixedOffsets();
        if (offsets.size() != 2 ||
            offsets[0].dyn_cast<Value>() != loop.getInductionVar() ||
            !offsets[1].dyn_cast<Value>())
          continue;

        Operation *read = *subview->getUsers().begin();
        if (read->getName().getStringRef() != "vector.transfer_read" ||
            read->getBlock() != loop.getBody() || read->getNumResults() != 1)
          continue;
        if (!llvm::all_of(subview->getOperands(), [&](Value operand) {
              return availableOutsideLoop(operand, loop);
            }))
          continue;
        if (!llvm::all_of(read->getOperands(), [&](Value operand) {
              return availableOutsideLoop(operand, loop, subview);
            }))
          continue;
        result = Candidate{loop, subview, read};
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  static Value cloneReadAt(OpBuilder &builder, Candidate candidate, Value row) {
    IRMapping mapping;
    mapping.map(candidate.loop.getInductionVar(), row);
    builder.clone(*candidate.subview, mapping);
    Operation *read = builder.clone(*candidate.read, mapping);
    return read->getResult(0);
  }

  static void pipelineCandidate(Candidate candidate) {
    scf::ForOp oldLoop = candidate.loop;
    Location loc = oldLoop.getLoc();
    OpBuilder builder(oldLoop);

    Value first = cloneReadAt(builder, candidate, oldLoop.getLowerBound());
    SmallVector<Value> initArgs(oldLoop.getInitArgs());
    initArgs.push_back(first);
    scf::ForOp newLoop = builder.create<scf::ForOp>(
        loc, oldLoop.getLowerBound(), oldLoop.getUpperBound(),
        oldLoop.getStep(), initArgs);
    newLoop->setAttr("prefetch.explicit_pipeline", builder.getUnitAttr());

    Block *newBody = newLoop.getBody();
    if (!newBody->empty() && isa<scf::YieldOp>(newBody->back()))
      newBody->back().erase();
    builder.setInsertionPointToStart(newBody);
    IRMapping mapping;
    mapping.map(oldLoop.getInductionVar(), newLoop.getInductionVar());
    for (auto [oldArg, newArg] :
         llvm::zip(oldLoop.getRegionIterArgs(),
                   newLoop.getRegionIterArgs().take_front(
                       oldLoop.getNumRegionIterArgs())))
      mapping.map(oldArg, newArg);
    Value ready = newLoop.getRegionIterArgs().back();
    mapping.map(candidate.read->getResult(0), ready);

    // Issue the following iteration's load before cloning the current
    // iteration's compute.  The carried `ready` vector feeds current compute,
    // while this independent load can overlap it.
    Value nextRow = builder.create<arith::AddIOp>(
        loc, newLoop.getInductionVar(), newLoop.getStep());
    Value hasNext = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, nextRow, newLoop.getUpperBound());
    scf::IfOp next = builder.create<scf::IfOp>(
        loc, TypeRange{ready.getType()}, hasNext, /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&next.getThenRegion().front());
    Value loaded = cloneReadAt(builder, candidate, nextRow);
    builder.create<scf::YieldOp>(loc, ValueRange{loaded});
    builder.setInsertionPointToStart(&next.getElseRegion().front());
    builder.create<scf::YieldOp>(loc, ValueRange{ready});
    builder.setInsertionPointAfter(next);

    for (Operation &operation : oldLoop.getBody()->without_terminator()) {
      if (&operation == candidate.subview.getOperation() ||
          &operation == candidate.read)
        continue;
      builder.clone(operation, mapping);
    }

    SmallVector<Value> yields;
    auto oldYield = cast<scf::YieldOp>(oldLoop.getBody()->getTerminator());
    for (Value value : oldYield.getOperands())
      yields.push_back(mapping.lookupOrDefault(value));
    yields.push_back(next.getResult(0));
    builder.create<scf::YieldOp>(loc, yields);

    for (auto [oldResult, newResult] :
         llvm::zip(oldLoop.getResults(),
                   newLoop.getResults().take_front(oldLoop.getNumResults())))
      oldResult.replaceAllUsesWith(newResult);
    oldLoop.erase();
  }

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    unsigned transformed = 0;
    while (std::optional<Candidate> candidate = findCandidate(function)) {
      pipelineCandidate(*candidate);
      ++transformed;
    }
    if (transformed == 0) {
      function.emitError("could not find a constant packed GEMM operand "
                         "vector-read loop to pipeline");
      return signalPassFailure();
    }
    function->setAttr(
        "prefetch.explicit_pipeline_count",
        IntegerAttr::get(IntegerType::get(function.getContext(), 64),
                         transformed));
  }
};

/// Prefetch the original source matrix used by the outer BMM reduction loop.
///
/// The FlagGems/Triton-CPU payload first creates a ranked 4x4 view of an
/// unranked function argument and then copies that view into a private alloc:
///
///   %view = memref.reinterpret_cast %arg0 ...
///   %tile = memref.alloc() : memref<4x4xbf16>
///   memref.copy %view, %tile
///
/// Prefetching %tile is too late: the copy has already read the original
/// matrix.  This pass deliberately follows the source argument instead.  It
/// creates a guarded future view before the current view/copy and emits one
/// prefetch per source row so that a strided 4x4 tile covers all four cache
/// lines used by the real load.
struct PrefetchBmmSourcePass
    : public PassWrapper<PrefetchBmmSourcePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefetchBmmSourcePass)

  PrefetchBmmSourcePass() = default;
  PrefetchBmmSourcePass(const PrefetchBmmSourcePass &pass)
      : PassWrapper(pass) {}

  Option<unsigned> argumentIndex{
      *this, "argument-index",
      llvm::cl::desc("Original unranked BMM matrix argument to prefetch"),
      llvm::cl::init(0)};
  Option<int64_t> distance{
      *this, "distance",
      llvm::cl::desc("Prefetch distance in outer K-loop iterations"),
      llvm::cl::init(8)};
  Option<unsigned> locality{
      *this, "locality",
      llvm::cl::desc("LLVM/MLIR locality hint in the range 0..3"),
      llvm::cl::init(2)};
  Option<unsigned> issueEvery{
      *this, "issue-every",
      llvm::cl::desc("Issue prefetches every N outer K-loop iterations"),
      llvm::cl::init(8)};
  Option<int64_t> expectedRows{*this, "expected-rows",
                               llvm::cl::desc("Required source tile row count"),
                               llvm::cl::init(4)};
  Option<int64_t> expectedTileK{
      *this, "expected-tile-k",
      llvm::cl::desc("Required source tile reduction width"),
      llvm::cl::init(4)};

  StringRef getArgument() const final { return "prefetch-bmm-source"; }
  StringRef getDescription() const final {
    return "Prefetch a guarded future tile from an original BMM source";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  static Value createIntegerLikeConstant(OpBuilder &builder, Location loc,
                                         Type type, int64_t value) {
    if (isa<IndexType>(type))
      return builder.create<arith::ConstantIndexOp>(loc, value);
    auto integerType = dyn_cast<IntegerType>(type);
    if (!integerType)
      return {};
    return builder.create<arith::ConstantOp>(
        loc, type, builder.getIntegerAttr(integerType, value));
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
    if (issueEvery == 0) {
      function.emitError("prefetch issue-every must be positive");
      return signalPassFailure();
    }
    if (expectedRows <= 0 || expectedTileK <= 0) {
      function.emitError("expected tile dimensions must be positive");
      return signalPassFailure();
    }
    if (argumentIndex >= function.getNumArguments()) {
      function.emitError("prefetch source argument index is out of range");
      return signalPassFailure();
    }

    Value sourceArgument = function.getArgument(argumentIndex);
    if (!isa<UnrankedMemRefType>(sourceArgument.getType())) {
      function.emitError(
          "prefetch-bmm-source requires an original unranked memref argument");
      return signalPassFailure();
    }

    struct Candidate {
      scf::ForOp loop;
      memref::ReinterpretCastOp view;
      Value currentOffset;
      MemRefType viewType;
    };
    SmallVector<Candidate> candidates;

    function.walk([&](scf::ForOp loop) {
      for (Operation &operation : loop.getBody()->without_terminator()) {
        auto view = dyn_cast<memref::ReinterpretCastOp>(operation);
        if (!view || view.getSource() != sourceArgument)
          continue;

        auto viewType = view.getResult().getType();
        if (viewType.getRank() != 2 || !viewType.hasStaticShape() ||
            viewType.getShape()[0] != expectedRows ||
            viewType.getShape()[1] != expectedTileK)
          continue;

        // The onsite Triton may bufferize the A/B panel either as a private
        // memref.copy -> alloc (local reproduction) or as a direct vector
        // transfer from the strided view (no packing copy).  The source
        // argument + static 4x4 checks already uniquely identify the panel,
        // so only require the view to be consumed at all.
        if (view.getResult().use_empty())
          continue;

        OpFoldResult mixedOffset = view.getConstifiedMixedOffset();
        Value currentOffset = mixedOffset.dyn_cast<Value>();
        if (!currentOffset || !isa<IndexType>(currentOffset.getType()))
          continue;

        candidates.push_back({loop, view, currentOffset, viewType});
      }
    });

    if (candidates.size() != 1) {
      // Diagnose why the strict source-A structure did not match, so an
      // unexpected onsite IR shape can be fixed without dumping the file.
      int64_t loopCount = 0;
      int64_t reinterpretCount = 0;
      int64_t fromSourceArgCount = 0;
      int64_t staticTileCount = 0;
      int64_t copyToAllocCount = 0;
      function.walk([&](scf::ForOp loop) {
        ++loopCount;
        for (Operation &operation : loop.getBody()->without_terminator()) {
          auto view = dyn_cast<memref::ReinterpretCastOp>(operation);
          if (!view)
            continue;
          ++reinterpretCount;
          if (view.getSource() == sourceArgument)
            ++fromSourceArgCount;
          auto viewType = view.getResult().getType();
          if (viewType.getRank() == 2 && viewType.hasStaticShape() &&
              viewType.getShape()[0] == expectedRows &&
              viewType.getShape()[1] == expectedTileK)
            ++staticTileCount;
          if (!view.getResult().use_empty())
            ++copyToAllocCount;
        }
      });
      function.emitError()
          << "expected exactly one original-source 2-D BMM tile feeding a "
             "private memref.copy for argument "
          << argumentIndex.getValue() << ", found " << candidates.size()
          << "; diagnostic: scf_for_loops=" << loopCount
          << " reinterpret_cast=" << reinterpretCount
          << " from_source_arg=" << fromSourceArgCount << " static_"
          << expectedRows.getValue() << "x" << expectedTileK.getValue() << "="
          << staticTileCount << " consumed=" << copyToAllocCount;
      return signalPassFailure();
    }

    Candidate candidate = candidates.front();
    OpBuilder builder(candidate.view);
    Location loc = candidate.view.getLoc();
    Type ivType = candidate.loop.getInductionVar().getType();
    Value distanceValue =
        createIntegerLikeConstant(builder, loc, ivType, distance);
    Value frequencyValue =
        createIntegerLikeConstant(builder, loc, ivType, issueEvery);
    Value zeroValue = createIntegerLikeConstant(builder, loc, ivType, 0);
    if (!distanceValue || !frequencyValue || !zeroValue) {
      function.emitError(
          "outer K-loop induction type must be index or integer");
      return signalPassFailure();
    }

    Value iterationDelta = builder.create<arith::MulIOp>(
        loc, candidate.loop.getStep(), distanceValue);
    Value futureIteration = builder.create<arith::AddIOp>(
        loc, candidate.loop.getInductionVar(), iterationDelta);
    Value inBounds = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ult, futureIteration,
        candidate.loop.getUpperBound());

    Value elapsed = builder.create<arith::SubIOp>(
        loc, candidate.loop.getInductionVar(), candidate.loop.getLowerBound());
    Value ordinal =
        builder.create<arith::DivUIOp>(loc, elapsed, candidate.loop.getStep());
    Value remainder =
        builder.create<arith::RemUIOp>(loc, ordinal, frequencyValue);
    Value onFrequency = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, remainder, zeroValue);
    Value shouldIssue =
        builder.create<arith::AndIOp>(loc, inBounds, onFrequency);

    Value sourceElementDelta =
        builder.create<arith::ConstantIndexOp>(loc, distance * expectedTileK);
    Value futureOffset = builder.create<arith::AddIOp>(
        loc, candidate.currentOffset, sourceElementDelta);

    SmallVector<OpFoldResult> sizes = candidate.view.getConstifiedMixedSizes();
    SmallVector<OpFoldResult> strides =
        candidate.view.getConstifiedMixedStrides();
    auto guard = builder.create<scf::IfOp>(
        loc, shouldIssue, [&](OpBuilder &nestedBuilder, Location nestedLoc) {
          auto futureView = nestedBuilder.create<memref::ReinterpretCastOp>(
              nestedLoc, candidate.viewType, sourceArgument, futureOffset,
              sizes, strides);
          futureView->setAttr("prefetch.source_argument",
                              nestedBuilder.getI64IntegerAttr(argumentIndex));
          futureView->setAttr("prefetch.distance_iterations",
                              nestedBuilder.getI64IntegerAttr(distance));

          Value column =
              nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0);
          for (int64_t row = 0; row < expectedRows; ++row) {
            Value rowValue =
                nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, row);
            auto prefetch = nestedBuilder.create<memref::PrefetchOp>(
                nestedLoc, futureView.getResult(), ValueRange{rowValue, column},
                /*isWrite=*/false, static_cast<uint32_t>(locality),
                /*isDataCache=*/true);
            (void)prefetch;
          }
          nestedBuilder.create<scf::YieldOp>(nestedLoc);
        });
    guard->setAttr("prefetch.issue_every",
                   builder.getI64IntegerAttr(issueEvery));
  }
};

void registerPrefetchPasses() {
  PassRegistration<PrefetchSnapshotPass>();
  PassRegistration<AnalyzeGemmRhsPass>();
  PassRegistration<PrefetchMaterializePass>();
  PassRegistration<PrefetchGemmRhsPass>();
  PassRegistration<PipelineGemmRhsLoadPass>();
  PassRegistration<PrefetchBmmSourcePass>();
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "PrefetchPassPlugin", LLVM_VERSION_STRING,
          []() { registerPrefetchPasses(); }};
}
