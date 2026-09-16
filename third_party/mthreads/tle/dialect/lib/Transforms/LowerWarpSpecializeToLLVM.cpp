#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/BarrierUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>
#include <utility>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERWARPSPECIALIZE
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttmg = triton::musa;

static constexpr StringLiteral kStaticWarpSpecializeAttr =
    "musa_tle.static_warp_specialize";
static constexpr StringLiteral kLocalSyncIntrinsic = "llvm.musa.syncthreads.lm";
static constexpr StringLiteral kExplicitCtaSyncAttr =
    "musa_tle.explicit_cta_sync";
static constexpr StringLiteral kBarRecordIntrinsic =
    "llvm.musa.async.bar.record";
static constexpr StringLiteral kDefaultProducerAttr =
    "musa_tle.default_producer";

struct PartitionSync {
  LLVM::CallIntrinsicOp op;
  int32_t numWarps;
  Region *partition;
};

static bool isSqmmaMmaIntrinsic(StringRef name) {
  return name.starts_with("llvm.musa.sqmma.") && name.ends_with(".mma");
}

static FailureOr<Value> getForwardedBlockArgument(Block *predecessor,
                                                  Block *successor,
                                                  unsigned argument) {
  auto branch = dyn_cast<BranchOpInterface>(predecessor->getTerminator());
  if (!branch)
    return failure();
  auto successorIt = llvm::find(branch->getSuccessors(), successor);
  if (successorIt == branch->getSuccessors().end())
    return failure();
  unsigned successorIndex =
      std::distance(branch->getSuccessors().begin(), successorIt);
  SuccessorOperands operands = branch.getSuccessorOperands(successorIndex);
  if (argument < operands.getProducedOperandCount())
    return failure();
  ValueRange forwarded = operands.getForwardedOperands();
  unsigned forwardedIndex = argument - operands.getProducedOperandCount();
  if (forwardedIndex >= forwarded.size())
    return failure();
  return forwarded[forwardedIndex];
}

static bool isConstant(Value value, int64_t expected) {
  Attribute constant;
  if (!matchPattern(value, m_Constant(&constant)))
    return false;
  auto integer = dyn_cast<IntegerAttr>(constant);
  return integer && integer.getInt() == expected;
}

static bool isUnitIncrement(Value value, Value induction) {
  auto matches = [&](Value lhs, Value rhs) {
    return (lhs == induction && isConstant(rhs, 1)) ||
           (rhs == induction && isConstant(lhs, 1));
  };
  if (auto add = value.getDefiningOp<arith::AddIOp>())
    return matches(add.getLhs(), add.getRhs());
  if (auto add = value.getDefiningOp<LLVM::AddOp>())
    return matches(add.getLhs(), add.getRhs());
  return false;
}

// scf.for has already become CFG when static warp specialization is lowered.
// Recover the canonical 0-based unit-step induction variable so a partition
// barrier reused by the loop waits on alternating hardware phases.
static Value resolvePartitionSyncPhase(LLVM::CallIntrinsicOp sync,
                                       Region &partition,
                                       DominanceInfo &dominance,
                                       IRRewriter &rewriter,
                                       Value initialPhase) {
  Block *loopHeader = nullptr;
  for (Block &candidate : partition.getBlocks()) {
    if (candidate.getNumArguments() == 0 ||
        !dominance.dominates(&candidate, sync->getBlock()))
      continue;
    Value induction = candidate.getArgument(0);
    auto inductionType = dyn_cast<IntegerType>(induction.getType());
    if (!inductionType || inductionType.getWidth() != 32)
      continue;

    SmallVector<Block *> entries;
    SmallVector<Block *> backedges;
    for (Block *predecessor : candidate.getPredecessors()) {
      (dominance.dominates(&candidate, predecessor) ? backedges : entries)
          .push_back(predecessor);
    }
    if (entries.size() != 1 || backedges.size() != 1)
      continue;
    FailureOr<Value> initial =
        getForwardedBlockArgument(entries.front(), &candidate, 0);
    FailureOr<Value> next =
        getForwardedBlockArgument(backedges.front(), &candidate, 0);
    if (failed(initial) || failed(next) || !isConstant(*initial, 0) ||
        !isUnitIncrement(*next, induction))
      continue;

    if (!loopHeader || dominance.dominates(loopHeader, &candidate))
      loopHeader = &candidate;
  }

  if (!loopHeader)
    return initialPhase;
  rewriter.setInsertionPoint(sync);
  Value one = arith::ConstantIntOp::create(rewriter, sync.getLoc(), 1, 32);
  return arith::AndIOp::create(rewriter, sync.getLoc(),
                               loopHeader->getArgument(0), one);
}

static LogicalResult lowerWarpGroupBarriers(LLVM::LLVMFuncOp func,
                                            ttg::WarpSpecializeOp ws,
                                            IRRewriter &rewriter) {
  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto defaultWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  if (!defaultWarpsAttr || defaultWarpsAttr.getInt() <= 0 ||
      defaultWarpsAttr.getInt() > std::numeric_limits<int32_t>::max())
    return ws.emitOpError(
        "MUSA TLE default partition requires a positive int32 "
        "ttg.num-warps");
  int32_t defaultWarps = static_cast<int32_t>(defaultWarpsAttr.getInt());

  SmallVector<PartitionSync> syncs;
  auto collect = [&](Region &region, int32_t warps) {
    region.walk([&](LLVM::CallIntrinsicOp call) {
      if (call.getIntrin() == kLocalSyncIntrinsic)
        syncs.push_back({call, warps, &region});
    });
  };
  collect(ws.getDefaultRegion(), defaultWarps);
  for (auto [region, warps] :
       llvm::zip_equal(ws.getPartitionRegions(), ws.getPartitionNumWarps()))
    collect(*region, warps);

  llvm::DenseSet<Region *> partitions;
  for (auto &sync : syncs)
    partitions.insert(sync.partition);
  if (syncs.empty())
    return success();

  auto reserved = ttmg::reserveBarrierIdRange(
      syncs.front().op, static_cast<int32_t>(partitions.size()));
  if (failed(reserved))
    return syncs.front().op.emitOpError(
        "MUSA TLE partition synchronization exhausted hardware barrier "
        "ids");

  LLVM::CallIntrinsicOp initializationRendezvous;
  for (Operation *op = ws->getPrevNode(); op; op = op->getPrevNode()) {
    auto call = dyn_cast<LLVM::CallIntrinsicOp>(op);
    if (call && call.getIntrin() == kLocalSyncIntrinsic) {
      initializationRendezvous = call;
      break;
    }
  }

  Location loc = initializationRendezvous ? initializationRendezvous.getLoc()
                                          : ws.getLoc();
  if (initializationRendezvous)
    rewriter.setInsertionPoint(initializationRendezvous);
  else
    rewriter.setInsertionPoint(ws);
  Value phase = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  llvm::DenseMap<Region *, Value> barrierIds;
  SmallVector<std::pair<Value, Value>> initializationArgs;
  int32_t nextId = *reserved;
  for (PartitionSync &sync : syncs) {
    if (barrierIds.count(sync.partition))
      continue;
    Value id = arith::ConstantIntOp::create(rewriter, loc, nextId++, 32);
    Value count =
        arith::ConstantIntOp::create(rewriter, loc, sync.numWarps, 32);
    initializationArgs.push_back({id, count});
    barrierIds[sync.partition] = id;
  }

  Value tid =
      LLVM::CallIntrinsicOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"), ValueRange{})
          .getResult(0);
  Value issueInit = arith::CmpIOp::create(rewriter, loc,
                                          arith::CmpIPredicate::eq, tid, phase);
  auto initIf = scf::IfOp::create(rewriter, loc, issueInit, false);
  rewriter.setInsertionPointToStart(&initIf.getThenRegion().front());
  for (auto [id, count] : initializationArgs)
    LLVM::CallIntrinsicOp::create(
        rewriter, loc, rewriter.getStringAttr("llvm.musa.async.init.arrival"),
        ValueRange{id, count, phase});

  rewriter.setInsertionPointAfter(initIf);
  if (!initializationRendezvous)
    LLVM::CallIntrinsicOp::create(rewriter, loc,
                                  rewriter.getStringAttr(kLocalSyncIntrinsic),
                                  ValueRange{});

  for (PartitionSync &sync : syncs) {
    rewriter.setInsertionPoint(sync.op);
    Value id = barrierIds.lookup(sync.partition);
    // A partition reuses this resource at every rendezvous. Use the phase
    // returned by arrival, not a fixed phase that only works on the first use.
    Value arrived =
        LLVM::CallIntrinsicOp::create(
            rewriter, sync.op.getLoc(), rewriter.getI32Type(),
            rewriter.getStringAttr("llvm.musa.async.arrive"), ValueRange{id})
            .getResult(0);
    LLVM::CallIntrinsicOp::create(
        rewriter, sync.op.getLoc(),
        rewriter.getStringAttr("llvm.musa.async.wait"),
        ValueRange{id, arrived});
    rewriter.eraseOp(sync.op);
  }

  SmallVector<LLVM::CallIntrinsicOp> oldBarRecords;
  func.walk([&](LLVM::CallIntrinsicOp call) {
    if (call.getIntrin() == kBarRecordIntrinsic)
      oldBarRecords.push_back(call);
  });
  for (LLVM::CallIntrinsicOp record : oldBarRecords)
    rewriter.eraseOp(record);
  rewriter.setInsertionPointToStart(&func.getBody().front());
  Value barCount = arith::ConstantIntOp::create(
      rewriter, func.getLoc(), ttmg::getReservedBarrierCount(func), 32);
  LLVM::CallIntrinsicOp::create(rewriter, func.getLoc(),
                                rewriter.getStringAttr(kBarRecordIntrinsic),
                                ValueRange{barCount});
  return success();
}

static LogicalResult lowerMultiStaticWarpSpecialize(
    LLVM::LLVMFuncOp func, ttg::WarpSpecializeOp ws,
    IRRewriter &rewriter) {
  SmallVector<Region *> partitionRegions(ws.getPartitionRegions().begin(),
                                         ws.getPartitionRegions().end());
  ArrayRef<int32_t> partitionNumWarps = ws.getPartitionNumWarps();
  if (ws.getNumResults() != 0 || partitionRegions.size() < 2 ||
      partitionRegions.size() != partitionNumWarps.size())
    return ws.emitOpError(
        "mthreads TLE multi-partition lowering requires at least two worker "
        "partitions, matching warp counts, and no results");
  if (llvm::any_of(partitionNumWarps,
                   [](int32_t count) { return count <= 0; }))
    return ws.emitOpError(
        "mthreads TLE multi-partition lowering requires positive worker warp "
        "counts");
  if (ws.getDefaultRegion().empty() ||
      llvm::any_of(partitionRegions,
                   [](Region *region) { return region->empty(); }))
    return ws.emitOpError("mthreads TLE static partitions must not be empty");

  auto partitions = ws.getPartitionOp();
  ValueRange captures = partitions.getExplicitCaptures();
  SmallVector<Block *> partitionRoots;
  SmallVector<ttg::WarpReturnOp> partitionReturns;
  for (Region *partition : partitionRegions) {
    Block &entry = partition->front();
    if (entry.getNumArguments() != captures.size())
      return ws.emitOpError(
          "worker capture count changed during multi-partition lowering");
    for (auto [argument, capture] : llvm::zip(entry.getArguments(), captures)) {
      if (argument.getType() != capture.getType())
        return ws.emitOpError(
            "worker capture types were not converted consistently");
      argument.replaceAllUsesWith(capture);
    }
    entry.eraseArguments([](BlockArgument) { return true; });
    partitionRoots.push_back(&entry);
    partition->walk(
        [&](ttg::WarpReturnOp op) { partitionReturns.push_back(op); });
  }

  Region &consumerRegion = ws.getDefaultRegion();
  SmallVector<ttg::WarpYieldOp> consumerYields;
  consumerRegion.walk(
      [&](ttg::WarpYieldOp op) { consumerYields.push_back(op); });
  if (partitionReturns.empty() || consumerYields.empty())
    return ws.emitOpError("static partitions lost their terminators");

  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto consumerWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  auto threadsPerWarpAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  if (!consumerWarpsAttr || !threadsPerWarpAttr ||
      consumerWarpsAttr.getInt() <= 0 || threadsPerWarpAttr.getInt() <= 0)
    return ws.emitOpError("multi-partition lowering requires positive "
                          "ttg.num-warps and ttg.threads-per-warp");
  int64_t workerStartWarp = consumerWarpsAttr.getInt();
  int64_t threadsPerWarp = threadsPerWarpAttr.getInt();
  if (workerStartWarp >
      std::numeric_limits<int32_t>::max() / threadsPerWarp)
    return ws.emitOpError("static partition boundary exceeds int32 range");

  Block *dispatch = ws->getBlock();
  Block *continuation =
      rewriter.splitBlock(dispatch, std::next(ws->getIterator()));
  Region &funcBody = func.getBody();
  auto &funcBlocks = funcBody.getBlocks();
  for (Region *partition : partitionRegions)
    funcBlocks.splice(continuation->getIterator(), partition->getBlocks());
  Block *consumerRoot = &consumerRegion.front();
  funcBlocks.splice(continuation->getIterator(), consumerRegion.getBlocks());

  for (ttg::WarpReturnOp op : partitionReturns) {
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op.getLoc(), continuation);
    rewriter.eraseOp(op);
  }
  for (ttg::WarpYieldOp op : consumerYields) {
    if (op.getNumOperands() != 0)
      return op.emitOpError(
          "mthreads TLE static consumer must not yield values");
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op.getLoc(), continuation);
    rewriter.eraseOp(op);
  }

  Location loc = ws.getLoc();
  rewriter.eraseOp(ws);
  rewriter.setInsertionPointToEnd(dispatch);
  Value tid =
      LLVM::CallIntrinsicOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"), ValueRange{})
          .getResult(0);
  Value firstWorkerThread = arith::ConstantIntOp::create(
      rewriter, loc, workerStartWarp * threadsPerWarp, 32);
  Value isDefault = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ult, tid, firstWorkerThread);
  Block *workerDispatch =
      rewriter.createBlock(&funcBody, continuation->getIterator());
  rewriter.setInsertionPointToEnd(dispatch);
  cf::CondBranchOp::create(rewriter, loc, isDefault, consumerRoot, ValueRange{},
                           workerDispatch, ValueRange{});

  for (unsigned index = 0; index + 1 < partitionRegions.size(); ++index) {
    workerStartWarp += partitionNumWarps[index];
    if (workerStartWarp <= 0 ||
        workerStartWarp >
            std::numeric_limits<int32_t>::max() / threadsPerWarp) {
      emitError(loc, "mthreads TLE static partition boundary exceeds int32 "
                     "range");
      return failure();
    }
    int64_t upperThread = workerStartWarp * threadsPerWarp;
    rewriter.setInsertionPointToEnd(workerDispatch);
    Value upper =
        arith::ConstantIntOp::create(rewriter, loc, upperThread, 32);
    Value inPartition = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, tid, upper);
    Block *nextDispatch =
        rewriter.createBlock(&funcBody, continuation->getIterator());
    rewriter.setInsertionPointToEnd(workerDispatch);
    cf::CondBranchOp::create(rewriter, loc, inPartition,
                             partitionRoots[index], ValueRange{}, nextDispatch,
                             ValueRange{});
    workerDispatch = nextDispatch;
  }
  rewriter.setInsertionPointToEnd(workerDispatch);
  cf::BranchOp::create(rewriter, loc, partitionRoots.back());
  return success();
}

static LogicalResult lowerStaticWarpSpecialize(LLVM::LLVMFuncOp func,
                                               ttg::WarpSpecializeOp ws,
                                               IRRewriter &rewriter) {
  auto workerRegions = ws.getPartitionRegions();
  ArrayRef<int32_t> workerWarps = ws.getPartitionNumWarps();
  if (ws.getNumResults() != 0 || workerRegions.empty() ||
      workerRegions.size() != workerWarps.size())
    return ws.emitOpError(
        "MUSA TLE late lowering requires a default partition, one or more "
        "worker partitions, and no results");

  Region &defaultRegion = ws.getDefaultRegion();
  if (defaultRegion.empty())
    return ws.emitOpError(
        "MUSA TLE static default partition must not be empty");
  for (auto [workerIndex, workerRegion] : llvm::enumerate(workerRegions)) {
    if (workerRegion->empty())
      return ws.emitOpError() << "MUSA TLE static worker partition #"
                              << workerIndex << " must not be empty";
  }

  auto partitions = ws.getPartitionOp();
  ValueRange captures = partitions.getExplicitCaptures();
  for (auto [workerIndex, workerRegion] : llvm::enumerate(workerRegions)) {
    Block &workerEntry = workerRegion->front();
    if (workerEntry.getNumArguments() != captures.size())
      return ws.emitOpError() << "MUSA TLE worker partition #" << workerIndex
                              << " capture count changed during lowering";
    for (auto [argument, capture] :
         llvm::zip(workerEntry.getArguments(), captures)) {
      if (argument.getType() != capture.getType())
        return ws.emitOpError()
               << "MUSA TLE worker partition #" << workerIndex
               << " capture types were not converted consistently";
      argument.replaceAllUsesWith(capture);
    }
    workerEntry.eraseArguments([](BlockArgument) { return true; });
  }

  ModuleOp module = func->getParentOfType<ModuleOp>();
  auto defaultWarpsAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  auto threadsPerWarpAttr =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  auto totalWarpsAttr =
      module->getAttrOfType<IntegerAttr>("ttg.total-num-warps");
  auto workerStartIds = ws.getWarpGroupStartIds();
  if (!defaultWarpsAttr || !threadsPerWarpAttr || !totalWarpsAttr ||
      !workerStartIds || defaultWarpsAttr.getInt() <= 0 ||
      threadsPerWarpAttr.getInt() <= 0 || totalWarpsAttr.getInt() <= 0)
    return ws.emitOpError(
        "MUSA TLE late lowering requires positive ttg.num-warps, "
        "ttg.threads-per-warp, ttg.total-num-warps, and worker start ids");
  if (workerStartIds->size() != workerRegions.size())
    return ws.emitOpError(
        "MUSA TLE late lowering requires one start id per worker partition");

  int64_t nextWarp = defaultWarpsAttr.getInt();
  SmallVector<int32_t> threadEnds;
  threadEnds.reserve(workerRegions.size() + 1);
  auto appendThreadEnd = [&](int64_t warpEnd) -> LogicalResult {
    if (warpEnd <= 0 || warpEnd > std::numeric_limits<int32_t>::max() /
                                      threadsPerWarpAttr.getInt())
      return ws.emitOpError(
          "MUSA TLE static partition boundary exceeds int32 range");
    threadEnds.push_back(
        static_cast<int32_t>(warpEnd * threadsPerWarpAttr.getInt()));
    return success();
  };
  if (failed(appendThreadEnd(nextWarp)))
    return failure();
  for (auto [workerIndex, startAndCount] :
       llvm::enumerate(llvm::zip(*workerStartIds, workerWarps))) {
    auto [startId, numWarps] = startAndCount;
    if (startId != nextWarp || numWarps <= 0)
      return ws.emitOpError() << "MUSA TLE worker partition #" << workerIndex
                              << " has a non-contiguous static warp range";
    nextWarp += numWarps;
    if (failed(appendThreadEnd(nextWarp)))
      return failure();
  }
  if (nextWarp != totalWarpsAttr.getInt())
    return ws.emitOpError(
        "MUSA TLE worker warp ranges do not match ttg.total-num-warps");

  SmallVector<SmallVector<ttg::WarpReturnOp>> workerReturns;
  SmallVector<Block *> workerRoots;
  workerReturns.reserve(workerRegions.size());
  workerRoots.reserve(workerRegions.size());
  for (Region *workerRegion : workerRegions) {
    workerRoots.push_back(&workerRegion->front());
    auto &returns = workerReturns.emplace_back();
    workerRegion->walk([&](ttg::WarpReturnOp op) { returns.push_back(op); });
    if (returns.empty())
      return ws.emitOpError(
          "MUSA TLE static worker partition lost its terminator");
  }
  SmallVector<ttg::WarpYieldOp> defaultYields;
  defaultRegion.walk([&](ttg::WarpYieldOp op) { defaultYields.push_back(op); });
  if (defaultYields.empty())
    return ws.emitOpError(
        "MUSA TLE static default partition lost its terminator");

  Block *dispatch = ws->getBlock();
  Block *continuation =
      rewriter.splitBlock(dispatch, std::next(ws->getIterator()));
  Region &funcBody = func.getBody();
  auto &funcBlocks = funcBody.getBlocks();
  Block *defaultRoot = &defaultRegion.front();

  if (workerRegions.size() == 1) {
    Region &workerRegion = *workerRegions.front();
    bool defaultProducer = ws->hasAttr(kDefaultProducerAttr);
    Region &producerExecRegion = defaultProducer ? defaultRegion : workerRegion;
    Region &consumerExecRegion = defaultProducer ? workerRegion : defaultRegion;
    Block *producerRoot = &producerExecRegion.front();
    Block *consumerRoot = &consumerExecRegion.front();
    SmallVector<Operation *> producerTerminators;
    producerExecRegion.walk([&](Operation *op) {
      if (isa<ttg::WarpReturnOp, ttg::WarpYieldOp>(op))
        producerTerminators.push_back(op);
    });
    SmallVector<Operation *> consumerTerminators;
    consumerExecRegion.walk([&](Operation *op) {
      if (isa<ttg::WarpReturnOp, ttg::WarpYieldOp>(op))
        consumerTerminators.push_back(op);
    });
    if (producerTerminators.empty() || consumerTerminators.empty())
      return ws.emitOpError("static partitions lost their terminators");

    funcBlocks.splice(continuation->getIterator(), workerRegion.getBlocks());
    Block *workerJoin =
        rewriter.createBlock(&funcBody, continuation->getIterator());
    funcBlocks.splice(continuation->getIterator(), defaultRegion.getBlocks());

    for (Operation *op : producerTerminators) {
      rewriter.setInsertionPoint(op);
      cf::BranchOp::create(rewriter, op->getLoc(), workerJoin);
      rewriter.eraseOp(op);
    }
    for (Operation *op : consumerTerminators) {
      if (auto yield = dyn_cast<ttg::WarpYieldOp>(op);
          yield && yield.getNumOperands() != 0)
        return op->emitOpError(
            "MUSA TLE static consumer must not yield values");
      rewriter.setInsertionPoint(op);
      cf::BranchOp::create(rewriter, op->getLoc(), continuation);
      rewriter.eraseOp(op);
    }

    Location loc = ws.getLoc();
    rewriter.eraseOp(ws);
    rewriter.setInsertionPointToEnd(dispatch);
    Value tid = LLVM::CallIntrinsicOp::create(
                    rewriter, loc, rewriter.getI32Type(),
                    rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"),
                    ValueRange{})
                    .getResult(0);
    Value boundaryValue =
        arith::ConstantIntOp::create(rewriter, loc, threadEnds.front(), 32);
    Value isProducer = arith::CmpIOp::create(
        rewriter, loc,
        defaultProducer ? arith::CmpIPredicate::ult
                        : arith::CmpIPredicate::uge,
        tid, boundaryValue);
    cf::CondBranchOp::create(rewriter, loc, isProducer, producerRoot,
                             ValueRange{}, workerJoin, ValueRange{});

    rewriter.setInsertionPointToEnd(workerJoin);
    Value isConsumer = arith::CmpIOp::create(
        rewriter, loc,
        defaultProducer ? arith::CmpIPredicate::uge
                        : arith::CmpIPredicate::ult,
        tid, boundaryValue);
    cf::CondBranchOp::create(rewriter, loc, isConsumer, consumerRoot,
                             ValueRange{}, continuation, ValueRange{});
    return success();
  }

  for (Region *workerRegion : workerRegions)
    funcBlocks.splice(continuation->getIterator(), workerRegion->getBlocks());
  funcBlocks.splice(continuation->getIterator(), defaultRegion.getBlocks());

  for (auto &returns : workerReturns) {
    for (ttg::WarpReturnOp op : returns) {
      rewriter.setInsertionPoint(op);
      cf::BranchOp::create(rewriter, op.getLoc(), continuation);
      rewriter.eraseOp(op);
    }
  }
  for (ttg::WarpYieldOp op : defaultYields) {
    if (op.getNumOperands() != 0)
      return op.emitOpError(
          "MUSA TLE static default partition must not yield values");
    rewriter.setInsertionPoint(op);
    cf::BranchOp::create(rewriter, op->getLoc(), continuation);
    rewriter.eraseOp(op);
  }

  SmallVector<Block *> rangeTargets;
  rangeTargets.reserve(workerRoots.size() + 1);
  rangeTargets.push_back(defaultRoot);
  rangeTargets.append(workerRoots.begin(), workerRoots.end());
  SmallVector<Block *> fallbackDispatches;
  fallbackDispatches.reserve(rangeTargets.size() - 1);
  for (size_t index = 1; index < rangeTargets.size(); ++index)
    fallbackDispatches.push_back(
        rewriter.createBlock(&funcBody, continuation->getIterator()));

  Location loc = ws.getLoc();
  rewriter.eraseOp(ws);
  rewriter.setInsertionPointToEnd(dispatch);
  Value tid =
      LLVM::CallIntrinsicOp::create(
          rewriter, loc, rewriter.getI32Type(),
          rewriter.getStringAttr("llvm.musa.read.ptx.sreg.tid.x"), ValueRange{})
          .getResult(0);
  for (size_t index = 0; index < rangeTargets.size(); ++index) {
    if (index != 0)
      rewriter.setInsertionPointToEnd(fallbackDispatches[index - 1]);
    Value end =
        arith::ConstantIntOp::create(rewriter, loc, threadEnds[index], 32);
    Value inRange = arith::CmpIOp::create(rewriter, loc,
                                          arith::CmpIPredicate::ult, tid, end);
    Block *fallback = index + 1 < rangeTargets.size()
                          ? fallbackDispatches[index]
                          : continuation;
    cf::CondBranchOp::create(rewriter, loc, inRange, rangeTargets[index],
                             ValueRange{}, fallback, ValueRange{});
  }
  return success();
}

class LowerWarpSpecializePass
    : public impl::TritonMUSAGPUTLELowerWarpSpecializeBase<
          LowerWarpSpecializePass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto stripExplicitCtaSyncMarkers = [&]() {
      module.walk([&](LLVM::CallIntrinsicOp call) {
        call->removeAttr(kExplicitCtaSyncAttr);
      });
    };
    SmallVector<ttg::WarpSpecializeOp> marked;
    module.walk([&](ttg::WarpSpecializeOp ws) {
      if (ws->hasAttr(kStaticWarpSpecializeAttr))
        marked.push_back(ws);
    });
    if (marked.empty()) {
      stripExplicitCtaSyncMarkers();
      return;
    }
    if (marked.size() != 1) {
      marked[1].emitOpError(
          "MUSA TLE static warp_specialize supports exactly one marked "
          "operation per module");
      return signalPassFailure();
    }

    auto func = marked.front()->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func) {
      marked.front().emitOpError(
          "MUSA TLE late lowering requires an LLVM function");
      return signalPassFailure();
    }

    // Unknown CTA-wide synchronization cannot be narrowed safely. Diagnose it
    // while the static partition and source location are still available.
    WalkResult checked = marked.front()->walk([&](LLVM::CallIntrinsicOp call) {
      if (call.getIntrin() != "llvm.musa.barrier0")
        return WalkResult::advance();
      call.emitOpError(
          "CTA barrier inside MUSA TLE static warp_specialize partition "
          "would wait for unrelated warps; use partition synchronization");
      return WalkResult::interrupt();
    });
    if (checked.wasInterrupted())
      return signalPassFailure();

    IRRewriter rewriter(&getContext());
    if (failed(lowerWarpGroupBarriers(func, marked.front(), rewriter)) ||
        failed(lowerStaticWarpSpecialize(func, marked.front(), rewriter))) {
      signalPassFailure();
      return;
    }
    stripExplicitCtaSyncMarkers();
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
