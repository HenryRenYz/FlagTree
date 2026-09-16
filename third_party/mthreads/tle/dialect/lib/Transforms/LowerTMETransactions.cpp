#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "MUSATLE/Transforms/PipePartitionUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cstdint>
#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERTMETRANSACTIONS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttmg = triton::musa;

static FailureOr<int32_t>
resolveIssueThread(ttmg::AsyncTMECopyGlobalToLocalOp copy) {
  // Standalone completion TME keeps its pre-existing static-WS contract.
  // Only the marked MUSA TLE static warp-specialize path uses arbitrary
  // default/worker partition starts.
  auto ws = copy->getParentOfType<ttg::WarpSpecializeOp>();
  if (ws && !ws->hasAttr("musa_tle.static_warp_specialize")) {
    Region *copyRegion = copy->getParentRegion();
    if (ws.getDefaultRegion().isAncestor(copyRegion))
      return copy.emitOpError(
          "mthreads TLE completion TME copy must be in the producer partition");
    std::optional<unsigned> worker;
    for (auto [index, region] : llvm::enumerate(ws.getPartitionRegions())) {
      if (region->isAncestor(copyRegion)) {
        worker = index;
        break;
      }
    }
    if (!worker || *worker != 0)
      return copy.emitOpError(
          "mthreads TLE completion TME copy must be in the producer partition");
    ModuleOp module = copy->getParentOfType<ModuleOp>();
    auto numWarps = module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
    auto threadsPerWarp =
        module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
    if (!numWarps || !threadsPerWarp || numWarps.getInt() <= 0 ||
        threadsPerWarp.getInt() <= 0)
      return copy.emitOpError(
          "mthreads TLE producer issue thread requires ttg.num-warps and "
          "ttg.threads-per-warp");
    int64_t value = numWarps.getInt() * threadsPerWarp.getInt();
    if (value > std::numeric_limits<int32_t>::max())
      return copy.emitOpError(
          "mthreads TLE producer issue thread exceeds int32 range");
    return static_cast<int32_t>(value);
  }
  FailureOr<int32_t> startThread =
      triton::musa_tle::getPipePartitionStartThread(copy);
  if (failed(startThread))
    return failure();
  return *startThread;
}

struct CompletionGroupMetadata {
  int64_t id = 0;
  int64_t memberIndex = 0;
  int64_t memberCount = 0;
  int64_t totalTransactionBytes = 0;
};

struct CompletionGroupInstance {
  CompletionGroupMetadata metadata;
  Value barrier;
  Value predicate;
  int32_t issueThread = 0;
  int64_t accumulatedBytes = 0;
  SmallVector<bool> seenMembers;
  SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> members;
};

static FailureOr<std::optional<CompletionGroupMetadata>>
getCompletionGroupMetadata(ttmg::AsyncTMECopyGlobalToLocalOp copy) {
  Attribute raw = copy->getAttr(ttmg::kTLECompletionGroupAttr);
  if (!raw)
    return std::optional<CompletionGroupMetadata>{};

  auto group = dyn_cast<DenseI64ArrayAttr>(raw);
  if (!group || group.size() != 4) {
    copy.emitOpError("MUSA TLE completion group metadata requires group id, "
                     "member index, member count, and total bytes");
    return failure();
  }

  ArrayRef<int64_t> values = group.asArrayRef();
  CompletionGroupMetadata metadata{values[0], values[1], values[2], values[3]};
  if (metadata.id < 0 || metadata.memberIndex < 0 ||
      metadata.memberCount <= 0 ||
      metadata.memberCount > std::numeric_limits<int32_t>::max() ||
      metadata.memberIndex >= metadata.memberCount ||
      metadata.totalTransactionBytes <= 0 ||
      metadata.totalTransactionBytes > std::numeric_limits<int32_t>::max()) {
    copy.emitOpError("MUSA TLE completion group metadata is out of range");
    return failure();
  }
  return std::optional<CompletionGroupMetadata>{metadata};
}

static FailureOr<int32_t>
getTransactionBytes(ttmg::AsyncTMECopyGlobalToLocalOp copy) {
  auto expectBytes =
      copy->getAttrOfType<IntegerAttr>(ttmg::kTLEExpectBytesAttr);
  if (!expectBytes || !expectBytes.getType().isInteger(32) ||
      expectBytes.getInt() <= 0) {
    copy.emitOpError(
        "MUSA TLE completion TME copy requires positive expect_bytes");
    return failure();
  }
  return static_cast<int32_t>(expectBytes.getInt());
}

static bool haveEquivalentPredicates(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  auto lhsConstant = lhs.getDefiningOp<arith::ConstantOp>();
  auto rhsConstant = rhs.getDefiningOp<arith::ConstantOp>();
  return lhsConstant && rhsConstant &&
         lhsConstant.getValue() == rhsConstant.getValue();
}

// A multi-field pipe commit may be wrapped by one predicated prefetch
// `scf.if`.  The payload copies are still one logical transaction, but the
// producer arrival must be emitted after the region so it is not stranded in
// a branch block.  Keep this relaxation narrow: all fields must be in the same
// branch of one `scf.if` directly nested in the same `scf.for` body.  A group
// split across sibling branches, loops, or arbitrary regions remains
// rejected because an unconditional arrival would be able to deadlock the
// opposite path.
struct GroupBranchContext {
  scf::IfOp ifOp;
  bool thenBranch;
  scf::ForOp forOp;
};

static bool isConstantTrue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return constant.value() == 1;
  return false;
}

static std::optional<GroupBranchContext>
getGroupedBranchContext(ArrayRef<ttmg::AsyncTMECopyGlobalToLocalOp> copies) {
  if (copies.empty())
    return std::nullopt;

  auto findNearestIf = [](Operation *op)
      -> std::optional<std::pair<scf::IfOp, bool>> {
    for (Region *region = op->getParentRegion(); region;) {
      Operation *parent = region->getParentOp();
      if (!parent)
        break;
      if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
        if (region == &ifOp.getThenRegion())
          return std::make_pair(ifOp, true);
        if (ifOp.elseBlock() && region == &ifOp.getElseRegion())
          return std::make_pair(ifOp, false);
        return std::nullopt;
      }
      region = parent->getParentRegion();
    }
    return std::nullopt;
  };

  auto firstCopy = copies.front();
  auto first = findNearestIf(firstCopy.getOperation());
  if (!first)
    return std::nullopt;
  auto [ifOp, thenBranch] = *first;
  // The branch operation itself must be in the body of one loop.  This avoids
  // treating a warp-specialize/SCF region or a nested loop as a single phase
  // domain when the barrier state is actually carried elsewhere.
  auto forOp = ifOp->getParentOfType<scf::ForOp>();
  if (!forOp || ifOp->getParentRegion() != &forOp.getRegion())
    return std::nullopt;

  Block *copyBlock = copies.front()->getBlock();
  if (!copyBlock)
    return std::nullopt;
  for (auto copy : copies) {
    auto context = findNearestIf(copy.getOperation());
    if (!context || context->first != ifOp || context->second != thenBranch ||
        copy->getBlock() != copyBlock)
      return std::nullopt;
    if (!copy->getBlock()->getParent() ||
        copy->getParentOfType<scf::ForOp>() != forOp)
      return std::nullopt;
  }
  return GroupBranchContext{ifOp, thenBranch, forOp};
}

static Value combineBranchAndCopyPred(OpBuilder &builder, Location loc,
                                      GroupBranchContext context, Value pred) {
  Value branchPred = context.ifOp.getCondition();
  if (!context.thenBranch) {
    Value one = arith::ConstantIntOp::create(builder, loc, 1, 1);
    branchPred = arith::XOrIOp::create(builder, loc, branchPred, one);
  }
  if (isConstantTrue(branchPred))
    return pred;
  if (isConstantTrue(pred))
    return branchPred;
  return arith::AndIOp::create(builder, loc, branchPred, pred);
}

// Return the contiguous grouped payloads immediately preceding `final` in
// the same lexical copy stream.  Barrier SSA values are commonly reused by
// every loop iteration, so grouping all copies by barId alone would merge
// distinct commits.  Contiguity plus the final marker is the existing pipe
// protocol's group boundary.
static SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp>
collectLegacyGroupedCopies(
    ttmg::AsyncTMECopyGlobalToLocalOp final,
    ArrayRef<ttmg::AsyncTMECopyGlobalToLocalOp> copies) {
  SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> group;
  if (!final->hasAttr(ttmg::kTLEGroupedCompletionFinalAttr))
    return group;
  auto finalIt = llvm::find(copies, final);
  if (finalIt == copies.end())
    return group;
  group.push_back(final);
  while (finalIt != copies.begin()) {
    auto previous = std::prev(finalIt);
    auto previousCopy = *previous;
    if (!previousCopy->hasAttr(ttmg::kTLEGroupedCompletionAttr) ||
        previousCopy.getBarId() != final.getBarId())
      break;
    group.push_back(*previous);
    finalIt = previous;
  }
  std::reverse(group.begin(), group.end());
  return group;
}

class LowerTMETransactionsPass
    : public impl::TritonMUSAGPUTLELowerTMETransactionsBase<
          LowerTMETransactionsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<Block *> blocks;
    llvm::SmallPtrSet<Block *, 8> seenBlocks;
    module.walk([&](ttmg::AsyncTMECopyGlobalToLocalOp copy) {
      if ((copy->hasAttr(ttmg::kTLEExpectBytesAttr) ||
           copy->hasAttr(ttmg::kTLECompletionGroupAttr)) &&
          seenBlocks.insert(copy->getBlock()).second)
        blocks.push_back(copy->getBlock());
    });
    // The experimental per-copy protocol is intentionally handled as a
    // separate pass over the copies.  This keeps the established grouped
    // completion contract (one aggregate add.trans emitted by its leader)
    // byte-for-byte unchanged.  A group is identified by the SSA barrier
    // value; LowerPipe reuses one barrier.index value for all payload fields.
    // Every copy contributes its own transaction, and exactly one final copy
    // emits the producer arrival after all asynchronous copies have been
    // issued.
    for (ttmg::AsyncTMECopyGlobalToLocalOp copy : perCopyCopies) {
      if (!copy->hasAttr(ttmg::kTLEExpectBytesAttr)) {
        copy.emitOpError("per-copy grouped completion requires positive "
                         "expect_bytes on every payload");
        signalPassFailure();
        return;
      }
    }

    struct PerCopyGroup {
      Value barrier;
      SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> copies;
      SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> finals;
    };
    SmallVector<PerCopyGroup> perCopyGroups;
    for (ttmg::AsyncTMECopyGlobalToLocalOp copy : perCopyCopies) {
      PerCopyGroup *group = nullptr;
      for (PerCopyGroup &candidate : perCopyGroups) {
        if (candidate.barrier == copy.getBarId()) {
          group = &candidate;
          break;
        }
      }
      if (!group) {
        perCopyGroups.push_back(PerCopyGroup{copy.getBarId(), {}, {}});
        group = &perCopyGroups.back();
      }
      group->copies.push_back(copy);
      if (copy->hasAttr(ttmg::kTLEGroupedCompletionFinalAttr))
        group->finals.push_back(copy);
    }

    for (PerCopyGroup &group : perCopyGroups) {
      if (group.copies.size() < 2) {
        group.copies.front().emitOpError(
            "per-copy grouped completion requires at least two payloads");
        signalPassFailure();
        return;
      }
      if (group.finals.size() != 1) {
        group.copies.front().emitOpError(
            "per-copy grouped completion requires exactly one final payload");
        signalPassFailure();
        return;
      }
      auto final = group.finals.front();
      auto branchContext = getGroupedBranchContext(group.copies);
      // Arriving before a later payload has been issued would permit the
      // barrier to become ready too early.  The ordinary form is restricted
      // to one block.  A narrowly proven predicated branch may defer the
      // arrival until after its `scf.if`; the branch predicate is retained on
      // the deferred op so a not-taken prefetch does not signal a transaction
      // that was never issued.
      Block *finalBlock = final->getBlock();
      for (auto copy : group.copies) {
        bool orderedInBlock = copy->getBlock() == finalBlock &&
                              (copy == final || copy->isBeforeInBlock(final));
        bool orderedInBranch = branchContext &&
                               (copy == final || copy->isBeforeInBlock(final));
        if ((!branchContext && !orderedInBlock) ||
            (branchContext && !orderedInBranch)) {
          final.emitOpError("per-copy grouped completion final payload must "
                            "follow all other payload copies in one block");
          signalPassFailure();
          return;
        }
      }

      FailureOr<int32_t> issueThread = resolveIssueThread(final);
      if (failed(issueThread)) {
        signalPassFailure();
        return;
      }
      auto issueThreadAttr = rewriter.getI32IntegerAttr(*issueThread);
      auto explicitCompletionAttr = rewriter.getUnitAttr();
      for (auto copy : group.copies) {
        auto expectBytes =
            copy->getAttrOfType<IntegerAttr>(ttmg::kTLEExpectBytesAttr);
        if (!expectBytes || !expectBytes.getType().isInteger(32) ||
            expectBytes.getInt() <= 0) {
          copy.emitOpError(
              "per-copy grouped completion requires positive expect_bytes");
          signalPassFailure();
          return;
        }
        Location loc = copy.getLoc();
        copy->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        copy->setAttr(ttmg::kTMEExplicitCompletionAttr,
                      explicitCompletionAttr);
        rewriter.setInsertionPoint(copy);
        Value bytes = arith::ConstantIntOp::create(
            rewriter, loc, expectBytes.getInt(), 32);
        auto addTrans = ttmg::BarrierAddTransOp::create(
            rewriter, loc, copy.getBarId(), bytes, copy.getPred());
        addTrans->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        addTrans->setAttr(ttmg::kTMEExplicitCompletionAttr,
                          explicitCompletionAttr);
        if (copy == final) {
          if (branchContext)
            rewriter.setInsertionPointAfter(branchContext->ifOp);
          else
            rewriter.setInsertionPointAfter(copy);
          Value arrivePred = copy.getPred();
          if (branchContext)
            arrivePred = combineBranchAndCopyPred(
                rewriter, loc, *branchContext, arrivePred);
          auto arrive = ttmg::ArriveBarrierNoRetOp::create(
              rewriter, loc, copy.getBarId(), arrivePred);
          arrive->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
          arrive->setAttr(ttmg::kTMEExplicitCompletionAttr,
                          explicitCompletionAttr);
        }
        copy->removeAttr(ttmg::kTLEExpectBytesAttr);
        copy->removeAttr(ttmg::kTLEGroupedPerCopyCompletionAttr);
        copy->removeAttr(ttmg::kTLEGroupedCompletionFinalAttr);
        // The legacy pass below iterates over the original `copies` snapshot.
        // Mark this operation explicitly because the per-copy lowering removes
        // its marker attributes before that second pass runs.
        loweredPerCopyCopies.insert(copy.getOperation());
      }
    }

    SmallVector<std::unique_ptr<CompletionGroupInstance>> groups;
    for (Block *block : blocks) {
      llvm::DenseMap<int64_t, std::unique_ptr<CompletionGroupInstance>> open;
      for (Operation &operation : *block) {
        auto copy = dyn_cast<ttmg::AsyncTMECopyGlobalToLocalOp>(&operation);
        if (!copy || (!copy->hasAttr(ttmg::kTLEExpectBytesAttr) &&
                      !copy->hasAttr(ttmg::kTLECompletionGroupAttr)))
          continue;

        FailureOr<int32_t> bytes = getTransactionBytes(copy);
        FailureOr<int32_t> issueThread = resolveIssueThread(copy);
        FailureOr<std::optional<CompletionGroupMetadata>> metadata =
            getCompletionGroupMetadata(copy);
        if (failed(bytes) || failed(issueThread) || failed(metadata)) {
          signalPassFailure();
          return;
        }

        if (!*metadata) {
          auto instance = std::make_unique<CompletionGroupInstance>();
          instance->metadata = CompletionGroupMetadata{0, 0, 1, *bytes};
          instance->barrier = copy.getBarId();
          instance->predicate = copy.getPred();
          instance->issueThread = *issueThread;
          instance->accumulatedBytes = *bytes;
          instance->seenMembers.push_back(true);
          instance->members.push_back(copy);
          groups.push_back(std::move(instance));
          continue;
        }

        const CompletionGroupMetadata &member = **metadata;
        std::unique_ptr<CompletionGroupInstance> &instance = open[member.id];
        if (!instance) {
          instance = std::make_unique<CompletionGroupInstance>();
          instance->metadata = member;
          instance->barrier = copy.getBarId();
          instance->predicate = copy.getPred();
          instance->issueThread = *issueThread;
          instance->seenMembers.assign(member.memberCount, false);
        }

        if (instance->metadata.memberCount != member.memberCount ||
            instance->metadata.totalTransactionBytes !=
                member.totalTransactionBytes) {
          copy.emitOpError(
              "MUSA TLE completion group members have inconsistent metadata");
          signalPassFailure();
          return;
        }
        if (instance->barrier != copy.getBarId() ||
            !haveEquivalentPredicates(instance->predicate, copy.getPred()) ||
            instance->issueThread != *issueThread) {
          copy.emitOpError("MUSA TLE completion group members must use the "
                           "same barrier, predicate, and issue thread");
          signalPassFailure();
          return;
        }
        if (instance->seenMembers[member.memberIndex]) {
          copy.emitOpError(
              "MUSA TLE completion group contains a duplicate member index");
          signalPassFailure();
          return;
        }
        instance->seenMembers[member.memberIndex] = true;
        instance->members.push_back(copy);
        instance->accumulatedBytes += *bytes;
        if (instance->accumulatedBytes > std::numeric_limits<int32_t>::max()) {
          copy.emitOpError(
              "MUSA TLE completion group bytes exceed the positive i32 range");
          signalPassFailure();
          return;
        }

        if (instance->members.size() ==
            static_cast<size_t>(member.memberCount)) {
          if (instance->accumulatedBytes != member.totalTransactionBytes) {
            copy.emitOpError(
                "MUSA TLE completion group member bytes do not match the "
                "declared total");
            signalPassFailure();
            return;
          }
          groups.push_back(std::move(instance));
          open.erase(member.id);
        }
      }
      if (!expectBytes && !groupedCompletion) {
        copy.emitOpError("mthreads TLE completion TME copy requires transaction "
                         "bytes or a grouped completion marker");
        signalPassFailure();
        return;
      }

      if (!open.empty()) {
        open.begin()->second->members.front().emitOpError(
            "MUSA TLE completion group is incomplete within its block");
        signalPassFailure();
        return;
      }
    }

    for (const std::unique_ptr<CompletionGroupInstance> &ownedGroup : groups) {
      CompletionGroupInstance &group = *ownedGroup;
      ttmg::AsyncTMECopyGlobalToLocalOp first = group.members.front();
      ttmg::AsyncTMECopyGlobalToLocalOp last = group.members.back();
      Location loc = first.getLoc();
      auto issueThreadAttr = rewriter.getI32IntegerAttr(group.issueThread);
      auto explicitCompletionAttr = rewriter.getUnitAttr();

      rewriter.setInsertionPoint(first);
      Value bytes = arith::ConstantIntOp::create(
          rewriter, loc, group.metadata.totalTransactionBytes, 32);
      auto addTrans = ttmg::BarrierAddTransOp::create(
          rewriter, loc, group.barrier, bytes, group.predicate);

      ttmg::ArriveBarrierNoRetOp arrive;
      if (!first->hasAttr(ttmg::kTLEPipeDeferredArrivalAttr)) {
        rewriter.setInsertionPointAfter(last);
        arrive = ttmg::ArriveBarrierNoRetOp::create(
            rewriter, last.getLoc(), group.barrier, group.predicate);
      }

      for (ttmg::AsyncTMECopyGlobalToLocalOp copy : group.members) {
        copy->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        copy->setAttr(ttmg::kTMEExplicitCompletionAttr, explicitCompletionAttr);
        copy->removeAttr(ttmg::kTLEExpectBytesAttr);
        copy->removeAttr(ttmg::kTLECompletionGroupAttr);
        copy->removeAttr(ttmg::kTLEPipeDeferredArrivalAttr);
      }
      for (Operation *op : {addTrans.getOperation(), arrive.getOperation()}) {
        if (!op)
          continue;
        op->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        op->setAttr(ttmg::kTMEExplicitCompletionAttr, explicitCompletionAttr);
      }
    }

  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
