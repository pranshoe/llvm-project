//===-- MLFusionPass.cpp - Machine Learning Loop Fusion Pass ----*- C++ -*-===//
/// \file
/// This pass uses an embedded zero-latency XGBoost polyhedral cost model to predict
/// loop fusion profitability on adjacent sibling loops, preventing register thrashing
/// and eliminating intermediate buffer memory roundtrips.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/MLFusionPass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <vector>
#include <set>
#include <algorithm>

// Include transpiled C++ decision tree models for loop fusion
#include "predict_fusion_gate.h"

#define DEBUG_TYPE "ml-fusion"

using namespace llvm;

namespace {

/// Collects pointer operands accessed in a loop
std::set<Value*> collectPointers(Loop *L) {
    std::set<Value*> pointers;
    for (BasicBlock *BB : L->blocks()) {
        for (Instruction &I : *BB) {
            if (auto *LI = dyn_cast<LoadInst>(&I))
                pointers.insert(LI->getPointerOperand());
            else if (auto *SI = dyn_cast<StoreInst>(&I))
                pointers.insert(SI->getPointerOperand());
        }
    }
    return pointers;
}

/// Attach loop fusion directive metadata
void applyFusionMetadata(Loop *L1, Loop *L2) {
    if (!L1 || !L1->getHeader() || !L1->getHeader()->getTerminator() ||
        !L2 || !L2->getHeader() || !L2->getHeader()->getTerminator())
        return;
    LLVMContext &Ctx = L1->getHeader()->getContext();

    Metadata *FuseVals[] = {
        MDString::get(Ctx, "llvm.loop.fusion.enable")
    };
    MDNode *FuseMD = MDNode::get(Ctx, FuseVals);
    L1->getHeader()->getTerminator()->setMetadata("llvm.loop.fuse", FuseMD);
    L2->getHeader()->getTerminator()->setMetadata("llvm.loop.fuse", FuseMD);
}

} // end anonymous namespace

PreservedAnalyses MLFusionPass::run(Function &F, FunctionAnalysisManager &FAM) {
    if (F.isDeclaration()) return PreservedAnalyses::all();

    auto &SE  = FAM.getResult<ScalarEvolutionAnalysis>(F);
    auto &LI  = FAM.getResult<LoopAnalysis>(F);

    bool Modified = false;

    // Iterate through adjacent pairs of top-level loops
    auto &TopLoops = LI.getTopLevelLoops();
    if (TopLoops.size() < 2)
        return PreservedAnalyses::all();

    for (size_t idx = 0; idx + 1 < TopLoops.size(); ++idx) {
        Loop *L1 = TopLoops[idx];
        Loop *L2 = TopLoops[idx + 1];

        // 1. Trip Count Compatibility
        unsigned TC1 = SE.getSmallConstantTripCount(L1);
        unsigned TC2 = SE.getSmallConstantTripCount(L2);
        if (TC1 == 0 || TC2 == 0 || TC1 != TC2) {
            // Incompatible iteration domains
            continue;
        }

        // 2. Data Reuse & Buffer Sharing Analysis
        std::set<Value*> Pointers1 = collectPointers(L1);
        std::set<Value*> Pointers2 = collectPointers(L2);

        unsigned SharedBuffers = 0;
        for (Value *P : Pointers1) {
            if (Pointers2.count(P))
                SharedBuffers++;
        }

        // 3. Combined Opcode & Register Pressure Analysis
        unsigned numComputations = 0, numReductions = 0, totalAccesses = 0;
        unsigned opAdd = 0, opSub = 0, opMul = 0, opDiv = 0, opOther = 0;
        unsigned InstCount1 = 0, InstCount2 = 0;

        for (BasicBlock *BB : L1->blocks()) {
            InstCount1 += BB->size();
            for (Instruction &I : *BB) {
                if (isa<BinaryOperator>(&I)) {
                    numComputations++;
                    switch (I.getOpcode()) {
                        case Instruction::Add:
                        case Instruction::FAdd: opAdd++; break;
                        case Instruction::Sub:
                        case Instruction::FSub: opSub++; break;
                        case Instruction::Mul:
                        case Instruction::FMul: opMul++; break;
                        case Instruction::SDiv:
                        case Instruction::UDiv:
                        case Instruction::FDiv: opDiv++; break;
                        default: opOther++; break;
                    }
                } else if (isa<LoadInst>(&I)) {
                    totalAccesses++;
                } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
                    totalAccesses++;
                    if (isa<BinaryOperator>(SI->getValueOperand())) numReductions++;
                }
            }
        }

        for (BasicBlock *BB : L2->blocks()) {
            InstCount2 += BB->size();
            for (Instruction &I : *BB) {
                if (isa<BinaryOperator>(&I)) {
                    numComputations++;
                    switch (I.getOpcode()) {
                        case Instruction::Add:
                        case Instruction::FAdd: opAdd++; break;
                        case Instruction::Sub:
                        case Instruction::FSub: opSub++; break;
                        case Instruction::Mul:
                        case Instruction::FMul: opMul++; break;
                        case Instruction::SDiv:
                        case Instruction::UDiv:
                        case Instruction::FDiv: opDiv++; break;
                        default: opOther++; break;
                    }
                } else if (isa<LoadInst>(&I)) {
                    totalAccesses++;
                } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
                    totalAccesses++;
                    if (isa<BinaryOperator>(SI->getValueOperand())) numReductions++;
                }
            }
        }

        std::set<Value*> allPointers = Pointers1;
        allPointers.insert(Pointers2.begin(), Pointers2.end());
        unsigned numBuffers = std::max(1u, (unsigned)allPointers.size());
        uint64_t workingSetBytes = (uint64_t)numBuffers * (uint64_t)TC1 * sizeof(float);

        unsigned CombinedInsts = InstCount1 + InstCount2;
        float estimatedRegPressure = (float)CombinedInsts * 0.25f;
        float arithmeticOps = (float)(opAdd + opSub + opMul + opDiv);

        // 4. Build 27-D Feature Vector for Stage 1 XGBoost Fusion Gate Model
        float fusion_features[27] = {0.0f};
        fusion_features[0]  = std::log1p((float)workingSetBytes);
        fusion_features[1]  = (float)L1->getLoopDepth();
        fusion_features[2]  = std::log1p((float)TC1);
        fusion_features[3]  = 0.0f; // has_parametric_bounds
        fusion_features[4]  = (float)numComputations;
        fusion_features[5]  = (float)numReductions;
        fusion_features[6]  = (float)numBuffers;
        fusion_features[7]  = (float)totalAccesses;
        fusion_features[8]  = 0.0f; // max_access_offset
        fusion_features[9]  = (float)opAdd;
        fusion_features[10] = (float)opSub;
        fusion_features[11] = (float)opMul;
        fusion_features[12] = (float)opDiv;
        fusion_features[13] = (float)opOther;
        fusion_features[14] = (float)totalAccesses;
        fusion_features[15] = (float)numComputations / std::max(1u, (unsigned)L1->getLoopDepth());
        fusion_features[16] = (float)totalAccesses / std::max(1.0f, (float)numComputations);
        fusion_features[17] = (float)numReductions / std::max(1.0f, (float)numComputations);
        fusion_features[18] = estimatedRegPressure;
        fusion_features[19] = (estimatedRegPressure > 16.0f) ? 1.0f : 0.0f;
        fusion_features[20] = arithmeticOps;
        fusion_features[21] = (float)totalAccesses / std::max(1.0f, arithmeticOps);
        fusion_features[22] = (float)TC1;
        fusion_features[23] = 1.0f;
        fusion_features[24] = 1.0f;
        fusion_features[25] = 1.0f;
        fusion_features[26] = 1.0f;

        // Fast C++ Decision Tree Inference (< 10 nanoseconds)
        float p_fuse = model_fusion::predict_fusion_gate(fusion_features);

        std::string Name1 = L1->getHeader()->getName().str();
        std::string Name2 = L2->getHeader()->getName().str();
        if (Name1.empty()) Name1 = "loop_1";
        if (Name2.empty()) Name2 = "loop_2";

        LLVM_DEBUG(dbgs() << "[MLFusion] Function: " << F.getName() 
                          << " | Loops: " << Name1 << " & " << Name2 
                          << " | Fusion Prob: " << p_fuse << "\n");

        // Gated Decision (p_fuse >= GateThreshold && SharedBuffers > 0)
        if (p_fuse >= GateThreshold && SharedBuffers > 0) {
            outs() << ">>> [ML FUSION] Function: " << F.getName() 
                   << " | Fusing: " << Name1 << " & " << Name2 
                   << " (Shared Buffers: " << SharedBuffers 
                   << ", Confidence: " << format("%.1f", p_fuse * 100.0f) << "%)\n";

            applyFusionMetadata(L1, L2);
            Modified = true;
        } else {
            LLVM_DEBUG(dbgs() << ">>> [ML FUSION] Keeping loops separate: " 
                              << Name1 << " & " << Name2 
                              << " (Confidence: " << format("%.1f", p_fuse * 100.0f) << "%)\n");
        }
    }

    if (!Modified)
        return PreservedAnalyses::all();

    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
}
