#ifndef LLVM_TRANSFORMS_SCALAR_MLFUSIONPASS_H
#define LLVM_TRANSFORMS_SCALAR_MLFUSIONPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

class MLFusionPass : public OptionalPassInfoMixin<MLFusionPass> {
private:
    float GateThreshold;

public:
    MLFusionPass(float Threshold = 0.50f) : GateThreshold(Threshold) {}

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

    static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_MLFUSIONPASS_H
