#ifndef LLVM_TRANSFORMS_SCALAR_MLUNROLLPASS_H
#define LLVM_TRANSFORMS_SCALAR_MLUNROLLPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

class MLUnrollPass : public OptionalPassInfoMixin<MLUnrollPass> {
private:
    float GateThreshold;

public:
    MLUnrollPass(float Threshold = 0.45f) : GateThreshold(Threshold) {}

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

    static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_MLUNROLLPASS_H
