#ifndef LLVM_TRANSFORMS_SCALAR_MLTILINGPASS_H
#define LLVM_TRANSFORMS_SCALAR_MLTILINGPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm{
    class Function;

    class MLTilingPass : public OptionalPassInfoMixin<MLTilingPass> {
        private:
            float GateThreshold;

        public:
            MLTilingPass(float Threshold = 0.40f) : GateThreshold(Threshold){}

            PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

            static bool isRequired() { return true; }

    };

}

#endif