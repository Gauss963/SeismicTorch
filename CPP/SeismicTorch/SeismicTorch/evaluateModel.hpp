//
//  time_evaluation.hpp
//  SeismicTorch
//
//  Created by Gauss on 2023/7/31.
//

#ifndef evaluateModel_hpp
#define evaluateModel_hpp

#include <torch/script.h>
#include <stdio.h>

void evaluateModel(torch::jit::script::Module module);

#endif /* time_evaluation_hpp */
