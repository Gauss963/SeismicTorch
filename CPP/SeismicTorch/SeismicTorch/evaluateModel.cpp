#include <torch/script.h>
#include <iostream>
#include <memory>

#include <vector>
#include <chrono>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include "evaluateModel.hpp"

void evaluateModel()
{
    
    torch::jit::script::Module module;
    module = torch::jit::load("../../model/EarthquakeCNN_TS.pt");
    
    
    std::vector<double> durations(10000); // 儲存每次執行的時間

    for(int i = 0; i < 10000; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        
        at::Tensor input = torch::rand({1, 1, 80, 3});
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        at::Tensor output = module.forward(inputs).toTensor();
        at::Tensor softmax_output = torch::softmax(output, -1);
        
        
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        durations[i] = diff.count(); // 將時間轉換為毫秒
    }
    
    // 計算平均時間
    double sum = std::accumulate(durations.begin(), durations.end(), 0.0);
    double average_time = sum / durations.size();
    
    // 找出最大時間
    double max_time = *std::max_element(durations.begin(), durations.end());
    double sq_sum = std::inner_product(durations.begin(), durations.end(), durations.begin(), 0.0);
    
    // 計算標準差
    double std_dev = std::sqrt(sq_sum / durations.size() - average_time * average_time);
    double pi_conversion_factor = 0.4072483731;

    std::cout << "If execute on the Macbook 2017, check here\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << std::right << "Maximum time on Macbook is: " << std::setw(15) << std::fixed << std::setprecision(6) << max_time << " ms.\n";
    std::cout << std::right << "Average time on Macbook is: " << std::setw(15) << std::fixed << std::setprecision(6) << average_time << " ms.\n";
    std::cout << std::right << "Standard dev on Macbook is: " << std::setw(15) << std::fixed << std::setprecision(6) << std_dev << " ms.\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::right << "Maximum time on Raspberry Pi 4b should be: " << std::setw(15) << std::fixed << std::setprecision(6) << max_time / pi_conversion_factor << " ms.\n";
    std::cout << std::right << "Average time on Raspberry Pi 4b should be: " << std::setw(15) << std::fixed << std::setprecision(6) << average_time / pi_conversion_factor << " ms.\n";
    std::cout << std::right << "Standard dev on Raspberry Pi 4b should be: " << std::setw(15) << std::fixed << std::setprecision(6) << std_dev / std::sqrt(pi_conversion_factor) << " ms.\n";
    std::cout << std::string(80, '=') << "\n";
}
