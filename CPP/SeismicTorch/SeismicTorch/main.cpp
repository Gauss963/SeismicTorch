#include <torch/script.h>
#include <iostream>
#include <memory>

#include <vector>
#include <chrono>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iomanip>

torch::jit::script::Module ReadModel()
{
    torch::jit::script::Module module;
    module = torch::jit::load("../../model/EarthquakeCNN_TS.pt");
    return module;
    
}

int main() {
    torch::jit::script::Module module;
    try {
        // 載入模型，"EarthquakeCNN_TS.pt"應該在您的執行目錄下
        module = torch::jit::load("../../model/EarthquakeCNN_TS.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "model loaded\n";

    // 創建一個輸入張量
    at::Tensor input = torch::rand({1, 1, 80, 3});

    // 將輸入張量放入一個IValue vector中
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // 進行前向運算
    at::Tensor output = module.forward(inputs).toTensor();
    
    // 使用softmax獲得最後的輸出
    // at::Tensor softmax_output = torch::softmax(output, /*dim=*/-1);
    at::Tensor softmax_output = torch::softmax(output, -1);
    std::cout << softmax_output << '\n';
    
    
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

//    // 計算平均時間
//    double sum = std::accumulate(durations.begin(), durations.end(), 0.0);
//    double average = sum / durations.size();
//
//    // 找出最大時間
//    double max_duration = *std::max_element(durations.begin(), durations.end());
//
//    // 計算標準差
//    double sq_sum = std::inner_product(durations.begin(), durations.end(), durations.begin(), 0.0);
//    double stddev = std::sqrt(sq_sum / durations.size() - average * average);
//
//    std::cout << "Average time: " << average << " ms\n";
//    std::cout << "Max time: " << max_duration << " ms\n";
//    std::cout << "Standard deviation: " << stddev << " ms\n";
    
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
    
    return 0;
}
