#include <torch/script.h>
#include <iostream>
#include <memory>

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
    
    return 0;
}
