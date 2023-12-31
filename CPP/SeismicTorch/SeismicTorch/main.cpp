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
#include "sacio.hpp"

int main()
{
    torch::jit::script::Module module;
    try
    {
        module = torch::jit::load("../../model/EarthquakeCNN_TS.pt");
    }
    catch (const c10::Error& e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "model loaded" << std::endl;

    // 創建一個輸入張量
    at::Tensor input = torch::rand({1, 1, 80, 3});

    // 將輸入張量放入一個IValue vector中
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // 進行前向運算
    at::Tensor output = module.forward(inputs).toTensor();
    
    // 使用softmax獲得最後的輸出
    at::Tensor softmax_output = torch::softmax(output, -1);
    std::cout << softmax_output << '\n';
    
    std::vector<float> softmax_vector(softmax_output.data_ptr<float>(), softmax_output.data_ptr<float>() + softmax_output.numel());
    
    evaluateModel(module);
    
    
    SACHEAD hd;
    const char *filenameX = "../../data_QSIS_Event/5AFE5/RCEC.08f.5AFE5.TW.C2.HLX.2022.01.03.09.46.37.sac";
    const char *filenameY = "../../data_QSIS_Event/5AFE5/RCEC.08f.5AFE5.TW.C2.HLY.2022.01.03.09.46.37.sac";
    const char *filenameZ = "../../data_QSIS_Event/5AFE5/RCEC.08f.5AFE5.TW.C2.HLZ.2022.01.03.09.46.37.sac";
    
    float *dataX = read_sac(filenameX, &hd);
    float *dataY = read_sac(filenameY, &hd);
    float *dataZ = read_sac(filenameZ, &hd);
    
    if (dataX == NULL || dataY == NULL || dataZ == NULL)
    {
        std::cerr << "Error reading the SAC file." << std::endl;
        return -1;
    }
    else
    {
        std::cout << "sac file loaded" << std::endl;
    }
    
    std::vector<double> vecDataX(dataX, dataX + hd.npts);
    std::vector<double> vecDataY(dataY, dataY + hd.npts);
    std::vector<double> vecDataZ(dataZ, dataZ + hd.npts);
    
    std::cout << "X direction vector size: " << vecDataX.size() << std::endl;
    std::cout << "Y direction vector size: " << vecDataY.size() << std::endl;
    std::cout << "Z direction vector size: " << vecDataZ.size() << std::endl;
    
    for (int i = 0; i < vecDataX.size(); i++)
    {
        std::cout << vecDataX[i] << " ";
        std::cout << vecDataY[i] << " ";
        std::cout << vecDataZ[i] << std::endl;
    }
    
    free(dataX);
    free(dataY);
    free(dataZ);
    return 0;
}
