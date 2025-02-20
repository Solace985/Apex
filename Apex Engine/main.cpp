#include <torch/script.h>
#include <iostream>
#include <vector>

extern "C" {
    float* predict(float* input_data, int size) {
        torch::jit::script::Module model = torch::jit::load("models/lstm.onnx");

        // Convert input to tensor
        std::vector<float> input_vector(input_data, input_data + size);
        torch::Tensor input_tensor = torch::from_blob(input_vector.data(), {1, 30, 8});

        // Make a prediction
        torch::Tensor output = model.forward({input_tensor}).toTensor();

        // Convert to C-style array
        float* result = new float[3];
        std::memcpy(result, output.data_ptr<float>(), 3 * sizeof(float));
        return result;
    }
}
