#ifndef AI_ENGINE_HPP
#define AI_ENGINE_HPP

#include <torch/torch.h>
#include <torch/script.h>
#include <string>

// Base AI Engine class
class AIEngine {
public:
    AIEngine(const std::string& model_path);
    torch::Tensor predict(torch::Tensor input);

private:
    torch::jit::script::Module model;
};

// Model selection function
AIEngine* load_model(const std::string& model_type);

#endif
