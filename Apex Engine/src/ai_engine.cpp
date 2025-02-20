#include "ai_engine.hpp"

AIEngine::AIEngine(const std::string& model_path) {
    try {
        model = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }
}

torch::Tensor AIEngine::predict(torch::Tensor input) {
    input = input.unsqueeze(0);  // Add batch dimension
    torch::Tensor output = model.forward({input}).toTensor();
    return output;
}

// Dynamic Model Loader
AIEngine* load_model(const std::string& model_type) {
    if (model_type == "lstm") return new AIEngine("models/lstm.onnx");
    if (model_type == "transformer") return new AIEngine("models/transformer.onnx");
    if (model_type == "cnn") return new AIEngine("models/cnn.onnx");
    if (model_type == "hybrid") return new AIEngine("models/hybrid.onnx");
    if (model_type == "maddpg") return new AIEngine("models/rl_maddpg.onnx");
    if (model_type == "qlearning") return new AIEngine("models/rl_qlearning.onnx");
    if (model_type == "sentiment") return new AIEngine("models/sentiment.onnx");
    if (model_type == "fundamental") return new AIEngine("models/fundamental.onnx");
    return nullptr;
}
