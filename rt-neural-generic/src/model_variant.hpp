#include <variant>
#include <RTNeural/RTNeural.h>

struct NullModel { static constexpr int input_size = 0; static constexpr int output_size = 0; };
// from RTNeural template gen
using ModelType_GRU_8_1 = RTNeural::ModelT<float, 1, 1, RTNeural::GRULayerT<float, 1, 8>, RTNeural::DenseT<float, 8, 1>>;
using ModelType_LSTM_32_2 = RTNeural::ModelT<float, 2, 2, RTNeural::LSTMLayerT<float, 2, 32>, RTNeural::DenseT<float, 32, 2>>;
using ModelType_LSTM_16_2 = RTNeural::ModelT<float, 2, 2, RTNeural::LSTMLayerT<float, 2, 16>, RTNeural::DenseT<float, 16, 2>>;
using ModelType_LSTM_8_2 = RTNeural::ModelT<float, 2, 2, RTNeural::LSTMLayerT<float, 2, 8>, RTNeural::DenseT<float, 8, 2>>;
using ModelType_GRU_16_1 = RTNeural::ModelT<float, 1, 1, RTNeural::GRULayerT<float, 1, 16>, RTNeural::DenseT<float, 16, 1>>;
using ModelType_GRU_32_1 = RTNeural::ModelT<float, 1, 1, RTNeural::GRULayerT<float, 1, 32>, RTNeural::DenseT<float, 32, 1>>;
using ModelType_GRU_16_2 = RTNeural::ModelT<float, 2, 2, RTNeural::GRULayerT<float, 2, 16>, RTNeural::DenseT<float, 16, 2>>;
using ModelType_GRU_32_2 = RTNeural::ModelT<float, 2, 2, RTNeural::GRULayerT<float, 2, 32>, RTNeural::DenseT<float, 32, 2>>;
using ModelType_LSTM_8_1 = RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, 8>, RTNeural::DenseT<float, 8, 1>>;
using ModelType_LSTM_32_1 = RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, 32>, RTNeural::DenseT<float, 32, 1>>;
using ModelType_LSTM_16_1 = RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, 16>, RTNeural::DenseT<float, 16, 1>>;
using ModelType_GRU_8_2 = RTNeural::ModelT<float, 2, 2, RTNeural::GRULayerT<float, 2, 8>, RTNeural::DenseT<float, 8, 2>>;
// extra from aidadsp
using ModelType_GRU_12_1 = RTNeural::ModelT<float, 1, 1, RTNeural::GRULayerT<float, 1, 12>, RTNeural::DenseT<float, 12, 1>>;
using ModelType_LSTM_12_1 = RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, 12>, RTNeural::DenseT<float, 12, 1>>;
using ModelType_LSTM_20_1 = RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, 20>, RTNeural::DenseT<float, 20, 1>>;
using ModelType_LSTM_40_1 = RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, 40>, RTNeural::DenseT<float, 40, 1>>;

using ModelVariantType = std::variant<
// null
NullModel,
// from RTNeural template gen
ModelType_GRU_8_1,ModelType_LSTM_32_2,ModelType_LSTM_16_2,ModelType_LSTM_8_2,
ModelType_GRU_16_1,ModelType_GRU_32_1,ModelType_GRU_16_2,ModelType_GRU_32_2,
ModelType_LSTM_8_1,ModelType_LSTM_32_1,ModelType_LSTM_16_1,ModelType_GRU_8_2,
// from aidadsp
ModelType_GRU_12_1,ModelType_LSTM_12_1,ModelType_LSTM_20_1,ModelType_LSTM_40_1
>;

enum ModelUnionType {
    // null
    kNullModel,
    // from RTNeural template gen
    kModelType_GRU_8_1,
    kModelType_LSTM_32_2,
    kModelType_LSTM_16_2,
    kModelType_LSTM_8_2,
    kModelType_GRU_16_1,
    kModelType_GRU_32_1,
    kModelType_GRU_16_2,
    kModelType_GRU_32_2,
    kModelType_LSTM_8_1,
    kModelType_LSTM_32_1,
    kModelType_LSTM_16_1,
    kModelType_GRU_8_2,
    // from aidadsp
    kModelType_GRU_12_1,
    kModelType_LSTM_12_1,
    kModelType_LSTM_20_1,
    kModelType_LSTM_40_1,
};

union ModelUnion {
    void* ptr;
    // from RTNeural template gen
    ModelType_GRU_8_1* gru_8_1;
    ModelType_LSTM_32_2* lstm_32_2;
    ModelType_LSTM_16_2* lstm_16_2;
    ModelType_LSTM_8_2* lstm_8_2;
    ModelType_GRU_16_1* gru_16_1;
    ModelType_GRU_32_1* gru_32_1;
    ModelType_GRU_16_2* gru_16_2;
    ModelType_GRU_32_2* gru_32_2;
    ModelType_LSTM_8_1* lstm_8_1;
    ModelType_LSTM_32_1* lstm_32_1;
    ModelType_LSTM_16_1* lstm_16_1;
    ModelType_GRU_8_2* gru_8_2;
    // from aidadsp
    ModelType_GRU_12_1* gru_12_1;
    ModelType_LSTM_12_1* lstm_12_1;
    ModelType_LSTM_20_1* lstm_20_1;
    ModelType_LSTM_40_1* lstm_40_1;
};

// from RTNeural template gen
inline bool is_model_type_ModelType_GRU_8_1 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "gru";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 8;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 1;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_LSTM_32_2 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "lstm";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 32;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 2;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_LSTM_16_2 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "lstm";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 16;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 2;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_LSTM_8_2 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "lstm";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 8;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 2;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_GRU_16_1 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "gru";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 16;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 1;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_GRU_32_1 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "gru";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 32;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 1;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_GRU_16_2 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "gru";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 16;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 2;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_GRU_32_2 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "gru";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 32;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 2;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_LSTM_8_1 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "lstm";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 8;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 1;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_LSTM_32_1 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "lstm";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 32;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 1;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_LSTM_16_1 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "lstm";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 16;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 1;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_GRU_8_2 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "gru";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 8;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 2;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

// extra from aidadsp
inline bool is_model_type_ModelType_GRU_12_1 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "gru";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 12;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 1;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_LSTM_12_1 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "lstm";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 12;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 1;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_LSTM_20_1 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "lstm";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 20;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 1;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool is_model_type_ModelType_LSTM_40_1 (const nlohmann::json& model_json) {
    const auto rnn_layer_type = model_json.at ("layers").at (0).at ("type").get<std::string>();
    const auto is_layer_type_correct = rnn_layer_type == "lstm";
    const auto rnn_dim = model_json.at ("layers").at (0).at ("shape").back().get<int>();
    const auto is_rnn_dim_correct = rnn_dim == 40;
    const auto io_dim = model_json.at ("in_shape").back().get<int>();
    const auto is_io_dim_correct = io_dim == 1;
    return is_layer_type_correct && is_rnn_dim_correct && is_io_dim_correct;
}

inline bool custom_model_creator (const nlohmann::json& model_json, ModelVariantType& model) {
    // from RTNeural template gen
    if (is_model_type_ModelType_GRU_8_1 (model_json)) {
        model.emplace<ModelType_GRU_8_1>();
        return true;
    }
    else if (is_model_type_ModelType_LSTM_32_2 (model_json)) {
        model.emplace<ModelType_LSTM_32_2>();
        return true;
    }
    else if (is_model_type_ModelType_LSTM_16_2 (model_json)) {
        model.emplace<ModelType_LSTM_16_2>();
        return true;
    }
    else if (is_model_type_ModelType_LSTM_8_2 (model_json)) {
        model.emplace<ModelType_LSTM_8_2>();
        return true;
    }
    else if (is_model_type_ModelType_GRU_16_1 (model_json)) {
        model.emplace<ModelType_GRU_16_1>();
        return true;
    }
    else if (is_model_type_ModelType_GRU_32_1 (model_json)) {
        model.emplace<ModelType_GRU_32_1>();
        return true;
    }
    else if (is_model_type_ModelType_GRU_16_2 (model_json)) {
        model.emplace<ModelType_GRU_16_2>();
        return true;
    }
    else if (is_model_type_ModelType_GRU_32_2 (model_json)) {
        model.emplace<ModelType_GRU_32_2>();
        return true;
    }
    else if (is_model_type_ModelType_LSTM_8_1 (model_json)) {
        model.emplace<ModelType_LSTM_8_1>();
        return true;
    }
    else if (is_model_type_ModelType_LSTM_32_1 (model_json)) {
        model.emplace<ModelType_LSTM_32_1>();
        return true;
    }
    else if (is_model_type_ModelType_LSTM_16_1 (model_json)) {
        model.emplace<ModelType_LSTM_16_1>();
        return true;
    }
    else if (is_model_type_ModelType_GRU_8_2 (model_json)) {
        model.emplace<ModelType_GRU_8_2>();
        return true;
    }
    // extra from aidadsp
    else if (is_model_type_ModelType_GRU_12_1 (model_json)) {
        model.emplace<ModelType_GRU_12_1>();
        return true;
    }
    else if (is_model_type_ModelType_LSTM_12_1 (model_json)) {
        model.emplace<ModelType_LSTM_12_1>();
        return true;
    }
    else if (is_model_type_ModelType_LSTM_20_1 (model_json)) {
        model.emplace<ModelType_LSTM_20_1>();
        return true;
    }
    else if (is_model_type_ModelType_LSTM_40_1 (model_json)) {
        model.emplace<ModelType_LSTM_40_1>();
        return true;
    }
    model.emplace<NullModel>();
    return false;
}

inline ModelUnion custom_model_creator (const nlohmann::json& model_json, ModelUnionType& type) {
    // from RTNeural template gen
    if (is_model_type_ModelType_GRU_8_1 (model_json)) {
        type = kModelType_GRU_8_1;
        return { new ModelType_GRU_8_1 };
    }
    else if (is_model_type_ModelType_LSTM_32_2 (model_json)) {
        type = kModelType_LSTM_32_2;
        return { new ModelType_LSTM_32_2 };
    }
    else if (is_model_type_ModelType_LSTM_16_2 (model_json)) {
        type = kModelType_LSTM_16_2;
        return { new ModelType_LSTM_16_2 };
    }
    else if (is_model_type_ModelType_LSTM_8_2 (model_json)) {
        type = kModelType_LSTM_8_2;
        return { new ModelType_LSTM_8_2 };
    }
    else if (is_model_type_ModelType_GRU_16_1 (model_json)) {
        type = kModelType_GRU_16_1;
        return { new ModelType_GRU_16_1 };
    }
    else if (is_model_type_ModelType_GRU_32_1 (model_json)) {
        type = kModelType_GRU_32_1;
        return { new ModelType_GRU_32_1 };
    }
    else if (is_model_type_ModelType_GRU_16_2 (model_json)) {
        type = kModelType_GRU_16_2;
        return { new ModelType_GRU_16_2 };
    }
    else if (is_model_type_ModelType_GRU_32_2 (model_json)) {
        type = kModelType_GRU_32_2;
        return { new ModelType_GRU_32_2 };
    }
    else if (is_model_type_ModelType_LSTM_8_1 (model_json)) {
        type = kModelType_LSTM_8_1;
        return { new ModelType_LSTM_8_1 };
    }
    else if (is_model_type_ModelType_LSTM_32_1 (model_json)) {
        type = kModelType_LSTM_32_1;
        return { new ModelType_LSTM_32_1 };
    }
    else if (is_model_type_ModelType_LSTM_16_1 (model_json)) {
        type = kModelType_LSTM_16_1;
        return { new ModelType_LSTM_16_1 };
    }
    else if (is_model_type_ModelType_GRU_8_2 (model_json)) {
        type = kModelType_GRU_8_2;
        return { new ModelType_GRU_8_2 };
    }
    // extra from aidadsp
    else if (is_model_type_ModelType_GRU_12_1 (model_json)) {
        type = kModelType_GRU_12_1;
        return { new ModelType_GRU_12_1 };
    }
    else if (is_model_type_ModelType_LSTM_12_1 (model_json)) {
        type = kModelType_LSTM_12_1;
        return { new ModelType_LSTM_12_1 };
    }
    else if (is_model_type_ModelType_LSTM_20_1 (model_json)) {
        type = kModelType_LSTM_20_1;
        return { new ModelType_LSTM_20_1 };
    }
    else if (is_model_type_ModelType_LSTM_40_1 (model_json)) {
        type = kModelType_LSTM_40_1;
        return { new ModelType_LSTM_40_1 };
    }
    return { nullptr };
}
