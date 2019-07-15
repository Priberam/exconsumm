#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <list>
#include <libconfig/libconfig.hh>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif
#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"
#include "dynet/rnn.h"
#include "dynet/io.h"
#include "dynet/model.h"
#include "dynet/gru.h"
#include "dynet_binary_io.h"
#include "OSListdir.h"
#include "Corpus.h"

namespace pba_summarization
{
const unsigned CHAR_SIZE = 255;

typedef struct Span
{
    std::string type;
    int start = 0;
    int end = 0;
} Span;

class NoNameYetModelConfig
{
public:
    bool InitFromConfig(libconfig::Config& config, const std::string& logging_folder)
    {
        libconfig::Setting &cfg = config.lookup("ModelParams");
        cfg.lookupValue("Name", m_name);
        cfg.lookupValue("Layers", LAYERS);
        cfg.lookupValue("UseDropout", USE_DROPOUT);
        cfg.lookupValue("Dropout", DROPOUT);
        cfg.lookupValue("LSTMInputDim", LSTM_INPUT_DIM);
        cfg.lookupValue("PretrainnedDim", PRETRAINED_DIM);
        cfg.lookupValue("HiddenDim", HIDDEN_DIM);
        cfg.lookupValue("ActionDim", ACTION_DIM);
        cfg.lookupValue("UpdatePretrained", UPDATE_PRETRAINED);
        cfg.lookupValue("Noise", NOISE);
        cfg.lookupValue("BatchSize", BATCH_SIZE);
        cfg.lookupValue("Compressive", COMPRESSIVE);
        cfg.lookupValue("CompressiveBOW", COMPRESSIVE_BOW);
        cfg.lookupValue("Relation", MAX_RELATION);
        cfg.lookupValue("UseROUGE", USE_ROUGE);
        cfg.lookupValue("LearningRate", LEARNING_RATE);


        std::ostringstream os;
        os << m_name << '_' << LAYERS
            << '_' << LSTM_INPUT_DIM
            << '_' << HIDDEN_DIM
            << '_' << ACTION_DIM
            << '_' << LSTM_INPUT_DIM
            << "-pid" << getpid();
        // int best_correct_heads = 0;
        m_model_name = os.str();
        m_logs_dir = logging_folder;
        return true;
    }

    std::ofstream Logger(const std::string& group) {
        return std::ofstream(LoggerFilePath(group), std::ios_base::out | std::ios_base::app);
    }

    std::string LoggerFilePath(const std::string& group) {
        return m_logs_dir + OSSEP + model_name() + ".params" + group + ".log";
    }

    void num_layers(unsigned num) { LAYERS = num; }
    void dopout(float rate) { DROPOUT = rate; }
    void action_size(unsigned size) { ACTION_SIZE = size; }
    void vocab_size(unsigned size) { VOCAB_SIZE = size; }

    bool load_params(const std::string pfile)
    {
        dynet::BinaryFileLoader loader(pfile);
        loader.populate(m_model);
        return true;
    }
    const std::string& model_name() const { return m_model_name; }

public:
    unsigned LAYERS = 2;
    unsigned SENTENCE_LAYERS = 1;
    unsigned HIDDEN_DIM = 60;
    unsigned ACTION_DIM = 36;
    unsigned ACTION_SIZE = 36;
    unsigned PRETRAINED_DIM = 50;
    unsigned LSTM_INPUT_DIM = 60;
    bool USE_DROPOUT = true;
    float DROPOUT = 0.0f;
    unsigned VOCAB_SIZE = 0;
    bool  UPDATE_PRETRAINED = true;
    double NOISE = 0;
    int BATCH_SIZE = 10;
    bool COMPRESSIVE = false;
    bool COMPRESSIVE_BOW = false;
    bool ENABLE_REINFORCEMENT_LEARNING = true;
    float MAX_RELATION = 2;
    bool USE_ROUGE = false;
    float LEARNING_RATE = 0.001f;
    dynet::ParameterCollection m_model;
    std::string m_model_name;
    std::string m_logs_dir;
    std::string m_name;

};

class EmbededDocument
{
public:
    std::vector<std::vector<dynet::Expression>> m_sentences_words;
    std::vector<std::vector<dynet::Expression>> m_embeded_sentences_words;
    std::vector<std::vector<dynet::Expression>> m_embeded_sentences_words_fwd;
    std::vector<std::vector<dynet::Expression>> m_embeded_sentences_words_rev;
    std::vector<dynet::Expression> m_embeded_sentences;
    std::vector<int> m_summary_sentences;
    std::vector<std::vector<unsigned>> m_result_summary_sentence_words;
    std::vector<dynet::Expression> bufferf;
    std::vector<dynet::Expression> buffer_lstm_out;
    std::vector<dynet::Expression> output_lstm_out;
    dynet::Expression doc;

};


class WordLevelExtractorModel
{
    friend class NoNameYetModel;

public:
    DataModel * data_model() { return m_data_model; }
protected:
    NoNameYetModelConfig * m_config;
    DataModel* m_data_model;

protected:


public:
    explicit WordLevelExtractorModel(DataModel* data_model, NoNameYetModelConfig* config) : m_data_model(data_model),
        m_config(config),
        stack_lstm(m_config->LAYERS, m_config->HIDDEN_DIM, m_config->HIDDEN_DIM, m_config->m_model),

        p_pbias(m_config->m_model.add_parameters({ m_config->ACTION_DIM })),
        p_A(m_config->m_model.add_parameters({ m_config->ACTION_DIM, m_config->HIDDEN_DIM })),
        p_B(m_config->m_model.add_parameters({ m_config->ACTION_DIM, m_config->HIDDEN_DIM })),
        p_O(m_config->m_model.add_parameters({ m_config->ACTION_DIM, m_config->HIDDEN_DIM })),
        p_S(m_config->m_model.add_parameters({ m_config->ACTION_DIM, m_config->HIDDEN_DIM })),
        p_PS(m_config->m_model.add_parameters({ m_config->ACTION_DIM, m_config->HIDDEN_DIM })),
        p_DC(m_config->m_model.add_parameters({ m_config->ACTION_DIM, m_config->HIDDEN_DIM })),
        p_w2l(m_config->m_model.add_parameters({ m_config->HIDDEN_DIM,  m_config->PRETRAINED_DIM })), 
        p_ib(m_config->m_model.add_parameters({ m_config->HIDDEN_DIM })),
        p_p2a(m_config->m_model.add_parameters({ m_config->ACTION_SIZE, m_config->ACTION_DIM })),
        p_abias(m_config->m_model.add_parameters({ m_config->ACTION_SIZE })),
        p_buffer_guard(m_config->m_model.add_parameters({ m_config->PRETRAINED_DIM })),
        p_stack_guard(m_config->m_model.add_parameters({ m_config->HIDDEN_DIM })),
        p_output_guard(m_config->m_model.add_parameters({ m_config->PRETRAINED_DIM }))

    {  

    }
    void new_graph(dynet::ComputationGraph* hg)
    {
        stack_lstm.new_graph(*hg);
    }
    std::pair<std::vector<unsigned>, dynet::Expression> log_prob_parser(dynet::ComputationGraph* hg,
        const Document& document,
        bool is_evaluation, unsigned sentence, 
        double *right, EmbededDocument& embeded_doc, dynet::Expression& doc_context, dynet::Expression parent_stack, int iac);
    std::pair<std::vector<unsigned>, dynet::Expression> log_prob_parser2(dynet::ComputationGraph* hg,
                                                                         const Document& document,
                                                                         bool is_evaluation, unsigned sentence,
                                                                         double *right, EmbededDocument& embeded_doc, dynet::Expression& doc_context, dynet::Expression parent_stack, int iac, bool standalone_mode = false);
        

public:
    dynet::LSTMBuilder stack_lstm; // (layers, input, hidden, trainer)

    dynet::Parameter p_pbias; // state bias ok
    dynet::Parameter p_S; // stack lstm to parser state ok
    dynet::Parameter p_A; // sentence encoding to parser state ok
    dynet::Parameter p_B; // buffer+output bilstm to parser state ok 
    dynet::Parameter p_O; // doc output lstm to parser state
    dynet::Parameter p_DC; // ok
    dynet::Parameter p_PS; // stack lstm to parser state ok




    dynet::Parameter p_ib; // LSTM input bias
    dynet::Parameter p_cbias; // composition function bias
    dynet::Parameter p_p2a;   // parser state to action
    dynet::Parameter p_abias;  // action bias
    dynet::Parameter p_buffer_guard;  // end of buffer
    dynet::Parameter p_stack_guard;  // end of stack
    dynet::Parameter p_output_guard;  // end of output buffer

    dynet::Parameter p_w2l;
};
class NoNameYetModel
{
    friend class MultiDocumentModel;
public:
    const unsigned kUNK = 0;
    DataModel* data_model() { return m_data_model; }
protected:
    NoNameYetModelConfig * m_config;
    DataModel* m_data_model;
    class WordLevelExtractorModel m_word_level;
protected:


public:
    bool Train(Corpus& train_corpus, Corpus& dev_corpus, const std::string& output_path, const std::string& evaluate_command);

 
    bool Test(Corpus& test_corpus,
              const std::string& output_path,
              const std::string& evaluate_command);

    explicit NoNameYetModel(DataModel* data_model, NoNameYetModelConfig* config) : m_data_model(data_model),
        m_config(config),
        stack_lstm(m_config->LAYERS, m_config->HIDDEN_DIM, m_config->HIDDEN_DIM, m_config->m_model),
        output_lstm(m_config->LAYERS, m_config->LSTM_INPUT_DIM, m_config->HIDDEN_DIM / 2, m_config->m_model),
        buffer_lstm(m_config->LAYERS, m_config->LSTM_INPUT_DIM, m_config->HIDDEN_DIM / 2, m_config->m_model),
        sentence_lstm_fwd(m_config->SENTENCE_LAYERS, m_config->PRETRAINED_DIM, m_config->LSTM_INPUT_DIM / 2, m_config->m_model),
        sentence_lstm_rev(m_config->SENTENCE_LAYERS, m_config->PRETRAINED_DIM, m_config->LSTM_INPUT_DIM / 2, m_config->m_model),
        p_pbias(m_config->m_model.add_parameters({ m_config->ACTION_DIM })),
        p_A(m_config->m_model.add_parameters({ m_config->ACTION_DIM, m_config->HIDDEN_DIM })),
        p_B(m_config->m_model.add_parameters({ m_config->ACTION_DIM, m_config->HIDDEN_DIM })),
        p_O(m_config->m_model.add_parameters({ m_config->ACTION_DIM, m_config->HIDDEN_DIM })),
        p_S(m_config->m_model.add_parameters({ m_config->ACTION_DIM, m_config->HIDDEN_DIM })),

        p_w2l(m_config->m_model.add_parameters({ m_config->HIDDEN_DIM,  m_config->LSTM_INPUT_DIM })), //m_config->LSTM_CHAR_OUTPUT_DIM + m_config->PRETRAINED_DIM })),



        p_ib(m_config->m_model.add_parameters({ m_config->HIDDEN_DIM })),
        p_p2a(m_config->m_model.add_parameters({ m_config->ACTION_SIZE, m_config->ACTION_DIM })),
        p_abias(m_config->m_model.add_parameters({ m_config->ACTION_SIZE })),

        p_buffer_guard(m_config->m_model.add_parameters({ m_config->LSTM_INPUT_DIM })),
        p_stack_guard(m_config->m_model.add_parameters({ m_config->HIDDEN_DIM })),
        m_word_level(data_model, config)

    { 
        if (m_config->VOCAB_SIZE > 0) {
            p_t = m_config->m_model.add_lookup_parameters(m_config->VOCAB_SIZE, { m_config->PRETRAINED_DIM });
            for (auto it : m_data_model->word_embeddings())
                p_t.initialize(it.first, it.second);
        }

    }



    dynet::Expression sentence_encoder_lstm(dynet::ComputationGraph* hg, const std::vector<unsigned>& sentence, bool apply_dropout, EmbededDocument* embeded_doc);

    void encode_document(dynet::ComputationGraph* hg, const Document &document, bool apply_dropout, EmbededDocument& emb);


    std::pair<std::vector<int>, dynet::Expression> log_prob_parser(dynet::ComputationGraph* hg,
                                                                   const Document& document,
                                                                   Corpus *corpus,
                                                                   bool is_evaluation,
                                                                   double *right, EmbededDocument& embeded_doc, int iac = 0, 
                                                                   bool standalone_mode = false);


    void new_graph(dynet::ComputationGraph* hg)
    {
        stack_lstm.new_graph(*hg);
        buffer_lstm.new_graph(*hg);
        output_lstm.new_graph(*hg);
        sentence_lstm_fwd.new_graph(*hg);
        sentence_lstm_rev.new_graph(*hg);
        m_word_level.new_graph(hg);
        stack_lstm.start_new_sequence();
        stack_lstm.add_input(parameter(*hg, p_stack_guard));

    }


private:
    dynet::LSTMBuilder stack_lstm; // (layers, input, hidden, trainer) ok
    dynet::LSTMBuilder output_lstm; // (layers, input, hidden, trainer) ok
    dynet::LSTMBuilder buffer_lstm; // input buffer lstm ok

    dynet::LSTMBuilder sentence_lstm_fwd; // span lstm forward
    dynet::LSTMBuilder sentence_lstm_rev; // span lstm backward

    dynet::Parameter p_pbias; // state bias
    dynet::Parameter p_S; // stack lstm to parser state
    dynet::Parameter p_A; // sentence encoding to parser state
    dynet::Parameter p_B; // buffer+output bilstm to parser state
    dynet::Parameter p_O; // doc output lstm to parser state

    dynet::LookupParameter p_t; // pretrained word embeddings (not updated)


    dynet::Parameter p_w2l; // word to LSTM input


    dynet::Parameter p_ib; // LSTM input bias
    dynet::Parameter p_p2a;   // parser state to action
    dynet::Parameter p_abias;  // action bias
    dynet::Parameter p_buffer_guard;  // end of buffer
    dynet::Parameter p_stack_guard;  // end of stack
};



}
