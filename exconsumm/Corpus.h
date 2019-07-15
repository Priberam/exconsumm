#pragma once
#include <string>
#include <libconfig/libconfig.hh>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <random>
#include "DataModel.h"

namespace pba_summarization
{

typedef struct Path
{
    unsigned sentence;
    float avrR;
    std::vector<struct Path> sons;
} Path;

typedef struct Tmp
{
    std::vector<int> m_compressed_sentences_to_sentences;
    std::vector<std::vector<unsigned int>> m_compressed_sentences;
    std::vector<std::vector<int>> m_permutations;

    std::vector<std::vector< int>> m_sentences_unigrams;
    std::vector< std::vector<int>> m_sentences_bigrams;
    std::vector<int> m_abstractive_unigrams;
    std::vector<int> m_abstractive_bigrams;
    std::unordered_map<uint64_t, unsigned> m_document_words;
    double m_max;
} Tmp;

class Document
{
public:
    Document(DataModel& data_model) : m_data_model(data_model)
    {}

    ~Document() {
        if (m_highlight_doc)
            delete m_highlight_doc;
    }
    void AddSentence(const std::vector<std::string>& sentence, int action)
    {

        if (m_keep_words)
            m_str_sentences.push_back(sentence);
        std::vector<unsigned> ids;
        ids.reserve(sentence.size());
        for (auto &s : sentence)
        {
            ids.push_back(m_data_model.get_word(s/*, true*/));

        }
        m_sentences.push_back(ids);
        m_sentences.back().shrink_to_fit();
    }
    void AddRemoveSegments(const std::vector<std::pair<int, int>> segments)
    {
        m_sentences_remove_segments.push_back(segments);
    }
    void AddSentence(const std::vector<unsigned>& sentence, int action)
    {

        m_sentences.push_back(sentence);
        m_sentences.back().shrink_to_fit();
    }
    void AddStrSentence(const std::vector<std::string>& sentence)
    {

        m_str_sentences.push_back(sentence);
    }
    void AddAbstractiveSentence(const std::vector<std::string>& sentence, int action)
    {
        std::vector<unsigned> ids;
        ids.reserve(sentence.size());
        for (auto &s : sentence)
        {
            ids.push_back(m_data_model.get_word(s));
        }
        m_abstractive_sentences.push_back(ids);
        m_abstractive_sentences.back().shrink_to_fit();
    }
    void SetAbstractiveSentences(const std::vector<std::vector<unsigned>>& sentences)
    {
        m_abstractive_sentences = sentences;
        
    }
    void AddRemoveWords(std::vector<unsigned> word_actions)
    {
        int i = 0;
        if (m_actions_sentences.size() == 0)
            return;
        m_action_sentences_remove_words.resize(m_actions_sentences[0].size());
        while (i < m_actions_sentences[0].size() && (m_action_sentences_remove_words[i].size() > 0 ||m_actions_sentences[0][i] != 1) )
        {
            i++;
        }
        if (i == m_actions_sentences[0].size())
        {
            std::cerr << "Error words i= " << i << " sentences=" << m_actions_sentences[0].size() << std::endl;
            return;
            //exit(1);
        }
        if (word_actions.size() != m_sentences[i].size())
        {
            std::cerr << "Oracle mismatch" << std::endl;
            word_actions.resize(m_sentences[i].size());
            //exit(1);

        }
        m_action_sentences_remove_words[i]= word_actions;
    }
    void OutputSummary(const std::vector<int>& sentences_actions,
                       const std::vector<std::vector<unsigned>>& sentences_word_actions, 
                       const std::string& abstractive_summary, 
                       const std::string & output_path) const;


    void OutputOracle(const std::string& output_path) const ;
    std::vector<int> empty;
    const std::vector<int>& sentences_actions() const { if (m_actions_sentences.size() == 0) return empty; return m_actions_sentences[0]; }
    std::vector<int>& sentences_actions_nc()  { if (m_actions_sentences.size() == 0) return empty; return m_actions_sentences[0]; }
    const std::vector<unsigned>& sentences_word_actions(unsigned sentence) const { return m_action_sentences_remove_words[sentence]; }
    int n_sentences_actions() const {
        return (int) m_actions_sentences
            .size();
    }
    const std::vector<int>& sentences_actions_rnd(int i, double *reward) const {

        //std::random_device rd;
        //std::uniform_int_distribution<int> distribution(0, (int) m_actions_sentences.size() - 1);
        //std::mt19937 engine(rd()); // Mersenne twister MT19937
        //int i = distribution(engine);
        *reward = m_rewards[i];  return m_actions_sentences[i];
    }
    void set_sentences_actions(const std::vector<int>&actions) { m_actions_sentences.push_back(actions); m_rewards.push_back(1.0); }
    void add_sentences_actions(const std::vector<int>&actions, double reward) { m_actions_sentences.push_back(actions); m_rewards.push_back(reward); }
    const std::vector<std::vector<unsigned>>& sentences() const { return m_sentences; }
    const std::vector<std::vector<std::string>>& str_sentences() const { return m_str_sentences; }

    const std::vector<std::vector<unsigned>>& abstractive_sentences() const { return m_abstractive_sentences; }

    void set_keep_words(bool save) { m_keep_words = save; }
    void set_path(const std::string path) { m_path = path; }
    const std::string &path() const { return m_path; }
    const std::vector<Path>& best_paths() const { return m_paths; }
    void SaveBestPaths(const std::string& output_path);
    void LoadBestPaths(const std::string& input);
    void BuildBestPaths(const std::string& output_path);
    float Recurse(const std::vector<float>& avgR, int current, const std::vector<int>& base_unigrams, std::vector<int>& base_bigrams, Path* path, int level, bool dec);
    double ComputeFastRouge(const std::vector<int>& selected, const std::vector<std::vector<unsigned>>& sentences_words, bool eval=false) const ;
    void set_best_rouge(double best) {
        best_rouge = best;
    }

    void set_highlight_doc(Document* doc) {
        m_highlight_doc = doc;
    }

    Document * highlight_doc() {
        return m_highlight_doc;
    }

    const Document * highlight_doc() const{
        return m_highlight_doc;
    }
    static void setRougeFx(float x) { BETA = x; }
    static double RougeFx() { return BETA; }
protected:
    //std::vector<int> m_actions_sentences;
    std::vector<std::vector<unsigned>> m_sentences;
    std::vector<std::vector<std::pair<int, int>>> m_sentences_remove_segments;
    std::vector<std::vector<unsigned>> m_abstractive_sentences;
    std::vector<std::vector<std::string>> m_str_sentences; // probably just for debug
    std::vector<std::vector<int>> m_actions_sentences;
    std::vector<std::vector<unsigned>> m_action_sentences_remove_words; // Just for the first action_sentences
    std::vector<double> m_rewards;
    DataModel &m_data_model;
    bool m_keep_words = false;
    std::string m_path;
    Document * m_highlight_doc { nullptr };

    //for the rouge
    Tmp  *m_tmp;
    
    std::vector<Path> m_paths;
    double best_rouge = 0;
    static float BETA;

};



class GenericCorpus
{
public:
    GenericCorpus(DataModel &data_model) : m_data_model(data_model) {};
    void set_keep_words(bool save) { m_keep_words = save; }
    virtual bool LoadFiles(libconfig::Setting& config) = 0;
    virtual void clear() = 0;
    virtual unsigned NumberOfDocuments() = 0;

protected:
    DataModel & m_data_model;
    bool m_keep_words = false;

};

class Corpus : public GenericCorpus
{
public:
    Corpus(DataModel &data_model) : GenericCorpus(data_model) {};
    ~Corpus()
    {
        for (auto d : m_documents)
            delete d;
    }

    virtual bool LoadFiles(libconfig::Setting& config) = 0;
    virtual unsigned NumberOfDocuments() { return (unsigned)m_documents.size(); }
    const Document* GetDocument(int i) { return m_documents[i]; }
    const Document* GetDocument(const std::string& id) { 
        auto it = m_document_by_id.find(id);
        if (it == m_document_by_id.end()) {
            std::cerr << "no document " << id << std::endl;
            return nullptr;
        }
        return m_documents_2[it->second];
    }
    virtual void BuildBestPaths() = 0;
    void set_eval_corpus() { eval_corpus = true; }

    virtual void clear()
    {
        for (auto d : m_documents)
            delete d;
        m_documents.clear();
        m_files.clear();
    }

protected:
    std::list<std::string> m_files;
    bool eval_corpus = false;
    std::vector<Document*> m_documents;
    std::vector<Document*> m_documents_2;
    std::unordered_map<std::string, int> m_document_by_id;

};


}