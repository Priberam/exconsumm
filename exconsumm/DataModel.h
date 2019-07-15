#pragma once
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "binary_stream.h"

namespace pba_summarization
{
class MapVec
{
public:
    unsigned get_string(const std::string& str) const
    {
        auto it = m_str_map.find(str);
        if (it != m_str_map.end())
            return it->second;
        return 0;
    }
    unsigned get_string(const std::string& str, bool add)
    {
        if (m_str_map.find(str) != m_str_map.end())
            return m_str_map[str];

        if (add)
        {
            m_str_map.insert(std::pair<std::string, unsigned>(str, (unsigned)m_str_map.size()));
            m_str_vec.push_back(str);
            return (unsigned)m_str_vec.size() - 1;
        }
        return 0;

    }
    void clear()
    {
        m_str_vec.clear();
        m_str_map.clear();
    }
    const std::string BAD = "BAD";
    const std::string& str(unsigned str_id) const {
        if (str_id >= m_str_vec.size())
            return BAD;
        return m_str_vec[str_id];
    }
public:
    std::unordered_map<std::string, unsigned> m_str_map;
    std::vector<std::string> m_str_vec;
};

class DataModel
{
private:
    bool force_add = false;
public:
   
    void set_force_add(bool f) { force_add = true; }
    typedef enum ACTION_TYPE
    {
        IN_SUMMARY,
        IS_OUT,
    } ACTION_TYPE;
    ACTION_TYPE action_type(unsigned a) const
    {
        
        switch (a)
        {
        case 2: return IN_SUMMARY;
        case 1: return IN_SUMMARY;
        case 0: return IS_OUT;
        }
        return IS_OUT;
    }
    unsigned get_word(const std::string& action, bool add = false)
    {
        std::string word = action;
        if (m_lower_case_emebedings)
        {
            std::string lower = word;
            std::cerr << "No lower case function available" << std::endl;
            //pba_local::Utf8StringToLowercase(word, &lower);
            word = lower;
        }
        unsigned id =  m_words.get_string(word, (add || force_add) && !m_locked);
        if ((!add || !force_add) && id != 0)
        {
            if (m_word_counts.find(word) == m_word_counts.end())
                m_word_counts.insert(std::pair<std::string, unsigned>(word, 1));
            else
                m_word_counts[word]++;
        }
        return id;
    }
    void limit_words(int nwords)
    {
        std::cerr << "Total number of words in corpus:" << m_word_counts.size() << std::endl;
        std::vector<std::pair<std::string, unsigned>> wds(m_word_counts.begin(), m_word_counts.end());
        std::sort(wds.begin(), wds.end(), [](std::pair<std::string, unsigned> p1, std::pair<std::string, unsigned> p2) {return p1.second > p2.second; });
        m_words.clear();
        const unsigned kUNK = get_word("UNK", true);
        const unsigned kEOS = get_word("EOS", true);
        const unsigned kEOSUM = get_word("EOSUM", true);
        for (unsigned i = 0; i < wds.size() && i< (unsigned) nwords-3; i++)
        {
            get_word(wds[i].first, true);
        }
        if (nwords < wds.size())
            std::cerr << "Frequency  of first removed word = " << wds[nwords].second << std::endl;
    }
    const std::string& word(unsigned action_id) const {
        return m_words.str(action_id);
    }
    const MapVec& words() { return m_words; }

    void set_lower_case_emebedings(bool lower) { m_lower_case_emebedings = lower; }

    bool load_pretrainned_word_embeddings(const std::string path, unsigned dim)
    {
        const unsigned kUNK = get_word("UNK", true);
        const unsigned kEOS = get_word("EOS", true);
        const unsigned kEOSUM = get_word("EOSUM", true);

        m_word_embeddings.clear();
        m_word_embeddings[kUNK] = std::vector<float>(dim, 0);
        m_word_embeddings[kEOS] = std::vector<float>(dim, 0);
        m_word_embeddings[kEOSUM] = std::vector<float>(dim, 0);

        std::cerr << "Loading from " << path << " with " << dim << " dimensions\n";
        std::ifstream in(path.c_str());
        std::string line;
        std::getline(in, line);
        std::vector<float> v(dim, 0);
        std::string word;
        while (getline(in, line)) {
            std::istringstream lin(line);
            lin >> word;
            
            unsigned id = get_word(word, true);
            if (id != 0)
            {
                for (unsigned i = 0; i < dim; ++i) lin >> v[i];

                m_word_embeddings[id] = v;
            }
        }
        return true;
    }
    const std::vector<unsigned> &actions() const { return m_actions; }
    const std::unordered_map<unsigned, std::vector<float>>& word_embeddings() const { return m_word_embeddings; }
    bool save_model(std::string model_path)
    {
        pba_local::binary_stream fp;
        if (!fp.open(model_path, true))
            return false;
        uint16_t n_actions = (uint16_t)m_actions.size();
        fp.put(n_actions);
        for (unsigned i = 0; i < m_actions.size(); i++)
        {
            fp.put((uint32_t)m_actions[i]);
        }
        uint32_t n_words = (uint32_t)m_words.m_str_vec.size();
        fp.put(n_words);
        for (unsigned i = 0; i < m_words.m_str_vec.size(); i++)
        {
            fp.put(m_words.m_str_vec[i]);
        }
        return true;
    }
    bool load_model(std::string model_path)
    {
        pba_local::binary_stream fp;
        if (!fp.open(model_path, false))
            return false;
        m_actions.clear();
        uint16_t n_actions;
        fp.get(&n_actions);
        uint32_t iaux;
        for (unsigned i = 0; i < n_actions; i++)
        {
            fp.get(&iaux);
            m_actions.push_back(iaux);
        }
 
        uint32_t n_words;
        fp.get(&n_words);
        std::string aux;
        for (unsigned i = 0; i < n_words; i++)
        {
            fp.get(&aux);
            m_words.get_string(aux, true);
        }
        return true;
    }
    void lock() { m_locked = true; }
protected:
    std::vector<unsigned> m_actions = { 0, 1/*, 2 */};
    std::unordered_map<unsigned, std::vector<float>> m_word_embeddings;
    MapVec m_words;
    std::unordered_map<std::string, unsigned> m_word_counts;
    bool m_locked = false;
    bool m_lower_case_emebedings = false;
};

}