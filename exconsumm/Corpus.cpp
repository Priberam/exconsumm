#include "Corpus.h"
#include "OSListdir.h"
#include <iostream>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include "FastRouge.h"
#include <omp.h>
#include "binary_stream.h"


namespace pba_summarization
{

void SavePath(pba_local::binary_stream& out, Path& p)
{
    out.put(p.avrR);
    out.put(p.sentence);
    out.put(p.sons.size());
    for (auto &p1 : p.sons)
    {
        SavePath(out, p1);
    }
}
void LoadPath(pba_local::binary_stream& out, Path& p, int level)
{
    if (level >= 5)
        return;
    out.get(&p.avrR);
    out.get(&p.sentence);
    uint64_t sz;
    out.get(&sz);
    p.sons.resize(sz);
    for (unsigned i = 0; i < sz; i++)
    {
        if (sz > 8)
            LoadPath(out, p.sons[i], level + 1);
    }

    std::sort(p.sons.begin(), p.sons.end(), [](Path& p1, Path& p2) {return p1.avrR > p2.avrR; });
    if (p.sons.size() > 4)
        p.sons.resize(4);
}
void Document::LoadBestPaths(const std::string& input)
{
    //std::string path = m_path;
    //if (m_path.find_last_of("\\/") != std::string::npos)
    //    path = path.substr(m_path.find_last_of("\\/") + 1);
    //path = output_path + path;
    //std::ofstream sum_file(path + ".bpath");
    pba_local::binary_stream out;
    if (!out.open(input, false))
        return;
    uint64_t sz;
    out.get(&sz);
    m_paths.resize(sz);
    for (unsigned i = 0; i < sz; i++)
    {
        LoadPath(out, m_paths[i], 0);
    }
    std::sort(m_paths.begin(), m_paths.end(), [](Path& p1, Path& p2) {return p1.avrR > p2.avrR; });
}

void Document::SaveBestPaths(const std::string& output_path)
{
    std::string path = m_path;
    if (m_path.find_last_of("\\/") != std::string::npos)
        path = path.substr(m_path.find_last_of("\\/") + 1);
    path = output_path + path;
    pba_local::binary_stream out;
    if (!out.open(path + ".bpath", true))
        return;
    out.put(m_paths.size());
    for (auto &p : m_paths)
    {
        SavePath(out, p);
    }
    m_paths.clear();
    out.close();

}
float Document::BETA = 1.0f;
double Document::ComputeFastRouge(const std::vector<int>& selected, const std::vector<std::vector<unsigned>>& sentences_words, bool eval) const
{
    Tmp *tmp = new Tmp;


    std::unordered_set<unsigned> stopwords = { 0, m_data_model.get_word(",") , m_data_model.get_word("."),
        m_data_model.get_word(":"), m_data_model.get_word(";"), m_data_model.get_word("a"), m_data_model.get_word("the") };
    unsigned i = 0;
    for (auto &s : m_abstractive_sentences)
    {

        uint64_t prev = 0xffffff;
        for (auto w : s)
        {
            if (stopwords.find(w) != stopwords.end())
                continue;
            if (tmp->m_document_words.find(w) == tmp->m_document_words.end())
                tmp->m_document_words.insert(std::pair<uint64_t, unsigned>(w, (unsigned int)tmp->m_document_words.size()));
            if (tmp->m_document_words.find(prev << 32 | w) == tmp->m_document_words.end())
                tmp->m_document_words.insert(std::pair<uint64_t, unsigned>(prev << 32 | w, (unsigned int)tmp->m_document_words.size()));
            prev = w;
        }
    }


    tmp->m_abstractive_unigrams.resize(tmp->m_document_words.size() + 2);
    tmp->m_abstractive_bigrams.resize(tmp->m_document_words.size() + 2);
    for (auto &s : m_abstractive_sentences)
    {
        uint64_t prev = 0xffffff;
        for (auto w : s)
        {
            if (stopwords.find(w) != stopwords.end())
                continue;
            
            if (tmp->m_document_words.find(w) != tmp->m_document_words.end())
                tmp->m_abstractive_unigrams[tmp->m_document_words[w]] += 1;
            else
                tmp->m_abstractive_unigrams[tmp->m_document_words.size()] += 1;
            if (tmp->m_document_words.find(prev << 32 | w) != tmp->m_document_words.end())
                tmp->m_abstractive_bigrams[tmp->m_document_words[prev << 32 | w]] += 1;
            else
                tmp->m_abstractive_bigrams[tmp->m_document_words.size() + 1] += 1;
            prev = w;
        }
    }

    // Now, do the same for each sentence
    std::vector<int>  unigrams(tmp->m_document_words.size() + 2);
    std::vector<int>  bigrams(tmp->m_document_words.size() + 2);
    for (unsigned i = 0; i < m_sentences.size(); i++)
    {
        if (selected[i] == 0)
            continue;

        uint64_t prev = 0xffffff;
        int iw = 0;
        for (auto w : m_sentences[i])
        {
            iw++;
            if (stopwords.find(w) != stopwords.end())
                continue;
            if (sentences_words.size() != 0 && sentences_words[i].size() != 0 && sentences_words[i][iw-1] == 0)
                continue;
            if (tmp->m_document_words.find(w) != tmp->m_document_words.end())
                unigrams[tmp->m_document_words[w]] += 1;
            else
                unigrams[tmp->m_document_words.size()] += 1;

            if (tmp->m_document_words.find(prev << 32 | w) != tmp->m_document_words.end())
                bigrams[tmp->m_document_words[prev << 32 | w]] += 1;
            else
                bigrams[tmp->m_document_words.size() + 1] += 1;

            prev = w;
        }
    }

    pba_summarization::FastRouge fr;
    // For each sentence calculate rouge
   
    auto uni_Pc = fr.single_doc_ROUGE_P(unigrams, tmp->m_abstractive_unigrams);
    auto uni_Rc = fr.single_doc_ROUGE_R(unigrams, tmp->m_abstractive_unigrams);
    float p = uni_Pc.first / (float)uni_Pc.second;
    float r = uni_Rc.first / (float)uni_Rc.second;
    float uni_F = (1 + BETA * BETA) * r*p / (r + (BETA * BETA) *p);
    if (r == 0 || p == 0 || uni_Pc.second == 0)
        uni_F = 0;
    auto bi_Pc = fr.single_doc_ROUGE_P(bigrams, tmp->m_abstractive_bigrams);
    auto bi_Rc = fr.single_doc_ROUGE_R(bigrams, tmp->m_abstractive_bigrams);
    p = bi_Pc.first / (float)bi_Pc.second;
    r = bi_Rc.first / (float)bi_Rc.second;
    float bi_F = (1 + BETA * BETA) * r*p / (r + (BETA * BETA) *p);        
    if (r == 0 || p == 0 || bi_Pc.second == 0)
        bi_F = 0;
    

    delete tmp;
    double val;
    if (best_rouge != 0 && !eval)
        val = 1 - (best_rouge - ((bi_F + uni_F) / 2.0f));
    else
        val = ((bi_F + uni_F) / 2.0f);
    //if (val > 1 - 0.0000001)
    //    val = 1 - 0.0000001;
    if (bi_F + uni_F == 0) return 0;
    return val;
    
}

static std::vector<unsigned int> remove_segments(const std::vector<int>& inds, const std::vector<std::pair<int, int>>& segments, const std::vector<unsigned int> original)
{
    std::vector<unsigned int> result;
    int j = 0;
    for (int i = 0; i < inds.size(); i++)
    {
        for (; j < segments[inds[i]].first; j++)
            result.push_back(original[j]);
        j = segments[inds[i]].second;
    }
    for (;j<original.size(); j++)
        result.push_back(original[j]);
    return result;
}
void Document::BuildBestPaths(const std::string& output_path)
{
    m_tmp = new Tmp;
    for (unsigned i = 0; i < m_sentences.size(); i++)
    {
        m_tmp->m_compressed_sentences.push_back(m_sentences[i]);
        m_tmp->m_compressed_sentences_to_sentences.push_back(i);
        m_tmp->m_permutations.push_back({});
        unsigned N = (unsigned) m_sentences_remove_segments[i].size();
        for (unsigned K = 1; K <= N; K++)
        {

            std::string bitmask(K, 1); // K leading 1's
            bitmask.resize(N, 0); // N-K trailing 0's

            // print integers and permute bitmask
            do {
                std::vector<int> elems;
                for (unsigned i = 0; i < N; ++i) // [0..N-1] integers
                {
                    if (bitmask[i])
                        elems.push_back(i);
                }

                m_tmp->m_compressed_sentences.push_back(remove_segments(elems, m_sentences_remove_segments[i], m_sentences[i]));
                if (m_tmp->m_compressed_sentences.back().size() == 0)
                {
                    m_tmp->m_compressed_sentences.pop_back();
                }
                else
                {
                    m_tmp->m_compressed_sentences_to_sentences.push_back(i);
                    m_tmp->m_permutations.push_back(elems);
                }
            } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
        }
    }


    // Calculate unigram and bigrams for the reference summary
    

    std::unordered_set<unsigned> stopwords = { 0, m_data_model.get_word(",") , m_data_model.get_word("."),
        m_data_model.get_word(":"), m_data_model.get_word(";"), m_data_model.get_word("a"), m_data_model.get_word("the") };
    for (auto &s : m_abstractive_sentences)
    {

        uint64_t prev = 0xffffff;
        for (auto w : s)
        {
            if (stopwords.find(w) != stopwords.end())
                continue;
            if (m_tmp->m_document_words.find(w) == m_tmp->m_document_words.end())
                m_tmp->m_document_words.insert(std::pair<uint64_t, unsigned>(w, (unsigned int)m_tmp->m_document_words.size()));
            if (m_tmp->m_document_words.find(prev << 32 | w) == m_tmp->m_document_words.end())
                m_tmp->m_document_words.insert(std::pair<uint64_t, unsigned>(prev << 32 | w, (unsigned int)m_tmp->m_document_words.size()));
            prev = w;
        }
    }
    if (m_tmp->m_document_words.size() == 0)
        return;
    


    m_tmp->m_abstractive_unigrams.resize(m_tmp->m_document_words.size()+2);
    m_tmp->m_abstractive_bigrams.resize(m_tmp->m_document_words.size()+2);
    for (auto &s : m_abstractive_sentences)
    {
        uint64_t prev = 0xffffff;
        for (auto w : s)
        {
            if (stopwords.find(w) != stopwords.end())
                continue;

            if (m_tmp->m_document_words.find(w) != m_tmp->m_document_words.end())
                m_tmp->m_abstractive_unigrams[m_tmp->m_document_words[w]] += 1;
            else
                m_tmp->m_abstractive_unigrams[m_tmp->m_document_words.size()] += 1;
            if (m_tmp->m_document_words.find(prev << 32 | w) != m_tmp->m_document_words.end())
                m_tmp->m_abstractive_bigrams[m_tmp->m_document_words[prev << 32 | w]] += 1;
            else
                m_tmp->m_abstractive_bigrams[m_tmp->m_document_words.size() + 1] += 1;
            prev = w;
        }
    }

    // Now, do the same for each sentence
    m_tmp->m_sentences_unigrams.resize(m_tmp->m_compressed_sentences.size());
    m_tmp->m_sentences_bigrams.resize(m_tmp->m_compressed_sentences.size());
    for (unsigned i = 0; i < m_tmp->m_compressed_sentences.size(); i++)
    {
        uint64_t prev = 0xffffff;
        m_tmp->m_sentences_unigrams[i].resize(m_tmp->m_document_words.size() + 2);
        m_tmp->m_sentences_bigrams[i].resize(m_tmp->m_document_words.size() + 2);
        for (auto w : m_tmp->m_compressed_sentences[i])
        {
            if (stopwords.find(w) != stopwords.end())
                continue;

            if (m_tmp->m_document_words.find(w) != m_tmp->m_document_words.end())
                m_tmp->m_sentences_unigrams[i][m_tmp->m_document_words[w]] += 1;
            else
                m_tmp->m_sentences_unigrams[i][m_tmp->m_document_words.size()] += 1;

            if (m_tmp->m_document_words.find(prev << 32 | w) != m_tmp->m_document_words.end())
                m_tmp->m_sentences_bigrams[i][m_tmp->m_document_words[prev << 32 | w]] += 1;
            else
                m_tmp->m_sentences_bigrams[i][m_tmp->m_document_words.size() + 1] += 1;

            prev = w;
        }
    }

    pba_summarization::FastRouge fr;
    // For each sentence calculate rouge
    std::vector<float> avgR;
    for (unsigned i = 0; i < m_tmp->m_compressed_sentences.size(); i++)
    {
        auto uni_Pc = fr.single_doc_ROUGE_P(m_tmp->m_sentences_unigrams[i], m_tmp->m_abstractive_unigrams);
        auto uni_Rc = fr.single_doc_ROUGE_R(m_tmp->m_sentences_unigrams[i], m_tmp->m_abstractive_unigrams);
        float p = uni_Pc.first / (float)uni_Pc.second;
        float r = uni_Rc.first / (float)uni_Rc.second;
        float uni_F = (1 + BETA * BETA) * r*p / (r + (BETA * BETA) *p);
        if (r == 0 || p == 0)
            uni_F = 0;
        auto bi_Pc = fr.single_doc_ROUGE_P(m_tmp->m_sentences_bigrams[i], m_tmp->m_abstractive_bigrams);
        auto bi_Rc = fr.single_doc_ROUGE_R(m_tmp->m_sentences_bigrams[i], m_tmp->m_abstractive_bigrams);
        p = bi_Pc.first / (float)bi_Pc.second;
        r = bi_Rc.first / (float)bi_Rc.second;
        float bi_F = (1 + BETA * BETA) * r*p / (r + (BETA * BETA) *p);        if (r == 0 || p == 0)
            bi_F = 0;
        avgR.push_back((bi_F + uni_F) / 2.0f);
    }
    double min = 0;
    if (avgR.size() > m_abstractive_sentences.size() +80)
    {
        auto tmp = avgR;
        std::sort(tmp.begin(), tmp.end(), [](double p1, double p2) {return p1 > p2; });
        min = tmp[m_abstractive_sentences.size()+80];
    }
    for (int i = 0; i < (int)m_tmp->m_compressed_sentences.size(); i++)
    {
        if (avgR[i] == 0 || avgR[i] < min)
            m_tmp->m_compressed_sentences_to_sentences[i] = -1;

    }
    m_tmp->m_max = 0;
    // Lets build the paths
    m_paths.resize(m_tmp->m_compressed_sentences.size());
#pragma omp parallel for num_threads( 8 )  shared(avgR )
    for (int i = 0; i <(int)m_tmp->m_compressed_sentences.size(); i++)
    {
        if (avgR[i] == 0 || avgR[i] < min )
            continue;        
        Path p;
        p.avrR = avgR[i];
        p.sentence = i;
        m_paths[i] = (p);
        float ret = Recurse(avgR, i, m_tmp->m_sentences_unigrams[i], m_tmp->m_sentences_bigrams[i], &m_paths[i], 0, false);
        if (ret > m_paths[i].avrR)
            m_paths[i].avrR = ret;
        if (ret > m_tmp->m_max)
            m_tmp->m_max = ret;
    }
    if (m_tmp->m_max > 0)
    {
        std::sort(m_paths.begin(), m_paths.end(), [](Path& p1, Path& p2) {return p1.avrR > p2.avrR; });
        OutputOracle(output_path);
    }
    m_paths.clear();
    delete m_tmp;

}

float Document::Recurse(const std::vector<float>& pavgR, int current, const std::vector<int>& base_unigrams, std::vector<int>& base_bigrams, Path *path, int level, bool dec)
{

    if (level >= m_abstractive_sentences.size() +4)
    {
        return 0;
    }
    // Calculate the union of the unigrams and bigrams of the 
    // last sentence and the current one
    std::vector<std::vector<int>> sentences_unigrams(m_tmp->m_compressed_sentences.size());
    std::vector<std::vector<int>> sentences_bigrams(m_tmp->m_compressed_sentences.size());

    for (unsigned i = current + 1; i < m_tmp->m_compressed_sentences.size(); i++)
    {
        if (m_tmp->m_compressed_sentences_to_sentences[i] <= m_tmp->m_compressed_sentences_to_sentences[current])
            continue;
        if (pavgR[i] == 0)
            continue;
        sentences_unigrams[i] = base_unigrams;
        for (int j = 0; j < m_tmp->m_sentences_unigrams[i].size(); j++)
        {
            sentences_unigrams[i][j] += m_tmp->m_sentences_unigrams[i][j];
        }
        sentences_bigrams[i] = base_bigrams;
        for (int j = 0; j < m_tmp->m_sentences_bigrams[i].size(); j++)
        {
            sentences_bigrams[i][j] += m_tmp->m_sentences_bigrams[i][j];
        }
    }
    pba_summarization::FastRouge fr;
    std::vector<float> avgR(m_tmp->m_compressed_sentences.size());
    for (unsigned i = current + 1; i < m_tmp->m_compressed_sentences.size(); i++)
    {
        if (m_tmp->m_compressed_sentences_to_sentences[i] <= m_tmp->m_compressed_sentences_to_sentences[current])
            continue;
        if (pavgR[i] == 0)
            continue;

        auto uni_Pc = fr.single_doc_ROUGE_P(sentences_unigrams[i], m_tmp->m_abstractive_unigrams);
        auto uni_Rc = fr.single_doc_ROUGE_R(sentences_unigrams[i], m_tmp->m_abstractive_unigrams);
        float p = uni_Pc.first / (float)uni_Pc.second;
        float r = uni_Rc.first / (float)uni_Rc.second;
        float uni_F = (1 + BETA * BETA) * r*p / (r + (BETA * BETA) *p);
        if (r == 0 || p == 0)
            uni_F = 0;
        auto bi_Pc = fr.single_doc_ROUGE_P(sentences_bigrams[i], m_tmp->m_abstractive_bigrams);
        auto bi_Rc = fr.single_doc_ROUGE_R(sentences_bigrams[i], m_tmp->m_abstractive_bigrams);
        p = bi_Pc.first / (float)bi_Pc.second;
        r = bi_Rc.first / (float)bi_Rc.second;
        float bi_F = (1 + BETA * BETA) * r*p / (r + (BETA * BETA) *p);
        if (r == 0 || p == 0)
            bi_F = 0;
        avgR[i] = ((bi_F + uni_F) / 2.0f);
    }
    
    double min = 0;
    if (avgR.size() > 10)
    {
        auto tmp = avgR;
        std::sort(tmp.begin(), tmp.end(), [](double p1, double p2) {return p1 > p2; });
        min = tmp[10];
    }
    
    float max = path->avrR;
    for (unsigned i = current + 1; i < m_tmp->m_compressed_sentences.size(); i++)
    {
        if (m_tmp->m_compressed_sentences_to_sentences[i] <= m_tmp->m_compressed_sentences_to_sentences[current])
            continue;

        if (avgR[i] == 0 || avgR[i] < min || avgR[i] <= path->avrR)
            continue;
        double sum = 0;
        for (int j = i+1; j < m_tmp->m_compressed_sentences.size(); j++)
        {
            if (m_tmp->m_compressed_sentences_to_sentences[i] <= m_tmp->m_compressed_sentences_to_sentences[current])
                continue;

            if (avgR[i] == 0 || avgR[i] < min || avgR[i] <= path->avrR)
                continue;
            sum += avgR[j] - pavgR[j];
        }
        if (sum + pavgR[i] < m_tmp->m_max)
            continue;
        bool next_dec = false;
        Path p;
        p.avrR = avgR[i];
        p.sentence = i;
        //path->sons.push_back(p);

        float ret = Recurse(avgR, i, sentences_unigrams[i], sentences_bigrams[i], &p, level + 1, next_dec);
        //path->sons.shrink_to_fit();
        if (ret > max)
        {
            p.avrR = ret;
            if (path->sons.size() == 0)
                path->sons.push_back(p);
            else
                path->sons.back() = p;
            max = ret;
        }
        if (ret > m_tmp->m_max)
            m_tmp->m_max = ret;
        //if (path->sons.back().avrR > max)
        //    max = path->sons.back().avrR;

    }
    //auto tmp = path->sons;
    //std::sort(path->sons.begin(), path->sons.end(), [](Path& p1, Path& p2) {return p1.avrR > p2.avrR; });
    //if (path->sons.size() > 10)
    //{
    //    path->sons.resize(10);
    //    path->sons.shrink_to_fit();
    //}
    //if (path->sons.size() > 20)
    //{
    //    
    //    for (auto &p : path->sons)
    //    {
    //        if (p.avrR < path->sons[20].avrR)
    //        {
    //            p.sons.clear();
    //        }
    //    }
    //}
    return max;


}

void Document::OutputOracle(const std::string& output_path) const
{
    //std::string output_path = "./oracles/";
    std::string path = m_path;
    if (m_path.find_last_of("\\/") != std::string::npos)
        path = path.substr(m_path.find_last_of("\\/") + 1);
    path = output_path + path;
    std::ofstream f1_file(path + ".compressed_oracle");
    std::ofstream f3_file(path + ".moracle");
    std::ofstream f2_file(path + ".oracle.csum");
    const Path *opath = &(m_paths)[0];
    while (opath != NULL)
    {
        int tk = 0;
        int comp = 0;
        bool remove = false;
        f3_file << m_tmp->m_compressed_sentences_to_sentences[opath->sentence] << " ";
        for (auto &w : m_str_sentences[m_tmp->m_compressed_sentences_to_sentences[opath->sentence]])
        {
            if (m_tmp->m_permutations[opath->sentence].size() > comp &&
                m_sentences_remove_segments[m_tmp->m_compressed_sentences_to_sentences[opath->sentence]][m_tmp->m_permutations[opath->sentence][comp]].first == tk)
            {
                f1_file << "[ ";
                remove = true;
            }
            if (m_tmp->m_permutations[opath->sentence].size() > comp &&
                m_sentences_remove_segments[m_tmp->m_compressed_sentences_to_sentences[opath->sentence]][m_tmp->m_permutations[opath->sentence][comp]].second == tk)
            {
                f1_file << "] ";
                remove = false;
                comp++;
            }
            f1_file << w << " ";
            if (!remove)
                f2_file << w << " ";

            tk++;
        }
        if (remove)
            f1_file << "] ";
        f1_file << std::endl;
        f2_file << std::endl;
        if (opath->sons.size())
            opath = &opath->sons[0];
        else
            opath = NULL;
    }
    f3_file << m_paths[0].avrR << std::endl;
    
}



void Document::OutputSummary(const std::vector<int>& sentences_actions, const std::vector<std::vector<unsigned>>& sentences_word_actions,
    const std::string &abstractive_summary, const std::string& output_path) const
{
    std::string path = m_path;
    if (m_path.find_last_of("\\/") != std::string::npos)
        path = path.substr(m_path.find_last_of("\\/") + 1);
    path = output_path + path;
    std::ofstream sum_file(path + ".gsum");

    for (unsigned i = 0; i < sentences_actions.size(); i++)
    {
        if (sentences_actions[i] == 1)
        {
            for (auto &w : m_str_sentences[i])
            {
                if (w == "-LRB-")
                    sum_file << "(";
                else if (w == "-RRB-")
                    sum_file << ")";
                else
                    sum_file << w << " ";
            }
            sum_file << std::endl;
        }
    }
    std::ofstream csum_file(path + ".csum");

    for (unsigned i = 0; i < sentences_actions.size(); i++)
    {
        if (sentences_actions.size() > i && sentences_actions[i] == 1)
        {
            int j = 1;
            for (auto &w : m_str_sentences[i])
            {
                if (sentences_word_actions[i].size() > j && sentences_word_actions[i][j++] == 1)
                {
                    if (w == "-LRB-")
                        csum_file << "(";
                    else if (w == "-RRB-")
                        csum_file << ")";
                    else
                        csum_file << w << " ";
                }
            }
            csum_file << std::endl;
        }
    }
    std::ofstream orig_file(path + ".body");

    for (unsigned i = 0; i < sentences_actions.size(); i++)
    {
        for (auto &w : m_str_sentences[i])
        {
            orig_file << w << " ";
        }
        orig_file << std::endl;
    }
    std::ofstream or_file(path + ".osum");
    if (m_actions_sentences.size())
        for (unsigned i = 0; i < m_actions_sentences[0].size(); i++)
        {
            if (m_actions_sentences[0][i] == 1)
            {
                int iw = 0;
                for (auto &w : m_str_sentences[i])
                {
                    if (m_action_sentences_remove_words.size() == 0 || m_action_sentences_remove_words[i][iw] == 1)
                        or_file << w << " ";
                    iw++;
                }
                or_file << std::endl;
            }
        }

}



}