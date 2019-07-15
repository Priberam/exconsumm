
#include "FastRouge.h"
#include <unordered_map>
#include <algorithm> 
//#stopwords = set(['.', ',', ';', ':', '``', "''"])
//stopwords = set(['.', ',', ';', ':'])
//#stopwords = set()
//
namespace pba_summarization
{
FastRouge::FastRouge()
{
}


FastRouge::~FastRouge()
{
}



std::pair<int, int> FastRouge::single_doc_ROUGE_R(const std::unordered_map<uint64_t, int>& guess_summ_dict, const std::unordered_map<uint64_t, int>& gold_summ_dict)
{
    int total_items = 0;
    int found_items = 0;
    for (auto &item : gold_summ_dict)
    {
        //if (m_stopwords.count(item.first))
        //    continue;
        total_items += item.second; 
        auto it = guess_summ_dict.find(item.first);
        if (it == guess_summ_dict.end())
            continue;
        found_items += std::min<int>(item.second, it->second);
    }
    return std::make_pair(found_items, total_items);
}
std::pair<int, int> FastRouge::single_doc_ROUGE_P(const std::unordered_map<uint64_t, int>& guess_summ_dict, const std::unordered_map<uint64_t, int>& gold_summ_dict)
{
    int total_items = 0;
    int found_items = 0;
    for (auto &item : guess_summ_dict)
    {
        //if (m_stopwords.count(item.first))
        //    continue;
        total_items += item.second;
        auto it = gold_summ_dict.find(item.first);
        if (it == gold_summ_dict.end())
            continue;
        found_items += std::min<int>(item.second, it->second);

            
    }
    return std::make_pair(found_items, total_items);
}
std::pair<int, int> FastRouge::single_doc_ROUGE_R(const std::vector<int>& guess_summ_dict, const std::vector<int>& gold_summ_dict)
{
    int total_items = 0;
    int found_items = 0;
    int i = 0;
    for (auto &item : gold_summ_dict)
    {
        //if (m_stopwords.count(item.first))
        //    continue;
        total_items += item;
        found_items += std::min<int>(item, guess_summ_dict[i++]);
    }
    return std::make_pair(found_items, total_items);
}
std::pair<int, int> FastRouge::single_doc_ROUGE_P(const std::vector<int>& guess_summ_dict, const std::vector<int>& gold_summ_dict)
{
    int total_items = 0;
    int found_items = 0;
    int i = 0;
    for (auto &item : guess_summ_dict)
    {
        //if (m_stopwords.count(item.first))
        //    continue;
        total_items += item;
        found_items += std::min<int>(item, gold_summ_dict[i++]);


    }
    return std::make_pair(found_items, total_items);
}
}