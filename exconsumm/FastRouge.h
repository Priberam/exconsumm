#pragma once
#include <unordered_set>
#include <unordered_map>
#include <vector>
namespace pba_summarization
{
class FastRouge
{
public:
    FastRouge();
    ~FastRouge();

    std::pair<int, int> single_doc_ROUGE_R(const std::unordered_map<uint64_t, int>& guess_summ_dict, const std::unordered_map<uint64_t, int>& gold_summ_dict);
    std::pair<int, int> single_doc_ROUGE_P(const std::unordered_map<uint64_t, int>& guess_summ_dict, const std::unordered_map<uint64_t, int>& gold_summ_dict);
    std::pair<int, int> single_doc_ROUGE_R(const std::vector<int>& guess_summ_dict, const std::vector<int>& gold_summ_dict);
    std::pair<int, int> single_doc_ROUGE_P(const std::vector<int>& guess_summ_dict, const std::vector<int>& gold_summ_dict);

protected:
    std::unordered_set<unsigned> m_stopwords;
};
}
