#ifndef REFRESHCNNDMCORPUS_H
#define REFRESHCNNDMCORPUS_H

#include "Corpus.h"
#include <libconfig/libconfig.hh>
#include <unordered_set>

namespace pba_summarization
{
class RefreshCNNDMCorpus : public Corpus
{
public:
    RefreshCNNDMCorpus(DataModel &data_model)  : Corpus(data_model) {};
    ~RefreshCNNDMCorpus();
  


public:
    bool LoadFiles(libconfig::Setting& config);
    bool BuildOracles(libconfig::Setting& config, const std::string  &output_path);
    bool LoadHighlightsAsDocument(const std::string& id, Document *doc);

    bool LoadDocument(const std::string& path, Document *doc, bool build_oracles);
    virtual void clear()
    {
        Corpus::clear();
        m_valid_file_ids.clear();
    }
    void BuildBestPaths();
public:
    
    std::string m_ids_file;
    std::string m_path;
    std::unordered_set<std::string> m_valid_file_ids;
    std::string m_oracles_folder;
    std::string m_mainbody_folder;
    std::string m_highlights_folder;
    std::string m_mainbody_extension = ".compressed.mainbody";
    std::string m_highlight_extension = ".highlight";
    bool m_use_compressions = true;

};


}
#endif

