#include "RefreshCNNDMCorpus.h"
#include "OSListdir.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>


namespace pba_summarization
{



RefreshCNNDMCorpus::~RefreshCNNDMCorpus()
{
}
void RefreshCNNDMCorpus::BuildBestPaths()
{
    int i = 0;
    for (auto d : m_documents)
    {
        d->BuildBestPaths("");
        d->SaveBestPaths(m_path + OSSEP + "bestpaths" + OSSEP);
        if (i++ % 100 == 0)
            std::cerr << "p";

    }
}

bool RefreshCNNDMCorpus::LoadFiles(libconfig::Setting& config)
{
    config.lookupValue("Compressive", m_use_compressions);
    config.lookupValue("Path", m_path);
    config.lookupValue("IDSFile", m_ids_file);
    m_oracles_folder = "mainbody-multipleoracles";
    m_mainbody_folder = "mainbody_compressive";
    m_mainbody_extension = ".compressed.mainbody";
    m_highlight_extension = ".highlights";
    m_highlights_folder = "highlights";
    config.lookupValue("OracleFolder", m_oracles_folder);
    config.lookupValue("MainbodyFolder", m_mainbody_folder);
    config.lookupValue("HighlightsFolder", m_highlights_folder);
    config.lookupValue("MainbodyExtension", m_mainbody_extension);
    config.lookupValue("HighlightExtension", m_highlight_extension);
    config.lookupValue("OracleFolder", m_oracles_folder);

    int max_docs = 1501;
    config.lookupValue("MaxFilesCutoff", max_docs);

    std::cerr << "Loading " << m_ids_file << std::endl;
    std::ifstream data_file(m_ids_file);

    if (!data_file.good()) {
        std::cerr << "Cannot open: " << m_ids_file << std::endl;
        return false;
    }


    std::string line;
    while (std::getline(data_file, line)) {
        m_valid_file_ids.insert(line);
    }

    std::cerr << " " << m_valid_file_ids.size() << " files";
    std::cerr <<  "max_docs from " << m_ids_file << " : " << max_docs;

    int i = 0;
    for (auto &f : m_valid_file_ids)
    {

        Document *doc = new Document(m_data_model);
        doc->set_path(f);
        doc->set_keep_words(m_keep_words);
        if (LoadDocument(f, doc, false) )
        {

            m_documents_2.push_back(doc);
            m_document_by_id[f] = (int) m_documents_2.size() - 1;

            if (i <= max_docs) { 
                m_documents.push_back(doc);
                m_files.push_back(f);
            }/*otherwise we only populate the NN doc reprs needed anyway*/

            if (doc->sentences_actions().size() > 0)
            {
                std::vector<int> a(doc->sentences_actions().begin(), doc->sentences_actions().end());
                doc->set_best_rouge(doc->ComputeFastRouge(a, std::vector<std::vector<unsigned>>(a.size())));
            }
            
            //doc->LoadBestPaths(m_path + OSSEP + "bestpaths" + OSSEP + f + ".bpath");
        }
        else
            delete doc;
        if (i++ % 100 == 0)
            std::cerr << ".";
        
    }
    std::cerr << std::endl;
    return (m_valid_file_ids.size() > 0);
}

bool RefreshCNNDMCorpus::BuildOracles(libconfig::Setting& config, const std::string  &output_path)
{
    config.lookupValue("Compressive", m_use_compressions);
    config.lookupValue("Path", m_path);
    config.lookupValue("IDSFile", m_ids_file);
    m_oracles_folder = "mainbody-multipleoracles";
    m_mainbody_folder = "mainbody_compressive";
    m_mainbody_extension = ".compressed.mainbody";
    m_highlight_extension = ".highlights";
    m_highlights_folder = "highlights";
    config.lookupValue("OracleFolder", m_oracles_folder);
    config.lookupValue("MainbodyFolder", m_mainbody_folder);
    config.lookupValue("HighlightsFolder", m_highlights_folder);
    config.lookupValue("MainbodyExtension", m_mainbody_extension);
    config.lookupValue("HighlightExtension", m_highlight_extension);
    config.lookupValue("OracleFolder", m_oracles_folder);

    config.lookupValue("Path", m_path);
    config.lookupValue("IDSFile", m_ids_file);
    m_oracles_folder = "mainbody-multipleoracles";
    m_mainbody_folder = "mainbody_compressive";
    m_highlights_folder = "highlights";
    config.lookupValue("OracleFolder", m_oracles_folder);
    config.lookupValue("MainbodyFolder", m_mainbody_folder);
    config.lookupValue("HighlightsFolder", m_highlights_folder);
    config.lookupValue("HighlightExtension", m_highlight_extension);

    std::cerr << "Loading " << m_ids_file;
    std::ifstream data_file(m_ids_file);
    std::string line;
    while (std::getline(data_file, line)) {
        m_valid_file_ids.insert(line);
    }

    std::cerr << " " << m_valid_file_ids.size() << " files";

    int i = 0;
    for (auto &f : m_valid_file_ids)
    {
        //if (i > 4500)
        //    break;
        Document *doc = new Document(m_data_model);
        doc->set_path(f);
        doc->set_keep_words(m_keep_words);
        if (LoadDocument(f, doc, true) && doc->abstractive_sentences().size() > 0)
        {
            doc->BuildBestPaths(output_path);
            delete doc;
        }
        if (i++ % 100 == 0)
            std::cerr << ".";

    }
    std::cerr << std::endl;
    return (m_valid_file_ids.size() > 0);
}


bool RefreshCNNDMCorpus::LoadHighlightsAsDocument(const std::string& id, 
                                                  Document *doc) {
    std::string path = m_path + OSSEP + m_highlights_folder + OSSEP + id + m_highlight_extension;
    std::ifstream hihglights_file(path);

    int nwords = 0;
    int nlines = 0;
    std::string line;
    while (std::getline(hihglights_file, line) && nlines < 200) {

        std::istringstream iss(line);
        std::vector<std::string> sentence;
        std::vector<std::pair<int, int>> remove_segments;
        unsigned keep = 1;
        int ntk = 0;
        nlines++;
        do
        {
            std::string word;
            iss >> word;
            if (word.size() == 0) { continue; }
            if (word == "[")
                remove_segments.push_back(std::pair<int, int>(ntk, 0));
            else if (word == "]")
                remove_segments.back().second = ntk;
            else
            {
                ntk++;
                sentence.push_back(word);
                nwords++;
            }

        } while (iss);


        //sentence.push_back("</s>");
        doc->AddSentence(sentence, 0);
        doc->AddRemoveSegments(remove_segments);
    }
    if (doc->sentences().size() == 0)
        return false;

    return true;
}

bool RefreshCNNDMCorpus::LoadDocument(const std::string& id, Document* doc, bool build_oracles)
{
    std::string path = m_path + OSSEP + m_mainbody_folder + OSSEP + id + m_mainbody_extension;

    std::ifstream data_file(path);
    int nwords = 0;
    std::string line;
    int nlines = 0;
    if (!data_file.good())
        std::cerr << "Bad file (" << path << ")" << std::endl;
    int max_lines = 200000; // 200
    while (std::getline(data_file, line) && nlines < max_lines) {

        std::istringstream iss(line);
        std::vector<std::string> sentence;
        std::vector<std::pair<int, int>> remove_segments;
        unsigned keep = 1;
        int ntk = 0;
        nlines++;
        do
        {
            std::string word;
            iss >> word;
            if (word.size() == 0) { continue; }
            if (word == "[" && m_use_compressions)
                remove_segments.push_back(std::pair<int, int>(ntk, 0));
            else if (word == "]" && m_use_compressions)
                remove_segments.back().second = ntk;
            else
            {
                ntk++;
                sentence.push_back(word);
                nwords++;
            }

        } while (iss);

        
        //sentence.push_back("</s>");
        doc->AddSentence(sentence, 0);
        doc->AddRemoveSegments(remove_segments);
    }
    if (doc->sentences().size() == 0)
        return false;
    path = m_path + OSSEP + m_highlights_folder+ OSSEP + id + m_highlight_extension;
    std::ifstream hihglights_file(path);

    nlines = 0;
    while (std::getline(hihglights_file, line) && nlines++ < max_lines) {

        std::istringstream iss(line);
        std::vector<std::string> sentence;
        do
        {
            std::string word;
            iss >> word;
            if (word.size() == 0) { continue; }
            sentence.push_back(word);
        } while (iss);


        //sentence.push_back("</s>");
        doc->AddAbstractiveSentence(sentence, 0);
    }

    if (build_oracles)
        return nwords > 0;
    path = m_path + OSSEP + m_oracles_folder + OSSEP + id + ".moracle";
    //path = m_path + OSSEP + "compressive_oracle_model2" + OSSEP + id + ".moracle";

    
    std::ifstream oracle_file(path);
    int nactions = 0;
    int np = 0;
    std::vector<int> actions(doc->sentences().size(), 0);
    while (oracle_file.good() && np++ < 10 )
    {
        
        
        if (std::getline(oracle_file, line)) {
            double rate = std::stof(line.substr(line.find_last_of("\t ")));
            line = line.substr(0, line.find_last_of("\t "));
            std::istringstream iss(line);
            do {
                int action;
                iss >> action;
                if (action < max_lines)
                    actions[action] = 1;
                
            } while (iss);
            nactions++;
            doc->add_sentences_actions(actions, rate);
       }
    }

    path = m_path + OSSEP + "compressive_oracle" + OSSEP + id + ".compressed_oracle";
    path = m_path + OSSEP + m_oracles_folder + OSSEP + id + ".compressed_oracle";

    std::ifstream comp_data_file(path);
    nlines = 0;
    while (std::getline(comp_data_file, line) && nlines++ <max_lines) {

        std::istringstream iss(line);
        std::vector<unsigned> sentence;
        unsigned keep = 1;
        do
        {
            std::string word;
            iss >> word;
            if (word.size() == 0) { continue; }
            if (word == "[" && m_use_compressions)
                keep = 0;
            else if (word == "]" && m_use_compressions)
                keep = 1;
            else
                sentence.push_back(keep);
        
        } while (iss);


        //sentence.push_back("</s>");
        doc->AddRemoveWords(sentence);
    }



    return nwords > 0 /*&& nactions > 0*/;
}




}

