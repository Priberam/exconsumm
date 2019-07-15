// HNMTrainer.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include "dynet/training.h"
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/nodes.h"
#include "dynet/lstm.h"
#include "dynet/rnn.h"
#include "dynet/io.h"
#include "OSListdir.h"
#include <optionparser/optionparser.h>
#include <libconfig/libconfig.hh>
#include "RefreshCNNDMCorpus.h"
#include "DataModel.h"
#include "NoNameYetModel.h"

using namespace pba_summarization;
enum options { UNKNOWN, CONFIGFILE, TRAIN, RUNTEST, MODEL, ORACLES };
option::Descriptor usage[] = {
    { UNKNOWN, 0,"" , ""    ,option::Arg::None, "USAGE: Train [options]\nOptions:" },
{ CONFIGFILE, 0, "c", "config", option::Arg::Optional, "-c, --config\t Config file (required)" },
{ TRAIN, 0, "t", "train", option::Arg::None, "-t, --train\t Train new model" },
{ MODEL, 0, "m", "model", option::Arg::Optional, "-m, --model model_name" },
{ ORACLES, 0, "o", "oracles", option::Arg::Optional, "-o, --oracles\t Build Oracles " },

{ 0, 0, 0, 0, 0, 0 }
};


int main(int argc, char** argv)
{
  argc -= (argc > 0); argv += (argc > 0); // skip program name argv[0] if present
  option::Stats  stats(usage, argc, argv);
  option::Option options[256], buffer[256];
  option::Parser parse(usage, argc, argv, options, buffer);
  int a = options[CONFIGFILE].count();
  const char *z = options[CONFIGFILE].arg;
  if (argc == 0 || options[CONFIGFILE].count() == 0 || options[CONFIGFILE].arg == NULL) {
    option::printUsage(std::cout, usage);
    return 0;
  }
  dynet::initialize(argc, argv);
  libconfig::Config config;

  try {
    config.readFile(options[CONFIGFILE].arg);
  }

  catch (libconfig::FileIOException e) {
    std::cerr << e.what() << std::endl;
    return 0;
  } catch (libconfig::ParseException e) {
    std::cerr << e.what() << std::endl << e.getError() << " at " << e.getLine() << std::endl;
    return 0;
  } catch (libconfig::ConfigException e) {
    std::cerr << e.what() << std::endl;
    return 0;
  }

  std::string  output_path;
  if (!config.lookupValue("OutputPath", output_path))
  {
    std::cerr << "Output path not defined" << std::endl;
    return 0;
  }

  if (!config.exists("ModelParams")) {
      std::cerr << "Missing config ModelParams section" << std::endl;
      return 0;
  }

  auto& mparams = config.lookup("ModelParams");
      
  std::string model_name;
  if (!mparams.lookupValue("Name", model_name))
  {
      std::cerr << "ModelParams/Name not defined" << std::endl;
      return 0;
  }

  

  if (!FileDirectoryExists(output_path.c_str())) {
      CreateFileDirectory(output_path.c_str());
  }

  auto this_run = output_path + OSSEP + model_name;
  if (!FileDirectoryExists(this_run.c_str())) {
      CreateFileDirectory(this_run.c_str());
  } else {
      std::cout << "Directory already exists !! " << this_run << std::endl;
      return 0;
  }

  output_path = this_run;
 
  auto models_dir = output_path + std::string(OSSEP) + "models";
  auto dev_dir = output_path + std::string(OSSEP) + "dev";
  auto logs_dir = output_path + std::string(OSSEP) + "logs";
  auto cfgs_dir = output_path + std::string(OSSEP) + "cfgs";
  auto oracles_dir = output_path + std::string(OSSEP) + "oracles";

  if (!FileDirectoryExists(models_dir.c_str())) {
      CreateFileDirectory(models_dir.c_str());
  }
  if (!FileDirectoryExists(dev_dir.c_str())) {
      CreateFileDirectory(dev_dir.c_str());
  }
  if (!FileDirectoryExists(logs_dir.c_str())) {
      CreateFileDirectory(logs_dir.c_str());
  }
  if (!FileDirectoryExists(cfgs_dir.c_str())) {
      CreateFileDirectory(cfgs_dir.c_str());
  }
  if (!FileDirectoryExists(oracles_dir.c_str())) {
      CreateFileDirectory(oracles_dir.c_str());
  }
  std::cout << "model name: " << model_name << std::endl;
  std::cout << "output path: " << output_path << std::endl;

  std::string evaluate_command = "python3 ROUGE_REFRESH.py";
  config.lookupValue("EvaluateCommand", evaluate_command);


  bool lower_case_emebedings = false;
  config.lookupValue("LowerCaseEmbeddings", lower_case_emebedings);
  double rougeF = 1;
  config.lookupValue("RougeFx", rougeF);
  Document::setRougeFx(rougeF);
  if (options[ORACLES].count())
  {

      libconfig::Setting &cfg_training = config.lookup("TrainingCorpus");

      pba_summarization::DataModel  data_model;
      data_model.set_lower_case_emebedings(lower_case_emebedings);
      data_model.set_force_add(true);

      pba_summarization::RefreshCNNDMCorpus corpus(data_model);
      corpus.set_keep_words(true);
      if (!corpus.BuildOracles(cfg_training, output_path + "/oracles/"))
      {
          std::cerr << "Fail to load training files" << std::endl;
          return 0;
      }
      return 0;
  }
  else  if (options[TRAIN].count())
  {
      libconfig::Setting &cfg_training = config.lookup("TrainingCorpus");
      libconfig::Setting &cfg_validation = config.lookup("ValidationCorpus");
      libconfig::Setting &cfg_test = config.lookup("TestCorpus");

      pba_summarization::DataModel  data_model;

      data_model.set_lower_case_emebedings(lower_case_emebedings);

      std::string  file_embeddings;
      if (!config.lookupValue("PretrainedEmbeddings", file_embeddings))
      {
          std::cerr << "Embeddings file not defined in config" << std::endl;
          return 0;
      }
      int embeddings_dim = 0;
      if (!config.lookupValue("EmbeddingsDim", embeddings_dim))
      {
          std::cerr << "Embeddings dimension defined in config" << std::endl;
          return 0;
      }
      bool lower_case_emebedings = false;
      config.lookupValue("LowerCaseEmbeddings", lower_case_emebedings);
      data_model.set_lower_case_emebedings(lower_case_emebedings);

      if (!data_model.load_pretrainned_word_embeddings(file_embeddings, embeddings_dim))
      {
          std::cerr << "Fail to load embedings" << std::endl;
          return 0;
      }
      std::string corpus_type = "Sidenet";
      if (!config.lookupValue("CorpusType", corpus_type))
      {
          std::cerr << "Train path not defined in config" << std::endl;
          return 0;
      }

      std::cerr << "Corpus type is " << corpus_type << std::endl;
      pba_summarization::GenericCorpus *train_corpus = NULL, *validation_corpus = NULL, *test_corpus = NULL;


      if (corpus_type == "Refresh")
      {
          train_corpus = new pba_summarization::RefreshCNNDMCorpus(data_model);
          validation_corpus = new pba_summarization::RefreshCNNDMCorpus(data_model);
          test_corpus = new pba_summarization::RefreshCNNDMCorpus(data_model);

      }
      else
      {
          std::cerr << "Invalid corpus type !" << std::endl;
          return 0;
      }

      if (!train_corpus->LoadFiles(cfg_training))
      {
          std::cerr << "Fail to load training files" << std::endl;
          return 0;
      }
      auto model_config = std::make_shared<pba_summarization::NoNameYetModelConfig>();
      model_config->InitFromConfig(config, logs_dir);

      model_config->Logger("csum") << "Output path: " << output_path << std::endl;
      model_config->Logger("gsum") << "Output path: " << output_path << std::endl;
      model_config->Logger("asum") << "Output path: " << output_path << std::endl;

      validation_corpus->set_keep_words(true);

      if (!validation_corpus->LoadFiles(cfg_validation))
      {
          std::cerr << "Fail to load validation files" << std::endl;
          return 0;
      }


      {
          int nwords = 100000;
          config.lookupValue("MaxWords", nwords);
          data_model.limit_words(nwords);
          data_model.lock();
          data_model.load_pretrainned_word_embeddings(file_embeddings, embeddings_dim);
          train_corpus->clear();
          train_corpus->LoadFiles(cfg_training);
          validation_corpus->set_keep_words(true);
          validation_corpus->clear();
          validation_corpus->LoadFiles(cfg_validation);
      }
      model_config->vocab_size((unsigned)data_model.word_embeddings().size());
      model_config->action_size((unsigned)data_model.actions().size());



      if (!config.getRoot().exists("ModelFile"))
          config.getRoot().add("ModelFile", libconfig::Setting::Type::TypeString);
      config.getRoot()["ModelFile"] = output_path + "/models/" + model_config->model_name();
      data_model.save_model(output_path + "/models/" + model_config->model_name() + ".dat");

      std::string cfg = cfgs_dir + OSSEP + model_config->model_name() + ".cfg";
      config.writeFile(cfg.c_str());
      std::string file_model;

      try {
          pba_summarization::NoNameYetModel model(&data_model, model_config.get());
          if (options[MODEL].count() != 0)
              model_config->load_params(options[MODEL].arg);
          model.Train(*(pba_summarization::Corpus*)train_corpus, *(pba_summarization::Corpus*)validation_corpus, output_path, evaluate_command);
      }
      catch (std::runtime_error e) {
          std::cerr << "main - runtime_error.what(): " << e.what() << std::endl;
      }
      catch (std::exception e) {
          std::cerr << "main - exception.what(): " << e.what() << std::endl;
      }
      catch (...) {
          std::cerr << "main - unknown exception" << std::endl;
      }

  }
  else
  {
      std::string corpus_type = "Sidenet";
      if (!config.lookupValue("CorpusType", corpus_type))
      {
          std::cerr << "Train path not defined in config" << std::endl;
          return 0;
      }

      std::cerr << "Corpus type is " << corpus_type << std::endl;
      libconfig::Setting &cfg_test = config.lookup("TestCorpus");

      int embeddings_dim = 0;
      if (!config.lookupValue("EmbeddingsDim", embeddings_dim))
      {
          std::cerr << "Embeddings dimension defined in config" << std::endl;
          return 0;
      }
      std::string file_model;
      if (!config.lookupValue("ModelFile", file_model))
      {
          std::cerr << "Model file not defined in config" << std::endl;
          return 0;
      }

      pba_summarization::DataModel  data_model;
      data_model.set_lower_case_emebedings(lower_case_emebedings);
      data_model.load_model(file_model + ".dat");


      pba_summarization::GenericCorpus  *test_corpus = NULL;


      if (corpus_type == "Refresh")
      {
          test_corpus = new pba_summarization::RefreshCNNDMCorpus(data_model);
      }
      else
      {
          std::cerr << "Invalid corpus type !" << std::endl;
          return 0;
      }
      test_corpus->set_keep_words(true);
      if (!test_corpus->LoadFiles(cfg_test))
      {
          std::cerr << "Fail to load test files" << std::endl;
          return 0;
      }
      auto model_config = std::make_shared<pba_summarization::NoNameYetModelConfig>();
      model_config->InitFromConfig(config, logs_dir);

      model_config->Logger("csum") << "Output path: " << output_path << std::endl;
      model_config->Logger("gsum") << "Output path: " << output_path << std::endl;
      model_config->Logger("asum") << "Output path: " << output_path << std::endl;

      model_config->vocab_size((unsigned)data_model.words().m_str_vec.size());
      model_config->action_size((unsigned)data_model.actions().size());


      pba_summarization::NoNameYetModel model(&data_model, model_config.get());
      if (options[MODEL].count() == 0)
          model_config->load_params(file_model + ".params");
      else
          model_config->load_params(options[MODEL].arg);


      model.Test(*(pba_summarization::Corpus*)test_corpus, output_path, evaluate_command);

  }

  return 0;
}

