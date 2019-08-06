
#include <chrono>
#include <random>
#include <fstream>  
#include <iostream>  
#include <signal.h>
#include <algorithm>
#include "util.h"
#include "Corpus.h"
#include "NoNameYetModel.h"


#undef min

namespace pba_summarization
{


   dynet::Expression NoNameYetModel::sentence_encoder_lstm(dynet::ComputationGraph* hg, const std::vector<unsigned>& sentence, bool apply_dropout, EmbededDocument* embeded_doc)
   {
      if (apply_dropout) {
         sentence_lstm_fwd.set_dropout(m_config->DROPOUT);
         sentence_lstm_rev.set_dropout(m_config->DROPOUT);
      }
      else {
         sentence_lstm_fwd.disable_dropout();
         sentence_lstm_fwd.disable_dropout();
      }
      dynet::Expression ib = parameter(*hg, p_ib);
      dynet::Expression w2l = parameter(*hg, p_w2l);

      sentence_lstm_fwd.start_new_sequence();
      sentence_lstm_rev.start_new_sequence();
      if (embeded_doc)
      {
         embeded_doc->m_embeded_sentences_words.push_back({});
         embeded_doc->m_sentences_words.push_back({});
         embeded_doc->m_embeded_sentences_words_fwd.push_back({});
         embeded_doc->m_embeded_sentences_words_rev.push_back({});
      }
      std::vector<dynet::Expression> ws;
      std::vector<dynet::Expression> out;
      ws.reserve(sentence.size() + 2);
      int eos = m_data_model->get_word("EOS");
      auto eeos = /*const_*/lookup(*hg, p_t, eos);
      ws.push_back(eeos);
      sentence_lstm_fwd.add_input(eeos);
      if (embeded_doc)
      {
         embeded_doc->m_sentences_words.back().push_back(eeos);
         embeded_doc->m_embeded_sentences_words_fwd.back().push_back(sentence_lstm_fwd.back());
      }
      out.push_back(sentence_lstm_fwd.back());
      for (unsigned i = 0; i < sentence.size(); ++i) {
         dynet::Expression w;
         if (m_config->UPDATE_PRETRAINED || sentence[i] == 0)
            w = /*const_*/lookup(*hg, p_t, sentence[i]);
         else
            w = const_lookup(*hg, p_t, sentence[i]);
         sentence_lstm_fwd.add_input(w);
         ws.push_back(w);
         if (embeded_doc)
         {
            embeded_doc->m_sentences_words.back().push_back(w);
            embeded_doc->m_embeded_sentences_words_fwd.back().push_back(sentence_lstm_fwd.back());
         }
         out.push_back(sentence_lstm_fwd.back());

      }
      sentence_lstm_fwd.add_input(eeos);
      if (embeded_doc)
      {
         embeded_doc->m_sentences_words.back().push_back(eeos);
         embeded_doc->m_embeded_sentences_words_fwd.back().push_back(sentence_lstm_fwd.back());
      }
      out.push_back(sentence_lstm_fwd.back());
      ws.push_back(eeos);
      unsigned i = (unsigned)ws.size() - 1;
      if (embeded_doc)
         embeded_doc->m_embeded_sentences_words_rev.back().resize(embeded_doc->m_embeded_sentences_words_fwd.back().size());
      while (!ws.empty())
      {
         sentence_lstm_rev.add_input(ws.back());
         if (embeded_doc)
         {
            embeded_doc->m_embeded_sentences_words.back().push_back(dynet::concatenate({ out[i], sentence_lstm_rev.back() }));
            embeded_doc->m_embeded_sentences_words_rev.back()[i--] = sentence_lstm_rev.back();
         }
         ws.pop_back();
      }
      dynet::Expression efwd = sentence_lstm_fwd.back();
      dynet::Expression erev = sentence_lstm_rev.back();
      if (apply_dropout) {
         efwd = dropout(efwd, m_config->DROPOUT);
         erev = dropout(erev, m_config->DROPOUT);
      }
      dynet::Expression c = dynet::concatenate({ efwd, erev });
      return c;

   }

   void NoNameYetModel::encode_document(dynet::ComputationGraph* hg, const Document &document, bool apply_dropout, EmbededDocument& embed_doc)
   {
      if (apply_dropout) {
         output_lstm.set_dropout(m_config->DROPOUT);
         buffer_lstm.set_dropout(m_config->DROPOUT);
      }
      else {
         output_lstm.disable_dropout();
         buffer_lstm.disable_dropout();
      }
      buffer_lstm.start_new_sequence();
      output_lstm.start_new_sequence();

      // Build an hierarquial lstm for the sentences
      std::vector<dynet::Expression> buffer(document.sentences().size() + 1);  // variables representing word embeddings (possibly including POS info)
      embed_doc.bufferf.resize(document.sentences().size());  // variables representing word embeddings (possibly including POS info)

      for (unsigned j = 0; j < document.sentences().size(); j++)
      {
         dynet::Expression composed;
         composed = /*dynet::nobackprop*/(sentence_encoder_lstm(hg, document.sentences()[j], apply_dropout, &embed_doc));

         buffer[document.sentences().size() - j] = composed;
         embed_doc.bufferf[j] = composed;
      }
      // dummy symbol to represent the empty buffer
      buffer[0] = parameter(*hg, p_buffer_guard);
      embed_doc.bufferf.push_back(buffer[0]);
      embed_doc.buffer_lstm_out.resize(document.sentences().size() + 1);
      int j = 0;
      for (auto& b : buffer)
      {
         buffer_lstm.add_input(b);
         embed_doc.buffer_lstm_out[document.sentences().size() - j++] = buffer_lstm.back();
      }
      embed_doc.output_lstm_out.clear();
      for (auto& b : embed_doc.bufferf)
      {
         output_lstm.add_input(b);
         embed_doc.output_lstm_out.push_back(output_lstm.back());
      }

      embed_doc.doc = dynet::concatenate({ buffer_lstm.back(), output_lstm.back() });

   }
#define ST 6000
   std::pair<std::vector<int>, dynet::Expression> NoNameYetModel::log_prob_parser(dynet::ComputationGraph* hg,
      const Document &document,
      Corpus* corpus,
      bool is_evaluation,
      double *right, EmbededDocument& embed_doc, int iac, bool standalone_mode)
   {
      std::vector<int> results;

      bool apply_dropout = (m_config->USE_DROPOUT && !is_evaluation);
      static int n = 0;
      std::vector<int> actions;
      double reward = 1;

      if (!is_evaluation && !standalone_mode)
      {
         if (m_config->NOISE == 0)
            actions = document.sentences_actions_rnd(iac, &reward);
         else
            actions = document.sentences_actions();
      }


      std::random_device rd;
      std::uniform_real_distribution<double> distribution(1, 100);
      std::mt19937 engine(rd()); // Mersenne twister MT19937
      bool change = false;
      bool add_one = false;
      if (distribution(engine) < m_config->NOISE && iac != 0)
      {
         change = true;
         if (distribution(engine) < 50)
            add_one = true;

      }

      if (!is_evaluation && change) {
         for (auto &a : actions)
         {
            if (a == 1 /*&& !add_one*/)
            {
               double value = distribution(engine);
               if (value < 300 / 3.0)
                  a = 0;
            }
            else if (a == 0 /*&& add_one*/)
            {
               double value = distribution(engine);
               if (value < 300 / (double)(actions.size() - 3))
                  a = 1;
            }
         }
      }
      const bool build_training_graph = actions.size() > 0;

      if (apply_dropout) {
         stack_lstm.set_dropout(m_config->DROPOUT);
      }
      else {
         stack_lstm.disable_dropout();
      }
      // variables in the computation graph representing the parameters
      dynet::Expression pbias = parameter(*hg, p_pbias);
      dynet::Expression S = parameter(*hg, p_S);
      dynet::Expression O = parameter(*hg, p_O);
      dynet::Expression NNs;




      dynet::Expression B = parameter(*hg, p_B);

      dynet::Expression A = parameter(*hg, p_A);
      dynet::Expression p2a = parameter(*hg, p_p2a);
      dynet::Expression abias = parameter(*hg, p_abias);

      dynet::Expression ib = parameter(*hg, p_ib);
      dynet::Expression w2l = parameter(*hg, p_w2l);



      std::vector< dynet::Expression> log_probs;
      std::vector< dynet::Expression> log_probs_compressive;

      stack_lstm.start_new_sequence();
      stack_lstm.add_input(parameter(*hg, p_stack_guard));

      unsigned action_count = 0;  // incremented at each prediction
      std::vector<unsigned> current_valid_actions = { 0, 1 };
      int ii = 0;

      bool has_nns{ false };

      for (unsigned i = 0; i < embed_doc.bufferf.size() - 1; i++) {
         dynet::Expression composed = dynet::affine_transform({ ib, w2l, embed_doc.bufferf[action_count] });
         dynet::Expression bb = dynet::concatenate({ embed_doc.buffer_lstm_out[action_count], embed_doc.output_lstm_out[action_count] });
         dynet::Expression p_t;
         dynet::Expression attend_vector;

         //if (parent_doc == NULL)
         if (!m_config->COMPRESSIVE)
         {
            std::vector<dynet::Expression> aff_vars;

            aff_vars = { pbias, O, embed_doc.doc, S, stack_lstm.back(), B, bb, A, composed };
            p_t = dynet::affine_transform(aff_vars);
         }
         else
         {
            std::vector<dynet::Expression> aff_vars = { pbias, O, embed_doc.doc, S, stack_lstm.back(), B, bb, A, composed };
            p_t = dynet::affine_transform(aff_vars);
         }

         dynet::Expression nlp_t = dynet::tanh(p_t);

         dynet::Expression r_t = dynet::affine_transform({ abias, p2a, nlp_t });
         std::vector<float> adist;
         double best_score = 0;
         unsigned best_a = current_valid_actions[0];
         dynet::Expression adiste = log_softmax(r_t); //, current_valid_actions);

         if (!build_training_graph)
         {
            dynet::Expression adiste = softmax(r_t); //, current_valid_actions);

            adist = as_vector(hg->incremental_forward(adiste));
            best_score = adist[current_valid_actions[0]];
            for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
               if (adist[current_valid_actions[i]] > best_score) {
                  best_score = adist[current_valid_actions[i]];
                  best_a = current_valid_actions[i];
               }
            }
         }


         unsigned action = best_a;

         if (build_training_graph) {  // if we have reference actions (for training) use the reference action
            action = actions[action_count];

            if (best_a == action) { (*right)++; }
         }
         if (is_evaluation)
            action = best_a;

         log_probs.push_back(pick(adiste, action));

         if (action == 1)
         {
            embed_doc.m_summary_sentences.push_back(action_count);
         }
         ++action_count;

         results.push_back(best_a);

         if (action == 2 && is_evaluation)
            break;
         DataModel::ACTION_TYPE action_type = m_data_model->action_type(action);
         if (action_type == DataModel::ACTION_TYPE::IN_SUMMARY && m_config->COMPRESSIVE)
         {
            std::pair<std::vector<unsigned>, dynet::Expression> r;
            if (!m_config->COMPRESSIVE_BOW)
               r = m_word_level.log_prob_parser2(hg, document, is_evaluation, action_count - 1, NULL, embed_doc, embed_doc.doc, stack_lstm.back(), iac);
            else
               r = m_word_level.log_prob_parser(hg, document, is_evaluation, action_count - 1, NULL, embed_doc, embed_doc.doc, stack_lstm.back(), iac);
            log_probs_compressive.push_back(r.second);

         }


         // vou colocar a frase no output com a relação e vou colocar no stack

         if (action_type == DataModel::ACTION_TYPE::IN_SUMMARY) {
            if (!m_config->COMPRESSIVE)
            {
               stack_lstm.add_input(composed);
            }
            else
            {
               stack_lstm.add_input(m_word_level.stack_lstm.back());
            }
         }


      }

      int pos = 0;
      for (unsigned i = 0; i < results.size(); i++)
      {
         if (results[i] == 1)
         {
            pos++;
         }
      }

      std::vector<dynet::Expression> a;
      std::vector<dynet::Expression> b;

      for (unsigned i = 0; i < actions.size(); i++)
      {
         if (actions[i] == 1)
         {
            a.push_back(log_probs[i]);
         }
         else
            b.push_back(log_probs[i]);
      }

      //float rel2 = 1.0f / (a.size() / (float)(pos + 0.01));
      dynet::Expression tot_neglogprob = -sum(log_probs);

      if (a.size() && m_config->MAX_RELATION > 0)
      {
         tot_neglogprob = -sum(a) / (float)a.size(); // / document.sentences().size()  /*+  -log(dynet::logistic(1- dynet::constant(*hg, { 1 },nwords / (double)total_words)+ 0.0001))*/;

         float rel = std::min(b.size() / (float)a.size(), m_config->MAX_RELATION);
         if (b.size() > 0)
            tot_neglogprob = (tot_neglogprob - sum(b) / (float)b.size() * rel);
      }
      if (m_config->COMPRESSIVE && !is_evaluation && log_probs_compressive.size() > 0)
         tot_neglogprob = tot_neglogprob + sum(log_probs_compressive) / (float)log_probs_compressive.size() * 2;


      dynet::Expression p;


      if (m_config->USE_ROUGE)
      {
         float rouge = 0;
         if (!is_evaluation)
         {
            rouge = (float)document.ComputeFastRouge(actions, embed_doc.m_result_summary_sentence_words);
            //results = actions;
         }
         else
            rouge = (float)document.ComputeFastRouge(results, embed_doc.m_result_summary_sentence_words);
         std::cerr << "R:" << rouge << std::endl;
         tot_neglogprob = tot_neglogprob * ((rouge));

         if (iac == 0 && rouge == 0 && !is_evaluation)
         {
            results.clear();
            return std::make_pair(results, tot_neglogprob);
         }
      }
      else
         tot_neglogprob = (tot_neglogprob)*(float)reward;



      return std::make_pair(results, tot_neglogprob/** actions.size()*/);
   }



   static volatile bool requested_stop = false;
   static void signal_callback_handler(int /* signum */) {
      if (requested_stop) {
         std::cerr << "\nReceived SIGINT again, quitting.\n";
         _exit(1);
      }
      std::cerr << "\nReceived SIGINT terminating optimization early...\n";
      requested_stop = true;
   }

   std::pair<std::vector<unsigned>, dynet::Expression> WordLevelExtractorModel::log_prob_parser(dynet::ComputationGraph* hg,
      const Document &document,
      bool is_evaluation, unsigned sentence,
      double *right, EmbededDocument& embed_doc, dynet::Expression& doc_context, dynet::Expression parent_stack, int iac)
   {
      std::vector<unsigned> results;

      bool apply_dropout = (m_config->USE_DROPOUT && !is_evaluation);

      std::vector<unsigned> actions;
      std::unordered_set<unsigned> abstractive_word_ids;


      if (!is_evaluation)
      {
         for (auto &s : document.abstractive_sentences())
         {
            for (auto w : s)
               abstractive_word_ids.insert(w);
         }
         actions.resize(document.sentences()[sentence].size() + 1);
         actions[0] = 1;
         for (unsigned i = 1; i < document.sentences()[sentence].size(); i++)
         {
            auto w = document.sentences()[sentence][i - 1];
            if (abstractive_word_ids.find(w) != abstractive_word_ids.end() && w != 0)
               actions[i] = 1;
            else
               actions[i] = 0;
         }
         //actions.back() = 1;

         std::random_device rd;
         std::uniform_real_distribution<double> distribution(0, 100);
         std::mt19937 engine(rd()); // Mersenne twister MT19937
         bool change = false;
         bool add_one = false;
         if (distribution(engine) < m_config->NOISE && iac != 0)
         {
            change = true;
         }

         if (!is_evaluation && change)
            for (auto &a : actions)
            {
               double value = distribution(engine);
               if (value < 100.0 * 2 / (double)actions.size())
                  a = (a == 1) ? 0 : 1;
            }


      }

      const bool build_training_graph = actions.size() > 0;



      if (apply_dropout) {
         stack_lstm.set_dropout(m_config->DROPOUT);
      }
      else {
         stack_lstm.disable_dropout();
      }

      // variables in the computation graph representing the parameters
      dynet::Expression pbias = parameter(*hg, p_pbias);
      dynet::Expression S = parameter(*hg, p_S);
      dynet::Expression PS = parameter(*hg, p_PS);

      dynet::Expression B = parameter(*hg, p_B);
      dynet::Expression O = parameter(*hg, p_O);
      dynet::Expression DC = parameter(*hg, p_DC);


      dynet::Expression A = parameter(*hg, p_A);
      dynet::Expression p2a = parameter(*hg, p_p2a);
      dynet::Expression abias = parameter(*hg, p_abias);
      //dynet::Expression action_start = parameter(*hg, p_action_start);

      dynet::Expression ib = parameter(*hg, p_ib);
      dynet::Expression w2l = parameter(*hg, p_w2l);

      std::vector<dynet::Expression> buffer(embed_doc.m_sentences_words[sentence].size());  // variables representing word embeddings (possibly including POS info)
      std::vector<dynet::Expression> bufferf(embed_doc.m_sentences_words[sentence].size());  // variables representing word embeddings (possibly including POS info)
      std::vector<int> bufferi(embed_doc.m_sentences_words[sentence].size());  // position of the words in the sentence
                                                                                   // precompute buffer representation from left to right

                                                                                   // Build an hierarquial lstm for the sentences
      for (unsigned j = 0; j < embed_doc.m_sentences_words[sentence].size(); j++)
      {
         bufferf[j] = embed_doc.m_sentences_words[sentence][j];
      }


      dynet::Expression doc = dynet::concatenate({ embed_doc.m_embeded_sentences_words_fwd[sentence].back(), embed_doc.m_embeded_sentences_words_rev[sentence][0] });
      std::vector< dynet::Expression> log_probs;

      stack_lstm.start_new_sequence();
      stack_lstm.add_input(parameter(*hg, p_stack_guard));

      unsigned action_count = 0;  // incremented at each prediction
      std::vector<unsigned> current_valid_actions = { 0, 1 };
      std::vector<dynet::Expression> sms;
      int count_in = 0;
      while (buffer.size() > 1) {
         dynet::Expression composed = dynet::affine_transform({ ib, w2l, bufferf[action_count] });
         dynet::Expression bb = dynet::concatenate({ embed_doc.m_embeded_sentences_words_fwd[sentence][action_count],embed_doc.m_embeded_sentences_words_rev[sentence][action_count] });
         dynet::Expression p_t;

         p_t = dynet::affine_transform({ pbias, O, doc, S, stack_lstm.back(), B, bb, A,
                 composed, DC, doc_context/*, PS, parent_stack */ });

         dynet::Expression nlp_t = dynet::tanh(p_t);
         dynet::Expression r_t = dynet::affine_transform({ abias, p2a, nlp_t });

         dynet::Expression adiste = log_softmax(r_t); //, current_valid_actions);
         std::vector<float> adist = as_vector(hg->incremental_forward(adiste));
         double best_score = adist[current_valid_actions[0]];
         unsigned best_a = current_valid_actions[0];
         for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
            if (adist[current_valid_actions[i]] > best_score) {
               best_score = adist[current_valid_actions[i]];
               best_a = current_valid_actions[i];
            }
         }
         unsigned action = best_a;
         if (build_training_graph) {
            action = actions[action_count];
            //if (best_a == action) { (*right)++; }
         }

         auto w = 0;
         if (action_count)
            w = document.sentences()[sentence][action_count - 1];
         //if (done_words.find(w) == done_words.end() || w == 0)
         log_probs.push_back(pick(adiste, action));

         if (is_evaluation)
            action = best_a;


         ++action_count;

         results.push_back(best_a);
         DataModel::ACTION_TYPE action_type = m_data_model->action_type(action);


         if (action_type == DataModel::ACTION_TYPE::IN_SUMMARY) {
            stack_lstm.add_input(composed);

         }
         buffer.pop_back();
      }



      std::vector<dynet::Expression> a;
      std::vector<dynet::Expression> b;

      for (unsigned i = 0; i < actions.size(); i++)
      {
         if (actions[i] == 1)
         {
            a.push_back(log_probs[i]);
         }
         else
            b.push_back(log_probs[i]);
      }
      dynet::Expression tot_neglogprob = -sum(log_probs) / (float)actions.size() /*/ (1.5f)*/;

      if (a.size())
      {
         tot_neglogprob = -sum(a) / (float)a.size(); // / document.sentences().size()  /*+  -log(dynet::logistic(1- dynet::constant(*hg, { 1 },nwords / (double)total_words)+ 0.0001))*/;

         if (b.size() > 0)
            tot_neglogprob = tot_neglogprob - sum(b) / (float)b.size();
      }
      if (is_evaluation)
         embed_doc.m_result_summary_sentence_words[sentence] = results;
      else
         embed_doc.m_result_summary_sentence_words[sentence] = actions;
      return std::make_pair(results, tot_neglogprob /*/ (1 + 0.5)*/);
   }


   std::pair<std::vector<unsigned>, dynet::Expression> WordLevelExtractorModel::log_prob_parser2(dynet::ComputationGraph* hg,
      const Document &document,
      bool is_evaluation, unsigned sentence,
      double *right, EmbededDocument& embed_doc, dynet::Expression& doc_context, dynet::Expression parent_stack, int iac, bool standalone_mode)
   {
      std::vector<unsigned> results;

      bool apply_dropout = (m_config->USE_DROPOUT && !is_evaluation);

      std::vector<unsigned> actions;

      if (!standalone_mode) {
         actions = document.sentences_word_actions(sentence);
      }
      else {
         actions.resize(document.sentences()[sentence].size() + 1, 0);
      }

      actions.insert(actions.begin(), 1);
      actions.resize(document.sentences()[sentence].size() + 1);

      const bool build_training_graph = actions.size() > 1;



      if (apply_dropout) {
         stack_lstm.set_dropout(m_config->DROPOUT);
      }
      else {
         stack_lstm.disable_dropout();
      }

      // variables in the computation graph representing the parameters
      dynet::Expression pbias = parameter(*hg, p_pbias);
      dynet::Expression S = parameter(*hg, p_S);
      dynet::Expression PS = parameter(*hg, p_PS);

      dynet::Expression B = parameter(*hg, p_B);
      dynet::Expression O = parameter(*hg, p_O);
      dynet::Expression DC = parameter(*hg, p_DC);


      dynet::Expression A = parameter(*hg, p_A);
      dynet::Expression p2a = parameter(*hg, p_p2a);
      dynet::Expression abias = parameter(*hg, p_abias);
      //dynet::Expression action_start = parameter(*hg, p_action_start);

      dynet::Expression ib = parameter(*hg, p_ib);
      dynet::Expression w2l = parameter(*hg, p_w2l);


      std::vector<dynet::Expression> bufferf(embed_doc.m_sentences_words[sentence].size());  // variables representing word embeddings (possibly including POS info)

                                                                  // precompute buffer representation from left to right

                                                                  // Build an hierarquial lstm for the sentences
      for (unsigned j = 0; j < embed_doc.m_sentences_words[sentence].size(); j++)
      {
         bufferf[j] = embed_doc.m_sentences_words[sentence][j];
      }
#ifdef DONT_USE_MAXPOL
      std::vector<dynet::Expression> rep;
      for (unsigned i = 0; i < embed_doc.m_embeded_sentences_words_fwd[sentence].size(); i++)
         rep.push_back(dynet::concatenate({ embed_doc.m_embeded_sentences_words_fwd[sentence][i], embed_doc.m_embeded_sentences_words_rev[sentence][i] }));
      dynet::Expression ar;
      ar = dynet::concatenate(rep, 1);

      dynet::Expression doc = dynet::max_dim(ar, 1);
#else
      dynet::Expression doc = dynet::concatenate({ embed_doc.m_embeded_sentences_words_fwd[sentence].back(), embed_doc.m_embeded_sentences_words_rev[sentence][0] });
#endif
      std::vector< dynet::Expression> log_probs;
      stack_lstm.start_new_sequence();
      stack_lstm.add_input(parameter(*hg, p_stack_guard));
      unsigned action_count = 0;  // incremented at each prediction
      std::vector<unsigned> current_valid_actions = { 0, 1 };
      int count_in = 0;
      for (unsigned j = 0; j < actions.size(); j++) {
         dynet::Expression composed = dynet::affine_transform({ ib, w2l, bufferf[action_count] });
         dynet::Expression bb = dynet::concatenate({ embed_doc.m_embeded_sentences_words_fwd[sentence][action_count],
             embed_doc.m_embeded_sentences_words_rev[sentence][action_count] });
         dynet::Expression p_t;

         p_t = dynet::affine_transform({ pbias, O, doc, S, stack_lstm.back(), B, bb, A, composed, DC, doc_context, PS, parent_stack });


         dynet::Expression nlp_t = dynet::tanh(p_t);
         //if (apply_dropout)
         //    nlp_t = dynet::dropout(nlp_t, m_config.DROPOUT);

         dynet::Expression r_t = dynet::affine_transform({ abias, p2a, nlp_t });

         dynet::Expression adiste = log_softmax(r_t); //, current_valid_actions);
         unsigned best_a = current_valid_actions[0];

         if (is_evaluation) {
            std::vector<float> adist = as_vector(hg->incremental_forward(adiste));
            double best_score = adist[current_valid_actions[0]];
            for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
               if (adist[current_valid_actions[i]] > best_score) {
                  best_score = adist[current_valid_actions[i]];
                  best_a = current_valid_actions[i];
               }
            }
         }
         unsigned action = best_a;
         if (build_training_graph) {
            action = actions[action_count];
            //if (best_a == action) { (*right)++; }
         }

         log_probs.push_back(pick(adiste, action));

         if (is_evaluation)
            action = best_a;

         ++action_count;

         results.push_back(best_a);
         DataModel::ACTION_TYPE action_type = m_data_model->action_type(action);

         if (action_type == DataModel::ACTION_TYPE::IN_SUMMARY) {
            stack_lstm.add_input(composed);

         }
      }

      std::vector<dynet::Expression> a;
      std::vector<dynet::Expression> b;

      for (unsigned i = 0; i < actions.size(); i++)
      {
         if (actions[i] == 1)
         {
            a.push_back(log_probs[i]);
         }
         else
            b.push_back(log_probs[i]);
      }
      dynet::Expression tot_neglogprob = -sum(log_probs) / (float)actions.size();

      if (a.size())
      {
         tot_neglogprob = -sum(a) / (float)a.size();

         if (b.size() > 0)
            tot_neglogprob = tot_neglogprob - sum(b) / (float)b.size()/**0.5*/;
      }

      if (!standalone_mode && tot_neglogprob.pg == nullptr) {
         std::cout << "error: tot_neglogprob null" << std::endl;
      }

      if (is_evaluation)
         embed_doc.m_result_summary_sentence_words[sentence] = results;
      else
         embed_doc.m_result_summary_sentence_words[sentence] = actions;
      return std::make_pair(results, tot_neglogprob);
   }


   bool NoNameYetModel::Train(Corpus& train_corpus, Corpus& dev_corpus, const std::string& output_path, const std::string& evaluate_command)
   {
#ifdef ABS_DEVICECONTROL
      std::cerr << "ABS_DEVICECONTROL ON" << std::endl;
#else
      std::cerr << "ABS_DEVICECONTROL OFF" << std::endl;
#endif

      bool softlinkCreated = false;
      double best_f1_score = -1.0;

      const std::string fname_parameters = m_config->model_name() + ".params";
      std::cerr << "Writing parameters to file: " << fname_parameters << std::endl;

      std::unordered_set<unsigned> training_vocab; // words available in the training corpus

      bool next_epoch = 1.0;
      signal(SIGINT, signal_callback_handler);
      //dynet::SimpleSGDTrainer sgd(m_config.m_model);
      dynet::AdamTrainer sgd(m_config->m_model);
      sgd.learning_rate = m_config->LEARNING_RATE;
      float eta_decay = (float) 0.001f;
      unsigned status_every_i_iterations = m_config->BATCH_SIZE;
      std::cerr << "Training started." << "\n";
      std::vector<unsigned> order(train_corpus.NumberOfDocuments());
      for (unsigned i = 0; i < train_corpus.NumberOfDocuments(); ++i)
         order[i] = i;
      int tot_seen = 0;
      status_every_i_iterations = (std::min)(status_every_i_iterations, (unsigned)train_corpus.NumberOfDocuments());
      unsigned si = (unsigned)train_corpus.NumberOfDocuments();
      std::cerr << "NUMBER OF TRAINING DOCUMENTS: " << status_every_i_iterations << std::endl;
      std::cerr << "BATCH: " << si << std::endl;
      unsigned trs = 0;
      double right = 0;
      double llh = 0;
      bool first = true;
      int iter = -1;
      double best_f1 = 0;
      double best_f1_p = 0; double best_f1_r = 0; double best_f1_epoch = 0;
      std::random_device rd;
      int iepoch = -1;
      int MAX_PATHS = 1;
      float best_rouge = 0;
      double best_rouge_epoch = 0;

      int num_RL_tries = m_config->ENABLE_REINFORCEMENT_LEARNING ? 5 : 1;

      if (m_config->ENABLE_REINFORCEMENT_LEARNING) {
         std::cout << "USING RL" << std::endl;
      }
      else {
         std::cout << "NOT USING RL" << std::endl;
      }

      while (!requested_stop) {
         ++iter;
         bool eval = false;
         double f1 = 0;
         std::vector<dynet::Expression> losses;
         {
            dynet::ComputationGraph hg;
            new_graph(&hg);
            int one_ok = 0;
            int one_pred = 0;
            int one_total = 0;
            std::vector < std::pair<int, EmbededDocument>> docs;
            for (unsigned sii = 0; sii < status_every_i_iterations; ++sii, si++) {
               if (si == train_corpus.NumberOfDocuments()) {
                  si = 0;
                  eval = true;
                  iepoch++;
                  if (first) { first = false; }
                  else { sgd.learning_rate *= 1 - eta_decay; }
                  std::cerr << "**SHUFFLE\n";
                  std::shuffle(order.begin(), order.end(), rd);
                  if (iepoch % 1 == 0)
                     MAX_PATHS++;
               }

               if (si % 1000 == 0 && m_config->ENABLE_REINFORCEMENT_LEARNING) {
                  if (iepoch >= 1) {
                     num_RL_tries = 1;

                     m_config->Logger("csum") << "Seen all corpus once | " << "num_RL_tries: " << num_RL_tries << std::endl;
                     m_config->Logger("gsum") << "Seen all corpus once | " << "num_RL_tries: " << num_RL_tries << std::endl;
                     std::cout << "Seen all corpus once | num_RL_tries: " << num_RL_tries << std::endl;
                  }
                  else {
                     // Iteratively reduce RL impact
                     auto corpus_perc = (float)si / (float)train_corpus.NumberOfDocuments();
                     num_RL_tries = (int)std::ceil((1.0 - corpus_perc) * 5.0);
                     if (num_RL_tries <= 0)
                        num_RL_tries = 1;

                     m_config->Logger("csum") << "Seen " << corpus_perc << " % documents | " << "num_RL_tries: " << num_RL_tries << std::endl;
                     m_config->Logger("gsum") << "Seen " << corpus_perc << " % documents | " << "num_RL_tries: " << num_RL_tries << std::endl;
                     std::cout << "Seen " << corpus_perc << " % documents | " << "num_RL_tries: " << num_RL_tries << std::endl;
                  }
               }

               tot_seen += 1;
               const Document* document = train_corpus.GetDocument(order[si]);


               int doc_t = 0;
               for (int ia = 0; ia < document->sentences_actions().size(); ia++)
               {
                  if (document->sentences_actions()[ia] == 1)
                     doc_t++;
               }
               if (doc_t == 0)
                  continue;
               docs.push_back(std::pair<int, EmbededDocument>(order[si], EmbededDocument()));
               docs.back().second.m_result_summary_sentence_words.resize(document->sentences().size());
               encode_document(&hg, *document, m_config->DROPOUT, docs.back().second);

            }

            for (auto& d : docs) {

               const Document* document = train_corpus.GetDocument(d.first);


               for (int iac = 0; (iac < document->n_sentences_actions() || m_config->NOISE > 0) && iac < num_RL_tries; iac++)
               {
                  std::pair<std::vector<int>, dynet::Expression> pred_loss;
                  pred_loss = log_prob_parser(&hg, *document, &train_corpus, false, &right, d.second, NULL, NULL);

                  if (pred_loss.first.size() == 0)
                     continue;

                  std::cerr << ".";
                  losses.push_back(pred_loss.second);


                  for (int ia = 0; ia < document->sentences_actions().size(); ia++)
                  {
                     if (document->sentences_actions()[ia] == 1)
                        one_total++;
                     if (pred_loss.first[ia] == 1)
                     {
                        one_pred++;
                        if (document->sentences_actions()[ia] == 1)
                           one_ok++;
                     }
                  }

                  trs += (unsigned)document->sentences_actions().size();
               }
            }
            double p = one_ok / (double)one_pred;
            double r = one_ok / (double)one_total;
            f1 = 2 * (p*r) / (p + r);
            if (one_total == 0 || one_pred == 0 || one_ok == 0)
               f1 = 0;

            if (one_total == 0)
               continue;

            f1 += 0.0000001;
            auto batch_loss = dynet::sum(losses);
            llh = as_scalar(hg.incremental_forward(batch_loss)) / status_every_i_iterations;
            std::cerr << std::endl;
            sgd.status();
            std::cerr << "update #" << iter << " (epoch " << (tot_seen / (double)train_corpus.NumberOfDocuments()) << ")\tllh: " << llh << " ppl: " << exp(llh / trs) << " err: " << (trs - right) / trs << " f1: " << f1 << std::endl;
            llh = trs = 0;
            right = 0;
            hg.backward(batch_loss);
            sgd.update();
         }
         double epoch = (tot_seen / (double)train_corpus.NumberOfDocuments());


         static int logc = 0;
         static int last = 0;
         ++logc;
         if ((logc % 1000 == 1 || eval) && logc > last + 2) { // report on dev set
            unsigned dev_size = (unsigned)dev_corpus.NumberOfDocuments() > 300 ? 300 : (unsigned)dev_corpus.NumberOfDocuments();
            // dev_size = 100;

            double llh = 0;
            double trs = 0;
            double right = 0;
            int one_ok = 0;
            int one_pred = 0;
            int one_total = 0;
            last = logc;


            float total_rouge = 0;
            auto t_start = std::chrono::high_resolution_clock::now();
            for (unsigned sii = 0; sii < dev_size && sii < 300; ++sii) {
               const Document& document = *dev_corpus.GetDocument(sii);

               EmbededDocument embed_doc;
               embed_doc.m_result_summary_sentence_words.resize(document.sentences().size());

               dynet::ComputationGraph hg;
               new_graph(&hg);
               encode_document(&hg, document, false, embed_doc);
               std::pair<std::vector<int>, dynet::Expression> pred_loss;
               pred_loss = log_prob_parser(&hg, document, &dev_corpus, true, &right, embed_doc, NULL, NULL);

               std::string  abstractive_summary = "";


               float rouge = 0.0;

               std::vector<int> & pred = pred_loss.first;
               double lp = as_scalar(hg.incremental_forward(pred_loss.second));

               llh += lp;
               trs += (unsigned)document.sentences_actions().size();

               for (int ia = 0; ia < document.sentences_actions().size(); ia++)
               {
                  if (document.sentences_actions()[ia] == 1)
                     one_total++;
                  if (pred_loss.first[ia] == 1)
                  {
                     one_pred++;
                     if (document.sentences_actions()[ia] == 1)
                        one_ok++;
                  }
               }
               rouge = (float)document.ComputeFastRouge(pred_loss.first, embed_doc.m_result_summary_sentence_words, true);


               total_rouge += rouge;
               document.OutputSummary(pred_loss.first, embed_doc.m_result_summary_sentence_words, abstractive_summary, output_path + "/dev/");

               std::cerr << "e";
            }
            std::cerr << std::endl;

            auto t_end = std::chrono::high_resolution_clock::now();
            double nepochs = (tot_seen / (double)train_corpus.NumberOfDocuments());
            std::cerr << "  **dev (iter=" << iter << " epoch=" << nepochs << ")\trouge=" << total_rouge / dev_size << "\tbest rouge : " << best_rouge / dev_size << " err: " << (trs - right) / trs << " f1: " << best_f1 << "\t[" << dev_size << " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]" << std::endl;
            double p = one_ok / (double)one_pred;
            double r = one_ok / (double)one_total;
            double f1 = 2 * (p*r) / (p + r);
            std::cerr << "  **P = " << one_ok / (double)one_pred << " r = " << one_ok / (double)one_total << " f1= " << f1 << " n = " << one_pred / (double)dev_size << " MAX_RELATION = " << m_config->MAX_RELATION << std::endl;
            std::cerr << "  ** best P = " << best_f1_p << " best r = " << best_f1_r << " best f1 = " << best_f1 << " best epoch = " << best_f1_epoch << std::endl;
            //if (eval)

            std::string model_if_saved = "";
            if (total_rouge > best_rouge)
            {
               best_rouge = total_rouge;
               best_rouge_epoch = nepochs;

               std::string model_fname = output_path + "/models/" + fname_parameters + "_e_" + std::to_string(nepochs);
               m_config->Logger("gsum") << "saving to model_fname: " << model_fname << std::endl;
               m_config->Logger("csum") << "saving to model_fname: " << model_fname << std::endl;
               m_config->Logger("asum") << "saving to model_fname: " << model_fname << std::endl;
               model_if_saved = model_fname;
               dynet::BinaryFileSaver saver(model_fname);
               saver.save(m_config->m_model);
            }

            // evaluate extractive
            m_config->Logger("gsum") << " **P = " << one_ok / (double)one_pred << " r = " << one_ok / (double)one_total << " f1= " << f1 << " n = " << one_pred / (double)dev_size << " MAX_RELATION = " << m_config->MAX_RELATION << " at epoch =" << nepochs << std::endl;
            m_config->Logger("gsum") << " ** best P = " << best_f1_p << " best r = " << best_f1_r << " best f1 = " << best_f1 << " best epoch = " << best_f1_epoch << "\trouge = " << total_rouge / dev_size << "\tbest rouge : " << best_rouge / dev_size << std::endl;
            std::string command = evaluate_command + " --path " + output_path + "/dev/  --type gsum " +
               "--model " + fname_parameters + "_e_" + std::to_string(nepochs) + " --out " + m_config->LoggerFilePath("gsum") + " &";
            m_config->Logger("gsum") << command << std::endl;
            system(command.c_str());
            m_config->Logger("gsum") << std::endl;


            // evaluate compressive
            m_config->Logger("csum") << " **P = " << one_ok / (double)one_pred << " r = " << one_ok / (double)one_total << " f1= " << f1 << " n = " << one_pred / (double)dev_size << " MAX_RELATION = " << m_config->MAX_RELATION << std::endl;
            m_config->Logger("csum") << " ** best P = " << best_f1_p << " best r = " << best_f1_r << " best f1 = " << best_f1 << " best epoch = " << best_f1_epoch << std::endl;
            command = evaluate_command + " --path " + output_path + "/dev/  --type csum " +
               "--model " + fname_parameters + "_e_" + std::to_string(nepochs) + " --out " + m_config->LoggerFilePath("csum") + " &";
            m_config->Logger("csum") << command << std::endl;
            system(command.c_str());
            m_config->Logger("csum") << std::endl;





            eval = false;
            if (best_f1 < f1 && nepochs > 0) {
               best_f1 = f1;
               best_f1_p = p;
               best_f1_r = r;
               best_f1_epoch = nepochs;

               std::string model_fname = output_path + "/models/" + fname_parameters;
               m_config->Logger("gsum") << "saving new BEST-f1 to model_fname: " << model_fname << std::endl;
               m_config->Logger("csum") << "saving new BEST-f1 to model_fname: " << model_fname << std::endl;

               dynet::BinaryFileSaver saver(model_fname);
               saver.save(m_config->m_model);

               // Create a soft link to the most recent model in order to make it
               // easier to refer to it in a shell script.


            }

         }
      }
      return true;
   }

   bool NoNameYetModel::Test(Corpus& test_corpus,
      const std::string& output_path,
      const std::string& evaluate_command
   )
   {
      double llh = 0;
      double trs = 0;
      double right = 0;
      auto t_start = std::chrono::high_resolution_clock::now();
      unsigned corpus_size = (unsigned)test_corpus.NumberOfDocuments();

      for (unsigned sii = 0; sii < corpus_size; ++sii) {
         const Document* document = test_corpus.GetDocument(sii);
         dynet::ComputationGraph hg;
         new_graph(&hg);

         double lp = 0;
         EmbededDocument embed_doc;
         embed_doc.m_result_summary_sentence_words.resize(document->sentences().size());
         encode_document(&hg, *document, m_config->USE_DROPOUT, embed_doc);
         auto pred_loss = log_prob_parser(&hg,
            *document,
            &test_corpus,
            true,
            &right,
            embed_doc,
            0);

         std::vector<int> & pred = pred_loss.first;
         llh -= lp;
         trs += (unsigned)document->sentences_actions().size();

         // Evauation here

         // output_conll_nested(sentence, sentencePos, sentence_words, refs, hyps);



         document->OutputSummary(pred_loss.first, embed_doc.m_result_summary_sentence_words, "", output_path + "/test/");


      }


      return true;
   }



}
