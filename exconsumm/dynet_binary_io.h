#ifndef DYNET_BINARY_IO_H_
#define DYNET_BINARY_IO_H_

#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <iterator>

#include "dynet/dim.h"
#include "dynet/model.h"
#include "dynet/tensor.h"
#include "dynet/except.h"
#include "dynet/str-util.h"
#include "dynet/io.h"
#include "binary_stream.h"

namespace dynet {
    class BinaryFileSaver : public Saver {
    public:
        BinaryFileSaver(const std::string & filename, bool append = false);
        ~BinaryFileSaver() override;
        void save(const ParameterCollection & model,
            const std::string & key = "") override;
        void save(const Parameter & param, const std::string & key = "") override;
        void save(const LookupParameter & param, const std::string & key = "") override;

    protected:
        void save(const ParameterStorage & param, const std::string & key = "");
        void save(const LookupParameterStorage & param, const std::string & key = "");

        //std::unique_ptr<std::ostream> p_datastream;
        pba_local::binary_stream datastream;

    }; // class TextFileSaver

    class BinaryFileLoader : public Loader {
    public:
        BinaryFileLoader(const std::string & filename);
        ~BinaryFileLoader() override;
        void populate(ParameterCollection & model, const std::string & key = "") override;
        void populate(Parameter & param, const std::string & key = "") override;
        void populate(LookupParameter & lookup_param,
            const std::string & key = "") override;
        Parameter load_param(ParameterCollection & model, const std::string & key) override;
        LookupParameter load_lookup_param(ParameterCollection & model, const std::string & key) override;

    private:
        std::string dataname;
    }; // class TextFileLoader

} // namespace dynet

#endif
