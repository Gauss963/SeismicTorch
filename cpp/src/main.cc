#include "seismictorch/inference.hh"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <limits>
#include <stdexcept>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {
void help() {
  std::cout << "SeismicTorch " << seismictorch::version() << " - standalone CPU phase probabilities\n"
    "Usage: seismictorch-infer --sac X.sac Y.sac Z.sac [options]\n"
    "       seismictorch-infer --mseed input.mseed [options]\n"
    "Positional alternatives: three SAC paths, or one miniSEED path.\n"
    "Options:\n"
    "  --output PATH   CSV destination, default stdout\n"
    "  --stream ID     Select NET.STATION.LOCATION.CHANNEL_PREFIX in miniSEED\n"
    "  --stride N      Evaluate every N samples and interpolate, default 1\n"
    "  --threads N     CPU workers, default available cores (maximum 256)\n"
    "  --version       Print compiled model identity\n"
    "Outputs have exactly one row per input sample. Unsupported edges and\n"
    "gaps are nan. P, S and Noise are probabilities, not onset labels.\n";
}
std::size_t count(const std::string& text) {
  if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos) throw std::invalid_argument("Expected a nonnegative integer");
  std::size_t used=0;auto value=std::stoull(text,&used);
  if(used!=text.size() || value>std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("Integer option is out of range");
  return static_cast<std::size_t>(value);
}
int run(const std::vector<std::string>& args) {
  try {
    if(args.empty()) {help();return 0;}
    std::vector<std::filesystem::path> files,positional;
    bool mseed=false;std::string stream,output="-";seismictorch::InferOptions options;
    for(std::size_t i=0;i<args.size();++i) {
      const auto& arg=args[i];
      auto value=[&]()->const std::string& {if(i+1>=args.size()) throw std::invalid_argument("Missing value after "+arg);return args[++i];};
      if(arg=="--help"||arg=="-h") {help();return 0;}
      else if(arg=="--version") {std::cout<<seismictorch::version()<<" model="<<seismictorch::model_sha256()<<'\n';return 0;}
      else if(arg=="--sac") {
        if(!files.empty()) throw std::invalid_argument("Specify one input mode");
        for(int a=0;a<3;++a) files.push_back(std::filesystem::u8path(value()));
      }
      else if(arg=="--mseed") {
        if(!files.empty()) throw std::invalid_argument("Specify one input mode");
        files.push_back(std::filesystem::u8path(value()));mseed=true;
      }
      else if(arg=="--stream") stream=value();
      else if(arg=="--output") output=value();
      else if(arg=="--stride") options.stride=count(value());
      else if(arg=="--threads") options.threads=count(value());
      else if(!arg.empty()&&arg[0]=='-') throw std::invalid_argument("Unknown option "+arg);
      else positional.push_back(std::filesystem::u8path(arg));
    }
    if(!positional.empty()) {
      if(!files.empty()) throw std::invalid_argument("Cannot combine positional and flagged inputs");
      files=positional;mseed=files.size()==1;
    }
    if((mseed&&files.size()!=1)||(!mseed&&files.size()!=3)) throw std::invalid_argument("Provide three SAC files or one miniSEED file");
    if(options.stride==0 || options.threads>256) throw std::invalid_argument("Stride must be positive; threads must be 0 to 256");
    if(!mseed&&!stream.empty()) throw std::invalid_argument("--stream applies only to miniSEED");
    auto output_path=std::filesystem::u8path(output);
    if(output!="-") for(const auto& path:files)
      if(std::filesystem::weakly_canonical(path)==std::filesystem::weakly_canonical(output_path)) throw std::invalid_argument("Output must not overwrite an input waveform");
    auto waveform=mseed ? seismictorch::read_mseed(files[0],stream) : seismictorch::read_sac({files[0],files[1],files[2]});
    auto scores=seismictorch::infer(waveform,options);
    std::ofstream file;
    std::ostream* destination=&std::cout;
    if(output!="-") {file.open(output_path,std::ios::binary);if(!file) throw std::runtime_error("Cannot open output file");destination=&file;}
    auto& out=*destination;out.imbue(std::locale::classic());out<<std::setprecision(9);
    out<<"time_seconds,P,S,Noise\n";
    for(std::size_t i=0;i<waveform.size();++i) {
      out<<static_cast<double>(i)/waveform.sample_rate;
      if(!std::isfinite(scores.p[i])) out<<",nan,nan,nan\n";
      else out<<','<<scores.p[i]<<','<<scores.s[i]<<','<<scores.noise[i]<<'\n';
    }
    out.flush();if(!out) throw std::runtime_error("Failed to write probability vectors");
    return 0;
  } catch(const std::exception& e) {std::cerr<<"seismictorch-infer: "<<e.what()<<'\n';return 1;}
}
} // namespace
#ifdef _WIN32
int wmain(int argc,wchar_t** argv) {
  std::vector<std::string> args;
  for(int i=1;i<argc;++i) {
    int length=WideCharToMultiByte(CP_UTF8,0,argv[i],-1,nullptr,0,nullptr,nullptr);
    if(length<=0) {std::cerr<<"Invalid Windows command line\n";return 1;}
    std::string value(static_cast<std::size_t>(length),'\0');
    WideCharToMultiByte(CP_UTF8,0,argv[i],-1,value.data(),length,nullptr,nullptr);
    value.pop_back();args.push_back(std::move(value));
  }
  return run(args);
}
#else
int main(int argc,char** argv) {return run(std::vector<std::string>(argv+1,argv+argc));}
#endif
