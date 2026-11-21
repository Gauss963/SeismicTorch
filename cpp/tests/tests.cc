#include "seismictorch/inference.hh"
#include "libmseed.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
const auto fixtures=std::filesystem::path(FIXTURE_DIR);
void check(bool ok,const std::string& what) {if(!ok) throw std::runtime_error(what);}
void near(double actual,double expected,double tolerance,const std::string& what) {
  check(std::isfinite(actual)&&std::abs(actual-expected)<=tolerance,what+": "+std::to_string(actual)+" != "+std::to_string(expected));
}
struct Binary {
  std::ifstream input;
  explicit Binary(const std::filesystem::path& path):input(path,std::ios::binary) {check(bool(input),"Open fixture");}
  std::uint64_t integer(unsigned count) {
    std::uint64_t bits=0;
    for(unsigned i=0;i<count;++i) {auto c=input.get();check(c!=EOF,"Truncated fixture");bits|=std::uint64_t(static_cast<unsigned char>(c))<<(8*i);}
    return bits;
  }
  float f32() {auto bits=static_cast<std::uint32_t>(integer(4));float v;std::memcpy(&v,&bits,4);return v;}
  double f64() {auto bits=integer(8);double v;std::memcpy(&v,&bits,8);return v;}
  void magic(const char* expected) {char text[4];input.read(text,4);check(input&&std::memcmp(text,expected,4)==0,"Fixture signature");}
};
template<class F>void rejects(F f,const std::string& what) {bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}check(rejected,what);}

void model_cases() {
  Binary file(fixtures/"model-cases.bin");file.magic("NNC1");auto count=file.integer(4);check(count==141,"Reference count");double maximum=0;
  for(std::size_t i=0;i<count;++i) {
    seismictorch::ModelWindow x{};for(auto& v:x)v=file.f32();
    std::array<float,3> expected{};for(auto& v:expected)v=file.f32();
    auto actual=seismictorch::classify_window(x);
    for(std::size_t a=0;a<3;++a) {maximum=std::max(maximum,std::abs(double(actual[a])-expected[a]));near(actual[a],expected[a],2e-4,"CNN probability");}
    check(std::max_element(actual.begin(),actual.end())-actual.begin()==std::max_element(expected.begin(),expected.end())-expected.begin(),"CNN class parity");
  }
  std::cout<<"141 real model windows match Python, max probability error "<<maximum<<'\n';
}

seismictorch::Waveform pipeline(const std::string& name) {
  Binary file(fixtures/name);file.magic("PRE1");auto n=file.integer(4);seismictorch::Waveform w;
  w.sample_rate=file.f64();w.start_time_ns=static_cast<std::int64_t>(file.integer(8));auto count=file.integer(4);
  for(auto& a:w.xyz)a.resize(n);
  for(std::size_t i=0;i<n;++i)for(auto& a:w.xyz)a[i]=file.f64();
  for(std::size_t k=0;k<count;++k) {
    auto center=file.integer(4);seismictorch::ModelWindow expected{};for(auto& v:expected)v=file.f32();
    std::array<float,3> probabilities{};for(auto& v:probabilities)v=file.f32();
    auto actual=seismictorch::prepare_window(w,static_cast<std::size_t>(center));check(actual.has_value(),"Prepared window exists");
    for(std::size_t i=0;i<240;++i)near((*actual)[i],expected[i],2e-5,"Filter/rotation/Fourier parity");
    auto p=seismictorch::classify_window(*actual);
    for(std::size_t a=0;a<3;++a)near(p[a],probabilities[a],2e-4,"End-to-end probability");
  }
  return w;
}
void same_wave(const seismictorch::Waveform& a,const seismictorch::Waveform& b) {
  check(a.size()==b.size(),"Reader lengths");near(a.sample_rate,b.sample_rate,1e-8,"Reader rate");check(a.start_time_ns==b.start_time_ns,"Reader UTC start");
  for(std::size_t axis=0;axis<3;++axis)for(std::size_t i=0;i<a.size();++i)near(a.xyz[axis][i],b.xyz[axis][i],0,"Decoded sample");
}
void write_records(char* data,int length,void* target) {static_cast<std::ofstream*>(target)->write(data,length);}
void write_mseed3(const std::filesystem::path& path,const seismictorch::Waveform& w,const char* station="TEST",bool append=false) {
  std::ofstream output(path,std::ios::binary|(append ? std::ios::app : std::ios::trunc));
  for(std::size_t a=0;a<3;++a) {
    std::vector<std::int32_t> samples;for(auto value:w.xyz[a])samples.push_back(static_cast<std::int32_t>(value));
    auto* record=msr3_init(nullptr);check(record!=nullptr,"Allocate miniSEED record");
    std::string channel="HL";channel+="XYZ"[a];
    check(ms_nslc2sid(record->sid,LM_SIDLEN,0,"XX",station,"",channel.c_str())>=0,"Format source identifier");
    record->formatversion=3;record->reclen=512;record->starttime=w.start_time_ns;record->samprate=w.sample_rate;
    record->encoding=DE_STEIM2;record->samplecnt=static_cast<std::int64_t>(samples.size());record->numsamples=record->samplecnt;
    record->datasamples=samples.data();record->datasize=samples.size()*4;record->sampletype='i';record->pubversion=1;
    std::int64_t packed=0;auto records=msr3_pack(record,write_records,&output,&packed,MSF_FLUSHDATA,0);
    record->datasamples=nullptr;msr3_free(&record);
    check(records>0&&packed==static_cast<std::int64_t>(samples.size())&&output,"Pack miniSEED 3");
  }
}
void dense_cases(const seismictorch::Waveform& w) {
  auto result=seismictorch::infer(w,{1,2});
  check(result.p.size()==w.size()&&result.s.size()==w.size()&&result.noise.size()==w.size(),"Equal-length vectors");
  for(std::size_t i=0;i<100;++i)check(std::isnan(result.p[i]),"Leading edge NaN");
  for(std::size_t i=w.size()-99;i<w.size();++i)check(std::isnan(result.noise[i]),"Trailing edge NaN");
  for(std::size_t c=100;c<=w.size()-100;++c)near(result.p[c]+result.s[c]+result.noise[c],1,2e-6,"Probability sum");
  auto sparse=seismictorch::infer(w,{13,2});
  for(std::size_t c=100;c<=w.size()-100;++c)near(sparse.p[c]+sparse.s[c]+sparse.noise[c],1,2e-6,"Interpolated probability sum");
  auto dropout=w;std::fill(dropout.xyz[1].begin()+900,dropout.xyz[1].begin()+1000,0);
  check(!seismictorch::prepare_window(dropout,950),"QC before physical rotation");
  auto masked=seismictorch::infer(dropout,{1000,1});check(std::isnan(masked.p[950]),"Never interpolate across raw dropout");
  auto gap=w;for(auto& a:gap.xyz)std::fill(a.begin()+500,a.begin()+520,std::numeric_limits<double>::quiet_NaN());
  auto gapped=seismictorch::infer(gap,{10,2});check(std::isnan(gapped.p[510])&&std::isnan(gapped.p[550])&&std::isfinite(gapped.p[700]),"Gap segmentation");
  rejects([&]{seismictorch::infer(w,{0,1});},"Reject zero stride");
  auto malformed=w;malformed.xyz[1].pop_back();rejects([&]{seismictorch::infer(malformed);},"Reject shape mismatch");
}
} // namespace

int main() {
  auto temporary=std::filesystem::temp_directory_path()/std::filesystem::u8path("seismictorch-test-"+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  try {
    std::filesystem::create_directories(temporary);
    model_cases();auto expected=pipeline("pipeline.bin");
    for(const auto* name:{"pipeline-40.bin","pipeline-50.bin","pipeline-200.bin","pipeline-1000.bin"}) pipeline(name);
    same_wave(seismictorch::read_sac({fixtures/"x.sac",fixtures/"y.sac",fixtures/"z.sac"}),expected);
    same_wave(seismictorch::read_sac({fixtures/"x-big.sac",fixtures/"y-big.sac",fixtures/"z-big.sac"}),expected);
    auto unicode=temporary/std::filesystem::u8path(u8"測試.mseed");std::filesystem::copy_file(fixtures/"steim2.mseed",unicode);
    same_wave(seismictorch::read_mseed(unicode),expected);
    same_wave(seismictorch::read_mseed(fixtures/"float32.mseed","XX.TEST..HN"),expected);
    rejects([&]{seismictorch::read_mseed(fixtures/"float32.mseed","UNKNOWN");},"Reject missing group");
    auto mseed3=temporary/"version3.mseed";write_mseed3(mseed3,expected);same_wave(seismictorch::read_mseed(mseed3),expected);
    auto v7=temporary/"x-v7.sac";
    {std::ifstream source(fixtures/"x.sac",std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(source)),{});
     bytes[304]=7;bytes[305]=bytes[306]=bytes[307]=0;
     std::ofstream output(v7,std::ios::binary);output.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));
     for(int i=0;i<22;++i) {double value=i==0 ?.01 :(i==1 ?0 :-12345);std::uint64_t bits;std::memcpy(&bits,&value,8);for(int j=0;j<8;++j)output.put(static_cast<char>((bits>>(8*j))&255));}}
    same_wave(seismictorch::read_sac({v7,fixtures/"y.sac",fixtures/"z.sac"}),expected);
    auto gap=expected;
    for(auto& axis:gap.xyz)axis.erase(axis.begin()+400,axis.end());
    auto gappy=temporary/"gaps.mseed";write_mseed3(gappy,gap);
    gap=expected;gap.start_time_ns+=4200000000LL;
    for(auto& axis:gap.xyz)axis.erase(axis.begin(),axis.begin()+420);
    write_mseed3(gappy,gap,"TEST",true);
    auto decoded_gap=seismictorch::read_mseed(gappy);check(decoded_gap.size()==expected.size(),"miniSEED gap grid length");
    for(std::size_t a=0;a<3;++a)for(std::size_t i=0;i<expected.size();++i) {
      if(i>=400&&i<420)check(std::isnan(decoded_gap.xyz[a][i]),"miniSEED gap marker");
      else near(decoded_gap.xyz[a][i],expected.xyz[a][i],0,"miniSEED segment samples");
    }
    auto overlap=temporary/"overlap.mseed";write_mseed3(overlap,expected);
    auto conflicting=expected;conflicting.xyz[0][900]+=1;write_mseed3(overlap,conflicting,"TEST",true);
    rejects([&]{seismictorch::read_mseed(overlap);},"Conflicting overlapping records");
    write_mseed3(mseed3,expected,"OTHER",true);
    rejects([&]{seismictorch::read_mseed(mseed3);},"Ambiguous station groups");
    same_wave(seismictorch::read_mseed(mseed3,"XX.OTHER..HL"),expected);

    {std::fstream data(mseed3,std::ios::in|std::ios::out|std::ios::binary);data.seekg(40);char c;data.read(&c,1);c^=1;data.seekp(40);data.write(&c,1);}
    rejects([&]{seismictorch::read_mseed(mseed3);},"Validate miniSEED 3 CRC");
    rejects([&]{seismictorch::read_sac({fixtures/"steim2.mseed",fixtures/"y.sac",fixtures/"z.sac"});},"Reject malformed SAC");
    dense_cases(expected);
    std::filesystem::remove_all(temporary);std::cout<<"Reader, preprocessing, dense-vector, gap and QC tests passed\n";return 0;
  } catch(const std::exception& e) {std::filesystem::remove_all(temporary);std::cerr<<e.what()<<'\n';return 1;}
}
