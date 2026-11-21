#include "seismictorch/inference.hh"
#include "libmseed.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

namespace seismictorch {
namespace {
constexpr std::size_t max_samples = 50000000;

std::vector<char> read_bytes(const std::filesystem::path& path) {
  std::ifstream input(path,std::ios::binary|std::ios::ate);
  if (!input) throw std::runtime_error("Cannot open input: "+path.u8string());
  auto length=input.tellg();
  if (length<=0 || static_cast<std::uint64_t>(length)>2000000000ULL)
    throw std::runtime_error("Input is empty or exceeds the 2 GB reader limit");
  std::vector<char> bytes(static_cast<std::size_t>(length));
  input.seekg(0);input.read(bytes.data(),length);
  if (!input) throw std::runtime_error("Truncated input: "+path.u8string());
  return bytes;
}

std::uint32_t u32(const char* p,bool big) {
  const auto* b=reinterpret_cast<const unsigned char*>(p);
  if(big) return (std::uint32_t(b[0])<<24)|(std::uint32_t(b[1])<<16)|(std::uint32_t(b[2])<<8)|b[3];
  return (std::uint32_t(b[3])<<24)|(std::uint32_t(b[2])<<16)|(std::uint32_t(b[1])<<8)|b[0];
}
float f32(const char* p,bool big) {auto bits=u32(p,big);float value;std::memcpy(&value,&bits,4);return value;}
double f64(const char* p,bool big) {
  std::uint64_t bits=0;
  for(int i=0;i<8;++i) bits=(bits<<8)|static_cast<unsigned char>(p[big ? i : 7-i]);
  double value;std::memcpy(&value,&bits,8);return value;
}

std::int64_t year_start_days(int year) {
  // Gregorian civil date to days since Unix epoch, independent of local TZ.
  int y=year-1;
  auto before=[](std::int64_t v){return 365*v+v/4-v/100+v/400;};
  return before(y)-before(1969);
}

double canonical_rate(double rate) {
  if (!std::isfinite(rate) || rate<40 || rate>1000.01)
    throw std::runtime_error("Only sample rates from 40 to 1000 Hz are supported");
  double integer=std::round(rate);
  return std::abs(rate-integer)<1e-4 ? integer : rate;
}

struct Sac {
  double rate;
  std::int64_t start;
  std::vector<double> samples;
};

Sac sac(const std::filesystem::path& file) {
  auto data=read_bytes(file);
  if (data.size()<632) throw std::runtime_error("SAC header is truncated");
  bool big=false;
  auto integer=[&](std::size_t index){return static_cast<std::int32_t>(u32(data.data()+280+4*index,big));};
  if (integer(6)!=6 && integer(6)!=7) {
    big=true;
    if (integer(6)!=6 && integer(6)!=7) throw std::runtime_error("Not a supported binary SAC v6/v7 file");
  }
  int count=integer(9);
  if (count<=0 || static_cast<std::size_t>(count)>max_samples || integer(15)!=1 || integer(35)!=1)
    throw std::runtime_error("SAC must contain an evenly sampled time series of at most 50 million samples");
  auto payload=632+static_cast<std::size_t>(count)*4;
  if (data.size()<payload) throw std::runtime_error("SAC sample data is truncated");
  double delta=f32(data.data(),big),begin=f32(data.data()+5*4,big);
  if (integer(6)==7) {
    if (data.size()<payload+176) throw std::runtime_error("SAC v7 double footer is truncated");
    delta=f64(data.data()+payload,big);begin=f64(data.data()+payload+8,big);
  }
  if (!std::isfinite(delta) || delta<=0 || !std::isfinite(begin) || begin==-12345)
    throw std::runtime_error("Invalid SAC sample interval or begin time");
  int year=integer(0),day=integer(1),hour=integer(2),minute=integer(3),second=integer(4),millisecond=integer(5);
  bool leap=year%4==0 && (year%100!=0 || year%400==0);
  if (year<1900 || year>2200 || day<1 || day>(leap?366:365) || hour<0 || hour>23 || minute<0 || minute>59 || second<0 || second>60 || millisecond<0 || millisecond>999)
    throw std::runtime_error("Invalid SAC UTC reference time");
  std::int64_t seconds=(year_start_days(year)+day-1)*86400+hour*3600+minute*60+second;
  auto base=seconds*1000000000LL+millisecond*1000000LL;
  long double relative=static_cast<long double>(begin)*1e9L;
  if(relative<=static_cast<long double>(std::numeric_limits<std::int64_t>::min()) || relative>=static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
    throw std::runtime_error("SAC begin time is outside the supported range");
  auto offset=static_cast<std::int64_t>(std::llround(relative));
  if((offset>0 && base>std::numeric_limits<std::int64_t>::max()-offset) ||
     (offset<0 && base<std::numeric_limits<std::int64_t>::min()-offset))
    throw std::runtime_error("SAC absolute time is outside the supported range");
  auto start=base+offset;
  Sac result{canonical_rate(1/delta),start,std::vector<double>(static_cast<std::size_t>(count))};
  for (std::size_t i=0;i<result.samples.size();++i) result.samples[i]=f32(data.data()+632+4*i,big);
  return result;
}

double sample(const MS3TraceSeg& segment,std::size_t index) {
  switch(segment.sampletype) {
    case 'i': return static_cast<const std::int32_t*>(segment.datasamples)[index];
    case 'f': return static_cast<const float*>(segment.datasamples)[index];
    case 'd': return static_cast<const double*>(segment.datasamples)[index];
    default: throw std::runtime_error("miniSEED channel has unsupported nonnumeric sample data");
  }
}
int axis_index(char c) {
  if (c=='X'||c=='E'||c=='1') return 0;
  if (c=='Y'||c=='N'||c=='2') return 1;
  return c=='Z' ? 2 : -1;
}
struct Group {std::array<MS3TraceID*,3> axes{};};
double clock_tolerance(const MS3Record*) {return 1e-6;}
} // namespace

Waveform read_sac(const std::array<std::filesystem::path,3>& files) {
  Waveform result;
  for(std::size_t a=0;a<3;++a) {
    auto input=sac(files[a]);
    if(a==0) {result.sample_rate=input.rate;result.start_time_ns=input.start;}
    else if (std::abs(input.rate-result.sample_rate)>1e-6 ||
             std::abs(static_cast<double>(input.start-result.start_time_ns))*result.sample_rate/1e9>.001 ||
             input.samples.size()!=result.size())
      throw std::runtime_error("SAC components must have matching sample rates, UTC starts and lengths");
    result.xyz[a]=std::move(input.samples);
  }
  result.stream="SAC_XYZ";
  return result;
}

Waveform read_mseed(const std::filesystem::path& file,const std::string& selector) {
  auto bytes=read_bytes(file);
  MS3TraceList* raw=nullptr;
  MS3Tolerance tolerance{};tolerance.time=clock_tolerance;
  auto records=mstl3_readbuffer(&raw,bytes.data(),bytes.size(),0,MSF_UNPACKDATA|MSF_VALIDATECRC,&tolerance,0);
  auto cleanup=[](MS3TraceList* p){mstl3_free(&p,0);};
  std::unique_ptr<MS3TraceList,decltype(cleanup)> list(raw,cleanup);
  if (records<0 || !list || list->numtraceids==0) throw std::runtime_error("Invalid, corrupt or empty miniSEED file");
  std::map<std::string,Group> groups;
  for(auto* id=list->traces.next[0];id;id=id->next[0]) {
    char net[128]{},sta[128]{},loc[128]{},chan[128]{};
    if(ms_sid2nslc_n(id->sid,net,sizeof(net),sta,sizeof(sta),loc,sizeof(loc),chan,sizeof(chan))!=0)
      throw std::runtime_error("Unsupported miniSEED source identifier");
    std::string channel(chan);
    if(channel.empty()) continue;
    int a=axis_index(channel.back());
    if(a<0) continue;
    auto key=std::string(net)+"."+sta+"."+loc+"."+channel.substr(0,channel.size()-1);
    auto& group=groups[key];
    if(group.axes[static_cast<std::size_t>(a)]) throw std::runtime_error("Ambiguous axis in miniSEED group "+key);
    group.axes[static_cast<std::size_t>(a)]=id;
  }
  std::vector<std::string> complete;
  for(const auto& item:groups)
    if(item.second.axes[0] && item.second.axes[1] && item.second.axes[2] && (selector.empty()||selector==item.first)) complete.push_back(item.first);
  if(complete.size()!=1) {
    std::string message="Select exactly one complete XYZ, ENZ or 12Z miniSEED group with --stream. Complete groups:";
    for(const auto& item:groups) if(item.second.axes[0]&&item.second.axes[1]&&item.second.axes[2]) message+=" "+item.first;
    throw std::runtime_error(message);
  }
  const auto& group=groups[complete[0]];
  auto start=std::numeric_limits<std::int64_t>::max(),finish=std::numeric_limits<std::int64_t>::min();
  double rate=0;
  for(auto* id:group.axes) for(auto* seg=id->first;seg;seg=seg->next) {
    double current=canonical_rate(seg->samprate);
    if(rate==0) rate=current;
    if(std::abs(rate-current)>1e-6 || seg->numsamples<=0 || !seg->datasamples || seg->samplecnt!=seg->numsamples)
      throw std::runtime_error("miniSEED segments must have matching rates and fully decoded samples");
    start=std::min(start,seg->starttime);
    auto last=seg->starttime+static_cast<std::int64_t>(std::llround((seg->numsamples-1)*1e9/rate));
    finish=std::max(finish,last);
  }
  long double span=static_cast<long double>(finish)-start;
  if(rate==0 || span<0 || span*rate/1e9L>max_samples-1)
    throw std::runtime_error("miniSEED has no samples or an excessive time span");
  auto length=std::llround(span*rate/1e9L)+1;
  if(length<=0 || static_cast<std::uint64_t>(length)>max_samples) throw std::runtime_error("miniSEED time span exceeds the 50 million sample limit");
  Waveform result;result.sample_rate=rate;result.start_time_ns=start;result.stream=complete[0];
  for(std::size_t a=0;a<3;++a) {
    auto& output=result.xyz[a];output.assign(static_cast<std::size_t>(length),std::numeric_limits<double>::quiet_NaN());
    for(auto* seg=group.axes[a]->first;seg;seg=seg->next) {
      double position=static_cast<double>(seg->starttime-start)*rate/1e9;
      auto first=std::llround(position);
      if(std::abs(position-first)>.001 || first<0 || static_cast<std::uint64_t>(first)+seg->numsamples>output.size())
        throw std::runtime_error("miniSEED components are not aligned to a common sample grid");
      for(std::size_t i=0;i<static_cast<std::size_t>(seg->numsamples);++i) {
        double value=sample(*seg,i);auto index=static_cast<std::size_t>(first)+i;
        if(std::isfinite(output[index]) && output[index]!=value) throw std::runtime_error("Conflicting overlapping miniSEED samples");
        output[index]=value;
      }
    }
  }
  return result;
}
} // namespace seismictorch
