#include "seismictorch/inference.hh"
#include "model_weights.hh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

namespace seismictorch {
namespace {
constexpr double pi = 3.1415926535897932384626433832795;
using Matrix = std::array<double, 9>;
constexpr Matrix identity{1,0,0,0,1,0,0,0,1};

double median(std::vector<double> values) {
  auto middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  double result = values[middle];
  if (values.size() % 2 == 0)
    result = (result + *std::max_element(values.begin(), values.begin() + middle)) / 2;
  return result;
}

void validate(const Waveform& w) {
  if (!std::isfinite(w.sample_rate) || w.sample_rate < 40 || w.sample_rate > 1000)
    throw std::invalid_argument("Sample rate must be 40 to 1000 Hz");
  if (w.xyz[1].size() != w.size() || w.xyz[2].size() != w.size())
    throw std::invalid_argument("Three components must have equal lengths");
}

bool finite_at(const Waveform& w, std::size_t i) {
  return std::isfinite(w.xyz[0][i]) && std::isfinite(w.xyz[1][i]) && std::isfinite(w.xyz[2][i]);
}

struct Prepared {
  const Waveform& source;
  std::size_t begin, length, half;
  std::array<std::vector<double>, 3> filtered;
  std::vector<double> kernel;
  Matrix rotation = identity;

  Prepared(const Waveform& w, std::size_t start, std::size_t end)
      : source(w), begin(start), length(end-start), half(static_cast<std::size_t>(std::llround(w.sample_rate))) {
    std::array<double, 12> sos = embedded::highpass_sos;
    std::array<double, 4> zi = embedded::highpass_zi;
    if (std::abs(w.sample_rate-100) > 1e-8) {
      double k = std::tan(pi*.5/w.sample_rate);
      for (std::size_t s=0; s<2; ++s) {
        double q = 1/(2*std::cos((s==0 ? 1 : 3)*pi/8));
        double norm = 1/(1+k/q+k*k);
        sos[s*6]=norm; sos[s*6+1]=-2*norm; sos[s*6+2]=norm;
        sos[s*6+3]=1; sos[s*6+4]=2*(k*k-1)*norm; sos[s*6+5]=(1-k/q+k*k)*norm;
      }
      zi = {-sos[0], sos[0], 0, 0};
    }
    for (std::size_t axis=0; axis<3; ++axis) {
      auto& y=filtered[axis]; y.resize(length);
      long double sum=0;
      for (std::size_t i=0;i<length;++i) sum+=w.xyz[axis][begin+i];
      double mean=static_cast<double>(sum/length), mid=(length-1)*.5;
      long double covariance=0;
      for (std::size_t i=0;i<length;++i) covariance+=(static_cast<double>(i)-mid)*(w.xyz[axis][begin+i]-mean);
      double n=static_cast<double>(length);
      double slope=length>1 ? static_cast<double>(covariance/(n*(n*n-1)/12)) : 0;
      for (std::size_t i=0;i<length;++i) y[i]=w.xyz[axis][begin+i]-mean-slope*(static_cast<double>(i)-mid);
      std::array<double,4> state{};
      for (std::size_t i=0;i<4;++i) state[i]=zi[i]*y[0];
      for (auto& sample:y) {
        for (std::size_t s=0;s<2;++s) {
          double value=sos[s*6]*sample+state[s*2];
          state[s*2]=sos[s*6+1]*sample-sos[s*6+4]*value+state[s*2+1];
          state[s*2+1]=sos[s*6+2]*sample-sos[s*6+5]*value;
          sample=value;
        }
      }
    }
    // scipy.signal.resample's real, even-length Fourier truncation, including
    // the doubled output Nyquist bin when downsampling.
    const auto count=2*half;
    kernel.resize(80*count);
    for (std::size_t t=0;t<80;++t) for (std::size_t j=0;j<count;++j) {
      double value=1;
      auto minimum=std::min<std::size_t>(80,count);
      for (std::size_t k=1;k<(minimum+1)/2;++k)
        value+=2*std::cos(2*pi*k*(static_cast<double>(t)/80-static_cast<double>(j)/count));
      if (minimum%2==0) {
        auto k=minimum/2;
        if (count>80) value+=2*std::cos(2*pi*k*j/count)*std::cos(pi*t);
        else value+=std::cos(2*pi*k*(static_cast<double>(t)/80-static_cast<double>(j)/count));
      }
      kernel[t*count+j]=value/count;
    }
    if (length>=7*half) {
      std::array<std::array<double,6>,3> blocks{};
      std::array<double,3> offset{};
      for (std::size_t a=0;a<3;++a) {
        for (std::size_t b=0;b<6;++b) {
          auto first=w.xyz[a].begin()+static_cast<std::ptrdiff_t>(begin+(b+1)*half);
          blocks[a][b]=median(std::vector<double>(first,first+static_cast<std::ptrdiff_t>(half)));
        }
        offset[a]=median(std::vector<double>(blocks[a].begin(),blocks[a].end()));
      }
      double norm=std::hypot(offset[0],std::hypot(offset[1],offset[2])),drift=0;
      for (std::size_t b=0;b<6;++b)
        drift=std::max(drift,std::hypot(blocks[0][b]-offset[0],std::hypot(blocks[1][b]-offset[1],blocks[2][b]-offset[2])));
      if (norm>=900 && norm<=1100 && drift<=8) {
        double x=offset[0]/norm,y=offset[1]/norm,z=offset[2]/norm,s=std::hypot(x,y);
        if (s<1e-14) { if(z<0) rotation={-1,0,0,0,1,0,0,0,-1}; }
        else {
          double a=y/s,b=-x/s,angle=std::atan2(s,z),c=std::cos(angle),sn=std::sin(angle),d=1-c;
          rotation={c+a*a*d,a*b*d,b*sn, a*b*d,c+b*b*d,-a*sn, -b*sn,a*sn,c};
        }
      }
    }
  }

  bool usable(std::size_t c) const {
    if (c<half || c>length-half) return false;
    for (const auto& axis:source.xyz) {
      std::size_t run=1;
      for (std::size_t i=begin+c-half+1;i<begin+c+half;++i) {
        run=axis[i]==axis[i-1] ? run+1 : 1;
        if (run>=half) return false;
      }
    }
    return true;
  }

  std::optional<ModelWindow> window(std::size_t c) const {
    if (!usable(c)) return std::nullopt;
    std::array<double,240> values{};
    double peak=0;
    const auto count=2*half;
    const auto& r=c>=8*half ? rotation : identity;
    for (std::size_t t=0;t<80;++t) {
      std::array<double,3> v{};
      for (std::size_t a=0;a<3;++a)
        for (std::size_t j=0;j<count;++j) v[a]+=kernel[t*count+j]*filtered[a][c-half+j];
      for (std::size_t a=0;a<3;++a) {
        double value=0;
        for (std::size_t k=0;k<3;++k) value+=r[a*3+k]*v[k];
        values[t*3+a]=value; peak=std::max(peak,std::abs(value));
      }
    }
    if (!std::isfinite(peak) || peak==0) return std::nullopt;
    ModelWindow result{};
    for (std::size_t i=0;i<240;++i) result[i]=static_cast<float>(values[i]/peak);
    return result;
  }
};
} // namespace

std::optional<ModelWindow> prepare_window(const Waveform& w,std::size_t center) {
  validate(w);
  if (center>=w.size() || !finite_at(w,center)) return std::nullopt;
  std::size_t a=center,b=center+1;
  while(a>0 && finite_at(w,a-1)) --a;
  while(b<w.size() && finite_at(w,b)) ++b;
  if (b-a<2*static_cast<std::size_t>(std::llround(w.sample_rate))) return std::nullopt;
  return Prepared(w,a,b).window(center-a);
}

PhaseVectors infer(const Waveform& w,InferOptions options) {
  validate(w);
  if (options.stride==0 || options.threads>256) throw std::invalid_argument("Invalid stride or CPU thread count");
  float nan=std::numeric_limits<float>::quiet_NaN();
  PhaseVectors scores{std::vector<float>(w.size(),nan),std::vector<float>(w.size(),nan),std::vector<float>(w.size(),nan)};
  std::size_t begin=0,half=static_cast<std::size_t>(std::llround(w.sample_rate));
  while (begin<w.size()) {
    while(begin<w.size() && !finite_at(w,begin)) ++begin;
    auto end=begin;
    while(end<w.size() && finite_at(w,end)) ++end;
    if (end-begin>=2*half) {
      Prepared prepared(w,begin,end);
      std::vector<std::size_t> centers;
      for (std::size_t c=half;c<=end-begin-half;) {
        centers.push_back(c);
        if (options.stride>end-begin-half-c) break;
        c+=options.stride;
      }
      if (centers.back()!=end-begin-half) centers.push_back(end-begin-half);
      std::atomic<std::size_t> next{0};
      auto worker=[&] {
        for (auto j=next.fetch_add(1);j<centers.size();j=next.fetch_add(1)) {
          auto c=centers[j]; auto input=prepared.window(c);
          if (!input) continue;
          auto p=classify_window(*input); auto i=begin+c;
          scores.p[i]=p[0]; scores.s[i]=p[1]; scores.noise[i]=p[2];
        }
      };
      auto threads=std::min(centers.size(),options.threads ? options.threads : std::min<std::size_t>(256,std::max(1u,std::thread::hardware_concurrency())));
      std::vector<std::thread> workers;
      try {
        for (std::size_t i=1;i<threads;++i) workers.emplace_back(worker);
      } catch (...) {
        for (auto& thread:workers) thread.join();
        throw;
      }
      worker();
      for (auto& thread:workers) thread.join();
      if (options.stride>1) {
        for (std::size_t j=1;j<centers.size();++j) {
          auto left=begin+centers[j-1],right=begin+centers[j];
          if (!std::isfinite(scores.p[left]) || !std::isfinite(scores.p[right])) continue;
          for (auto i=left+1;i<right;++i) {
            if (!prepared.usable(i-begin)) continue;
            float f=static_cast<float>(i-left)/static_cast<float>(right-left);
            scores.p[i]=scores.p[left]*(1-f)+scores.p[right]*f;
            scores.s[i]=scores.s[left]*(1-f)+scores.s[right]*f;
            scores.noise[i]=scores.noise[left]*(1-f)+scores.noise[right]*f;
          }
        }
      }
    }
    begin=end;
  }
  return scores;
}
} // namespace seismictorch
