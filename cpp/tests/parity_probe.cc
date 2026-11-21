#include "seismictorch/inference.hh"
#include <iomanip>
#include <iostream>
int main(int argc,char** argv) {
  try {
    if(argc<5)return 2;
    auto w=seismictorch::read_sac({argv[1],argv[2],argv[3]});
    std::cout<<std::setprecision(10);
    for(int i=4;i<argc;++i) {
      auto window=seismictorch::prepare_window(w,std::stoull(argv[i]));
      if(!window)throw std::runtime_error("Invalid test window");
      auto p=seismictorch::classify_window(*window);std::cout<<p[0]<<' '<<p[1]<<' '<<p[2]<<'\n';
    }
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
