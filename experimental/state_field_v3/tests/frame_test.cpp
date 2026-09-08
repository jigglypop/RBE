#include "frame_api.h"
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
static void check(bool ok) { if(!ok) {std::cerr<<"frame test failed\n";std::exit(1);} }
int main() {
 std::mt19937 rng(20260909);uint64_t checks=0;
 for(uint32_t d=1;d<=5;++d) {
  const uint32_t n=1u<<d,mask=n-1;
  for(int k=0;k<1000;++k) {
   auto code=[&](){return (rng()&mask)|((rng()&mask)<<15)|((rng()&1u)<<30);};
   uint32_t p=code(),q=code(),r=0,tr=0;std::array<uint32_t,32>x{},a{},b{},z{};
   for(uint32_t i=0;i<n;++i)x[i]=rng();
   check(rbe_frame_compose(p,q,&r)==0);check(rbe_frame_transpose(r,&tr)==0);
   check(rbe_route32(x.data(),a.data(),n,d,q)==0);
   check(rbe_route32(a.data(),b.data(),n,d,p)==0);
   check(rbe_route32(x.data(),z.data(),n,d,r)==0);
   for(uint32_t i=0;i<n;++i){check(b[i]==z[i]);++checks;}
   check(rbe_route32(z.data(),b.data(),n,d,tr)==0);
   for(uint32_t i=0;i<n;++i){check(x[i]==b[i]);++checks;}
  }
 }
 check(rbe_frame_transpose(0x80000000u,nullptr)==-1);
 std::cout<<"{\"passed\":true,\"word_checks\":"<<checks<<"}\n";
}
