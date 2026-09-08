#include "frame_api.h"
#include <bit>
#include <limits>

namespace {
constexpr uint32_t mask = 32767u;
uint32_t composed(uint32_t u,uint32_t v) {
    const uint32_t a=u&mask,b=(u>>15)&mask,c=v&mask,d=(v>>15)&mask;
    const uint32_t s=((u^v)>>30)^uint32_t(std::popcount(b&c)&1);
    return (a^c)|((b^d)<<15)|((s&1u)<<30);
}
bool valid(uint32_t c) {return (c>>31)==0;}
template<class Word> int route(const Word* src,Word* dst,size_t words,uint32_t d,uint32_t code) {
    if(!src || !dst || d<1 || d>15 || !valid(code) || words==0) return -1;
    const size_t n=size_t(1)<<d;
    const uint32_t a=code&mask,b=(code>>15)&mask,s=code>>30;
    if((a|b)>=n || words%n || words>std::numeric_limits<size_t>::max()/sizeof(Word))return -1;
    const auto from=reinterpret_cast<uintptr_t>(src),to=reinterpret_cast<uintptr_t>(dst);
    const size_t bytes=words*sizeof(Word);
    if(from>std::numeric_limits<uintptr_t>::max()-bytes || to>std::numeric_limits<uintptr_t>::max()-bytes)return -1;
    if(from<to+bytes && to<from+bytes)return -1;
    constexpr unsigned shift=sizeof(Word)*8u-1u;
    for(size_t base=0;base<words;base+=n) {
        for(uint32_t i=0;i<n;++i) {
            const uint32_t j=i^a;
            const Word sign=Word(s^uint32_t(std::popcount(b&j)&1))<<shift;
            dst[base+i]=src[base+j]^sign;
        }
    }
    return 0;
}
}
extern "C" int rbe_frame_compose(uint32_t u,uint32_t v,uint32_t*out) {
    if(!out || !valid(u) || !valid(v)) return -1;
    *out=composed(u,v);
    return 0;
}
extern "C" int rbe_frame_transpose(uint32_t u,uint32_t*out) {
    if(!out || !valid(u))return -1;
    *out=u^(uint32_t(std::popcount((u&mask)&((u>>15)&mask))&1)<<30);return 0;
}
extern "C" int rbe_frame_fuse(const uint32_t*cs,size_t length,uint32_t*out) {
    if(!cs || !out || length==0) return -1;
    uint32_t q=0;
    for(size_t i=0;i<length;++i) {if(!valid(cs[i]))return -1;q=composed(cs[i],q);}
    *out=q;return 0;
}
extern "C" int rbe_route32(const uint32_t*s,uint32_t*d,size_t n,uint32_t k,uint32_t c) {return route(s,d,n,k,c);}
extern "C" int rbe_route64(const uint64_t*s,uint64_t*d,size_t n,uint32_t k,uint32_t c) {return route(s,d,n,k,c);}
