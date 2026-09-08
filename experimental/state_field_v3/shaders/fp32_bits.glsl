// Binary32 round-to-nearest, ties-to-even implemented with INTEGER words.
// Supports subnormals, signed zeros, infinity and canonical quiet NaN.
// Does NOT claim to outperform hardware floating point. No quantization.
uint jam(uint x, uint d) {
    if(d==0u) return x;
    if(d>=32u) return uint(x!=0u);
    return (x>>d)|uint((x & ((1u<<d)-1u))!=0u);
}
uint round_pack(uint sign, int e, uint sig) {
    if(e<1) {sig=jam(sig,uint(1-e));e=1;}
    uint frac=sig>>3u, rem=sig&7u;
    if(rem>4u || (rem==4u && (frac&1u)!=0u)) ++frac;
    if(frac>=0x1000000u) {frac>>=1u;++e;}
    if(e>=255) return sign|0x7f800000u;
    if(frac<0x800000u) e=0;
    return sign|(uint(e)<<23u)|(frac&0x7fffffu);
}
uint add32(uint a, uint b) {
    uint aa=a&0x7fffffffu, bb=b&0x7fffffffu;
    if(aa>0x7f800000u || bb>0x7f800000u) return 0x7fc00000u;
    if(aa==0x7f800000u || bb==0x7f800000u) {
        if(aa==bb && ((a^b)&0x80000000u)!=0u) return 0x7fc00000u;
        return aa==0x7f800000u?a:b;
    }
    if(aa<bb) {uint t=a;a=b;b=t;t=aa;aa=bb;bb=t;}
    uint sa=a&0x80000000u, sb=b&0x80000000u;
    int ea=int((a>>23u)&255u),eb=int((b>>23u)&255u);
    uint ma=(a&0x7fffffu)|(ea!=0?0x800000u:0u);
    uint mb=(b&0x7fffffu)|(eb!=0?0x800000u:0u);
    ea=max(ea,1);eb=max(eb,1);
    uint x=ma<<3u,y=jam(mb<<3u,uint(ea-eb)), z;
    if(sa==sb) {
        z=x+y;
        if(z>=0x8000000u) {z=jam(z,1u);++ea;}
    } else {
        z=x-y;
        if(z==0u) return 0u;
        while(z<0x4000000u && ea>1) {z<<=1u;--ea;}
    }
    return round_pack(sa,ea,z);
}
uint mul32(uint a, uint b) {
    uint aa=a&0x7fffffffu,bb=b&0x7fffffffu,sign=(a^b)&0x80000000u;
    if(aa>0x7f800000u || bb>0x7f800000u) return 0x7fc00000u;
    if(aa==0x7f800000u || bb==0x7f800000u) {
        if(aa==0u || bb==0u) return 0x7fc00000u;
        return sign|0x7f800000u;
    }
    if(aa==0u || bb==0u) return sign;
    int ea=int((a>>23u)&255u)-127,eb=int((b>>23u)&255u)-127;
    uint ma=a&0x7fffffu,mb=b&0x7fffffu;
    if(ea==-127) {int sh=23-findMSB(ma);ma<<=uint(sh);ea=-126-sh;} else ma|=0x800000u;
    if(eb==-127) {int sh=23-findMSB(mb);mb<<=uint(sh);eb=-126-sh;} else mb|=0x800000u;
    uint hi,lo;umulExtended(ma,mb,hi,lo);
    uint d=hi>=0x8000u?21u:20u;
    int e=ea+eb+(d==21u?1:0)+127;
    uint sig=(hi<<(32u-d))|(lo>>d);
    sig|=uint((lo&((1u<<d)-1u))!=0u);
    return round_pack(sign,e,sig);
}
