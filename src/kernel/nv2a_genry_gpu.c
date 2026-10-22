/*
 * nv2a_genry_gpu.c -- the executor's pixel work on D3D11. See the header.
 *
 * The pixel shader is a port of nv2a_combiner.c and the per-pixel half of
 * xf_rows() in nv2a_pb_exec.c, so a frame drawn here should match the CPU
 * rasteriser pixel for pixel apart from filtering and depth precision.
 */
#include "nv2a_genry_gpu.h"
#include "../d3d/d3d8_swizzle.h"

#if defined(_WIN32)
#define COBJMACROS
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

extern ptrdiff_t xbox_GetMemoryOffset(void);

#define GG_VB_BYTES     (16u << 20)
#define GG_CHUNK_VERTS  (64u * 1024u * 3u)  /* triangles per Draw at most */
#define GG_MAX_TARGETS  32
#define GG_MAX_DEPTHS   16
#define GG_TEX_SLOTS    4096
#define GG_TEX_PROBE    32
#define GG_MAX_STATES   128

/* Mirrors the cbuffer in the shader, 16-byte rows. */
typedef struct {
    uint32_t cicw[8], aicw[8], cocw[8], aocw[8];
    uint32_t fac0[8], fac1[8];
    uint32_t fin[4];        /* final0, final1, final_c0, final_c1 */
    uint32_t ctl[4];        /* control, stage_program, flags, alpha_func */
    uint32_t ctl2[4];       /* alpha_ref, clip_plane_mode, linear mask, cube mask */
    float    tsize[4][4];   /* width, height, bound, 0 */
    float    tscale[4][4];  /* uv scale into a larger target, force alpha 1 */
    float    fogc[4];
    float    vp[4];         /* surface width, height, depth scale, 0 */
} GgConsts;

enum { GGF_RC = 1, GGF_TEX0 = 2, GGF_ALPHA = 4 };

typedef struct {
    uint32_t va, pitch, bpp, w, h;
    uint32_t swz;                    /* Morton order in guest memory */
    ID3D11Texture2D *tex, *staging;
    ID3D11RenderTargetView *rtv;
    ID3D11ShaderResourceView *srv;   /* the target sampled as a texture */
    int gpu_valid;          /* the GPU copy is at least as new as guest RAM */
    int gpu_newer;          /* the GPU drew something guest RAM lacks */
    uint32_t used_frame;
} GgTarget;

typedef struct {
    uint32_t va, w, h;
    ID3D11Texture2D *tex;
    ID3D11DepthStencilView *dsv;
    uint32_t used_frame;
} GgDepth;

typedef struct {
    uint32_t offset, color, width, height, pitch, palette, cube, live;
    uint64_t hash;
    uint32_t checked_frame, used_frame;
    ID3D11Texture2D *tex;
    ID3D11ShaderResourceView *srv;
} GgTex;

typedef struct { uint32_t key[6]; void *obj; } GgState;

static int s_on = -1, s_dead, s_prof_on, s_dbg_flat, s_dbg_novcol;
static ID3D11Device *s_dev;
static ID3D11DeviceContext *s_ctx;
static ID3D11VertexShader *s_vs;
static ID3D11PixelShader *s_ps;
static ID3D11InputLayout *s_il;
static ID3D11Buffer *s_vb, *s_cb;
static ID3D11RasterizerState *s_rast;
static UINT s_vb_used;
static uint32_t s_frame = 1;

static GgTarget s_rt[GG_MAX_TARGETS];
static GgDepth  s_ds[GG_MAX_DEPTHS];
static GgTex    s_tex[GG_TEX_SLOTS];
static uint32_t s_tex_live;
static GgState  s_blend[GG_MAX_STATES], s_dss[GG_MAX_STATES], s_smp[GG_MAX_STATES];
static int      s_nblend, s_ndss, s_nsmp;

static GgVertex *s_verts;
static uint32_t  s_nverts;
static GgTarget *s_cur;
static uint32_t *s_decode;
static size_t    s_decode_cap;
static uint32_t *s_rows;
static size_t    s_rows_cap;

static struct {
    uint64_t draws, tris, uploads, readbacks, tex_decodes, clears, rt_direct;
    double   readback_ms;
} s_stat;

/* GENRY_PROF: which texture-stage programs and formats the batches use, and
 * which textures could not be decoded -- the first thing to know when a
 * surface comes out black or striped. */
#define GG_SEEN 24
typedef struct { uint32_t key, n; } GgSeen;
static GgSeen s_seen_sp[GG_SEEN], s_seen_fail[GG_SEEN];

static void seen_add(GgSeen *tab, uint32_t key)
{
    int i;
    for (i = 0; i < GG_SEEN; i++) {
        if (tab[i].n && tab[i].key == key) { tab[i].n++; return; }
        if (!tab[i].n) { tab[i].key = key; tab[i].n = 1; return; }
    }
}

static void seen_print(const char *what, GgSeen *tab)
{
    int i;
    for (i = 0; i < GG_SEEN && tab[i].n; i++)
        fprintf(stderr, "  [GGPU]   %s %08X x%u\n", what, tab[i].key, tab[i].n);
    memset(tab, 0, sizeof(GgSeen) * GG_SEEN);
}

/* ---- the shaders ------------------------------------------------------ */

static const char s_hlsl[] =
"cbuffer CB : register(b0) {\n"
"  uint4 cicw[2]; uint4 aicw[2]; uint4 cocw[2]; uint4 aocw[2];\n"
"  uint4 fac0[2]; uint4 fac1[2];\n"
"  uint4 fin; uint4 ctl; uint4 ctl2;\n"
"  float4 tsize[4]; float4 tscale[4]; float4 fogc; float4 vp;\n"
"};\n"
"Texture2D tx0 : register(t0); Texture2D tx1 : register(t1);\n"
"Texture2D tx2 : register(t2); Texture2D tx3 : register(t3);\n"
"TextureCube tq0 : register(t4); TextureCube tq1 : register(t5);\n"
"TextureCube tq2 : register(t6); TextureCube tq3 : register(t7);\n"
"SamplerState s0 : register(s0); SamplerState s1 : register(s1);\n"
"SamplerState s2 : register(s2); SamplerState s3 : register(s3);\n"
"struct VSIn { float4 pos : POSITION; float4 d0 : COLOR0; float4 d1 : COLOR1;\n"
"  float4 t0 : TEXCOORD0; float4 t1 : TEXCOORD1; float4 t2 : TEXCOORD2;\n"
"  float4 t3 : TEXCOORD3; float4 fg : TEXCOORD4; };\n"
"struct VSOut { float4 pos : SV_Position;\n"
"  noperspective float4 d0 : COLOR0; noperspective float4 d1 : COLOR1;\n"
"  float4 t0 : TEXCOORD0; float4 t1 : TEXCOORD1; float4 t2 : TEXCOORD2;\n"
"  float4 t3 : TEXCOORD3; noperspective float fog : TEXCOORD4;\n"
"  float clip : SV_ClipDistance0; };\n"
"VSOut VS(VSIn v) {\n"
"  VSOut o; float w = v.pos.w; float2 sp = v.pos.xy + PXO;\n"
"  o.pos = float4((sp.x / vp.x * 2.0 - 1.0) * w, (1.0 - sp.y / vp.y * 2.0) * w,\n"
"                 v.pos.z * vp.z * w, w);\n"
"  o.d0 = v.d0; o.d1 = v.d1; o.t0 = v.t0; o.t1 = v.t1; o.t2 = v.t2; o.t3 = v.t3;\n"
"  o.fog = v.fg.x;\n"
"  o.clip = vp.w != 0.0 ? o.pos.z + 1e-6 * w : 1.0;\n"
"  return o;\n"
"}\n"
"#define PICK(a, i) a[(i) >> 2][(i) & 3]\n"
"float4 unpack(uint c) {\n"
"  return float4((c >> 16) & 255, (c >> 8) & 255, c & 255, c >> 24) / 255.0;\n"
"}\n"
"float map_in(float x, uint b) {\n"
"  float p = max(x, 0.0);\n"
"  uint m = (b >> 5) & 7;\n"
"  if (m == 0) return p;\n"
"  if (m == 1) return 1.0 - saturate(x);\n"
"  if (m == 2) return 2.0 * p - 1.0;\n"
"  if (m == 3) return -2.0 * p + 1.0;\n"
"  if (m == 4) return p - 0.5;\n"
"  if (m == 5) return -p + 0.5;\n"
"  if (m == 6) return x;\n"
"  return -x;\n"
"}\n"
"float map_out(float x, uint m) {\n"
"  if (m == 0x08) return x - 0.5;\n"
"  if (m == 0x10) return x * 2.0;\n"
"  if (m == 0x18) return (x - 0.5) * 2.0;\n"
"  if (m == 0x20) return x * 4.0;\n"
"  if (m == 0x30) return x * 0.5;\n"
"  return x;\n"
"}\n"
"float3 map_out3(float3 x, uint m) {\n"
"  return float3(map_out(x.x, m), map_out(x.y, m), map_out(x.z, m));\n"
"}\n"
"float3 rd_rgb(float4 r, uint b) {\n"
"  float3 v = (b & 16) ? r.aaa : r.rgb;\n"
"  return float3(map_in(v.x, b), map_in(v.y, b), map_in(v.z, b));\n"
"}\n"
"float rd_a(float4 r, uint b) { return map_in((b & 16) ? r.a : r.b, b); }\n"
"float4 samp2d(uint st, float2 uv) {\n"
"  if (tsize[st].z == 0.0) return float4(1, 1, 1, 1);\n"
"  if ((ctl2.z >> st) & 1) uv /= tsize[st].xy;\n"
"  uv *= tscale[st].xy;\n"
"  float4 r;\n"
"  if (st == 0) r = tx0.SampleLevel(s0, uv, 0);\n"
"  else if (st == 1) r = tx1.SampleLevel(s1, uv, 0);\n"
"  else if (st == 2) r = tx2.SampleLevel(s2, uv, 0);\n"
"  else r = tx3.SampleLevel(s3, uv, 0);\n"
"  if (tscale[st].z != 0.0) r.a = 1.0;\n"
"  return r;\n"
"}\n"
"float4 sampcube(uint st, float3 d) {\n"
"  if (tsize[st].z == 0.0) return float4(1, 1, 1, 1);\n"
"  if (st == 0) return tq0.SampleLevel(s0, d, 0);\n"
"  if (st == 1) return tq1.SampleLevel(s1, d, 0);\n"
"  if (st == 2) return tq2.SampleLevel(s2, d, 0);\n"
"  return tq3.SampleLevel(s3, d, 0);\n"
"}\n"
"float4 PS(VSOut i) : SV_Target {\n"
"  uint flags = ctl.z;\n"
"  float4 outc;\n"
"  if (flags & 1) {\n"
"    float4 R[16];\n"
"    [unroll] for (uint k = 0; k < 16; k++) R[k] = float4(0, 0, 0, 0);\n"
"    float4 tc[4] = { i.t0, i.t1, i.t2, i.t3 };\n"
"    float4 t[4];\n"
"    [unroll] for (uint st = 0; st < 4; st++) {\n"
"      uint mode = (ctl.y >> (st * 5)) & 31;\n"
"      float4 c = tc[st];\n"
"      float4 r = float4(0, 0, 0, 1);\n"
"      if (mode == 1 || mode == 2) {\n"
"        float q = c.w != 0.0 ? c.w : 1.0;\n"
"        r = samp2d(st, c.xy / q);\n"
"      } else if (mode == 3) {\n"
"        r = ((ctl2.w >> st) & 1) ? sampcube(st, c.xyz) : samp2d(st, c.xy);\n"
"      } else if (mode == 4) {\n"
"        r = saturate(c);\n"
"      } else if (mode == 5) {\n"
"        [unroll] for (uint j = 0; j < 4; j++) {\n"
"          bool ge = ((ctl2.y >> (st * 4 + j)) & 1) != 0;\n"
"          if (ge ? c[j] >= 0.0 : c[j] < 0.0) discard;\n"
"        }\n"
"        r = float4(0, 0, 0, 0);\n"
"      } else if (mode != 0) {\n"
"        r = float4(0, 0, 0, 0);\n"
"      }\n"
"      t[st] = r;\n"
"    }\n"
"    R[3] = float4(fogc.rgb, i.fog);\n"
"    R[4] = i.d0; R[5] = i.d1;\n"
"    R[8] = t[0]; R[9] = t[1]; R[10] = t[2]; R[11] = t[3];\n"
"    R[12].a = (ctl.y & 31) ? t[0].a : 1.0;\n"
"    uint n = min(ctl.x & 255, 8u);\n"
"    uint cf = ctl.x >> 8;\n"
"    [loop] for (uint s = 0; s < n; s++) {\n"
"      uint icw = PICK(cicw, s), ocw = PICK(cocw, s);\n"
"      uint ai = PICK(aicw, s), ao = PICK(aocw, s);\n"
"      uint fl = ocw >> 12, afl = ao >> 12;\n"
"      R[1] = unpack(PICK(fac0, (cf & 0x010) ? s : 0u));\n"
"      R[2] = unpack(PICK(fac1, (cf & 0x100) ? s : 0u));\n"
"      bool mux = (cf & 1) ? (R[12].a >= 0.5) : ((((int)(R[12].a * 255.0)) & 1) != 0);\n"
"      float3 A = rd_rgb(R[(icw >> 24) & 15], icw >> 24);\n"
"      float3 B = rd_rgb(R[(icw >> 16) & 15], icw >> 16);\n"
"      float3 C = rd_rgb(R[(icw >> 8) & 15], icw >> 8);\n"
"      float3 D = rd_rgb(R[icw & 15], icw);\n"
"      float aA = rd_a(R[(ai >> 24) & 15], ai >> 24);\n"
"      float aB = rd_a(R[(ai >> 16) & 15], ai >> 16);\n"
"      float aC = rd_a(R[(ai >> 8) & 15], ai >> 8);\n"
"      float aD = rd_a(R[ai & 15], ai);\n"
"      float3 pab = (fl & 2) ? dot(A, B).xxx : A * B;\n"
"      float3 pcd = (fl & 1) ? dot(C, D).xxx : C * D;\n"
"      float3 sum = (fl & 4) ? (mux ? pcd : pab) : pab + pcd;\n"
"      float3 ab = clamp(map_out3(pab, fl & 0x38), -1.0, 1.0);\n"
"      float3 cd = clamp(map_out3(pcd, fl & 0x38), -1.0, 1.0);\n"
"      float3 ms = clamp(map_out3(sum, fl & 0x38), -1.0, 1.0);\n"
"      float apab = aA * aB, apcd = aC * aD;\n"
"      float asum = (afl & 4) ? (mux ? apcd : apab) : apab + apcd;\n"
"      float aab = clamp(map_out(apab, afl & 0x38), -1.0, 1.0);\n"
"      float acd = clamp(map_out(apcd, afl & 0x38), -1.0, 1.0);\n"
"      float ams = clamp(map_out(asum, afl & 0x38), -1.0, 1.0);\n"
"      uint dst = (ocw >> 4) & 15;\n"
"      if (dst != 0) { R[dst].rgb = ab; if (fl & 0x80) R[dst].a = ab.b; }\n"
"      dst = ocw & 15;\n"
"      if (dst != 0) { R[dst].rgb = cd; if (fl & 0x40) R[dst].a = cd.b; }\n"
"      dst = (ocw >> 8) & 15;\n"
"      if (dst != 0) R[dst].rgb = ms;\n"
"      dst = (ao >> 4) & 15; if (dst != 0) R[dst].a = aab;\n"
"      dst = ao & 15;        if (dst != 0) R[dst].a = acd;\n"
"      dst = (ao >> 8) & 15; if (dst != 0) R[dst].a = ams;\n"
"      R[0] = float4(0, 0, 0, 0);\n"
"    }\n"
"    if (fin.x != 0 || fin.y != 0) {\n"
"      uint f0 = fin.x, f1 = fin.y, ff = f1 & 255;\n"
"      R[1] = unpack(fin.z); R[2] = unpack(fin.w);\n"
"      float3 a = (ff & 0x40) ? 1.0 - R[5].rgb : R[5].rgb;\n"
"      float3 b = (ff & 0x20) ? 1.0 - R[12].rgb : R[12].rgb;\n"
"      R[14] = float4((ff & 0x80) ? saturate(a + b) : a + b, 0.0);\n"
"      float3 E = rd_rgb(R[(f1 >> 24) & 15], f1 >> 24);\n"
"      float3 F = rd_rgb(R[(f1 >> 16) & 15], f1 >> 16);\n"
"      R[15] = float4(E * F, 0.0);\n"
"      float3 A = rd_rgb(R[(f0 >> 24) & 15], f0 >> 24);\n"
"      float3 B = rd_rgb(R[(f0 >> 16) & 15], f0 >> 16);\n"
"      float3 C = rd_rgb(R[(f0 >> 8) & 15], f0 >> 8);\n"
"      float3 D = rd_rgb(R[f0 & 15], f0);\n"
"      outc.rgb = D + C * (1.0 - A) + B * A;\n"
"      outc.a = rd_a(R[(f1 >> 8) & 15], f1 >> 8);\n"
"    } else {\n"
"      outc = R[12];\n"
"    }\n"
"    outc = saturate(outc);\n"
"  } else {\n"
"    float4 c = i.d0;\n"
"    if (flags & 2) c *= samp2d(0, i.t0.xy);\n"
"    outc = saturate(c);\n"
"  }\n"
"  if (flags & 4) {\n"
"    int v = (int)(outc.a * 255.0 + 0.5), r = (int)(ctl2.x & 255);\n"
"    uint f = ctl.w;\n"
"    bool ok = true;\n"
"    if (f == 0x200) ok = false;\n"
"    else if (f == 0x201) ok = v < r;\n"
"    else if (f == 0x202) ok = v == r;\n"
"    else if (f == 0x203) ok = v <= r;\n"
"    else if (f == 0x204) ok = v > r;\n"
"    else if (f == 0x205) ok = v != r;\n"
"    else if (f == 0x206) ok = v >= r;\n"
"    if (!ok) discard;\n"
"  }\n"
"  return outc;\n"
"}\n";

/* ---- device ------------------------------------------------------------ */

int gg_enabled(void)
{
    if (s_on < 0) {
        const char *e = getenv("GENRY_GPU");
        s_on = e && *e && *e != '0';
        s_prof_on = getenv("GENRY_PROF") != NULL;
    }
    return s_on && !s_dead;
}

static ID3DBlob *compile(const char *entry, const char *target)
{
    ID3DBlob *code = NULL, *err = NULL;
    /* GENRY_PX_OFS=<px>: a shift of the NV2A's screen coordinates before they
     * go to D3D11's pixel grid (whose centres sit at .5) -- testing the
     * one-pixel column and row along the frame's left and top edges. */
    static char pxo[48];
    D3D_SHADER_MACRO defs[2] = { { "PXO", pxo }, { NULL, NULL } };
    if (!pxo[0]) {
        const char *e = getenv("GENRY_PX_OFS");
        snprintf(pxo, sizeof pxo, "float2(%s, %s)", e ? e : "0.0", e ? e : "0.0");
    }
    HRESULT hr = D3DCompile(s_hlsl, sizeof s_hlsl - 1, "genry_gpu", defs, NULL,
                            entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
                            &code, &err);
    if (FAILED(hr)) {
        fprintf(stderr, "[GGPU] %s did not compile (0x%08lX): %s\n", entry,
                (unsigned long)hr,
                err ? (const char *)ID3D10Blob_GetBufferPointer(err) : "?");
        if (err) ID3D10Blob_Release(err);
        return NULL;
    }
    if (err) ID3D10Blob_Release(err);
    return code;
}

static int fail(const char *what, HRESULT hr)
{
    fprintf(stderr, "[GGPU] %s failed (0x%08lX): the CPU rasteriser keeps"
                    " drawing\n", what, (unsigned long)hr);
    fflush(stderr);
    s_dead = 1;
    return 0;
}

static void print_adapter(D3D_DRIVER_TYPE type)
{
    IDXGIDevice *dxgi = NULL;
    IDXGIAdapter *ad = NULL;
    DXGI_ADAPTER_DESC desc;

    if (SUCCEEDED(ID3D11Device_QueryInterface(s_dev, &IID_IDXGIDevice,
                                              (void **)&dxgi))) {
        if (SUCCEEDED(IDXGIDevice_GetAdapter(dxgi, &ad))) {
            if (SUCCEEDED(IDXGIAdapter_GetDesc(ad, &desc)))
                fprintf(stderr, "[GGPU] D3D11 on %ls%s\n", desc.Description,
                        type == D3D_DRIVER_TYPE_WARP ? " (WARP)" : "");
            IDXGIAdapter_Release(ad);
        }
        IDXGIDevice_Release(dxgi);
    }
    fflush(stderr);
}

static int gg_ready(void)
{
    static const D3D_FEATURE_LEVEL want[] = { D3D_FEATURE_LEVEL_11_0 };
    static const D3D11_INPUT_ELEMENT_DESC il[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,   0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,  16, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "COLOR",    1, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,  32, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,  48, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,  64, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 2, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,  80, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 3, DXGI_FORMAT_R32G32B32A32_FLOAT, 0,  96, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 4, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 112, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    D3D_DRIVER_TYPE type = D3D_DRIVER_TYPE_HARDWARE;
    D3D_FEATURE_LEVEL got;
    D3D11_BUFFER_DESC bd;
    D3D11_RASTERIZER_DESC rd;
    ID3DBlob *vs, *ps;
    HRESULT hr;

    if (s_dead)
        return 0;
    if (s_dev)
        return 1;
    hr = D3D11CreateDevice(NULL, type, NULL, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                           want, 1, D3D11_SDK_VERSION, &s_dev, &got, &s_ctx);
    if (FAILED(hr)) {
        type = D3D_DRIVER_TYPE_WARP;
        hr = D3D11CreateDevice(NULL, type, NULL, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                               want, 1, D3D11_SDK_VERSION, &s_dev, &got, &s_ctx);
    }
    if (FAILED(hr))
        return fail("D3D11CreateDevice", hr);
    print_adapter(type);

    vs = compile("VS", "vs_5_0");
    ps = compile("PS", "ps_5_0");
    if (!vs || !ps)
        return fail("shader compile", E_FAIL);
    hr = ID3D11Device_CreateVertexShader(s_dev, ID3D10Blob_GetBufferPointer(vs),
                                         ID3D10Blob_GetBufferSize(vs), NULL, &s_vs);
    if (SUCCEEDED(hr))
        hr = ID3D11Device_CreatePixelShader(s_dev, ID3D10Blob_GetBufferPointer(ps),
                                            ID3D10Blob_GetBufferSize(ps), NULL, &s_ps);
    if (SUCCEEDED(hr))
        hr = ID3D11Device_CreateInputLayout(s_dev, il, sizeof il / sizeof il[0],
                                            ID3D10Blob_GetBufferPointer(vs),
                                            ID3D10Blob_GetBufferSize(vs), &s_il);
    ID3D10Blob_Release(vs);
    ID3D10Blob_Release(ps);
    if (FAILED(hr))
        return fail("shader objects", hr);

    memset(&bd, 0, sizeof bd);
    bd.ByteWidth = GG_VB_BYTES;
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    hr = ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_vb);
    if (FAILED(hr))
        return fail("vertex buffer", hr);
    memset(&bd, 0, sizeof bd);
    bd.ByteWidth = sizeof(GgConsts);
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    hr = ID3D11Device_CreateBuffer(s_dev, &bd, NULL, &s_cb);
    if (FAILED(hr))
        return fail("constant buffer", hr);

    memset(&rd, 0, sizeof rd);
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;          /* as the CPU rasteriser: both faces */
    rd.ScissorEnable = TRUE;
    rd.DepthClipEnable = FALSE;             /* depth is compared, never clipped */
    hr = ID3D11Device_CreateRasterizerState(s_dev, &rd, &s_rast);
    if (FAILED(hr))
        return fail("rasterizer state", hr);

    s_verts = (GgVertex *)malloc(sizeof(GgVertex) * GG_CHUNK_VERTS);
    if (!s_verts)
        return fail("vertex staging", E_OUTOFMEMORY);

    ID3D11DeviceContext_IASetPrimitiveTopology(s_ctx, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ID3D11DeviceContext_IASetInputLayout(s_ctx, s_il);
    ID3D11DeviceContext_VSSetShader(s_ctx, s_vs, NULL, 0);
    ID3D11DeviceContext_PSSetShader(s_ctx, s_ps, NULL, 0);
    ID3D11DeviceContext_VSSetConstantBuffers(s_ctx, 0, 1, &s_cb);
    ID3D11DeviceContext_PSSetConstantBuffers(s_ctx, 0, 1, &s_cb);
    ID3D11DeviceContext_RSSetState(s_ctx, s_rast);
    {
        UINT stride = sizeof(GgVertex), offset = 0;
        ID3D11DeviceContext_IASetVertexBuffers(s_ctx, 0, 1, &s_vb, &stride, &offset);
    }
    return 1;
}

/* ---- guest surfaces ----------------------------------------------------- */

static uint8_t *guest(uint32_t va)
{
    return (uint8_t *)xbox_GetMemoryOffset() + va;
}

static uint32_t *rows_buf(size_t texels)
{
    if (texels > s_rows_cap) {
        free(s_rows);
        s_rows = (uint32_t *)malloc(texels * 4);
        s_rows_cap = s_rows ? texels : 0;
    }
    return s_rows;
}

static void target_upload(GgTarget *t)
{
    uint32_t x, y, *buf = rows_buf((size_t)t->w * t->h * 2);
    const uint8_t *src = guest(t->va);

    if (!buf)
        return;
    if (t->swz) {
        /* Unswizzle into the back half of the buffer, then convert from
         * there as if it were a linear surface of pitch w * bpp. */
        uint8_t *lin = (uint8_t *)(buf + (size_t)t->w * t->h);
        xbox_unswizzle_rect(lin, src, t->w, t->h, t->bpp);
        src = lin;
    }
    for (y = 0; y < t->h; y++) {
        const uint8_t *row = src + (size_t)y * (t->swz ? t->w * t->bpp : t->pitch);
        uint32_t *dst = buf + (size_t)y * t->w;
        if (t->bpp == 4) {
            memcpy(dst, row, (size_t)t->w * 4);
        } else {
            const uint16_t *p = (const uint16_t *)row;
            for (x = 0; x < t->w; x++) {
                uint16_t v = p[x];
                uint32_t r = (uint32_t)((v >> 11) & 0x1F) * 255u / 31u;
                uint32_t g = (uint32_t)((v >>  5) & 0x3F) * 255u / 63u;
                uint32_t b = (uint32_t)( v        & 0x1F) * 255u / 31u;
                dst[x] = 0xFF000000u | (r << 16) | (g << 8) | b;
            }
        }
    }
    ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)t->tex, 0, NULL,
                                          buf, t->w * 4, 0);
    s_stat.uploads++;
}

static void flush_draw(void);

static void target_readback(GgTarget *t)
{
    D3D11_MAPPED_SUBRESOURCE m;
    LARGE_INTEGER a, b, f;
    uint8_t *dst = guest(t->va), *lin = NULL;
    uint32_t x, y, stride = t->pitch;

    if (s_nverts && s_cur == t)
        flush_draw();
    if (t->swz) {
        /* Rows to a packed linear copy first, then swizzled into place. */
        uint32_t *b = rows_buf((size_t)t->w * t->h);
        if (!b)
            return;
        lin = (uint8_t *)b;
        dst = lin;
        stride = t->w * t->bpp;
    }
    QueryPerformanceCounter(&a);
    ID3D11DeviceContext_CopyResource(s_ctx, (ID3D11Resource *)t->staging,
                                     (ID3D11Resource *)t->tex);
    /* Blocking on purpose: answering the flush-kick handshake while the
     * flip's surface is read back (or while the flip waits for vblank) hung
     * Shadow the Hedgehog after its second frame -- something in our flip
     * synchronisation leans on the late answer (2026-10-21, see
     * xbox_Nv2aAckFlushKick). */
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)t->staging, 0,
                                       D3D11_MAP_READ, 0, &m)))
        return;
    for (y = 0; y < t->h; y++) {
        const uint32_t *src = (const uint32_t *)((const uint8_t *)m.pData
                                                 + (size_t)y * m.RowPitch);
        uint8_t *row = dst + (size_t)y * stride;
        if (t->bpp == 4) {
            memcpy(row, src, (size_t)t->w * 4);
        } else {
            uint16_t *p = (uint16_t *)row;
            for (x = 0; x < t->w; x++) {
                uint32_t c = src[x];
                p[x] = (uint16_t)(((c >> 8) & 0xF800) | ((c >> 5) & 0x07E0)
                                  | ((c >> 3) & 0x001F));
            }
        }
    }
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)t->staging, 0);
    if (lin)
        xbox_swizzle_rect(guest(t->va), lin, t->w, t->h, t->bpp);
    QueryPerformanceCounter(&b);
    QueryPerformanceFrequency(&f);
    s_stat.readback_ms += (double)(b.QuadPart - a.QuadPart) * 1000.0 / (double)f.QuadPart;
    s_stat.readbacks++;
    t->gpu_newer = 0;
}

static void target_release(GgTarget *t)
{
    if (t->rtv) ID3D11RenderTargetView_Release(t->rtv);
    if (t->srv) ID3D11ShaderResourceView_Release(t->srv);
    if (t->tex) ID3D11Texture2D_Release(t->tex);
    if (t->staging) ID3D11Texture2D_Release(t->staging);
    memset(t, 0, sizeof *t);
}

static int ranges_meet(uint32_t a, uint32_t an, uint32_t b, uint32_t bn)
{
    return a < b + bn && b < a + an;
}

static GgTarget *target_get(uint32_t va, uint32_t pitch, uint32_t bpp, uint32_t h,
                            uint32_t swz, uint32_t sw_w, uint32_t sw_h)
{
    GgTarget *t = NULL, *lru = NULL;
    uint32_t w, i;
    D3D11_TEXTURE2D_DESC td;

    if (swz) {
        w = sw_w;
        h = sw_h;
        pitch = w * bpp;
    } else {
        w = pitch / bpp;
    }
    for (i = 0; i < GG_MAX_TARGETS; i++) {
        GgTarget *c = &s_rt[i];
        if (c->tex && c->va == va && c->pitch == pitch && c->bpp == bpp && c->swz == swz) {
            t = c;
            break;
        }
    }
    if (t && t->h >= h) {
        t->used_frame = s_frame;
        return t;
    }
    if (t) {                                /* taller now: rebuild */
        if (t->gpu_newer) target_readback(t);
        target_release(t);
    }
    /* Memory another target covers belongs to this one from now on. */
    for (i = 0; i < GG_MAX_TARGETS; i++) {
        GgTarget *c = &s_rt[i];
        if (c->tex && ranges_meet(c->va, c->pitch * c->h, va, pitch * h)) {
            if (c->gpu_newer) target_readback(c);
            target_release(c);
        }
    }
    for (i = 0; i < GG_MAX_TARGETS; i++) {
        GgTarget *c = &s_rt[i];
        if (!c->tex) { t = c; break; }
        if (c != s_cur && (!lru || c->used_frame < lru->used_frame)) lru = c;
    }
    if (!t) {
        t = lru;
        if (!t) return NULL;
        if (t->gpu_newer) target_readback(t);
        target_release(t);
    }
    memset(&td, 0, sizeof td);
    td.Width = w;
    td.Height = h;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &t->tex)))
        return NULL;
    td.Usage = D3D11_USAGE_STAGING;
    td.BindFlags = 0;
    td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &t->staging))
        || FAILED(ID3D11Device_CreateRenderTargetView(s_dev, (ID3D11Resource *)t->tex,
                                                      NULL, &t->rtv))
        || FAILED(ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)t->tex,
                                                        NULL, &t->srv))) {
        target_release(t);
        return NULL;
    }
    t->va = va; t->pitch = pitch; t->bpp = bpp; t->w = w; t->h = h; t->swz = swz;
    t->gpu_valid = 0; t->gpu_newer = 0;
    t->used_frame = s_frame;
    fprintf(stderr, "[GGPU] surface 0x%08X %ux%u %ubpp%s\n", va, w, h, bpp * 8,
            swz ? " swizzled" : "");
    return t;
}

static GgDepth *depth_get(uint32_t va, uint32_t w, uint32_t h)
{
    GgDepth *d = NULL, *lru = NULL;
    D3D11_TEXTURE2D_DESC td;
    uint32_t i;

    for (i = 0; i < GG_MAX_DEPTHS; i++) {
        GgDepth *c = &s_ds[i];
        if (c->tex && c->va == va && c->w == w && c->h == h) {
            c->used_frame = s_frame;
            return c;
        }
    }
    for (i = 0; i < GG_MAX_DEPTHS; i++) {
        GgDepth *c = &s_ds[i];
        if (!c->tex) { d = c; break; }
        if (!lru || c->used_frame < lru->used_frame) lru = c;
    }
    if (!d) {
        d = lru;
        ID3D11DepthStencilView_Release(d->dsv);
        ID3D11Texture2D_Release(d->tex);
        memset(d, 0, sizeof *d);
    }
    memset(&td, 0, sizeof td);
    td.Width = w;
    td.Height = h;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, NULL, &d->tex)))
        return NULL;
    if (FAILED(ID3D11Device_CreateDepthStencilView(s_dev, (ID3D11Resource *)d->tex,
                                                   NULL, &d->dsv))) {
        ID3D11Texture2D_Release(d->tex);
        memset(d, 0, sizeof *d);
        return NULL;
    }
    d->va = va; d->w = w; d->h = h; d->used_frame = s_frame;
    /* Fresh depth reads as the far plane, as a cleared Xbox buffer would. */
    ID3D11DeviceContext_ClearDepthStencilView(s_ctx, d->dsv,
        D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);
    return d;
}

void gg_sync_range(uint32_t va, uint32_t bytes, int invalidate)
{
    uint32_t i;

    if (!s_dev || !bytes)
        return;
    for (i = 0; i < GG_MAX_TARGETS; i++) {
        GgTarget *t = &s_rt[i];
        if (!t->tex || !ranges_meet(t->va, t->pitch * t->h, va, bytes))
            continue;
        if (t->gpu_newer)
            target_readback(t);
        if (invalidate)
            t->gpu_valid = 0;
    }
}

/* ---- textures ----------------------------------------------------------- */

static uint64_t mix64(uint64_t h, uint64_t v)
{
    h ^= v;
    h *= 0x100000001B3ull;
    return h ^ (h >> 29);
}

/* Content fingerprint. Small textures are hashed whole; big ones by 4096
 * evenly spread words, which still catches a new movie frame or a redrawn
 * render target and costs a few microseconds. */
static uint64_t hash_guest(uint32_t va, uint32_t bytes)
{
    const uint8_t *p = guest(va);
    uint64_t h = 0xCBF29CE484222325ull ^ bytes, v;
    uint32_t words = bytes / 8, i, step;

    if (words <= 4096) {
        for (i = 0; i < words; i++) {
            memcpy(&v, p + (size_t)i * 8, 8);
            h = mix64(h, v);
        }
    } else {
        step = words / 4096;
        for (i = 0; i < 4096; i++) {
            memcpy(&v, p + (size_t)i * step * 8, 8);
            h = mix64(h, v);
        }
        memcpy(&v, p + (size_t)(words - 1) * 8, 8);
        h = mix64(h, v);
    }
    return h;
}

static void tex_release(GgTex *e)
{
    if (e->srv) ID3D11ShaderResourceView_Release(e->srv);
    if (e->tex) ID3D11Texture2D_Release(e->tex);
    memset(e, 0, sizeof *e);
}

static int tex_overlaps_newer_target(const GgTexture *t)
{
    uint32_t i;
    for (i = 0; i < GG_MAX_TARGETS; i++) {
        GgTarget *r = &s_rt[i];
        if (r->tex && r->gpu_newer
            && ranges_meet(r->va, r->pitch * r->h, t->offset, t->bytes))
            return 1;
    }
    return 0;
}

/* GENRY_TEX_DUMP=<dir>: every texture as first decoded, face 0, as a 32-bit
 * BMP named <offset>_f<format>_<w>x<h>.bmp -- what the GPU will sample, so a
 * striped or black surface can be traced to the texture or ruled out. */
static void tex_dump(const GgTexture *t, const uint32_t *argb)
{
    static const char *dir = (const char *)-1;
    static unsigned count;
    char path[512];
    uint8_t hdr[54];
    uint32_t img, y;
    FILE *f;

    if (dir == (const char *)-1)
        dir = getenv("GENRY_TEX_DUMP");
    if (!dir || count >= 2000)
        return;
    snprintf(path, sizeof path, "%s/%08X_f%02X_%ux%u%s.bmp", dir, t->offset,
             t->color, t->width, t->height, t->cube ? "_cube" : "");
    f = fopen(path, "wb");
    if (!f)
        return;
    count++;
    img = t->width * t->height * 4;
    memset(hdr, 0, sizeof hdr);
    hdr[0] = 'B'; hdr[1] = 'M';
    *(uint32_t *)(hdr + 2) = 54 + img;
    hdr[10] = 54; hdr[14] = 40;
    *(int32_t *)(hdr + 18) = (int32_t)t->width;
    *(int32_t *)(hdr + 22) = -(int32_t)t->height;      /* top-down */
    hdr[26] = 1; hdr[28] = 32;
    *(uint32_t *)(hdr + 34) = img;
    fwrite(hdr, 1, sizeof hdr, f);
    for (y = 0; y < t->height; y++)
        fwrite(argb + (size_t)y * t->width, 4, t->width, f);
    fclose(f);
}

/* A texture that is a surface the GPU drew this frame or the last: sample
 * the target itself. Reading it back and decoding it on the CPU every frame
 * -- the general path -- cost Sonic Heroes' stage a third of its frame time
 * for four 256x256 shadow and reflection targets. 32-bit colour formats only,
 * not the target being drawn into (D3D11 cannot read and write one view at
 * once), and the texture must lie at the target's origin inside it. */
static GgTarget *texture_target(const GgTexture *t, const GgTarget *drawing)
{
    uint32_t i;
    switch (t->color) {
    case 0x06: case 0x07: case 0x12: case 0x1E: break;
    default: return NULL;
    }
    if (t->cube)
        return NULL;
    for (i = 0; i < GG_MAX_TARGETS; i++) {
        GgTarget *r = &s_rt[i];
        if (r->tex && r->srv && r->va == t->offset && r->bpp == 4
            && r->gpu_valid && r->used_frame + 1 >= s_frame && r != drawing
            && t->width <= r->w && t->height <= r->h)
            return r;
        if (r->tex && r->va == t->offset) {
            static uint32_t told[16];
            static int ntold;
            int k;
            for (k = 0; k < ntold && told[k] != r->va; k++)
                ;
            if (k == ntold && ntold < 16) {
                told[ntold++] = r->va;
                fprintf(stderr, "[GGPU] texture 0x%08X fmt %02X %ux%u not sampled from its"
                                " target (%ux%u bpp %u valid %d used %u/%u drawing %d)\n",
                        t->offset, t->color, t->width, t->height, r->w, r->h, r->bpp,
                        r->gpu_valid, r->used_frame, s_frame, r == drawing);
            }
        }
    }
    return NULL;
}

static ID3D11ShaderResourceView *tex_get(const GgTexture *t)
{
    uint32_t slot = ((t->offset >> 7) ^ (t->offset >> 17) ^ t->color * 31u
                     ^ t->width * 7u) & (GG_TEX_SLOTS - 1);
    GgTex *e = NULL, *hole = NULL;
    uint64_t h;
    uint32_t k, faces = t->cube ? 6 : 1, f;
    D3D11_TEXTURE2D_DESC td;
    D3D11_SUBRESOURCE_DATA init[6];
    D3D11_SHADER_RESOURCE_VIEW_DESC sd;

    for (k = 0; k < GG_TEX_PROBE; k++) {
        GgTex *c = &s_tex[(slot + k) & (GG_TEX_SLOTS - 1)];
        if (!c->live) { if (!hole) hole = c; continue; }
        if (c->offset == t->offset && c->color == t->color && c->width == t->width
            && c->height == t->height && c->pitch == t->pitch
            && c->palette == t->palette && c->cube == t->cube) {
            e = c;
            break;
        }
    }
    if (e && e->checked_frame == s_frame && !tex_overlaps_newer_target(t)) {
        e->used_frame = s_frame;
        return e->srv;
    }
    gg_sync_range(t->offset, t->bytes, 0);   /* a surface the GPU drew first */
    h = hash_guest(t->offset, t->bytes);
    if (t->palette)
        h = mix64(h, hash_guest(t->palette, 1024));
    if (e && e->hash == h) {
        e->checked_frame = e->used_frame = s_frame;
        return e->srv;
    }
    if (!e) {
        if (!hole) {                       /* the neighbourhood is full */
            hole = &s_tex[slot];
            tex_release(hole);
            s_tex_live--;
        }
        e = hole;
        e->offset = t->offset; e->color = t->color; e->width = t->width;
        e->height = t->height; e->pitch = t->pitch; e->palette = t->palette;
        e->cube = t->cube; e->live = 1;
        s_tex_live++;
    }
    /* Decode every face, then (re)build the D3D texture. */
    {
        size_t texels = (size_t)t->width * t->height;
        if (texels * faces > s_decode_cap) {
            free(s_decode);
            s_decode = (uint32_t *)malloc(texels * faces * 4);
            s_decode_cap = s_decode ? texels * faces : 0;
            if (!s_decode) return NULL;
        }
        for (f = 0; f < faces; f++) {
            if (!t->decode(t->ctx, f, s_decode + texels * f)) {
                tex_release(e);
                s_tex_live--;
                return NULL;
            }
            init[f].pSysMem = s_decode + texels * f;
            init[f].SysMemPitch = t->width * 4;
            init[f].SysMemSlicePitch = 0;
        }
    }
    s_stat.tex_decodes++;
    if (!e->tex)
        tex_dump(t, s_decode);
    if (e->tex) {
        ID3D11Texture2D_GetDesc(e->tex, &td);
        if (td.Width == t->width && td.Height == t->height) {
            for (f = 0; f < faces; f++)
                ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)e->tex,
                                                      f, NULL, init[f].pSysMem,
                                                      init[f].SysMemPitch, 0);
            e->hash = h;
            e->checked_frame = e->used_frame = s_frame;
            return e->srv;
        }
        ID3D11ShaderResourceView_Release(e->srv);
        ID3D11Texture2D_Release(e->tex);
        e->srv = NULL; e->tex = NULL;
    }
    memset(&td, 0, sizeof td);
    td.Width = t->width;
    td.Height = t->height;
    td.MipLevels = 1;
    td.ArraySize = faces;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    td.MiscFlags = t->cube ? D3D11_RESOURCE_MISC_TEXTURECUBE : 0;
    memset(&sd, 0, sizeof sd);
    sd.Format = td.Format;
    if (t->cube) {
        sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
        sd.TextureCube.MipLevels = 1;
    } else {
        sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MipLevels = 1;
    }
    if (FAILED(ID3D11Device_CreateTexture2D(s_dev, &td, init, &e->tex))
        || FAILED(ID3D11Device_CreateShaderResourceView(s_dev, (ID3D11Resource *)e->tex,
                                                        &sd, &e->srv))) {
        tex_release(e);
        s_tex_live--;
        return NULL;
    }
    e->hash = h;
    e->checked_frame = e->used_frame = s_frame;
    return e->srv;
}

/* ---- fixed-function state ----------------------------------------------- */

static void *state_find(GgState *tab, int n, const uint32_t key[6])
{
    int i;
    for (i = 0; i < n; i++)
        if (!memcmp(tab[i].key, key, sizeof tab[i].key))
            return tab[i].obj;
    return NULL;
}

static void state_add(GgState *tab, int *n, const uint32_t key[6], void *obj)
{
    if (*n < GG_MAX_STATES) {
        memcpy(tab[*n].key, key, sizeof tab[*n].key);
        tab[*n].obj = obj;
        (*n)++;
    }
}

static D3D11_BLEND map_blend(uint32_t f, int alpha)
{
    switch (f) {
    case 0x0000: return D3D11_BLEND_ZERO;
    case 0x0001: return D3D11_BLEND_ONE;
    case 0x0300: return alpha ? D3D11_BLEND_SRC_ALPHA : D3D11_BLEND_SRC_COLOR;
    case 0x0301: return alpha ? D3D11_BLEND_INV_SRC_ALPHA : D3D11_BLEND_INV_SRC_COLOR;
    case 0x0302: return D3D11_BLEND_SRC_ALPHA;
    case 0x0303: return D3D11_BLEND_INV_SRC_ALPHA;
    case 0x0304: return D3D11_BLEND_DEST_ALPHA;
    case 0x0305: return D3D11_BLEND_INV_DEST_ALPHA;
    case 0x0306: return alpha ? D3D11_BLEND_DEST_ALPHA : D3D11_BLEND_DEST_COLOR;
    case 0x0307: return alpha ? D3D11_BLEND_INV_DEST_ALPHA : D3D11_BLEND_INV_DEST_COLOR;
    case 0x0308: return alpha ? D3D11_BLEND_ONE : D3D11_BLEND_SRC_ALPHA_SAT;
    case 0x8001: case 0x8003: return D3D11_BLEND_BLEND_FACTOR;
    case 0x8002: case 0x8004: return D3D11_BLEND_INV_BLEND_FACTOR;
    default:     return D3D11_BLEND_ONE;
    }
}

static D3D11_BLEND_OP map_blend_op(uint32_t e)
{
    switch (e) {
    case 0x800A: return D3D11_BLEND_OP_SUBTRACT;
    case 0x800B: return D3D11_BLEND_OP_REV_SUBTRACT;
    case 0x8007: return D3D11_BLEND_OP_MIN;
    case 0x8008: return D3D11_BLEND_OP_MAX;
    default:     return D3D11_BLEND_OP_ADD;
    }
}

static D3D11_COMPARISON_FUNC map_cmp(uint32_t f)
{
    return (f >= 0x200 && f <= 0x207) ? (D3D11_COMPARISON_FUNC)(f - 0x200 + 1)
                                      : D3D11_COMPARISON_ALWAYS;
}

static D3D11_STENCIL_OP map_sop(uint32_t op)
{
    switch (op) {
    case 0x0000: return D3D11_STENCIL_OP_ZERO;
    case 0x1E01: return D3D11_STENCIL_OP_REPLACE;
    case 0x1E02: return D3D11_STENCIL_OP_INCR_SAT;
    case 0x1E03: return D3D11_STENCIL_OP_DECR_SAT;
    case 0x150A: return D3D11_STENCIL_OP_INVERT;
    case 0x8507: return D3D11_STENCIL_OP_INCR;
    case 0x8508: return D3D11_STENCIL_OP_DECR;
    default:     return D3D11_STENCIL_OP_KEEP;
    }
}

static UINT8 write_mask(uint32_t nv)
{
    return (UINT8)(((nv & 0x00010000u) ? D3D11_COLOR_WRITE_ENABLE_RED : 0)
                 | ((nv & 0x00000100u) ? D3D11_COLOR_WRITE_ENABLE_GREEN : 0)
                 | ((nv & 0x00000001u) ? D3D11_COLOR_WRITE_ENABLE_BLUE : 0)
                 | ((nv & 0x01000000u) ? D3D11_COLOR_WRITE_ENABLE_ALPHA : 0));
}

static ID3D11BlendState *blend_state(uint32_t enable, uint32_t src, uint32_t dst,
                                     uint32_t eq, UINT8 mask)
{
    uint32_t key[6] = { enable, src, dst, eq, mask, 0 };
    ID3D11BlendState *bs = (ID3D11BlendState *)state_find(s_blend, s_nblend, key);
    D3D11_BLEND_DESC d;

    if (bs)
        return bs;
    memset(&d, 0, sizeof d);
    d.RenderTarget[0].BlendEnable = enable && !(src == 1 && dst == 0);
    d.RenderTarget[0].SrcBlend = map_blend(src, 0);
    d.RenderTarget[0].DestBlend = map_blend(dst, 0);
    d.RenderTarget[0].BlendOp = map_blend_op(eq);
    d.RenderTarget[0].SrcBlendAlpha = map_blend(src, 1);
    d.RenderTarget[0].DestBlendAlpha = map_blend(dst, 1);
    d.RenderTarget[0].BlendOpAlpha = map_blend_op(eq);
    d.RenderTarget[0].RenderTargetWriteMask = mask;
    if (FAILED(ID3D11Device_CreateBlendState(s_dev, &d, &bs)))
        return NULL;
    state_add(s_blend, &s_nblend, key, bs);
    return bs;
}

static ID3D11DepthStencilState *ds_state(const Nv2aRenderState *rs, int depth)
{
    uint32_t key[6];
    ID3D11DepthStencilState *ds;
    D3D11_DEPTH_STENCIL_DESC d;

    key[0] = depth ? (rs->depth_test_enable ? 1u : 0u) | (rs->depth_write ? 2u : 0u)
                     | (rs->stencil_enable ? 4u : 0u) : 0u;
    key[1] = depth ? rs->depth_func : 0;
    key[2] = depth ? rs->stencil_func | (rs->stencil_func_mask & 0xFF) << 16
                     | (rs->stencil_write_mask & 0xFF) << 24 : 0;
    key[3] = depth ? rs->stencil_fail : 0;
    key[4] = depth ? rs->stencil_zfail : 0;
    key[5] = depth ? rs->stencil_zpass : 0;
    ds = (ID3D11DepthStencilState *)state_find(s_dss, s_ndss, key);
    if (ds)
        return ds;
    memset(&d, 0, sizeof d);
    if (depth) {
        /* NV2A as OpenGL: with the test off, depth is not written either. */
        d.DepthEnable = rs->depth_test_enable ? TRUE : FALSE;
        d.DepthWriteMask = rs->depth_write ? D3D11_DEPTH_WRITE_MASK_ALL
                                           : D3D11_DEPTH_WRITE_MASK_ZERO;
        d.DepthFunc = map_cmp(rs->depth_func);
        d.StencilEnable = rs->stencil_enable ? TRUE : FALSE;
        d.StencilReadMask = (UINT8)(rs->stencil_func_mask & 0xFF);
        d.StencilWriteMask = (UINT8)(rs->stencil_write_mask & 0xFF);
        d.FrontFace.StencilFunc = map_cmp(rs->stencil_func);
        d.FrontFace.StencilFailOp = map_sop(rs->stencil_fail);
        d.FrontFace.StencilDepthFailOp = map_sop(rs->stencil_zfail);
        d.FrontFace.StencilPassOp = map_sop(rs->stencil_zpass);
        d.BackFace = d.FrontFace;
    } else {
        d.DepthFunc = D3D11_COMPARISON_ALWAYS;
        d.FrontFace.StencilFunc = d.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
    }
    if (FAILED(ID3D11Device_CreateDepthStencilState(s_dev, &d, &ds)))
        return NULL;
    state_add(s_dss, &s_ndss, key, ds);
    return ds;
}

static D3D11_TEXTURE_ADDRESS_MODE map_addr(uint32_t m)
{
    switch (m) {
    case 2:  return D3D11_TEXTURE_ADDRESS_MIRROR;
    case 3:  case 5: return D3D11_TEXTURE_ADDRESS_CLAMP;
    case 4:  return D3D11_TEXTURE_ADDRESS_BORDER;
    default: return D3D11_TEXTURE_ADDRESS_WRAP;
    }
}

static ID3D11SamplerState *sampler(const GgTexture *t)
{
    uint32_t mag = (t->filter >> 24) & 0xF, min = (t->filter >> 16) & 0xFF;
    uint32_t lin = (mag == 2 || min == 2 || min == 4 || min == 6) ? 1u : 0u;
    uint32_t key[6] = { t->addr_u, t->addr_v, lin, 0, 0, 0 };
    ID3D11SamplerState *ss = (ID3D11SamplerState *)state_find(s_smp, s_nsmp, key);
    D3D11_SAMPLER_DESC d;

    if (ss)
        return ss;
    memset(&d, 0, sizeof d);
    d.Filter = lin ? D3D11_FILTER_MIN_MAG_MIP_LINEAR : D3D11_FILTER_MIN_MAG_MIP_POINT;
    d.AddressU = map_addr(t->addr_u);
    d.AddressV = map_addr(t->addr_v);
    d.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    d.ComparisonFunc = D3D11_COMPARISON_NEVER;
    d.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(ID3D11Device_CreateSamplerState(s_dev, &d, &ss)))
        return NULL;
    state_add(s_smp, &s_nsmp, key, ss);
    return ss;
}

/* ---- batches ------------------------------------------------------------ */

static float depth_scale(uint32_t zeta_format)
{
    return zeta_format == 1 ? 1.0f / 65535.0f : 1.0f / 16777215.0f;
}

static void bind_target(GgTarget *t, GgDepth *d, const GgDraw *g)
{
    D3D11_VIEWPORT vp;
    D3D11_RECT sc;
    uint32_t x1 = g->clip_x + g->clip_w, y1 = g->clip_y + g->clip_h;

    ID3D11DeviceContext_OMSetRenderTargets(s_ctx, 1, &t->rtv, d ? d->dsv : NULL);
    vp.TopLeftX = 0; vp.TopLeftY = 0;
    vp.Width = (float)t->w; vp.Height = (float)t->h;
    vp.MinDepth = 0.0f; vp.MaxDepth = 1.0f;
    ID3D11DeviceContext_RSSetViewports(s_ctx, 1, &vp);
    sc.left = (LONG)(g->clip_x < t->w ? g->clip_x : t->w);
    sc.top = (LONG)(g->clip_y < t->h ? g->clip_y : t->h);
    sc.right = (LONG)(x1 < t->w ? x1 : t->w);
    sc.bottom = (LONG)(y1 < t->h ? y1 : t->h);
    if (g->wclip) {
        if ((LONG)g->wx0 > sc.left) sc.left = (LONG)g->wx0;
        if ((LONG)g->wy0 > sc.top) sc.top = (LONG)g->wy0;
        if ((LONG)g->wx1 < sc.right) sc.right = (LONG)g->wx1;
        if ((LONG)g->wy1 < sc.bottom) sc.bottom = (LONG)g->wy1;
        if (sc.right < sc.left) sc.right = sc.left;
        if (sc.bottom < sc.top) sc.bottom = sc.top;
    }
    ID3D11DeviceContext_RSSetScissorRects(s_ctx, 1, &sc);
}

static GgConsts s_k;

int gg_begin(const GgDraw *g)
{
    GgTarget *t;
    GgDepth *d = NULL;
    ID3D11ShaderResourceView *srv[8];
    ID3D11SamplerState *smp[4];
    ID3D11BlendState *bs;
    ID3D11DepthStencilState *dss;
    float bf[4];
    uint32_t st, i;

    if (!gg_enabled() || !gg_ready())
        return 0;
    if ((g->bpp != 4 && g->bpp != 2) || (!g->swizzle && (!g->pitch || !g->height
        || g->pitch % g->bpp)))
        return 0;
    if (s_nverts)
        flush_draw();
    t = target_get(g->color_va, g->pitch, g->bpp, g->height, g->swizzle, g->sw_w, g->sw_h);
    if (!t)
        return 0;
    if (g->zeta_va)
        d = depth_get(g->zeta_va, t->w, t->h);

    memset(&s_k, 0, sizeof s_k);
    memset(srv, 0, sizeof srv);
    memset(smp, 0, sizeof smp);
    for (st = 0; st < 4; st++) {
        const GgTexture *x = &g->tex[st];
        if (!((g->stage_on >> st) & 1))
            continue;
        if (!x->swizzled)
            s_k.ctl2[2] |= 1u << st;
        if (x->cube)
            s_k.ctl2[3] |= 1u << st;
        s_k.tsize[st][0] = (float)x->width;
        s_k.tsize[st][1] = (float)x->height;
        s_k.tscale[st][0] = s_k.tscale[st][1] = 1.0f;
        {
            GgTarget *rt = texture_target(x, t);
            if (rt) {
                srv[st] = rt->srv;
                s_k.tsize[st][2] = 1.0f;
                s_k.tscale[st][0] = (float)x->width / (float)rt->w;
                s_k.tscale[st][1] = (float)x->height / (float)rt->h;
                s_k.tscale[st][2] = (x->color == 0x07 || x->color == 0x1E) ? 1.0f : 0.0f;
                smp[st] = sampler(x);
                s_stat.rt_direct++;
                continue;
            }
        }
        srv[x->cube ? 4 + st : st] = tex_get(x);
        if (srv[x->cube ? 4 + st : st]) {
            s_k.tsize[st][2] = 1.0f;
            smp[st] = sampler(x);
        } else if (s_prof_on) {
            seen_add(s_seen_fail, x->color | (st << 8) | (x->cube << 12));
        }
    }
    if (s_prof_on)
        seen_add(s_seen_sp, g->use_rc ? g->rc.stage_program : 0xFFFFFFFFu);
    /* Bring-up switches for a surface that comes out wrong:
     * GENRY_DBG_WHITE=<mask> samples the masked stages as white,
     * GENRY_DBG_FLAT=1 draws diffuse colour only (no textures, no combiners),
     * GENRY_DBG_NOVCOL=1 replaces diffuse and specular with white. */
    {
        static int init;
        static uint32_t white;
        static int flat, novcol;
        if (!init) {
            const char *e = getenv("GENRY_DBG_WHITE");
            init = 1;
            white = e ? (uint32_t)strtoul(e, NULL, 0) : 0;
            flat = getenv("GENRY_DBG_FLAT") != NULL;
            novcol = getenv("GENRY_DBG_NOVCOL") != NULL;
            s_dbg_novcol = novcol;
        }
        for (st = 0; st < 4; st++)
            if ((white >> st) & 1)
                s_k.tsize[st][2] = 0.0f;
        s_dbg_flat = flat;
    }
    /* Texture reads may have pulled surfaces back from the GPU; the target
     * itself must hold what guest memory holds before anything draws. */
    if (!t->gpu_valid) {
        target_upload(t);
        t->gpu_valid = 1;
    }
    bind_target(t, d, g);
    ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 8, srv);
    ID3D11DeviceContext_PSSetSamplers(s_ctx, 0, 4, smp);

    memcpy(s_k.cicw, g->rc.color_icw, 32); memcpy(s_k.aicw, g->rc.alpha_icw, 32);
    memcpy(s_k.cocw, g->rc.color_ocw, 32); memcpy(s_k.aocw, g->rc.alpha_ocw, 32);
    memcpy(s_k.fac0, g->rc.factor0, 32);   memcpy(s_k.fac1, g->rc.factor1, 32);
    s_k.fin[0] = g->rc.final0; s_k.fin[1] = g->rc.final1;
    s_k.fin[2] = g->rc.final_c0; s_k.fin[3] = g->rc.final_c1;
    s_k.ctl[0] = g->rc.control;
    s_k.ctl[1] = g->rc.stage_program;
    s_k.ctl[2] = (g->use_rc ? GGF_RC : 0)
               | (!g->use_rc && (g->stage_on & 1) ? GGF_TEX0 : 0)
               | (g->rs.alpha_test_enable ? GGF_ALPHA : 0);
    if (s_dbg_flat)
        s_k.ctl[2] &= GGF_ALPHA;
    s_k.ctl[3] = g->rs.alpha_func;
    s_k.ctl2[0] = g->rs.alpha_ref;
    s_k.ctl2[1] = g->clip_plane_mode;
    for (i = 0; i < 4; i++)
        s_k.fogc[i] = g->fog_color[i];
    s_k.vp[0] = (float)t->w;
    s_k.vp[1] = (float)t->h;
    s_k.vp[2] = depth_scale(g->zeta_format);
    /* The near plane (z >= 0 in clip space), as the NV2A clips: the CPU-side
     * clipper only stops at the eye (w > 0), and geometry between the eye and
     * the near plane projected into slabs across the screen -- a limb of a
     * Sonic Heroes model passing the camera, for a frame at a time.
     * GENRY_NO_NEAR_CLIP=1 turns it off. */
    {
        static int off = -1;
        if (off < 0)
            off = getenv("GENRY_NO_NEAR_CLIP") != NULL;
        s_k.vp[3] = off ? 0.0f : 1.0f;
    }
    ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)s_cb, 0, NULL,
                                          &s_k, 0, 0);

    bs = blend_state(g->rs.blend_enable, g->rs.blend_src, g->rs.blend_dst,
                     g->rs.blend_eq, write_mask(g->rs.color_mask));
    bf[0] = (float)((g->rs.blend_color >> 16) & 0xFF) / 255.0f;
    bf[1] = (float)((g->rs.blend_color >>  8) & 0xFF) / 255.0f;
    bf[2] = (float)( g->rs.blend_color        & 0xFF) / 255.0f;
    bf[3] = (float)( g->rs.blend_color >> 24        ) / 255.0f;
    if (g->rs.blend_src == 0x8003 || g->rs.blend_src == 0x8004
        || g->rs.blend_dst == 0x8003 || g->rs.blend_dst == 0x8004)
        bf[0] = bf[1] = bf[2] = bf[3];      /* CONSTANT_ALPHA everywhere */
    ID3D11DeviceContext_OMSetBlendState(s_ctx, bs, bf, 0xFFFFFFFFu);
    dss = ds_state(&g->rs, d != NULL);
    ID3D11DeviceContext_OMSetDepthStencilState(s_ctx, dss, g->rs.stencil_ref & 0xFF);

    s_cur = t;
    s_nverts = 0;
    return 1;
}

static void flush_draw(void)
{
    D3D11_MAPPED_SUBRESOURCE m;
    UINT bytes = s_nverts * (UINT)sizeof(GgVertex);
    D3D11_MAP how = D3D11_MAP_WRITE_NO_OVERWRITE;

    if (!s_nverts || !s_cur)
        return;
    if (s_vb_used + bytes > GG_VB_BYTES) {
        how = D3D11_MAP_WRITE_DISCARD;
        s_vb_used = 0;
    }
    if (FAILED(ID3D11DeviceContext_Map(s_ctx, (ID3D11Resource *)s_vb, 0, how, 0, &m))) {
        s_nverts = 0;
        return;
    }
    memcpy((uint8_t *)m.pData + s_vb_used, s_verts, bytes);
    ID3D11DeviceContext_Unmap(s_ctx, (ID3D11Resource *)s_vb, 0);
    ID3D11DeviceContext_Draw(s_ctx, s_nverts, s_vb_used / (UINT)sizeof(GgVertex));
    s_vb_used += bytes;
    s_stat.draws++;
    s_stat.tris += s_nverts / 3;
    s_cur->gpu_newer = 1;
    s_nverts = 0;
}

void gg_tri(const GgVertex *a, const GgVertex *b, const GgVertex *c)
{
    if (s_nverts + 3 > GG_CHUNK_VERTS)
        flush_draw();
    s_verts[s_nverts++] = *a;
    s_verts[s_nverts++] = *b;
    s_verts[s_nverts++] = *c;
    if (s_dbg_novcol) {
        uint32_t k, j;
        for (k = s_nverts - 3; k < s_nverts; k++)
            for (j = 0; j < 4; j++)
                s_verts[k].d0[j] = s_verts[k].d1[j] = 1.0f;
    }
}

void gg_end(void)
{
    flush_draw();
}

int gg_clear(const GgDraw *g, uint32_t flags, uint32_t argb, float depth01,
             uint32_t stencil)
{
    GgTarget *t = NULL;
    GgDepth *d;
    uint32_t w, h;

    if (!gg_enabled() || !gg_ready())
        return 0;
    if ((g->bpp != 4 && g->bpp != 2) || (!g->swizzle && (!g->pitch || !g->height
        || g->pitch % g->bpp)))
        return 0;
    if (s_nverts)
        flush_draw();
    w = g->swizzle ? g->sw_w : g->pitch / g->bpp;
    h = g->swizzle ? g->sw_h : g->height;
    s_stat.clears++;
    if (flags & 0xF0) {
        int whole = g->clip_x == 0 && g->clip_y == 0 && g->clip_w >= w
                 && g->clip_h >= h && (flags & 0xF0) == 0xF0;
        uint32_t i;
        t = NULL;
        /* D3D clears a swizzled render target through a linear view of the
         * same memory (a uniform fill does not care about texel order) and
         * then draws into it swizzled. Clearing the swizzled target itself
         * keeps one target per surface instead of two that evict and read
         * each other back every frame. */
        if (whole && !g->swizzle)
            for (i = 0; i < GG_MAX_TARGETS; i++) {
                GgTarget *r = &s_rt[i];
                if (r->tex && r->swz && r->va == g->color_va && r->bpp == g->bpp
                    && r->w * r->h == w * h) {
                    t = r;
                    t->used_frame = s_frame;
                    break;
                }
            }
        if (!t)
            t = target_get(g->color_va, g->pitch, g->bpp, h, g->swizzle, g->sw_w, g->sw_h);
        if (!t)
            return 0;
        if (whole) {
            float c[4];
            c[0] = (float)((argb >> 16) & 0xFF) / 255.0f;
            c[1] = (float)((argb >>  8) & 0xFF) / 255.0f;
            c[2] = (float)( argb        & 0xFF) / 255.0f;
            c[3] = (float)( argb >> 24        ) / 255.0f;
            ID3D11DeviceContext_ClearRenderTargetView(s_ctx, t->rtv, c);
            t->gpu_valid = 1;               /* nothing older survives a clear */
        } else {
            /* Part of the surface or some channels: a quad, the combiners
             * off, the clear colour as diffuse. */
            GgVertex v[4];
            ID3D11ShaderResourceView *none[8] = {0};
            ID3D11BlendState *bs = blend_state(0, 1, 0, 0x8006, (UINT8)((flags >> 4) & 0xF));
            float x0 = (float)g->clip_x, y0 = (float)g->clip_y;
            float x1 = (float)(g->clip_x + g->clip_w), y1 = (float)(g->clip_y + g->clip_h);
            int k;
            if (!t->gpu_valid) {
                target_upload(t);
                t->gpu_valid = 1;
            }
            bind_target(t, NULL, g);
            memset(&s_k, 0, sizeof s_k);
            s_k.vp[0] = (float)t->w; s_k.vp[1] = (float)t->h; s_k.vp[2] = 1.0f;
            ID3D11DeviceContext_UpdateSubresource(s_ctx, (ID3D11Resource *)s_cb, 0,
                                                  NULL, &s_k, 0, 0);
            ID3D11DeviceContext_PSSetShaderResources(s_ctx, 0, 8, none);
            ID3D11DeviceContext_OMSetBlendState(s_ctx, bs, NULL, 0xFFFFFFFFu);
            ID3D11DeviceContext_OMSetDepthStencilState(s_ctx, ds_state(&g->rs, 0), 0);
            memset(v, 0, sizeof v);
            for (k = 0; k < 4; k++) {
                v[k].x = (k & 1) ? x1 : x0;
                v[k].y = (k & 2) ? y1 : y0;
                v[k].w = 1.0f;
                v[k].d0[0] = (float)((argb >> 16) & 0xFF) / 255.0f;
                v[k].d0[1] = (float)((argb >>  8) & 0xFF) / 255.0f;
                v[k].d0[2] = (float)( argb        & 0xFF) / 255.0f;
                v[k].d0[3] = (float)( argb >> 24        ) / 255.0f;
            }
            s_cur = t;
            s_nverts = 0;
            gg_tri(&v[0], &v[1], &v[2]);
            gg_tri(&v[1], &v[3], &v[2]);
            flush_draw();
        }
        t->gpu_newer = 1;
    }
    if ((flags & 0x3) && g->zeta_va) {
        UINT f = ((flags & 1) ? D3D11_CLEAR_DEPTH : 0) | ((flags & 2) ? D3D11_CLEAR_STENCIL : 0);
        if (t) { w = t->w; h = t->h; }
        d = depth_get(g->zeta_va, w, h);
        if (d)
            ID3D11DeviceContext_ClearDepthStencilView(s_ctx, d->dsv, f,
                depth01 < 0.0f ? 0.0f : (depth01 > 1.0f ? 1.0f : depth01),
                (UINT8)stencil);
    }
    return 1;
}

void gg_frame_end(uint32_t present_va)
{
    static uint32_t last_report;
    uint32_t i;

    if (!s_dev)
        return;
    if (s_nverts)
        flush_draw();
    for (i = 0; i < GG_MAX_TARGETS; i++) {
        GgTarget *t = &s_rt[i];
        if (!t->tex)
            continue;
        /* The surface being shown goes back to guest RAM for the window;
         * the others stay on the GPU, where the next frame samples and
         * draws them, and come back only if the CPU side asks
         * (gg_sync_range). The shown one is re-read from guest RAM when
         * next drawn into, as anything outside the executor may have
         * written it meanwhile. */
        if (ranges_meet(t->va, t->pitch * t->h, present_va, 4)) {
            /* GENRY_DBG_NO_PRESENT_READBACK=1: measure a frame without the
             * synchronous read of the shown surface (the window then shows
             * stale memory) */
            static int skip_rb = -1;
            if (skip_rb < 0) skip_rb = getenv("GENRY_DBG_NO_PRESENT_READBACK") != NULL;
            if (skip_rb)
                continue;
            if (t->gpu_newer)
                target_readback(t);
            t->gpu_valid = 0;
        }
        if (t->used_frame + 600 < s_frame)
            target_release(t);
    }
    for (i = 0; i < GG_MAX_DEPTHS; i++) {
        GgDepth *d = &s_ds[i];
        if (d->tex && d->used_frame + 600 < s_frame) {
            ID3D11DepthStencilView_Release(d->dsv);
            ID3D11Texture2D_Release(d->tex);
            memset(d, 0, sizeof *d);
        }
    }
    if (s_tex_live > GG_TEX_SLOTS * 3 / 4) {
        for (i = 0; i < GG_TEX_SLOTS; i++)
            if (s_tex[i].live && s_tex[i].used_frame + 120 < s_frame) {
                tex_release(&s_tex[i]);
                s_tex_live--;
            }
    }
    s_frame++;
    if (getenv("GENRY_PROF") && s_frame - last_report >= 120) {
        fprintf(stderr, "  [GGPU] %u frames: draws %llu tris %llu clears %llu"
                        " uploads %llu readbacks %llu (%.1f ms) decodes %llu"
                        " textures %u targets-as-textures %llu\n", s_frame - last_report,
                (unsigned long long)s_stat.draws, (unsigned long long)s_stat.tris,
                (unsigned long long)s_stat.clears,
                (unsigned long long)s_stat.uploads,
                (unsigned long long)s_stat.readbacks, s_stat.readback_ms,
                (unsigned long long)s_stat.tex_decodes, s_tex_live,
                (unsigned long long)s_stat.rt_direct);
        seen_print("stage program", s_seen_sp);
        seen_print("undecodable texture (fmt|stage<<8|cube<<12)", s_seen_fail);
        fflush(stderr);
        memset(&s_stat, 0, sizeof s_stat);
        last_report = s_frame;
    }
}

#else /* !_WIN32: the CPU rasteriser draws everything */

int  gg_enabled(void) { return 0; }
int  gg_begin(const GgDraw *d) { (void)d; return 0; }
void gg_tri(const GgVertex *a, const GgVertex *b, const GgVertex *c) { (void)a; (void)b; (void)c; }
void gg_end(void) {}
int  gg_clear(const GgDraw *d, uint32_t f, uint32_t c, float z, uint32_t s)
{ (void)d; (void)f; (void)c; (void)z; (void)s; return 0; }
void gg_sync_range(uint32_t va, uint32_t bytes, int inv) { (void)va; (void)bytes; (void)inv; }
void gg_frame_end(uint32_t present_va) { (void)present_va; }

#endif
