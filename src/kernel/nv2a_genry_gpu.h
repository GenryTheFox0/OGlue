/*
 * The pixel half of the pushbuffer executor on a D3D11 device.
 *
 * nv2a_pb_exec.c still decodes every batch and runs every vertex -- fixed
 * function, vertex programs, near-plane clipping -- exactly as before. What
 * moves is the per-pixel work that made the CPU path slow: rasterising,
 * texture sampling, the register combiners, fog, alpha test, depth, stencil
 * and blending. The pixel shader is one interpreter over the combiner
 * registers, so no state ever compiles a new shader mid-game.
 *
 * Guest memory stays the truth between frames: a surface is uploaded when the
 * GPU first draws into it, read back when anything on the CPU side needs it
 * (a texture sampled from it, the CPU rasteriser, the flip), and the GPU copy
 * is dropped at every flip. Depth and stencil live only on the GPU.
 *
 * On with GENRY_GPU=1. Everything here runs on the executor's thread.
 */
#ifndef NV2A_GENRY_GPU_H
#define NV2A_GENRY_GPU_H

#include <stdint.h>
#include "nv2a_backend.h"     /* Nv2aRenderState */
#include "nv2a_combiner.h"    /* Nv2aCombiner */

/* One vertex as the CPU rasteriser would interpolate it. */
typedef struct {
    float x, y, z, w;   /* surface pixels, raw depth units, clip w (> 0) */
    float d0[4], d1[4]; /* diffuse, specular: linear in screen space */
    float t[4][4];      /* stage coordinates: perspective-correct */
    float fog;          /* fog factor: linear in screen space */
    float pad[3];
} GgVertex;

typedef struct {
    uint32_t offset;            /* guest VA of texel (0,0), face 0 */
    uint32_t width, height, pitch, color, palette;
    uint32_t addr_u, addr_v, filter;
    uint32_t swizzled;          /* coordinates arrive normalised, not texels */
    uint32_t cube;
    uint32_t bytes;             /* guest bytes the texture spans, all faces */
    /* One face to 0xAARRGGBB, width * height texels; 0 when it cannot. */
    int (*decode)(const void *ctx, uint32_t face, uint32_t *argb);
    const void *ctx;
} GgTexture;

typedef struct {
    uint32_t color_va, pitch, bpp;      /* the colour surface */
    uint32_t height;                    /* rows it spans: clip_y + clip_h */
    /* SURFACE_FORMAT type 2: the surface is stored in Morton (swizzled)
     * order, sw_w x sw_h, and pitch means nothing. A title renders its
     * shadow and reflection textures this way and samples them as
     * swizzled textures. */
    uint32_t swizzle, sw_w, sw_h;
    /* SET_WINDOW_CLIP, inclusive rectangle 0 (D3D SetScissors): pixels
     * outside [wx0, wx1) x [wy0, wy1) are not drawn. */
    uint32_t wclip, wx0, wx1, wy0, wy1;
    uint32_t clip_x, clip_y, clip_w, clip_h;
    uint32_t zeta_va;                   /* 0: no depth buffer */
    uint32_t zeta_format;               /* 1 Z16, 2 Z24S8 */
    Nv2aRenderState rs;
    uint32_t use_rc;                    /* the combiners decide the colour */
    Nv2aCombiner rc;
    uint32_t clip_plane_mode;
    float    fog_color[4];
    uint32_t stage_on;                  /* bit per stage that samples */
    GgTexture tex[4];
} GgDraw;

int  gg_enabled(void);
/* Start a batch; 0 means the GPU cannot take it and the CPU path should. */
int  gg_begin(const GgDraw *d);
void gg_tri(const GgVertex *a, const GgVertex *b, const GgVertex *c);
void gg_end(void);
/* CLEAR_SURFACE: flags Z 0x1, stencil 0x2, colour 0xF0. 0: not handled. */
int  gg_clear(const GgDraw *d, uint32_t flags, uint32_t argb, float depth01,
              uint32_t stencil);
/* Bring every GPU-drawn surface overlapping [va, va+bytes) back to guest
 * memory; with `invalidate` the CPU is about to write it, so the GPU copy is
 * dropped too. */
void gg_sync_range(uint32_t va, uint32_t bytes, int invalidate);
/* The flip: the surface at present_va back to guest memory for the window;
 * the rest stay on the GPU until something on the CPU side needs them. */
void gg_frame_end(uint32_t present_va);

#endif /* NV2A_GENRY_GPU_H */
