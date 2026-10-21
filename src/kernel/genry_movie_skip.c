/* GENRY 2026-10-21 (Шрам): hold A to skip a full-motion video -- see
 * genry_movie_skip.h. The overlay is a dark disc in the lower right corner with
 * the green A button in it, a red ring that fills while A is held, chevrons
 * that run towards the button and the word ПРОПУСК; when the ring closes it
 * flashes and fades out. */
#include "genry_movie_skip.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
static uint64_t ms_now(void) { return GetTickCount64(); }
#else
#include <time.h>
static uint64_t ms_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}
#endif

static volatile uint32_t s_token;       /* guest handle of the movie file, 0: none */
static volatile uint64_t s_opened_ms, s_read_ms;
static volatile int      s_skip;        /* reads report end of file */
static volatile uint64_t s_skip_ms;
static volatile uint64_t s_hold_ms;     /* A went down, 0: not held */
static volatile int      s_down;

static int skip_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("GENRY_MOVIE_SKIP");
        on = !(e && atoi(e) == 0);
    }
    return on;
}

static uint32_t hold_len(void)
{
    static int ms = -1;
    if (ms < 0) {
        const char *e = getenv("GENRY_SKIP_HOLD_MS");
        ms = e ? atoi(e) : 900;
        if (ms < 100) ms = 100;
    }
    return (uint32_t)ms;
}

static int is_movie_path(const char *p)
{
    static const char *ext[] = { ".sfd", ".bik", ".xmv", ".wmv" };
    size_t n = p ? strlen(p) : 0, i, k;
    for (i = 0; i < sizeof ext / sizeof ext[0]; i++) {
        size_t m = strlen(ext[i]);
        if (n < m) continue;
        for (k = 0; k < m && tolower((unsigned char)p[n - m + k]) == ext[i][k]; k++)
            ;
        if (k == m) return 1;
    }
    return 0;
}

void genry_movie_opened(uint32_t token, const char *guest_path)
{
    if (!skip_on() || !token || !is_movie_path(guest_path))
        return;
    s_skip = 0;
    s_hold_ms = 0;
    s_opened_ms = s_read_ms = ms_now();
    s_token = token;
}

void genry_movie_closed(uint32_t token)
{
    if (token && token == s_token) {
        s_token = 0;
        s_skip = 0;
        s_hold_ms = 0;
    }
}

int genry_movie_read(uint32_t token)
{
    if (!token || token != s_token)
        return 0;
    s_read_ms = ms_now();
    return s_skip;
}

/* The movie is on screen: its file is open and was read lately (the player
 * reads ahead about a second of video at a time). */
static int playing(uint64_t now)
{
    return s_token && now - s_read_ms < 4000u;
}

void genry_movie_skip_input(int down)
{
    uint64_t now = ms_now();

    s_down = down;
    if (!skip_on() || !playing(now) || s_skip || !down) {
        s_hold_ms = 0;
        return;
    }
    if (!s_hold_ms)
        s_hold_ms = now;
    if (now - s_hold_ms >= hold_len()) {
        s_skip = 1;
        s_skip_ms = now;
        s_hold_ms = 0;
        fprintf(stderr, "[MOVIE] skipped by the player: handle %08X reads now end of file\n",
                (unsigned)s_token);
        fflush(stderr);
    }
}

/* ---- drawing ----------------------------------------------------------- */

static float clampf(float v, float a, float b) { return v < a ? a : v > b ? b : v; }

/* 0..1 coverage of a pixel at distance d from an edge, 1 px soft. */
static float cover(float d) { return clampf(0.5f - d, 0.0f, 1.0f); }

static void blend(uint32_t *p, uint32_t rgb, float a)
{
    uint32_t d = *p;
    float r, g, b;
    if (a <= 0.0f) return;
    if (a > 1.0f) a = 1.0f;
    r = ((d >> 16) & 0xFF) * (1.0f - a) + ((rgb >> 16) & 0xFF) * a;
    g = ((d >> 8) & 0xFF) * (1.0f - a) + ((rgb >> 8) & 0xFF) * a;
    b = (d & 0xFF) * (1.0f - a) + (rgb & 0xFF) * a;
    *p = (d & 0xFF000000u) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}

/* 5x7 glyphs, top row first, bit 4 = left column: П Р О У С К and A. */
static const uint8_t G_PE[7] = { 0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 };
static const uint8_t G_ER[7] = { 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10 };
static const uint8_t G_O[7]  = { 0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E };
static const uint8_t G_U[7]  = { 0x11, 0x11, 0x0A, 0x04, 0x04, 0x08, 0x10 };
static const uint8_t G_ES[7] = { 0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E };
static const uint8_t G_KA[7] = { 0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11 };
static const uint8_t G_A[7]  = { 0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11 };

static void glyph(uint32_t *argb, uint32_t w, uint32_t h, int x0, int y0, int s,
                  const uint8_t g[7], uint32_t rgb, float a)
{
    int r, c, dx, dy;
    for (r = 0; r < 7; r++)
        for (c = 0; c < 5; c++)
            if (g[r] & (0x10 >> c))
                for (dy = 0; dy < s; dy++)
                    for (dx = 0; dx < s; dx++) {
                        int x = x0 + c * s + dx, y = y0 + r * s + dy;
                        if (x >= 0 && y >= 0 && (uint32_t)x < w && (uint32_t)y < h)
                            blend(&argb[(size_t)y * w + x], rgb, a);
                    }
}

void genry_movie_overlay(uint32_t *argb, uint32_t w, uint32_t h)
{
    const float PI = 3.14159265f;
    uint64_t now = ms_now();
    float prog, alpha, flash = 0.0f, t;
    int skipping = s_skip && s_token;
    float cx, cy, R, x, y;
    int ix, iy, x0, x1, y0, y1;

    if (!skip_on() || !argb || w < 64 || h < 64)
        return;
    if (skipping) {
        uint64_t since = now - s_skip_ms;
        if (since > 900u) return;                    /* faded out */
        prog = 1.0f;
        flash = 1.0f - since / 900.0f;
        alpha = since < 450u ? 1.0f : 1.0f - (since - 450u) / 450.0f;
    } else if (s_hold_ms && playing(now)) {
        prog = clampf((float)(now - s_hold_ms) / (float)hold_len(), 0.0f, 1.0f);
        alpha = clampf(0.35f + prog * 3.0f, 0.0f, 1.0f);
    } else if (playing(now) && now - s_opened_ms < 3500u) {
        /* a hint while the movie starts: the button, dim, no ring */
        uint64_t since = now - s_opened_ms;
        prog = 0.0f;
        alpha = since < 500u ? since / 500.0f * 0.55f
              : since > 2800u ? (3500u - since) / 700.0f * 0.55f : 0.55f;
    } else {
        return;
    }

    R = h * 0.055f;                                   /* ~26 px at 480 lines */
    cx = w - R * 2.2f;
    cy = h - R * 2.2f;
    x0 = (int)(cx - R * 1.6f); x1 = (int)(cx + R * 1.6f);
    y0 = (int)(cy - R * 1.6f); y1 = (int)(cy + R * 1.6f);
    for (iy = y0; iy <= y1; iy++) {
        if (iy < 0 || (uint32_t)iy >= h) continue;
        for (ix = x0; ix <= x1; ix++) {
            uint32_t *p;
            float d, ang, rr;
            if (ix < 0 || (uint32_t)ix >= w) continue;
            p = &argb[(size_t)iy * w + ix];
            x = ix + 0.5f - cx;
            y = iy + 0.5f - cy;
            d = sqrtf(x * x + y * y);
            /* glow when the ring closes */
            if (flash > 0.0f && d < R * 1.6f)
                blend(p, 0xFF5A3C, flash * 0.45f * clampf(1.0f - (d - R) / (R * 0.6f), 0.0f, 1.0f) * (d > R ? 1.0f : 0.0f));
            /* dark disc */
            blend(p, 0x000000, alpha * 0.62f * cover(d - R));
            /* ring track */
            rr = fabsf(d - R * 0.82f);
            blend(p, 0xFFFFFF, alpha * 0.22f * cover(rr - R * 0.09f));
            /* progress arc: clockwise from the top */
            ang = atan2f(x, -y);                      /* 0 at 12 o'clock */
            if (ang < 0) ang += 2.0f * PI;
            if (prog > 0.0f && ang <= prog * 2.0f * PI)
                blend(p, flash > 0.0f ? 0xFF8A70 : 0xE3241B, alpha * cover(rr - R * 0.11f));
            /* the A button */
            blend(p, 0x2E9E3E, alpha * cover(d - R * 0.56f));
            blend(p, 0x60D46C, alpha * 0.55f * cover(fabsf(d - R * 0.53f) - R * 0.04f) * (y < 0 ? 1.0f : 0.3f));
        }
    }
    {   /* letter A, centred */
        int s = (int)(R / 9.0f) < 1 ? 1 : (int)(R / 9.0f);
        glyph(argb, w, h, (int)(cx - 2.5f * s), (int)(cy - 3.5f * s), s, G_A, 0xFFFFFF, alpha);
    }
    {   /* chevrons running into the button, and ПРОПУСК above them */
        int s = (int)(R / 13.0f) < 1 ? 1 : (int)(R / 13.0f);
        const uint8_t *word[7] = { G_PE, G_ER, G_O, G_PE, G_U, G_ES, G_KA };
        int tw = 7 * 6 * s, k, i;
        int tx = (int)(cx - R * 1.25f) - tw, ty = (int)(cy - 3.5f * s);
        for (k = 0; k < 7; k++)
            glyph(argb, w, h, tx + k * 6 * s, ty - 9 * s, s, word[k], 0xFFFFFF, alpha * 0.9f);
        t = (float)(now % 900u) / 900.0f;
        for (i = 0; i < 3; i++) {
            float ph = fmodf(t + i / 3.0f, 1.0f), a = alpha * (0.25f + 0.75f * ph) * (prog > 0.0f || skipping ? 1.0f : 0.5f);
            float bx = tx + tw - 4.0f * s * 3 + i * 4.0f * s + ph * 2.0f * s, by = (float)ty + 3.5f * s;
            int py;
            for (py = -3 * s; py <= 3 * s; py++) {      /* a ">" of two strokes */
                float off = (float)(py < 0 ? -py : py);
                int px = (int)(bx + 3.0f * s - off), q;
                for (q = 0; q < s + 1; q++) {
                    int X = px + q, Y = (int)by + py;
                    if (X >= 0 && Y >= 0 && (uint32_t)X < w && (uint32_t)Y < h)
                        blend(&argb[(size_t)Y * w + X], 0xE3241B, a);
                }
            }
        }
    }
}
