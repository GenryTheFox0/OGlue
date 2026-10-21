/* GENRY 2026-10-21 (Шрам): hold A to skip a full-motion video.
 *
 * The title opens its movie as a file (Sofdec .sfd, Bink .bik, .xmv/.wmv) and
 * streams it. Holding A (keyboard Z) for GENRY_SKIP_HOLD_MS while that file is
 * being read fills a ring in the window; when it closes, every further read of
 * the movie file reports end of file, and the player ends the movie the way it
 * ends one that ran out -- the title then goes on as after a movie watched to
 * the end. Nothing in guest memory is touched; the ring is drawn on the
 * window's copy of the frame. GENRY_MOVIE_SKIP=0 turns it off. */
#ifndef GENRY_MOVIE_SKIP_H
#define GENRY_MOVIE_SKIP_H
#include <stdint.h>

/* kernel_bridge.c: a guest file handle opened / closed / about to be read.
 * genry_movie_read returns 1 when the read is to report end of file. */
void genry_movie_opened(uint32_t token, const char *guest_path);
void genry_movie_closed(uint32_t token);
int  genry_movie_read(uint32_t token);

/* The title's input poll: is A (or whatever skips) held right now. */
void genry_movie_skip_input(int skip_down);

/* fb_present.c: draw the ring over a finished frame (0xAARRGGBB rows). */
void genry_movie_overlay(uint32_t *argb, uint32_t w, uint32_t h);

#endif
