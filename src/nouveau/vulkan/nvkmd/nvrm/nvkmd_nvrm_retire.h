/*
 * Copyright © 2026 Techniix
 * SPDX-License-Identifier: MIT
 */

/* Deferred reuse of sync-pool slots.
 *
 * A vk_sync's pool slot is a GPU semaphore word. Host SEM acquires and
 * releases emitted into a channel name the slot by address and run when the
 * channel reaches them, which can be well after the vk_sync that owned the
 * slot has been destroyed: the Vulkan runtime destroys the temporary syncs it
 * moves binary waits into as soon as the submission has been handed to the
 * driver (vk_queue_submit_cleanup), not when the GPU has run it. If the slot
 * went straight back to the pool, the next allocation would zero it and the
 * queued acquire would wait for a value that never comes, stalling the
 * channel.
 *
 * So a freed slot records, for every channel, the fence by which that
 * channel will have executed everything already emitted (its "retire
 * point"), and is reused only once each of those channels has completed it.
 *
 * Kept free of driver types so tests can exercise it directly. */

#ifndef NVKMD_NVRM_RETIRE_H
#define NVKMD_NVRM_RETIRE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

struct nvrm_retire_point {
   const void *chan;   /* NULL once the channel has been destroyed */
   uint64_t point;     /* fence value the channel must complete */
};

struct nvrm_retire_entry {
   void *mem;          /* opaque: the pool page's (or dedicated) memory */
   uint64_t offset;    /* slot offset within mem */
   uint32_t kind;      /* opaque to this header: what to do once ready */
   uint32_t num_points;
   struct nvrm_retire_point points[];
};

typedef uint64_t (*nvrm_retire_completed_fn)(const void *chan);
typedef uint64_t (*nvrm_retire_point_fn)(const void *chan);

/* Build the retire entry for a slot (or memory) being freed, or return NULL
 * with *ready_now = true when no channel has outstanding work, so it can be
 * reused at once. Channels whose completed fence already covers their retire
 * point are left out. Each channel is read exactly once, so a channel that
 * advances concurrently can't push another one out of the entry. On
 * allocation failure, returns NULL with *ready_now = false: the caller must
 * then never reuse the slot (leak it) rather than risk it. */
static inline struct nvrm_retire_entry *
nvrm_retire_entry_create(void *mem, uint64_t offset, uint32_t kind,
                         const void *const *chans, uint32_t num_chans,
                         nvrm_retire_completed_fn completed,
                         nvrm_retire_point_fn retire_point,
                         bool *ready_now)
{
   *ready_now = false;
   struct nvrm_retire_entry *e =
      malloc(sizeof(*e) + (size_t)num_chans * sizeof(e->points[0]));
   if (e == NULL)
      return NULL;

   e->mem = mem;
   e->offset = offset;
   e->kind = kind;
   e->num_points = 0;
   for (uint32_t i = 0; i < num_chans; i++) {
      uint64_t point = retire_point(chans[i]);
      if (completed(chans[i]) < point) {
         e->points[e->num_points].chan = chans[i];
         e->points[e->num_points].point = point;
         e->num_points++;
      }
   }

   if (e->num_points == 0) {
      free(e);
      *ready_now = true;
      return NULL;
   }
   return e;
}

/* True once every still-existing channel recorded in the entry has completed
 * its retire point. */
static inline bool
nvrm_retire_entry_ready(const struct nvrm_retire_entry *e,
                        nvrm_retire_completed_fn completed)
{
   for (uint32_t i = 0; i < e->num_points; i++) {
      const struct nvrm_retire_point *p = &e->points[i];
      if (p->chan != NULL && completed(p->chan) < p->point)
         return false;
   }
   return true;
}

/* A channel is going away: its work can no longer touch any slot, so drop it
 * from the entry. */
static inline void
nvrm_retire_entry_forget_chan(struct nvrm_retire_entry *e, const void *chan)
{
   for (uint32_t i = 0; i < e->num_points; i++) {
      if (e->points[i].chan == chan)
         e->points[i].chan = NULL;
   }
}

#endif /* NVKMD_NVRM_RETIRE_H */
