# SatoSugu, "Under Your Spell" (Desire)

21.8s edit, gif-sourced footage (every image held 3 frames, so 10 new images a second at 30fps).

## Narrative

The edit is a mirror. One lyric, "I don't eat, I don't sleep, I do nothing but think of you," is sung twice: once by Suguru about Satoru, once by Satoru about Suguru. What each one can't stop thinking about shows how he sees the other.

Suguru sees Satoru as overwhelming power. His verse is depression. He's falling apart, haunted, losing faith in what he fights for. His "you" lands on Satoru at full strength, so what consumes him is how strong Satoru has become. The love and the distance arrive together: this is when he realizes they aren't equals anymore. Satoru's power is breathtaking to him and also leaves him behind, and he doesn't blame Satoru for it, so there's no one to be angry at.

Satoru sees Suguru as warmth. His "you" is Suguru smiling softly, the one person who understood and encouraged him, back when he was a person and not just the strongest.

The mirror is the tragedy. Suguru loves an image of power, Satoru loves an image of warmth, and each is fixed on a version of the other rather than who that person became.

The chorus shows where that love ended up. Suguru is dead, and Satoru's love still haunts the story years later. Kenjaku doesn't beat the strongest sorcerer with strength. He only has to wear Suguru's face and smile the way Suguru smiled. "You keep me under your spell" stays in Satoru's POV, but the spell was never Suguru's. It's Satoru's own love, outliving the person and turned into the trap. The sealing only works because of how deep it goes.

## Song reading and clip choice

Narrative decides how each lyric is read, and the reading decides the clip.

Suguru's verse is the strong half. "I don't eat, I don't sleep" is figurative depression with real signs behind it (Shoko asks if he's lost weight in the same S2E5 scene shot 3 comes from).

Satoru's verse is loose. The clips read more as "here's Gojo" than as matches to the lyric: a fan animation, the island, the prison realm, an aura shot. Only the prison realm earns its spot, and that's because it foreshadows the ending, not because of the lyric. The verse works because of the POV flip, not the clips.

Clips were likely limited to existing gifs, so the editor picked by mood and character rather than exact moments. Suguru's shots land partly because his popular gifs come from his breakdown arc. Don't read too much into individual clip choices; the structure gets the credit.

## Cutting

Each cut lands on the beat just before a lyric phrase, so the picture arrives a hair before the words. This holds in both verses.

| # | Time | Len | Shot | Lyric / cue |
|---|---|---|---|---|
| 1 | 0.0-2.4 | 2.4s | Geto stunned in the Yuki talk (S2E5) | fades up, "I don't eat" |
| 2 | 2.4-4.0 | 1.6s | Geto (S2E5, ~07:15, Yuki talk) | beat at 2.39, before "don't sleep" 2.68 |
| 3 | 4.0-6.5 | 2.5s | Geto after "Satoru had become the strongest" (S2E5 ~02:46) | "I do nothing but think of" |
| 4 | 6.5-9.2 | 2.7s | Gojo power, fan animation | "youuu" = Satoru |
| 5 | 9.2-11.4 | 2.2s | Teen Gojo on the island (Hidden Inventory) | before "don't eat" 9.48 |
| 6 | 11.4-13.0 | 1.6s | Gojo sealed in the prison realm | before "don't sleep" 11.76 |
| 7 | 13.0-15.6 | 2.6s | Gojo aura (popular scene) | before "do nothing" 13.22 |
| 8 | 15.6-18.2 | 2.6s | Teen Geto smiling at Gojo | as "you" 15.34 rings out = Suguru |
| 9 | 18.2-19.6 | 1.4s | Kenjaku smiling at Gojo | "you keep me" |
| 10 | 19.6-20.9 | 1.3s | Gojo shocked, right after 9 | "under your spell" |
| 11 | 20.9-21.8 | 0.9s | Gojo shocked | repeat "keep", audio cuts off |

The shorter shots at the end mostly come from the source scene's own fast cutting and the video ending, not a deliberate speed-up. The real choice was landing the chorus on that moment.

## Treatment

Minimal. The opening fades up from slightly dark over ~0.9s. Every other cut is hard, with no transitions or white flashes, just big brightness jumps between shots (bright Gojo, dark prison realm, bright aura). Shots 3, 9, 10 and 11 are nearly still. Shots 4 and 5 have real motion. Remaining noise and flicker look like gif quality, not editing effects. Footage is 10fps; whether any clip is slowed is unverified without the source episodes. No audio edits.

---

# Frame-level findings + what to verify against source

Edit: satosugu-underyourspell-desire.mp4 (21.81s, 960x720, file says 30fps). Full breakdown: breakdowns/satosugu-under-your-spell.md.

The aim is to (1) confirm these findings against the S2 episodes and (2) turn the measurements into a standard part of edit analysis (frame cadence, speed vs source, flicker/flash detection).

## How the measurements were taken

Per-frame stats came from ffmpeg signalstats:

    ffmpeg -i edit.mp4 -vf "signalstats,metadata=print:file=luma.txt" -f null -

- YAVG = average brightness of the frame (0-255). Use it for fades, flashes to white/black, and flicker.
- YDIF = average pixel difference from the previous frame. 0 means the frame repeats the previous one, small means noise or a slight drift, large means motion, and a spike means a cut.

Cuts are YDIF spikes (roughly >40) that come with a jump in YAVG.

## Findings

**Frame cadence.** Throughout, YDIF runs change, 0, 0, change, 0, 0. Each image is held for 3 frames, so this file has 10 new images per second.

This does NOT show the edit was slowed down. The 30fps is only the export rate. The episodes are ~24fps, and anime is often drawn on 2s or 3s (about 8-12 new drawings per second). A gif can drop frames and still play at real speed. "Slowed" means a moment lasts longer than in the source, and that can only be checked by comparing durations.

**Cuts (s):** 2.40, 4.00, 6.50, 9.20, 11.40, 13.00, 15.60, 18.20, 19.60, 20.90. Each one lands on the beat just before a lyric phrase (beat ~143.6 BPM, e.g. beat 2.39 then "don't sleep" at 2.68).

**Per shot**

| # | Time | Motion (median YDIF of changing frames) | Brightness | Notes |
|---|---|---|---|---|
| 1 | 0.0-2.4 | ~1 after fade | 98 to 121 | fades up over ~0.9s |
| 2 | 2.4-4.0 | ~8 steady | 58 | small, constant change, probably gif noise |
| 3 | 4.0-6.5 | ~0.7 | 56 | near-still |
| 4 | 6.5-9.2 | ~25 | 34-117 | real motion, dips dark at 6.6-8.7 (fan animation?) |
| 5 | 9.2-11.4 | ~16 | 125-138 | real motion |
| 6 | 11.4-13.0 | ~2.6 | 29 | near-still, dark (prison realm) |
| 7 | 13.0-15.6 | ~8 steady | 142-149 | same steady change as shot 2 |
| 8 | 15.6-18.2 | ~6 | 79 | only 12 changing frames out of 78, so possibly a lower rate or partly held |
| 9 | 18.2-19.6 | ~3 | 78-82 | near-still |
| 10 | 19.6-20.9 | ~4 | 75-78 | near-still |
| 11 | 20.9-21.8 | ~1.3 | 115 | near-still, video ends |

**Treatment:** a fade-up at the start, otherwise all hard cuts. There are no white/black flash frames and no transitions. Alissa thinks the noise is gif quality, not an effect. No audio edits.

## Please verify with the episodes

Shot sources (Alissa's IDs):
1. Geto stunned in the Yuki conversation, S2E5 (the "kill all non-sorcerers" line is at ~09:17)
2. S2E5 ~07:15 (Yuki's "world where cursed spirits aren't born")
3. S2E5 ~02:46 ("Satoru had become the strongest")
4. Gojo, probably fan animation (not in the episodes?)
5. Teen Gojo on the island (S2E1-3)
6. Gojo in the prison realm (S2E9-10)
7. Gojo aura scene (maybe S1)
8. Teen Geto smiling at Gojo
9. Kenjaku smiling at Gojo (S2E9)
10-11. Gojo shocked, directly after 9

For each shot:
- Find the exact source timestamp.
- **Speed:** time the same motion (start to end) in the source and in the edit. Equal means real speed, longer in the edit means slowed. Report the ratio.
- **Frame rate:** count distinct drawings per second in the source vs the edit. This tells us whether the gif dropped drawings or the source was already drawn on 2s/3s.
- **Trimming:** which part of the source shot was used, and whether it ends where the source cuts. This decides whether the short shots at 9-11 come from the scene's own pacing or the editor's choice.
- Check whether shot 4's dark dips and the near-still shots match the source, or were added (freeze, flicker).

## Make this standard for edit analysis

Add a per-edit "frame report" step:
1. Cadence per shot: the YDIF repeat pattern, giving new images per second.
2. Cuts: YDIF spike + YAVG jump.
3. Flashes/fades: brief YAVG spikes to near-white or near-black (flash), and gradual YAVG ramps (fade).
4. Flicker: alternating YAVG dips inside a shot.
5. Freezes: runs of YDIF ~0 longer than the cadence.
6. Speed: compare against the matched source segment, as a duration ratio.
7. Audio (when relevant): compare against the original song for volume/bass/edits.
Report the numbers next to the stills, and don't infer intent from them alone. A slow-looking or fast-cut clip may simply be how the source was animated or what gifs were available.
