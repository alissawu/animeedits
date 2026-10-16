# Word emphasis reference check

Run on the existing /home/kit/ref_edits corpus with tools/validate_word_stress.py.
The script only reads cached Demucs vocals; no inference or downloads. It adds
stress annotations to the five available checked-reference timeline.json files
and writes word_stress_validation.json and word_stress_hits.json outside git.

There are 217 aligned words across five edits. Four other edits have no
checked-reference alignment and were excluded, rather than scoring Whisper text.
The input all_cuts.csv is unchanged. For rows with specific_lyric_word_match=True,
the nearest aligned start/end within 80 ms is selected using the original
summarizer's ordering. The script checks that the CSV's text/edge/uncertainty
target matches exactly, including ties. Repeated cut hits on a word remain
repeated observations, not unique words.

| Edit | Word-boundary cut hits | High-stress hits | All-word baseline |
|---|---:|---:|---:|
| August 10 | 2 | 0 | 28.6% |
| Cliche | 37 | 11 | 22.7% |
| Meant To Be | 0 | 0 | 29.6% |
| Under Your Spell | 4 | 2 | 22.2% |
| Prom Song | 7 | 2 | 23.5% |

15 of 50 hits land on high-stress words, 30.0%, versus a within-edit
cut-count-weighted all-word baseline of 23.0% (11.49 expected hits). This is 1.31x
enrichment. A reproducible 20,000-draw uniform-word Monte Carlo null, sampling with
replacement within each edit with seed 20261016, gives one-sided p=0.1475.
Mean continuous stress is 0.527 at hits versus 0.507 under that null.

The CSV includes both starts and ends. Start hits are 7/22 high stress, and end
hits are 8/28. Excluding uncertain matched words leaves 7/23 high stress (30.4%)
versus 5.60 expected (24.3%) from that edit's certain-word pool.

This is weak directional evidence, not confirmation that editors choose stressed
words. Cliche supplies 37 of 50 hits, 27 hits are uncertain, and cut detection
undercounts Meant To Be. The heuristic is not calibrated against human labels.
The null controls edit identity, not rhythm, line position, duration, local
clustering or differing boundary coverage. It tests prominence among word hits,
not whether cuts select words more often than random times. There is no basis
here to promote this feature into a mandatory cut rule.
