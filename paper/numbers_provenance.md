# Where every number in `main.tex` comes from

Re-derived on 2026-10-16 from `results/octvla_results.tar.xz` (731 JSONs,
archived 2026-09-30). Nothing below was copied from a document.

```bash
mkdir -p /tmp/octvla && tar -C /tmp/octvla -xJf results/octvla_results.tar.xz
E=/tmp/octvla/eval
PC="env PYTHONPATH=src python3 scripts/paired_compare.py"
```

| Paper location | Numbers | Command |
|---|---|---|
| Table I, ACT | rgb_cont 21/0/39; kv 22/0/37; kv_adaln 22/7 (p=0.016)/33; kv_tokens 26/0/41; scratch_kv 21→2 | `$PC $E CF-rgb_cont CF-kv CF-kv_adaln CF-kv_tokens CF-scratch_kv` |
| Table I, GR00T; abstract | rgb_cont 9/0/36; kv 31/7/40; kv_adaln 35/11/49 | `$PC $E GF-rgb_cont GF-kv GF-kv_adaln` |
| Table I, pi0.5 | rgb_cont 0/0/9; kv 1/0/21 (p=0.023); kv_adaln 2/0/8 | `$PC $E PXF-rgb_cont PXF-kv PXF-kv_adaln` |
| Table I, SmolVLA | rgb_cont 0/0/13; kv 4/0/17; kv_adaln 3/1/12 | `$PC $E SXF-rgb_cont SXF-kv SXF-kv_adaln` |
| Sec. V, per-seed ranges | kv_adaln seen 11/12/12, held-out 4/6/1; rgb_cont seen 4/2/3; kv held-out 0/5/2 | `collect.load(E,"GF")`, successes grouped by train_seed |
| Sec. V, replications of the AdaLN held-out result | ACT atomic clips 0→6 (p=0.031) | `$PC $E F-rgb_cont F-kv_adaln` |
| Table II, controls | shuffled 8/3; scene_mean 8/1; scene_attn 12/3; shuffled→kv_adaln 8→35; scene_attn→kv 12→31 (p=0.001) | `$PC $E GF-rgb_cont GF-kv_adaln_shuffled GF-scene_mean GF-scene_attn`; `$PC $E GF-kv_adaln_shuffled GF-kv_adaln`; `$PC $E GF-scene_attn GF-kv` |
| Table II, visual shift | rgb_cont 9→2 (p=0.039); kv 31→25; kv_adaln 35→24 (p=0.071) | `$PC $E GF-<arm> vshift/GF-<arm>` |
| Table II, noise | kv/kv_adaln: 32/28, 28/33, 24/23, 25/24, 12/10 | `$PC $E GF-<arm> noise/<level>/GF-<arm>` |
| Table V, GR00T count | 2 objects: 14/47/37; 4 objects: 0/9/4; ≥1 transfer at 4: 37/31/19; kv vs kv_adaln at 4: 31 vs 19 (p=0.043) | `$PC $E GF-rgb_cont GF-kv GF-kv_adaln`; `$PC $E GF-kv GF-kv_adaln` |
| Table V, ACT count | 2 objects: 18/18/16; 4 objects: 0; ≥1 transfer at 4: 22/14/18 | `$PC $E CF-rgb_cont CF-kv CF-kv_adaln` |
| Table I, held-out ≥1 transfer column | ACT 15/17/19/15/3; GR00T 17/30/22; pi0.5 7/3/3; SmolVLA 3/1/2 | same commands as Table I |
| Sec. IV-D, training format | atomic clips → full runs: 1→9 (p=0.008), 2 objects 3→19 (p<0.001) | `$PC $E AF-rgb CF-rgb` |
| Sec. V-A, ACT budget effect | CF-rgb → CF-rgb_cont seen 9→21 (p=0.012) | `$PC $E CF-rgb CF-rgb_cont` |
| Fig. 2, stage survival | rgb_cont .97/.62/.36/.69; kv .87/.77/.85/.91; kv_adaln .95/.86/.78/.92 | `collect_final_results.py --prefix GF $E`, STAGES section |
| Table III, data size | 19 runs 2→8, 0→4; 38 runs 4→28, 1→15; held-out ≥1 transfer at 19 runs 7→22 (p=0.001) | `$PC $E GQF-rgb_cont GQF-kv_adaln`; `$PC $E GHF-rgb_cont GHF-kv_adaln` |
| Fig. 3, count | per-seed successes at 2/3/4 objects | `collect.load(E,"GF")`, grouped by train_seed |
| Sec. VII, ACT split at 4 objects on atomic clips | F-kv 23 vs F-kv_adaln 12 first transfers | `$PC $E F-rgb_cont F-kv F-kv_adaln` |
| Sec. VII, ACT four-object floor | 0 / 1380 | `collect.load` over F, CF, AF, D, J, X; `profile == "four_object"` |

Quoted from `docs/research_questions.md` and **not** re-derived here, because
they come from diagnostics rather than rollouts:
- grasp error (~9 mm horizontal, 8.6–10.7 mm for the arms vs 8.7 mm for the
  control; 3-D error 17.5 → 10.0 mm, 9.2–11.9 mm for the arms);
- dropped-before-shelf objects (42 for the control vs 9 for `kv_adaln`), from
  rollout events;
- the branch sizes (~150 M vs ~1 M);
- the same-recipe-retrain p = 0.008.

These should be re-derived from `diagnostics/` before submission.
