# VENDORING manifest — `stable_pretraining`

> **NOT YET VENDORED.** This subtree is *scaffolding*. The real upstream source (the pretraining
> scaffold) lands only when the maintainer confirms the candidate commit SHA below against upstream and
> clones the pristine snapshot in — the maintainer-gated real-vendor step of
> [#96](https://github.com/AbdelStark/Lensemble/issues/96). Until then this directory holds the manifest
> and an empty `patches/` dir, nothing more.
>
> This vendored subtree is **separately licensed** (MIT) and lives **outside** the `lensemble` import
> DAG of [RFC-0001 §3](../../docs/rfcs/RFC-0001-architecture.md#3-dependency-layering-no-cycles). No
> symbol from this package is re-exported through `lensemble.__init__`; Lensemble reaches it only through
> a thin internal shim ([RFC-0016 §2](../../docs/rfcs/RFC-0016-deployment-vendoring-topology.md)).

This manifest records the exact fields required by
[RFC-0016 §2](../../docs/rfcs/RFC-0016-deployment-vendoring-topology.md).

## Source URL

`https://github.com/galilai-group/stable-pretraining`

## Vendored commit SHA

`d83f1426bb34049403642e82c1ce9fed3aa06435`

`STATUS: UNCONFIRMED — candidate research lead; the maintainer must confirm against upstream before the
real vendor` ([#96](https://github.com/AbdelStark/Lensemble/issues/96)). This SHA is a research lead, not
an authoritative pin. Do **not** treat it as the vendored revision until it is verified at vendor time.

## Vendored date

`TBD — not yet vendored`

## License

- **SPDX:** `MIT`
- **In-tree LICENSE path:** `./LICENSE` — `STATUS: pending real vendor` (the file is synced in at vendor
  time, not present yet).
- **Maintainer-confirmation note (RFC-0016 §2):** upstream **ships a real MIT `LICENSE` file** at its
  repository root (unlike `stable_worldmodel`, whose missing file was a packaging bug). It is therefore
  clean to clone-and-modify. Confirm the `LICENSE` is still present and MIT at vendor time, then sync it
  in to `./LICENSE`.

## Local-modification log

No local modifications yet. (When vendored, every local change lands as a numbered `patches/*.patch`
applied over the pristine snapshot — the snapshot itself stays byte-identical to upstream at the recorded
SHA, so the diff against upstream is always inspectable.)

## Upstream-sync procedure

To re-vendor / bump the recorded revision:

1. **Bump the SHA** — confirm the new upstream commit SHA against the upstream remote and update the
   *Vendored commit SHA* field above (clearing the `UNCONFIRMED` status once verified).
2. **Re-clone pristine** — replace this subtree's source with a clean checkout of upstream at that SHA
   (byte-identical to upstream; no in-place edits).
3. **Re-apply `patches/*.patch`** — apply the local patch series in order over the pristine snapshot.
4. **Update this manifest** — refresh the *Vendored date*, the *Local-modification log*, and sync the
   upstream `./LICENSE` (clearing its `pending real vendor` status).

Contribute fixes back upstream (e.g. CPU-fallback fixes upstream accepts) so the local patch set shrinks
over time.
