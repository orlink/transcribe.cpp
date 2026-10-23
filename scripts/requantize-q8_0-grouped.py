#!/usr/bin/env python3
"""
requantize-q8_0-grouped.py - write a Q8_0 GGUF whose matrix weights share one
scale per group of values (grouped Q8_0 scales).

    uv run --with gguf --with numpy scripts/requantize-q8_0-grouped.py \
        canary-1b-v2-F32.gguf canary-1b-v2-Q8_0.gguf out.gguf --group 256

Inputs:
    SOURCE   a full-precision GGUF of the model (F32 or F16), the values the
             grouped weights are quantized from
    TEMPLATE the model's Q8_0 GGUF: every tensor and key not touched here is
             copied from it unchanged

Output:
    TEMPLATE with each selected Q8_0 matrix weight replaced by Q8_0 blocks
    whose scales are equal within every group of --group values of a row
    (0 = one scale per row), plus the keys stt.q8_0.weight_group and
    stt.q8_0.act_group (the activation group the runtime should use; 0 = one
    scale per row). The file is a valid Q8_0 file for every backend; on a CPU
    with AMX, transcribe.cpp keeps such weights in ggml's grouped layout,
    whose int8 products are scaled once per group instead of once per 32
    values.

Selected tensors: the Q8_0 tensors of TEMPLATE whose names match --include
(a regular expression; the default is every 2-D Q8_0 weight except token
embeddings) and that have the same shape in SOURCE.

Scales: absmax / 127 per group, rounded to FP16 (as Q8_0), values rounded to
nearest. --clip-search tries clipping each group to a fraction of its absmax
(--clip-min .. 1.0 in --clip-steps steps) and keeps the one with the smallest
squared error.
"""

import argparse
import re
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("requantize-q8_0-grouped.py needs the gguf package (pip install gguf)")

QK = 32
DEFAULT_INCLUDE = r"\.weight$"
DEFAULT_EXCLUDE = r"(token_emb|embed\.token|embedding)"


def full_precision(t):
    """A GGUF tensor as float32 (rows, K)."""
    if t.tensor_type == gguf.GGMLQuantizationType.F32:
        a = np.asarray(t.data, dtype=np.float32)
    elif t.tensor_type == gguf.GGMLQuantizationType.F16:
        a = np.asarray(t.data, dtype=np.float16).astype(np.float32)
    elif t.tensor_type == gguf.GGMLQuantizationType.BF16:
        a = (np.asarray(t.data, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        raise ValueError(f"{t.name}: source type {t.tensor_type.name} is not full precision")
    K = int(t.shape[0])
    return a.reshape(-1, K)


def quantize_groups(w, group, clip_search, clip_min, clip_steps):
    """(fp16 scales (rows, K / g), int8 values (rows, K)) of w in groups of g values (0 = the whole row)."""
    rows, K = w.shape
    g = K if group == 0 else group
    wg = w.reshape(rows, K // g, g)
    amax = np.abs(wg).max(axis=2)
    best_d = None
    best_q = None
    best_err = None
    ratios = np.linspace(1.0, clip_min, clip_steps) if clip_search else np.array([1.0])
    for c in ratios:
        dh = (amax * c / 127.0).astype(np.float16)
        d = dh.astype(np.float32)
        inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0).astype(np.float32)
        q = np.clip(np.rint(wg * inv[:, :, None]), -127, 127)
        if len(ratios) == 1:
            return dh, q.astype(np.int8).reshape(rows, K)
        err = ((q * d[:, :, None] - wg) ** 2).sum(axis=2)
        if best_err is None:
            best_d, best_q, best_err = dh, q, err
        else:
            better = err < best_err
            best_d = np.where(better, dh, best_d)
            best_q = np.where(better[:, :, None], q, best_q)
            best_err = np.where(better, err, best_err)
    return best_d, best_q.astype(np.int8).reshape(rows, K)


def q8_0_blocks(dh, q, group):
    """Q8_0 blocks (rows, K / 32 * 34 bytes) with each group's scale repeated in its blocks."""
    rows, K = q.shape
    g = K if group == 0 else group
    blocks = np.zeros((rows, K // QK, 2 + QK), dtype=np.uint8)
    d_blocks = np.repeat(dh, g // QK, axis=1)  # (rows, K / 32)
    blocks[:, :, :2] = d_blocks.view(np.uint8).reshape(rows, K // QK, 2)
    blocks[:, :, 2:] = q.view(np.uint8).reshape(rows, K // QK, QK)
    return blocks.reshape(rows, K // QK * (2 + QK))


def copy_metadata(reader, writer, skip):
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.") or field.name in skip:
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="full-precision GGUF (F32 / F16)")
    ap.add_argument("template", help="the model's Q8_0 GGUF")
    ap.add_argument("output")
    ap.add_argument("--group", type=int, required=True, help="values per weight scale (multiple of 64; 0 = per row)")
    ap.add_argument("--act-group", type=int, default=None,
                    help="activation group the runtime should use (0 = per row; default: --group)")
    ap.add_argument("--include", default=DEFAULT_INCLUDE, help="regex of tensor names to group")
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE, help="regex of tensor names to keep as they are")
    ap.add_argument("--clip-search", action="store_true", help="choose each group's clip by squared error")
    ap.add_argument("--clip-min", type=float, default=0.7)
    ap.add_argument("--clip-steps", type=int, default=31)
    args = ap.parse_args()
    if args.group < 0 or (args.group > 0 and args.group % 64 != 0):
        sys.exit("--group must be 0 or a multiple of 64")
    act_group = args.group if args.act_group is None else args.act_group

    src = {t.name: t for t in gguf.GGUFReader(args.source).tensors}
    reader = gguf.GGUFReader(args.template)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(args.output, arch=arch)
    copy_metadata(reader, writer, {"stt.q8_0.weight_group", "stt.q8_0.act_group"})
    writer.add_int32("stt.q8_0.weight_group", args.group)
    writer.add_int32("stt.q8_0.act_group", act_group)

    inc, exc = re.compile(args.include), re.compile(args.exclude)
    plan = []
    for t in reader.tensors:
        K = int(t.shape[0])
        grouped = (t.tensor_type == gguf.GGMLQuantizationType.Q8_0 and len(t.shape) == 2 and inc.search(t.name)
                   and not exc.search(t.name) and t.name in src
                   and [int(x) for x in src[t.name].shape] == [int(x) for x in t.shape]
                   and (args.group == 0 or K % args.group == 0) and K % 64 == 0)
        plan.append((t, bool(grouped)))
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    n = 0
    sq_err = sq_ref = 0.0
    for t, grouped in plan:
        if not grouped:
            writer.write_tensor_data(t.data)
            continue
        w = full_precision(src[t.name])
        dh, q = quantize_groups(w, args.group, args.clip_search, args.clip_min, args.clip_steps)
        g = w.shape[1] if args.group == 0 else args.group
        rec = (np.repeat(dh.astype(np.float32), g, axis=1) * q)
        sq_err += float(((rec - w) ** 2).sum())
        sq_ref += float((w ** 2).sum())
        data = q8_0_blocks(dh, q, args.group).reshape(t.data.shape)
        writer.write_tensor_data(data)
        n += 1
    writer.close()
    wdesc = "one scale per row" if args.group == 0 else f"one scale per {args.group} values"
    print(f"{args.output}: {n} grouped tensors, {wdesc}, activations per {act_group or 'row'}"
          f"{', clip search' if args.clip_search else ''}; relative squared error {sq_err / max(sq_ref, 1e-30):.3e}")


if __name__ == "__main__":
    main()
