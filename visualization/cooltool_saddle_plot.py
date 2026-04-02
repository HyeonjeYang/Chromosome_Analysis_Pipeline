#!/usr/bin/env python3
"""
polychrom trajectory → cooler → saddle plot 파이프라인
=====================================================

Usage:
    python saddle_from_polychrom.py --traj trajectory_folder --npoly 60000
    python saddle_from_polychrom.py --traj trajectory_folder --npoly 60000 --binsize 500 --nbins 30
    python saddle_from_polychrom.py --traj trajectory_folder --npoly 60000 --blist B_list.txt

Input:
    --traj      : polychrom HDF5 trajectory 폴더 경로
    --npoly     : 총 모노머 수
    --binsize   : cooler bin 크기 (default: 200)
    --cutoff    : contact distance cutoff (default: 2.0)
    --nbins     : saddle plot 그룹 수 (default: 38)
    --stride    : URI sampling stride (default: 1)
    --blist     : B-type 모노머 리스트 파일 (phasing용, 없으면 자동 생성)
    --alen      : A block 길이 (default: 500)
    --blen      : B block 길이 (default: 500)
    --outdir    : 출력 폴더 (default: saddle_output)

Output:
    outdir/
        contact_map.cool    : cooler 파일
        saddle_plot.png     : saddle plot
        saddle_with_marginals.png
        saddle_strength.png
        eigenvector_E1.png
        saddle_data.npz     : 원본 데이터
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix, triu

from polychrom.hdf5_format import list_URIs, load_URI
from polychrom.contactmaps import monomerResolutionContactMap

import cooler
import cooltools
from cooltools.api.saddle import saddle_strength


def parse_args():
    parser = argparse.ArgumentParser(
        description="polychrom trajectory → cooler → saddle plot")
    parser.add_argument("--traj", required=True, help="trajectory 폴더 경로")
    parser.add_argument("--npoly", type=int, required=True, help="총 모노머 수")
    parser.add_argument("--binsize", type=int, default=200, help="cooler bin 크기")
    parser.add_argument("--cutoff", type=float, default=2.0, help="contact cutoff")
    parser.add_argument("--nbins", type=int, default=38, help="saddle 그룹 수")
    parser.add_argument("--stride", type=int, default=1, help="URI sampling stride")
    parser.add_argument("--blist", type=str, default="", help="B-type 모노머 리스트 파일")
    parser.add_argument("--alen", type=int, default=500, help="A block 길이")
    parser.add_argument("--blen", type=int, default=500, help="B block 길이")
    parser.add_argument("--outdir", type=str, default="saddle_output", help="출력 폴더")
    return parser.parse_args()


def make_B_list(npoly, alen, blen):
    """기본 A/B 패턴으로 B-type 모노머 리스트 생성"""
    B_list = [ii * (alen + blen) + jj
              for ii in range(int(npoly // (alen + blen)) + 1)
              for jj in range(blen)
              if ii * (alen + blen) + jj < npoly]
    return B_list


def make_phasing_track(npoly, binsize, B_list):
    """
    B_list로부터 phasing track 생성.
    각 bin의 B-type 모노머 비율을 계산.
    B가 많은 bin → 낮은 값, A가 많은 bin → 높은 값.
    """
    n_bins = int(np.ceil(npoly / binsize))
    b_set = set(B_list)

    chrom = []
    start = []
    end = []
    values = []

    for i in range(n_bins):
        s = i * binsize
        e = min((i + 1) * binsize, npoly)
        chrom.append("chr1")
        start.append(s)
        end.append(e)
        # A fraction (1 - B fraction) → A가 많으면 높은 값
        n_b = sum(1 for m in range(s, e) if m in b_set)
        a_frac = 1.0 - n_b / (e - s)
        values.append(a_frac)

    track = pd.DataFrame({
        'chrom': chrom,
        'start': start,
        'end': end,
        'phasing': values,
    })
    return track


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cool_path = os.path.join(args.outdir, "contact_map.cool")

    npoly = args.npoly
    binsize = args.binsize

    # ---- 1. contact map 계산 ----
    print("=" * 60)
    print("STEP 1: Contact map 계산")
    print("=" * 60)

    uris = list_URIs(args.traj)
    uris_sampled = uris[::args.stride]
    print(f"  Total URIs: {len(uris)}, sampled: {len(uris_sampled)}")

    contact_map = monomerResolutionContactMap(
        uris_sampled, cutoff=args.cutoff, n=4
    )
    print(f"  Contact map shape: {contact_map.shape}")

    # ---- 2. binned contact map → cooler ----
    print("\nSTEP 2: Cooler 파일 생성")

    n_bins = int(np.ceil(npoly / binsize))

    # bin contact map (monomer resolution → binned)
    binned = np.zeros((n_bins, n_bins), dtype=float)
    for i in range(n_bins):
        for j in range(n_bins):
            rs = i * binsize
            re = min((i + 1) * binsize, npoly)
            cs = j * binsize
            ce = min((j + 1) * binsize, npoly)
            binned[i, j] = np.sum(contact_map[rs:re, cs:ce])

    # 상삼각만 추출 (cooler는 upper triangle 사용)
    sparse_upper = triu(coo_matrix(binned), k=0)

    # chromsizes & bins 정의
    chromsizes = pd.Series({"chr1": npoly})
    bins = cooler.binnify(chromsizes, binsize)

    # pixels
    pixels = pd.DataFrame({
        "bin1_id": sparse_upper.row.astype(int),
        "bin2_id": sparse_upper.col.astype(int),
        "count": sparse_upper.data.astype(int),
    })
    # count가 0인 것 제거
    pixels = pixels[pixels["count"] > 0].reset_index(drop=True)

    # cooler 생성
    cooler.create_cooler(
        cool_path,
        bins=bins,
        pixels=pixels,
        dtypes={"count": int},
        ordered=True,
    )
    print(f"  Cooler saved: {cool_path}")

    # balancing
    clr = cooler.Cooler(cool_path)
    try:
        cooler.balance_cooler(clr, cis_only=True, store=True)
        print("  Balancing complete.")
        weight_name = "weight"
    except Exception as e:
        print(f"  Balancing failed: {e}")
        print("  Proceeding without balancing.")
        weight_name = None

    # reload
    clr = cooler.Cooler(cool_path)

    # ---- 3. phasing track 준비 ----
    print("\nSTEP 3: Phasing track 준비")

    if len(args.blist) > 0 and os.path.exists(args.blist):
        with open(args.blist, "r") as f:
            B_list = [int(x.strip()) for x in f.readlines() if x.strip()]
        print(f"  B_list loaded from {args.blist}: {len(B_list)} monomers")
    else:
        B_list = make_B_list(npoly, args.alen, args.blen)
        print(f"  B_list auto-generated (alen={args.alen}, blen={args.blen}): "
              f"{len(B_list)} monomers")

    phasing_track = make_phasing_track(npoly, binsize, B_list)
    print(f"  Phasing track shape: {phasing_track.shape}")

    # ---- 4. view 정의 ----
    view_df = pd.DataFrame({
        "chrom": ["chr1"],
        "start": [0],
        "end": [npoly],
        "name": ["chr1"],
    })

    # ---- 5. eigenvector 계산 ----
    print("\nSTEP 4: Eigenvector 계산")

    cis_eigs = cooltools.eigs_cis(
        clr,
        phasing_track,
        view_df=view_df,
        n_eigs=3,
        clr_weight_name=weight_name,
    )

    eigenvector_track = cis_eigs[1][["chrom", "start", "end", "E1"]]
    print(f"  E1 range: [{eigenvector_track['E1'].min():.4f}, "
          f"{eigenvector_track['E1'].max():.4f}]")

    # E1 plot
    fig, ax = plt.subplots(figsize=(12, 3))
    x = eigenvector_track["start"].values / 1e3
    e1 = eigenvector_track["E1"].values
    ax.fill_between(x, e1, where=(e1 > 0), color="red", alpha=0.6, label="A")
    ax.fill_between(x, e1, where=(e1 < 0), color="blue", alpha=0.6, label="B")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Genomic position (kb)")
    ax.set_ylabel("E1")
    ax.set_title("Compartment Eigenvector (E1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "eigenvector_E1.png"), dpi=200)
    plt.close(fig)
    print("  Saved: eigenvector_E1.png")

    # ---- 6. expected 계산 ----
    print("\nSTEP 5: Expected 계산")

    cvd = cooltools.expected_cis(
        clr=clr,
        view_df=view_df,
        clr_weight_name=weight_name,
    )
    print(f"  Expected shape: {cvd.shape}")

    # ---- 7. saddle 계산 ----
    print("\nSTEP 6: Saddle plot 계산")

    Q_LO = 0.025
    Q_HI = 0.975
    N_GROUPS = args.nbins

    interaction_sum, interaction_count = cooltools.saddle(
        clr,
        cvd,
        eigenvector_track,
        "cis",
        n_bins=N_GROUPS,
        qrange=(Q_LO, Q_HI),
        view_df=view_df,
        clr_weight_name=weight_name,
    )

    saddledata = interaction_sum / interaction_count
    saddle_core = saddledata[1:-1, 1:-1]
    print(f"  Saddle data shape: {saddledata.shape}")

    # ---- 8. 시각화 ----
    print("\nSTEP 7: 시각화")

    # 기본 saddle plot
    fig, ax = plt.subplots(figsize=(7, 7))
    vmax = np.nanmax(np.abs(np.log2(saddle_core[np.isfinite(saddle_core)])))
    vmax = min(vmax, 2.0)
    im = ax.imshow(
        np.log2(saddle_core),
        cmap="coolwarm",
        vmin=-vmax, vmax=vmax,
        origin="lower",
    )
    n = saddle_core.shape[0]
    ax.set_xticks([0, n-1])
    ax.set_xticklabels(["B", "A"], fontsize=14)
    ax.set_yticks([0, n-1])
    ax.set_yticklabels(["B", "A"], fontsize=14)
    ax.set_xlabel("E1 rank (B → A)", fontsize=12)
    ax.set_ylabel("E1 rank (B → A)", fontsize=12)
    ax.set_title("Saddle Plot (log2 O/E)", fontsize=14)
    plt.colorbar(im, ax=ax, label="log2(O/E)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "saddle_plot.png"), dpi=200)
    plt.close(fig)
    print("  Saved: saddle_plot.png")

    # marginals 포함
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(2, 2, width_ratios=[1, 5], height_ratios=[5, 1],
                  wspace=0.05, hspace=0.05)

    ax_main = fig.add_subplot(gs[0, 1])
    im = ax_main.imshow(np.log2(saddle_core), cmap="coolwarm",
                        vmin=-vmax, vmax=vmax, origin="lower")
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    ax_main.set_title("Saddle Plot (log2 O/E)", fontsize=14, pad=10)

    # BB / AA 강도 표시
    bb = np.log2(saddle_core[0, 0]) if np.isfinite(np.log2(saddle_core[0, 0])) else 0
    aa = np.log2(saddle_core[-1, -1]) if np.isfinite(np.log2(saddle_core[-1, -1])) else 0
    ax_main.text(1, 1, f"BB\n{bb:.2f}", fontsize=9, color="white",
                 fontweight="bold", ha="left", va="bottom")
    ax_main.text(n-2, n-2, f"AA\n{aa:.2f}", fontsize=9, color="white",
                 fontweight="bold", ha="right", va="top")

    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    my = interaction_count[1:-1, 1:-1].sum(axis=1)
    ax_left.barh(np.arange(n), my, color="gray", alpha=0.7)
    ax_left.invert_xaxis()
    ax_left.set_ylabel("E1 rank (B → A)")

    ax_bot = fig.add_subplot(gs[1, 1], sharex=ax_main)
    mx = interaction_count[1:-1, 1:-1].sum(axis=0)
    ax_bot.bar(np.arange(n), mx, color="gray", alpha=0.7)
    ax_bot.set_xlabel("E1 rank (B → A)")

    cax = fig.add_subplot(gs[1, 0])
    plt.colorbar(im, cax=cax, label="log2(O/E)")

    fig.savefig(os.path.join(args.outdir, "saddle_with_marginals.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: saddle_with_marginals.png")

    # saddle strength
    x_arr = np.arange(N_GROUPS + 2)
    strength = saddle_strength(interaction_sum, interaction_count)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(x_arr, strength, where="pre", color="black", lw=2)
    ax.set_xlabel("Extent")
    ax.set_ylabel("(AA + BB) / (2 × AB)")
    ax.set_title("Saddle Strength")
    ax.axhline(1, color="gray", ls="--", lw=1)
    ax.set_xlim(0, len(x_arr) // 2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "saddle_strength.png"), dpi=200)
    plt.close(fig)
    print("  Saved: saddle_strength.png")

    # 데이터 저장
    np.savez(os.path.join(args.outdir, "saddle_data.npz"),
             saddledata=saddledata,
             interaction_sum=interaction_sum,
             interaction_count=interaction_count,
             strength=strength,
             E1=eigenvector_track["E1"].values)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print(f"Output: {args.outdir}")
    print(f"Saddle strength (extent=1): {strength[1]:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()