"""임계값 테스트 사진 쌍의 CLIP 유사도를 계산하고 임계값을 추천한다.

usage: python threshold/analyze.py [--data data/threshold] [--model ViT-B-32:openai ...]
촬영 방법은 threshold/photo-guide.md 참고.
"""

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 스크립트로 실행하면 threshold/가 import 경로가 되므로 서버 코드(app/)를 찾을 수 있게 레포 루트를 넣는다.
sys.path.insert(0, str(ROOT))

GROUPS = ("same", "angle", "nearby", "cross")
PHOTO_GROUPS = ("same", "angle", "nearby")
EXPECTED = {"same": "pass", "angle": "decide", "nearby": "fail", "cross": "fail"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
# app.main과 같은 값. app.main은 torch까지 불러오므로 여기서는 값만 둔다.
MAX_IMAGE_BYTES = 10 * 1024 * 1024

TEXT = {
    "ko": {
        "same": "같은 장소·같은 구도",
        "angle": "같은 장소·다른 구도",
        "nearby": "200m 안 다른 장소",
        "cross": "다른 장소(교차)",
        "pass": "통과해야 함",
        "decide": "팀에서 정할 부분",
        "fail": "떨어져야 함",
        "median": "중앙값",
        "threshold": "임계값",
        "xlabel": "코사인 유사도",
        "title": "사진 쌍별 유사도",
    },
    "en": {
        "same": "Same place, same framing",
        "angle": "Same place, other framing",
        "nearby": "Other place within 200m",
        "cross": "Other place (cross)",
        "pass": "Should pass",
        "decide": "Team decides",
        "fail": "Should fail",
        "median": "Median",
        "threshold": "Threshold",
        "xlabel": "Cosine similarity",
        "title": "Similarity per photo pair",
    },
}
KOREAN_FONTS = ("Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR")

# 차트 색: 기대 결과에 따라 칠한다. 파랑·주황은 색각 이상 검사를 통과한 조합이다.
COLORS = {"pass": "#2a78d6", "fail": "#eb6834", "decide": "#898781"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def find_images(folder: Path) -> tuple[dict[str, Path], list[str]]:
    """폴더 안의 사진을 {번호: 경로}로 반환한다. 번호는 확장자를 뺀 파일 이름이다."""
    images, warnings = {}, []
    if not folder.is_dir():
        return images, warnings
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in images:
            warnings.append(f"{folder.name}/{path.stem}: 같은 번호의 사진이 여러 장이라 {path.name}은 건너뜀")
            continue
        images[path.stem] = path
    return images, warnings


def place_of(pair_id: str) -> str:
    """번호 앞쪽 숫자가 장소다. 03a와 03b는 같은 장소 03이다."""
    digits = re.match(r"\d+", pair_id)
    return digits.group() if digits else pair_id


def build_pairs(data_dir: Path) -> tuple[list[dict], list[str]]:
    """ref/의 기준 사진과 같은 번호의 사진을 묶고, 번호가 다른 ref↔same 쌍을 교차 쌍으로 더한다."""
    refs, warnings = find_images(data_dir / "ref")
    photos = {}
    for group in PHOTO_GROUPS:
        photos[group], group_warnings = find_images(data_dir / group)
        warnings += group_warnings

    pairs = []
    for group in PHOTO_GROUPS:
        for pair_id, path in photos[group].items():
            if pair_id not in refs:
                warnings.append(f"{group}/{path.name}: 같은 번호의 기준 사진(ref/{pair_id}.*)이 없어 건너뜀")
                continue
            pairs.append({"group": group, "pair": pair_id, "ref": refs[pair_id], "photo": path})

    # 장소마다 대상이 다르므로, 번호가 다른 기준 사진과 수행 사진은 "다른 장소" 쌍이 된다.
    # 03a, 03b처럼 숫자가 같으면 한 장소의 여러 세트라서 교차 쌍에서 뺀다.
    ids = sorted(set(refs) & set(photos["same"]))
    for ref_id in ids:
        for photo_id in ids:
            if place_of(ref_id) != place_of(photo_id):
                pairs.append({"group": "cross", "pair": f"{ref_id}-{photo_id}", "ref": refs[ref_id], "photo": photos["same"][photo_id]})

    for path in sorted({p["ref"] for p in pairs} | {p["photo"] for p in pairs}):
        if path.stat().st_size > MAX_IMAGE_BYTES:
            warnings.append(f"{path.parent.name}/{path.name}: 10MB를 넘어 실제 서버는 거절함 (분석에는 포함)")
    return pairs, warnings


def recommend_threshold(positives: list[float], negative_groups: dict[str, list[float]]) -> dict | None:
    """통과해야 할 쌍과 떨어져야 할 쌍을 가장 잘 가르는 임계값을 고른다.

    떨어져야 할 쌍은 그룹마다 따로 비율을 내서 평균한다. 교차 쌍이 훨씬 많아서,
    합쳐 버리면 정작 중요한 200m 안 쌍의 실패가 묻히기 때문이다.
    점수가 같으면 가장 가까운 사진 쌍과 가장 멀리 떨어진 값을 고른다.
    """
    negative_groups = {group: scores for group, scores in negative_groups.items() if scores}
    if not positives or not negative_groups:
        return None
    scores = sorted(set(positives).union(*negative_groups.values()))
    candidates = [scores[0] - 0.01, scores[-1] + 0.01] + [(a + b) / 2 for a, b in zip(scores, scores[1:])]

    def balanced_accuracy(t: float) -> float:
        pass_rate = sum(p >= t for p in positives) / len(positives)
        reject_rate = statistics.mean(sum(n < t for n in group) / len(group) for group in negative_groups.values())
        return (pass_rate + reject_rate) / 2

    best = max(candidates, key=lambda t: (balanced_accuracy(t), min(abs(t - s) for s in scores)))
    threshold = round(best, 3)
    return {
        "threshold": threshold,
        "balanced_accuracy": round(balanced_accuracy(threshold), 3),
        "passed": {"count": sum(p >= threshold for p in positives), "total": len(positives)},
        "rejected": {
            group: {"count": sum(n < threshold for n in group_scores), "total": len(group_scores)}
            for group, group_scores in negative_groups.items()
        },
        # 양수면 두 분포가 겹치지 않는다는 뜻이다.
        "gap": round(min(positives) - max(max(group) for group in negative_groups.values()), 3),
    }


def embed_all(paths: set[Path], model: str, pretrained: str) -> tuple[dict[Path, list[float]], list[str]]:
    """서버와 같은 디코딩·임베딩 코드로 사진을 임베딩한다."""
    from app.embedder import Embedder
    from app.main import _decode_image

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass

    embedder = Embedder(model, pretrained, num_threads=os.cpu_count() or 2)
    embedder.load()
    vectors, failures = {}, []
    for path in sorted(paths):
        try:
            vectors[path] = embedder.embed(_decode_image(path.read_bytes()))
        except Exception as e:
            failures.append(f"{path.parent.name}/{path.name}: 읽을 수 없음 ({e})")
    return vectors, failures


def describe(scores: list[float]) -> dict:
    return {
        "count": len(scores),
        "min": round(min(scores), 4),
        "median": round(statistics.median(scores), 4),
        "max": round(max(scores), 4),
    }


def plot(spec: str, rows: list[dict], recommendation: dict | None, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    installed = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in KOREAN_FONTS if name in installed), None)
    text = TEXT["ko" if font else "en"]
    plt.rcParams.update({"font.family": font or "sans-serif", "axes.unicode_minus": False})

    order = [group for group in GROUPS if any(row["group"] == group for row in rows)]
    fig, ax = plt.subplots(figsize=(8, 1.0 * len(order) + 1.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    jitter = random.Random(0)
    for i, group in enumerate(order):
        values = [row["similarity"] for row in rows if row["group"] == group]
        y = len(order) - 1 - i
        ys = [y + jitter.uniform(-0.2, 0.2) for _ in values]
        # 겹치는 점이 구분되도록 배경색 테두리를 두른다.
        ax.scatter(values, ys, s=40, color=COLORS[EXPECTED[group]], edgecolors=SURFACE, linewidths=0.8, zorder=3)
        ax.vlines(statistics.median(values), y - 0.32, y + 0.32, color=INK_SECONDARY, linewidth=1.6, zorder=4)

    labels = [f"{text[group]} ({sum(row['group'] == group for row in rows)})" for group in order]
    ax.set_yticks(range(len(order)), list(reversed(labels)))
    ax.set_ylim(-0.6, len(order) - 0.4)

    if recommendation:
        threshold = recommendation["threshold"]
        ax.axvline(threshold, color=INK, linestyle=(0, (4, 3)), linewidth=1, zorder=2)
        ax.text(threshold, len(order) - 0.45, f" {text['threshold']} {threshold:.3f}", color=INK, fontsize=8, va="top", ha="left")

    ax.set_xlabel(text["xlabel"], color=INK_SECONDARY, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=9)

    shown = {EXPECTED[group] for group in order}
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=6, color=COLORS[kind], label=text[kind])
        for kind in ("pass", "decide", "fail")
        if kind in shown
    ]
    handles.append(Line2D([], [], marker="|", linestyle="", markersize=10, markeredgewidth=1.6, color=INK_SECONDARY, label=text["median"]))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles), frameon=False, fontsize=8, labelcolor=INK_SECONDARY, handletextpad=0.3, columnspacing=1.2)
    ax.set_title(f"{text['title']} · {spec}", loc="left", color=INK, fontsize=11, pad=26)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def print_report(spec: str, rows: list[dict], recommendation: dict | None, data_dir: Path) -> None:
    text = TEXT["ko"]
    print(f"\n== {spec}")
    for group in GROUPS:
        scores = [row["similarity"] for row in rows if row["group"] == group]
        if scores:
            stats = describe(scores)
            print(f"  {text[group]:<14} {stats['count']:>3}쌍  최소 {stats['min']:.3f}  중앙 {stats['median']:.3f}  최대 {stats['max']:.3f}")
    if not recommendation:
        print("  같은 구도 쌍과 다른 장소 쌍이 모두 있어야 임계값을 추천할 수 있어요.")
        return

    threshold = recommendation["threshold"]
    passed = recommendation["passed"]
    rejected = ", ".join(f"{text[g]} {r['count']}/{r['total']}" for g, r in recommendation["rejected"].items())
    print(f"\n  추천 임계값 {threshold:.3f}  (균형 정확도 {recommendation['balanced_accuracy']:.0%})")
    print(f"  통과: 같은 구도 {passed['count']}/{passed['total']}  ·  탈락: {rejected}")
    angle = [row for row in rows if row["group"] == "angle"]
    if angle:
        print(f"  다른 구도는 이 임계값에서 {sum(row['passed'] for row in angle)}/{len(angle)}쌍이 통과해요.")
    if recommendation["gap"] > 0:
        print(f"  두 분포가 겹치지 않아요 (간격 {recommendation['gap']:.3f}). 간격이 좁으면 실제 사용에서 틀릴 수 있어요.")
    else:
        print(f"  두 분포가 {-recommendation['gap']:.3f}만큼 겹쳐요. 어떤 임계값으로도 일부는 틀려요.")

    wrong = [
        row for row in rows
        if (EXPECTED[row["group"]] == "pass" and not row["passed"]) or (EXPECTED[row["group"]] == "fail" and row["passed"])
    ]
    if wrong:
        print("\n  틀린 쌍 (사진을 직접 비교해 보세요):")
        for row in sorted(wrong, key=lambda r: abs(r["similarity"] - threshold)):
            ref, photo = row["ref"].relative_to(data_dir), row["photo"].relative_to(data_dir)
            print(f"    {text[row['group']]:<14} {row['similarity']:.3f}  {ref} ↔ {photo}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "threshold", help="ref/, same/, angle/, nearby/ 폴더가 있는 곳")
    parser.add_argument("--model", action="append", help="open_clip 모델:가중치 (여러 번 지정 가능, 기본 ViT-B-32:openai)")
    parser.add_argument("--out", type=Path, help="결과를 저장할 폴더 (기본: <data>/results)")
    args = parser.parse_args()
    data_dir = args.data.resolve()
    out_dir = (args.out or data_dir / "results").resolve()
    specs = args.model or ["ViT-B-32:openai"]

    pairs, warnings = build_pairs(data_dir)
    for warning in warnings:
        print(f"⚠️ {warning}")
    if not pairs:
        print(f"{data_dir}에 사진 쌍이 없어요. threshold/photo-guide.md를 참고해 사진을 넣어 주세요.", file=sys.stderr)
        return 2
    counts = ", ".join(f"{TEXT['ko'][g]} {sum(p['group'] == g for p in pairs)}" for g in GROUPS)
    print(f"사진 쌍: {counts}")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {p["ref"] for p in pairs} | {p["photo"] for p in pairs}
    all_rows, summary = [], {}
    for spec in specs:
        model, _, pretrained = spec.partition(":")
        vectors, failures = embed_all(paths, model, pretrained or "openai")
        for failure in failures:
            print(f"⚠️ {failure}")
        rows = [
            {**pair, "model": spec, "similarity": round(sum(a * b for a, b in zip(vectors[pair["ref"]], vectors[pair["photo"]])), 4)}
            for pair in pairs
            if pair["ref"] in vectors and pair["photo"] in vectors
        ]
        by_group = {group: [row["similarity"] for row in rows if row["group"] == group] for group in GROUPS}
        recommendation = recommend_threshold(by_group["same"], {"nearby": by_group["nearby"], "cross": by_group["cross"]})
        for row in rows:
            row["passed"] = bool(recommendation) and row["similarity"] >= recommendation["threshold"]

        print_report(spec, rows, recommendation, data_dir)
        chart = out_dir / f"similarity-{spec.replace(':', '-').replace('/', '-')}.png"
        plot(spec, rows, recommendation, chart)
        print(f"\n  그래프: {chart}")
        summary[spec] = {
            "groups": {group: describe(scores) for group, scores in by_group.items() if scores},
            "recommendation": recommendation,
        }
        all_rows += rows

    with open(out_dir / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "group", "pair", "ref", "photo", "similarity", "passed"])
        for row in all_rows:
            writer.writerow([
                row["model"], row["group"], row["pair"],
                row["ref"].relative_to(data_dir), row["photo"].relative_to(data_dir),
                row["similarity"], row["passed"],
            ])
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
