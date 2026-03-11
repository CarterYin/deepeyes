import os
import re
import json
import argparse
import matplotlib.pyplot as plt


def load_from_summary(summary_path):
    with open(summary_path, "r") as f:
        data = json.load(f)
    rows = data.get("rows", [])
    valid = [r for r in rows if r.get("exists") and r.get("direct_attributes") is not None and r.get("overall") is not None]
    valid.sort(key=lambda x: x["repeat"])
    return valid


def load_from_final_files(result_root):
    rows = []
    pat = re.compile(r"^final_acc_repeat(\d+)\.json$")
    for name in os.listdir(result_root):
        m = pat.match(name)
        if not m:
            continue
        repeat_n = int(m.group(1))
        path = os.path.join(result_root, name)
        with open(path, "r") as f:
            data = json.load(f)
        rows.append({
            "repeat": repeat_n,
            "direct_attributes": data.get("direct_attributes"),
            "overall": data.get("overall"),
        })
    rows = [r for r in rows if r["direct_attributes"] is not None and r["overall"] is not None]
    rows.sort(key=lambda x: x["repeat"])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./results", help="Root save path")
    parser.add_argument("--model_name", type=str, required=True, help="Model name folder under save_path")
    parser.add_argument("--summary_file", type=str, default="patch_repeat_summary.json", help="Summary json file name")
    parser.add_argument("--output_file", type=str, default="patch_repeat_curve.png", help="Output png file name")
    parser.add_argument("--title", type=str, default="Patch Repeat vs Accuracy", help="Plot title")
    args = parser.parse_args()

    result_root = os.path.join(args.save_path, args.model_name)
    summary_path = os.path.join(result_root, args.summary_file)

    if os.path.exists(summary_path):
        rows = load_from_summary(summary_path)
    else:
        rows = load_from_final_files(result_root)

    if not rows:
        raise RuntimeError(f"No valid data found in {result_root}. Run judge first.")

    x = [r["repeat"] for r in rows]
    y_direct = [r["direct_attributes"] for r in rows]
    y_overall = [r["overall"] for r in rows]

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(x, y_direct, marker="o", linewidth=2, label="direct_attributes")
    plt.plot(x, y_overall, marker="s", linewidth=2, label="overall")
    plt.xlabel("Patch Repeat N")
    plt.ylabel("Accuracy (%)")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(result_root, args.output_file)
    plt.savefig(out_path)
    print(f"Saved figure to: {out_path}")
    print(f"N = {x}")
    print(f"direct_attributes = {y_direct}")
    print(f"overall = {y_overall}")


if __name__ == "__main__":
    main()
