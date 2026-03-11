import os
import json
import argparse


def load_metric(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("direct_attributes"), data.get("overall")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True, help="Root save path, e.g. ./results")
    parser.add_argument("--model_name", type=str, required=True, help="Model name folder under save_path")
    parser.add_argument("--repeat_list", type=str, default="1,2,4,8", help="Comma-separated repeats")
    parser.add_argument("--output_name", type=str, default="patch_repeat_summary.json", help="Output summary json name")
    args = parser.parse_args()

    result_root = os.path.join(args.save_path, args.model_name)
    repeats = [int(x.strip()) for x in args.repeat_list.split(",") if x.strip()]

    rows = []
    for n in repeats:
        tag = f"repeat{n}"
        final_path = os.path.join(result_root, f"final_acc_{tag}.json")
        if not os.path.exists(final_path):
            rows.append({
                "repeat": n,
                "tag": tag,
                "direct_attributes": None,
                "overall": None,
                "exists": False,
                "path": final_path,
            })
            continue

        direct_attr, overall = load_metric(final_path)
        rows.append({
            "repeat": n,
            "tag": tag,
            "direct_attributes": direct_attr,
            "overall": overall,
            "exists": True,
            "path": final_path,
        })

    print("repeat,direct_attributes,overall,file_exists")
    for r in rows:
        print(f"{r['repeat']},{r['direct_attributes']},{r['overall']},{r['exists']}")

    # Save machine-readable summary
    output_path = os.path.join(result_root, args.output_name)
    with open(output_path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nSaved summary to: {output_path}")

    # Emit simple arrays for quick plotting
    x = [r["repeat"] for r in rows if r["exists"]]
    y_direct = [r["direct_attributes"] for r in rows if r["exists"]]
    y_overall = [r["overall"] for r in rows if r["exists"]]
    print("\nCurve arrays:")
    print(f"N = {x}")
    print(f"direct_attributes = {y_direct}")
    print(f"overall = {y_overall}")


if __name__ == "__main__":
    main()
