import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")


def check_correct(row):
    expected = str(row["expected_answer"]).lower().strip()
    answer = str(row["model_answer"]).lower().strip()
    category = row["category"]

    if category == "factual":
        return expected in answer
    
    if category == "classification":
        return expected in answer
    
    return False



df["correct"] = df.apply(check_correct, axis=1)

summary = df.groupby("model").agg(
    total_questions=("correct", "count"),
    correct_answers=("correct", "sum"),
    accuracy=("correct", "mean"),
    avg_latency=("latency", "mean"),
)

summary["accuracy"] = (summary["accuracy"] * 100).round(1)
summary["avg_latency"] = summary["avg_latency"].round(2)
pd.set_option("display.max_columns", None)
print("\n=== Model Performance Summary ===\n")
print(summary)

model_names = [name.split("/")[1].split(":")[0] for name in summary.index]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#1a1a2e")

colors = ["#00d2ff", "#ff6b6b", "#feca57"]

ax1.set_facecolor("#16213e")
bars1 = ax1.bar(model_names, summary["accuracy"], color=colors, width=0.6, edgecolor="white", linewidth=0.5)
ax1.set_title("Model Accuracy (%)", color="white", fontsize=14, fontweight="bold", pad=15)
ax1.set_ylabel("Accuracy (%)", color="white", fontsize=11)
ax1.set_ylim(0, 100)
ax1.tick_params(colors="white")
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2, f'{height}%', ha='center', color='white', fontweight='bold', fontsize=12)
for spine in ax1.spines.values():
    spine.set_color("#444")

ax2.set_facecolor("#16213e")
bars2 = ax2.bar(model_names, summary["avg_latency"], color=colors, width=0.6, edgecolor="white", linewidth=0.5)
ax2.set_title("Average Latency (seconds)", color="white", fontsize=14, fontweight="bold", pad=15)
ax2.set_ylabel("Seconds", color="white", fontsize=11)
ax2.tick_params(colors="white")
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height}s', ha='center', color='white', fontweight='bold', fontsize=12)
for spine in ax2.spines.values():
    spine.set_color("#444")

fig.suptitle("LLM Evaluation Benchmark Results", color="white", fontsize=18, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Chart saved to model_comparison.png")