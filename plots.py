from Visualization import DataVisualization  # or whatever your file is named
viz = DataVisualization.from_json("questions_reclassified_v2.json")

viz.plot_difficulty_distribution(bins=15)
viz.plot_question_type_distribution()
viz.plot_category_distribution(top_n=15)
viz.plot_difficulty_by_question_type()
viz.plot_difficulty_by_category(top_n=8)
viz.plot_question_length_distribution()
viz.plot_type_category_heatmap(min_count=2)
viz.plot_difficulty_vs_question_length()
viz.plot_numeric_correlation_heatmap()
viz.print_numeric_descriptive_stats()
viz.plot_overview_subplots()
viz.plot_numeric_pairplot(hue="category")
viz.plot_difficulty_vs_length_regression(hue="question_type")
viz.facet_difficulty_by_category(top_n=6)
viz.jointplot_difficulty_vs_length(kind="hex")

import pandas as pd
df_before = pd.read_json("cleaned_questions_all_v2.json")
df_after  = pd.read_json("questions_reclassified_v2.json")
counts_before = df_before["category"].value_counts().sort_index()
counts_after  = df_after["category"].value_counts().sort_index()
comparison = pd.DataFrame({
    "before": counts_before,
    "after": counts_after
})
comparison = comparison.fillna(0).astype(int)

comparison["diff"] = comparison["after"] - comparison["before"]

# 4. 打印结果
print(" category comparison：")
print(comparison)

# 可选：总数检查一下
print("\ntotal（before）：", len(df_before))
print("total（after）：", len(df_after))
