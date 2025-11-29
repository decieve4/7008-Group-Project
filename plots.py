from Visualization import DataVisualization  # or whatever your file is named
viz = DataVisualization.from_json("questions_reclassified.json")

viz.plot_difficulty_distribution(bins=15)
viz.plot_question_type_distribution()
viz.plot_category_distribution(top_n=15)
viz.plot_difficulty_by_question_type()
viz.plot_difficulty_by_category(top_n=8)
viz.plot_question_length_distribution()
viz.plot_type_category_heatmap(min_count=2)
viz.plot_difficulty_vs_question_length()
viz.plot_numeric_correlation_heatmap()

import pandas as pd

# 1. 读入修改前、修改后的数据
df_before = pd.read_json("cleaned_questions_all.json")
df_after  = pd.read_json("cleaned_questions_reclassified_no_yesno_multichoice.json")

# 2. 计算各 category 的数量（按名称排序方便对比）
counts_before = df_before["category"].value_counts().sort_index()
counts_after  = df_after["category"].value_counts().sort_index()

# 3. 合并成一个对比表
comparison = pd.DataFrame({
    "before": counts_before,
    "after": counts_after
})

# 没出现过的类别会是 NaN，这里填 0 并转成 int
comparison = comparison.fillna(0).astype(int)

# 再加一列差值：after - before
comparison["diff"] = comparison["after"] - comparison["before"]

# 4. 打印结果
print("各 category 在修改前后数量对比：")
print(comparison)

# 可选：总数检查一下
print("\n总题目数（前）：", len(df_before))
print("总题目数（后）：", len(df_after))
