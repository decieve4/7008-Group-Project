import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional

class DataVisualization:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    @classmethod
    def from_json(cls, json_path: str):
        df = pd.read_json(json_path)
        return cls(df)

    def plot_difficulty_distribution(self, bins: int = 10):
        if 'difficulty_score' not in self.df.columns:
            raise KeyError("Column 'difficulty_score' not found in DataFrame.")

        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=self.df,
            x='difficulty_score',
            bins=bins,
            color='steelblue'  # no kde parameter -> no curve
        )
        plt.title('Distribution of Difficulty Score')
        plt.xlabel('Difficulty Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    def plot_question_type_distribution(self):
        if 'question_type' not in self.df.columns:
            raise KeyError("Column 'question_type' not found in DataFrame.")

        plt.figure(figsize=(8, 5))
        order = self.df['question_type'].value_counts().index
        sns.countplot(
            data=self.df,
            x='question_type',
            order=order,
            color='seagreen'
        )
        plt.title('Distribution of Question Type')
        plt.xlabel('Question Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_category_distribution(self, top_n: Optional[int] = None):
        if 'category' not in self.df.columns:
            raise KeyError("Column 'category' not found in DataFrame.")

        # Exclude 'general' category
        df_no_general = self.df[self.df['category'] != 'general']

        counts = df_no_general['category'].value_counts()
        if top_n is not None:
            counts = counts.head(top_n)

        plt.figure(figsize=(8, 5))
        sns.barplot(
            x=counts.index,
            y=counts.values,
            color='coral'
        )
        title = 'Distribution of Category'
        #if top_n is not None:
            #title += f' (Top {top_n})'
        plt.title(title)
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    def plot_difficulty_by_question_type(self):
        """
        Boxplot of difficulty_score grouped by question_type.
        """
        required_cols = {'difficulty_score', 'question_type'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=self.df,
            x='question_type',
            y='difficulty_score',
            color='skyblue'
        )
        plt.title('Difficulty by Question Type')
        plt.xlabel('Question Type')
        plt.ylabel('Difficulty Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    def plot_difficulty_by_category(self, top_n: int | None = 10):
        """
        Boxplot of difficulty_score grouped by category,
        excluding 'general'. Optionally limit to top_n categories by count.
        """
        required_cols = {'difficulty_score', 'category'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        df_no_general = self.df[self.df['category'] != 'general'].copy()
        if df_no_general.empty:
            raise ValueError("No rows left after excluding 'general' category.")

        # focus on most frequent categories
        if top_n is not None:
            top_cats = (
                df_no_general['category']
                .value_counts()
                .head(top_n)
                .index
            )
            df_no_general = df_no_general[df_no_general['category'].isin(top_cats)]

        plt.figure(figsize=(10, 5))
        sns.boxplot(
            data=df_no_general,
            x='category',
            y='difficulty_score',
            color='lightcoral'
        )
        plt.title('Difficulty by Category (excluding "general")')
        plt.xlabel('Category')
        plt.ylabel('Difficulty Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


    def plot_question_length_distribution(self, bins: int = 20):
        """
        Histogram of question_text length in characters.
        """
        if 'question_text' not in self.df.columns:
            raise KeyError("Column 'question_text' not found in DataFrame.")

        lengths = self.df['question_text'].astype(str).str.len()

        plt.figure(figsize=(8, 5))
        sns.histplot(lengths, bins=bins, color='mediumpurple')
        plt.title('Distribution of Question Length (characters)')
        plt.xlabel('Question Length (characters)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()


    def plot_type_category_heatmap(self, min_count: int = 1):
        """
        Heatmap of counts of questions by (question_type, category),
        excluding 'general' and optionally filtering out rare combos.
        """
        required_cols = {'question_type', 'category'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        df_no_general = self.df[self.df['category'] != 'general']

        crosstab = pd.crosstab(df_no_general['question_type'],
                               df_no_general['category'])

        # optionally zero out very rare combos
        crosstab = crosstab.where(crosstab >= min_count, other=0)

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            crosstab,
            annot=True,
            fmt='g',
            cmap='Blues'
        )
        plt.title('Question Type vs Category (counts, excluding "general")')
        plt.xlabel('Category')
        plt.ylabel('Question Type')
        plt.tight_layout()
        plt.show()


    def _get_question_length_series(self):
        """
        内部小工具：返回题目长度（字符数）的 Series。
        """
        if "question_text" not in self.df.columns:
            raise KeyError("DataFrame 中缺少 'question_text' 列")
        return self.df["question_text"].astype(str).str.len()

    def plot_difficulty_vs_question_length(self, sample: int | None = 500):
        """
        难度 vs 题目长度散点图。
        sample: 如果数据太多，可以随机采样一部分点来画。
        """
        if "difficulty_score" not in self.df.columns:
            raise KeyError("DataFrame 中缺少 'difficulty_score' 列")

        lengths = self._get_question_length_series()
        data = pd.DataFrame({
            "difficulty_score": self.df["difficulty_score"],
            "question_length": lengths
        }).dropna()

        if sample is not None and len(data) > sample:
            data = data.sample(sample, random_state=42)

        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=data,
            x="question_length",
            y="difficulty_score",
            alpha=0.6
        )
        plt.title("Difficulty vs Question Length")
        plt.xlabel("Question Length (characters)")
        plt.ylabel("Difficulty Score")
        plt.tight_layout()
        plt.show()


    def _get_option_count_series(self, delimiter: str = "||"):
        """
        内部小工具：返回每题选项数量的 Series。
        """
        if "options_text" not in self.df.columns:
            raise KeyError("DataFrame 中缺少 'options_text' 列")

        def count_options(val):
            if pd.isna(val):
                return 0
            if isinstance(val, (list, tuple)):
                return len(val)
            s = str(val).strip()
            if not s:
                return 0
            return len(s.split(delimiter))

        return self.df["options_text"].map(count_options)

    def plot_numeric_correlation_heatmap(self, delimiter: str = "||"):
        """
        画 difficulty_score、question_length、option_count 等数值特征的相关性热力图。
        """
        if "difficulty_score" not in self.df.columns:
            raise KeyError("DataFrame 中缺少 'difficulty_score' 列")

        lengths = self._get_question_length_series()
        option_counts = self._get_option_count_series(delimiter=delimiter)

        num_df = pd.DataFrame({
            "difficulty_score": self.df["difficulty_score"],
            "question_length": lengths,
            "option_count": option_counts
        }).dropna()

        corr = num_df.corr()

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True
        )
        plt.title("Correlation between Numeric Features")
        plt.tight_layout()
        plt.show()

    def _question_length(self):
        """Return a Series with question length in characters."""
        if "question_text" not in self.df.columns:
            raise KeyError("Missing 'question_text' column.")
        return self.df["question_text"].astype(str).str.len()



    def plot_numeric_pairplot(self, hue: str | None = "category"):
        """
        Pairplot of numeric variables (difficulty, question length, option count),
        optionally colored by 'category' or 'question_type'.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        data = {}

        if "difficulty_score" in self.df.columns:
            data["difficulty_score"] = self.df["difficulty_score"]

        if "question_text" in self.df.columns:
            data["question_length"] = self._question_length()

        if "option_count" in self.df.columns:
            data["option_count"] = self.df["option_count"]

        if not data:
            raise ValueError("No numeric data available for pairplot.")

        plot_df = pd.DataFrame(data)

        if hue is not None and hue in self.df.columns:
            plot_df[hue] = self.df[hue]
            g = sns.pairplot(plot_df, hue=hue, diag_kind="hist")
        else:
            g = sns.pairplot(plot_df, diag_kind="hist")

        plt.show()


    def plot_difficulty_vs_length_regression(self, hue: str = "question_type"):
        """
        Regression plot of difficulty_score vs question_length,
        colored by the chosen categorical variable (default: question_type).
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        required = {"difficulty_score", "question_text"}
        if not required.issubset(self.df.columns):
            raise KeyError(f"Missing columns: {required - set(self.df.columns)}")

        df_plot = self.df.copy()
        df_plot["question_length"] = self._question_length()

        if hue not in df_plot.columns:
            hue = None  # fall back to no hue if not available

        g = sns.lmplot(
            data=df_plot,
            x="question_length",
            y="difficulty_score",
            hue=hue,
            height=5,
            aspect=1.3,
            scatter_kws={"alpha": 0.6, "s": 40}
        )
        g.set_axis_labels("Question Length (chars)", "Difficulty Score")
        plt.title("Difficulty vs Question Length (with regression line)")
        plt.tight_layout()
        plt.show()
    def facet_difficulty_by_category(self, top_n: int = 6, bins: int = 8):
        """
        Faceted histograms of difficulty_score for the top-N categories
        (excluding 'general').
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        required = {"difficulty_score", "category"}
        if not required.issubset(self.df.columns):
            raise KeyError(f"Missing columns: {required - set(self.df.columns)}")

        df_no_general = self.df[self.df["category"] != "general"].copy()
        top_cats = df_no_general["category"].value_counts().head(top_n).index
        df_top = df_no_general[df_no_general["category"].isin(top_cats)]

        g = sns.FacetGrid(df_top, col="category", col_wrap=3, sharex=True, sharey=True)
        g.map(plt.hist, "difficulty_score", bins=bins, color="peru", edgecolor="black")
        g.set_axis_labels("Difficulty Score", "Count")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Difficulty Distribution by Category")
        plt.show()
    def jointplot_difficulty_vs_length(self, kind: str = "scatter"):
        """
        Joint plot of difficulty_score vs question_length.
        kind: 'scatter', 'hex', or 'kde'.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt

        required = {"difficulty_score", "question_text"}
        if not required.issubset(self.df.columns):
            raise KeyError(f"Missing columns: {required - set(self.df.columns)}")

        df_plot = self.df.copy()
        df_plot["question_length"] = self._question_length()

        g = sns.jointplot(
            data=df_plot,
            x="question_length",
            y="difficulty_score",
            kind=kind,
            height=6
        )
        g.set_axis_labels("Question Length (chars)", "Difficulty Score")
        plt.show()
    def print_numeric_descriptive_stats(self):
        """
        Print basic descriptive statistics for key numeric columns.
        """
        candidate_cols = ["difficulty_score", "question_length", "option_count"]
        # only keep the ones that actually exist
        cols = [c for c in candidate_cols if c in self.df.columns]
        if "question_text" in self.df.columns and "question_length" not in cols:
            # compute length on the fly
            self.df["question_length"] = self._question_length()
            cols.append("question_length")

        if not cols:
            raise ValueError("No numeric columns found among "
                             "'difficulty_score', 'question_length', 'option_count'.")

        print(self.df[cols].describe().T.round(2))

    def plot_overview_subplots(self, top_n_categories: int = 8):
        """
        2x2 overview figure:
        - difficulty histogram
        - question length histogram
        - top-N category counts
        - question type counts
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # 1) Difficulty
        if "difficulty_score" in self.df.columns:
            sns.histplot(self.df["difficulty_score"], bins=10,
                         color="steelblue", ax=axs[0, 0])
            axs[0, 0].set_title("Difficulty Score")
        else:
            axs[0, 0].set_visible(False)

        # 2) Question length
        try:
            qlen = self._question_length()
            sns.histplot(qlen, bins=15, color="mediumpurple", ax=axs[0, 1])
            axs[0, 1].set_title("Question Length (chars)")
        except KeyError:
            axs[0, 1].set_visible(False)

        # 3) Category counts (excluding 'general')
        if "category" in self.df.columns:
            counts_cat = (self.df[self.df["category"] != "general"]["category"]
                          .value_counts()
                          .head(top_n_categories))
            sns.barplot(x=counts_cat.index, y=counts_cat.values,
                        color="coral", ax=axs[1, 0])
            axs[1, 0].set_title(f"Top {top_n_categories} Categories (excl. 'general')")
            axs[1, 0].set_xticklabels(axs[1, 0].get_xticklabels(),
                                      rotation=45, ha="right")
        else:
            axs[1, 0].set_visible(False)

        # 4) Question type counts
        if "question_type" in self.df.columns:
            counts_type = self.df["question_type"].value_counts()
            sns.barplot(x=counts_type.index, y=counts_type.values,
                        color="seagreen", ax=axs[1, 1])
            axs[1, 1].set_title("Question Types")
            axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(),
                                      rotation=45, ha="right")
        else:
            axs[1, 1].set_visible(False)

        plt.tight_layout()
        plt.show()
