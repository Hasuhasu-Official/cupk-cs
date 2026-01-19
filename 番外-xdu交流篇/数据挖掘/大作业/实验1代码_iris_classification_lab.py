# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# =========================
# 全局配置（可复现 + 输出控制）
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.30

OUTPUT_DIR = "outputs"         # 图片输出目录
SAVE_FIGURES = True            # 是否保存 PNG 图（建议保留）
SAVE_AUX_FILES = False         # 是否保存 txt/json/csv（默认关闭，避免生成多余文件）
PRINT_DETAILS = True           # 是否打印详细结果（建议保留，便于截图）


def ensure_output_dir(path: str) -> None:
    """创建输出目录（若不存在）"""
    os.makedirs(path, exist_ok=True)


def load_iris_as_dataframe() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """加载 Iris 数据集并转为 DataFrame/Series"""
    iris = load_iris()
    X_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="label")
    feature_names = iris.feature_names
    class_names = iris.target_names.tolist()
    return X_df, y, feature_names, class_names


def describe_split(y_train: pd.Series, y_test: pd.Series, class_names: list[str]) -> pd.DataFrame:
    """统计训练/测试集中各类样本数，便于截图说明分层抽样效果"""
    train_counts = y_train.value_counts().sort_index()
    test_counts = y_test.value_counts().sort_index()

    df = pd.DataFrame({
        "Class": class_names,
        "Train Count": [int(train_counts.get(i, 0)) for i in range(len(class_names))],
        "Test Count": [int(test_counts.get(i, 0)) for i in range(len(class_names))]
    })
    df["Total"] = df["Train Count"] + df["Test Count"]
    return df


def train_and_evaluate_models(X_train, X_test, y_train, y_test, class_names: list[str]):
    """
    训练并评估：决策树（Gini/Entropy）、KNN、朴素贝叶斯
    输出：
    - results_df: 准确率汇总表
    - fitted_models: 训练后模型
    - reports: 分类报告（字符串）
    - cms_df: 混淆矩阵（DataFrame）
    """
    # 决策树：Gini
    dt_gini = DecisionTreeClassifier(
        criterion="gini",
        random_state=RANDOM_STATE
    )

    # 决策树：Entropy（常用于表示信息增益准则）
    dt_entropy = DecisionTreeClassifier(
        criterion="entropy",
        random_state=RANDOM_STATE
    )

    # KNN：对距离敏感，标准化 + KNN 用 Pipeline 保证流程一致
    knn = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

    # 朴素贝叶斯：高斯朴素贝叶斯（连续特征）
    nb = GaussianNB()

    models = {
        "DecisionTree(Gini)": dt_gini,
        "DecisionTree(Entropy)": dt_entropy,
        "KNN(k=5, scaled)": knn,
        "NaiveBayes(Gaussian)": nb
    }

    fitted_models = {}
    reports = {}
    cms_df = {}
    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # 混淆矩阵（做成 DataFrame）
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        cm_df = pd.DataFrame(cm, index=[f"True:{c}" for c in class_names],
                             columns=[f"Pred:{c}" for c in class_names])

        rep = classification_report(y_test, y_pred, target_names=class_names, digits=4)

        rows.append({"Model": name, "Test Accuracy": acc})
        cms_df[name] = cm_df
        reports[name] = rep

    results_df = pd.DataFrame(rows).sort_values(by="Test Accuracy", ascending=False).reset_index(drop=True)
    return results_df, fitted_models, reports, cms_df


def plot_and_save_decision_tree(dt_model: DecisionTreeClassifier,
                               feature_names: list[str],
                               class_names: list[str],
                               out_path: str,
                               title: str):
    """可视化并保存决策树 PNG"""
    plt.figure(figsize=(16, 10))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        fontsize=10
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_and_save_feature_importance(importances: np.ndarray,
                                     feature_names: list[str],
                                     out_path: str,
                                     title: str) -> pd.DataFrame:
    """绘制并保存特征重要性柱状图，同时返回排序后的重要性表"""
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    plt.bar(imp_df["feature"], imp_df["importance"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return imp_df


def print_block(title: str):
    """打印分隔块标题，便于截图对齐"""
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def save_text(path: str, content: str) -> None:
    """保存文本到文件（可选）"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    # 仅当需要保存图片/文件时创建目录
    if SAVE_FIGURES or SAVE_AUX_FILES:
        ensure_output_dir(OUTPUT_DIR)

    # ========== 1) 数据加载 ==========
    X_df, y, feature_names, class_names = load_iris_as_dataframe()

    # 分层抽样 + 固定随机种子（可复现）
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ========== 2) 训练评估 ==========
    results_df, fitted_models, reports, cms_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, class_names
    )

    # ========== 3) 终端打印（用于截图） ==========
    if PRINT_DETAILS:
        print_block("数据集划分统计（Train/Test 每类样本数）")
        split_df = describe_split(y_train, y_test, class_names)
        print(split_df.to_string(index=False))

        print_block("各模型测试集准确率对比（Accuracy）")
        # 保留 4 位小数，截图更清晰
        tmp = results_df.copy()
        tmp["Test Accuracy"] = tmp["Test Accuracy"].map(lambda x: f"{x:.4f}")
        print(tmp.to_string(index=False))

        print_block("混淆矩阵（Confusion Matrix）与分类报告（Classification Report）")
        for name in results_df["Model"].tolist():
            print(f"\n--- {name} ---")
            print("\n[Confusion Matrix]")
            print(cms_df[name].to_string())
            print("\n[Classification Report]")
            print(reports[name])

    # ========== 4) 决策树可视化（PNG） ==========
    dt_gini = fitted_models["DecisionTree(Gini)"]
    dt_entropy = fitted_models["DecisionTree(Entropy)"]

    if SAVE_FIGURES:
        tree_gini_path = os.path.join(OUTPUT_DIR, "decision_tree_gini.png")
        tree_entropy_path = os.path.join(OUTPUT_DIR, "decision_tree_entropy.png")

        plot_and_save_decision_tree(
            dt_model=dt_gini,
            feature_names=feature_names,
            class_names=class_names,
            out_path=tree_gini_path,
            title="Decision Tree (criterion = gini)"
        )
        plot_and_save_decision_tree(
            dt_model=dt_entropy,
            feature_names=feature_names,
            class_names=class_names,
            out_path=tree_entropy_path,
            title="Decision Tree (criterion = entropy / information gain)"
        )

    # ========== 5) 特征重要性（打印 + PNG） ==========
    gini_importances = dt_gini.feature_importances_
    entropy_importances = dt_entropy.feature_importances_

    if SAVE_FIGURES:
        gini_imp_path = os.path.join(OUTPUT_DIR, "feature_importance_gini.png")
        entropy_imp_path = os.path.join(OUTPUT_DIR, "feature_importance_entropy.png")
        gini_imp_df = plot_and_save_feature_importance(
            gini_importances, feature_names, gini_imp_path, "Feature Importance (Gini)"
        )
        entropy_imp_df = plot_and_save_feature_importance(
            entropy_importances, feature_names, entropy_imp_path, "Feature Importance (Entropy / Information Gain)"
        )
    else:
        # 不保存图时，也生成表用于打印
        gini_imp_df = pd.DataFrame({"feature": feature_names, "importance": gini_importances}).sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)
        entropy_imp_df = pd.DataFrame({"feature": feature_names, "importance": entropy_importances}).sort_values(
            "importance", ascending=False
        ).reset_index(drop=True)

    if PRINT_DETAILS:
        print_block("特征重要性（Feature Importance）- Entropy/信息增益")
        # 打印 4 位小数，便于报告引用
        df1 = entropy_imp_df.copy()
        df1["importance"] = df1["importance"].map(lambda x: f"{x:.4f}")
        print(df1.to_string(index=False))

        print_block("特征重要性（Feature Importance）- Gini")
        df2 = gini_imp_df.copy()
        df2["importance"] = df2["importance"].map(lambda x: f"{x:.4f}")
        print(df2.to_string(index=False))

        # 可选：打印决策树的文本规则（终端截图时很有用）
        print_block("决策树规则（文本形式，便于截图引用）- Entropy")
        print(export_text(dt_entropy, feature_names=feature_names))

        print_block("决策树规则（文本形式，便于截图引用）- Gini")
        print(export_text(dt_gini, feature_names=feature_names))

    # ========== 6) 可选：保存辅助文件（默认关闭） ==========
    if SAVE_AUX_FILES:
        # 分类报告汇总
        report_all = []
        report_all.append("===== Accuracy Table =====\n")
        report_all.append(results_df.to_string(index=False))
        report_all.append("\n\n===== Classification Reports =====\n")
        for name, rep in reports.items():
            report_all.append(f"\n--- {name} ---\n{rep}")
        save_text(os.path.join(OUTPUT_DIR, "classification_reports.txt"), "\n".join(report_all))

        # 混淆矩阵 JSON
        cms_serializable = {k: v.values.tolist() for k, v in cms_df.items()}
        with open(os.path.join(OUTPUT_DIR, "confusion_matrices.json"), "w", encoding="utf-8") as f:
            json.dump(cms_serializable, f, ensure_ascii=False, indent=2)

        # 特征重要性 CSV
        gini_imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_gini.csv"),
                           index=False, encoding="utf-8-sig")
        entropy_imp_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_entropy.csv"),
                              index=False, encoding="utf-8-sig")

        # 复现信息
        meta = {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "class_names": class_names,
            "feature_names": feature_names
        }
        with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # ========== 7) 末尾提示 ==========
    if SAVE_FIGURES:
        print_block("图片输出位置（用于插入报告）")
        print(f"- outputs/decision_tree_entropy.png")
        print(f"- outputs/decision_tree_gini.png")
        print(f"- outputs/feature_importance_entropy.png")
        print(f"- outputs/feature_importance_gini.png")

    if not SAVE_AUX_FILES:
        print("\n[提示] 当前未生成 txt/json/csv 等辅助文件（SAVE_AUX_FILES=False）。如需保存请改为 True。")


if __name__ == "__main__":
    main()
