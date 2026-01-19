# -*- coding: utf-8 -*-
import os
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE


# =========================
# 0) 参数区（默认参数）
# =========================
RANDOM_STATE = 42
TOP_K_HVG = 2000
PCA_DIM = 50
CLUSTER_METHOD = "kmeans"      # "kmeans" 或 "agglomerative"
DO_STANDARDIZE = True
EMBED_METHODS = ["pca2"]       # 可加 "tsne" / "umap"

DATASET_INFO = {
    "Klein": 4,
    "Lake": 16,
    "Romanov": 7,
    "Xin": 8,
    "Zeisel": 9,
}


# =========================
# 1) 表格读取（表达矩阵）
# =========================
def read_table_auto(path: str, index_col: int = 0) -> pd.DataFrame:
    """
    读取表达矩阵（基因×细胞）
    依次尝试：tab -> comma -> 自动推断
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")

    # tab
    try:
        df = pd.read_csv(path, sep="\t", header=0, index_col=index_col)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # comma
    try:
        df = pd.read_csv(path, sep=",", header=0, index_col=index_col)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    # auto
    df = pd.read_csv(path, sep=None, engine="python", header=0, index_col=index_col)
    return df


# =========================
# 2) 标签读取（鲁棒版）
# =========================
def _clean_token(tok: str) -> str:
    return str(tok).strip().strip('"').strip("'").strip()


def read_label_file_robust(path: str) -> pd.Series:
    """
    鲁棒读取label文件：
    - 逐行解析，允许每行字段数不一致（解决Romanov ParserError）
    - 若是“单列向量label”：返回 Series(labels)
    - 若是“含cell_id + label（可能还有其他列）”：自动判定哪一列是label
      判定策略：在第1..末列中，选择“唯一值数量较少且>1”的列作为label列（更符合类别标签特征）
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"标签文件不存在：{path}")

    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 简单跳过可能的表头
            low = line.lower()
            if ("label" in low and "cell" in low) or low in ("cell\tlabel", "cell,label"):
                continue

            # 选择分隔：优先tab，其次逗号，否则按任意空白
            if "\t" in line:
                parts = line.split("\t")
            elif "," in line:
                parts = line.split(",")
            else:
                parts = line.split()

            parts = [_clean_token(p) for p in parts if _clean_token(p) != ""]
            if len(parts) == 0:
                continue
            rows.append(parts)

    if len(rows) == 0:
        raise ValueError(f"标签文件为空或无法解析：{path}")

    # 统计最大列数
    max_cols = max(len(r) for r in rows)

    # 如果所有行都只有1列 -> 单列label向量
    if max_cols == 1:
        labels = [r[0] for r in rows]
        return pd.Series(labels, name="label")

    # 否则构造“可变列”表（用None补齐）
    data = []
    for r in rows:
        rr = r + [None] * (max_cols - len(r))
        data.append(rr)
    df = pd.DataFrame(data)

    # 默认：第0列是cell_id；label列在[1..max_cols-1]中自动判定
    n = len(df)

    # 计算每个候选列的唯一值数量（忽略None）
    cand_cols = list(range(1, max_cols))
    uniq_counts = {}
    for c in cand_cols:
        vals = df[c].dropna().astype(str)
        uniq_counts[c] = vals.nunique()

    # 选择“唯一值较少且>1”的列作为label列（类别数通常远小于细胞数）
    # 限制：唯一值不能太大（<= min(200, n)）
    label_col = None
    best = None
    for c in cand_cols:
        u = uniq_counts[c]
        if 1 < u <= min(200, n):
            if best is None or u < best:
                best = u
                label_col = c

    # 若没找到合适列，则退化：取最后一列
    if label_col is None:
        label_col = max_cols - 1

    cell_ids = df[0].astype(str).apply(_clean_token).values
    labels = df[label_col].astype(str).apply(_clean_token).values

    # 若label_col取到了None较多（极端情况），再退化到最后一列
    if pd.isna(df[label_col]).mean() > 0.5 and label_col != max_cols - 1:
        label_col = max_cols - 1
        labels = df[label_col].astype(str).apply(_clean_token).values

    return pd.Series(labels, index=cell_ids, name="label")


# =========================
# 3) 对齐增强：多策略ID归一化
# =========================
def norm_id_basic(s: str) -> str:
    return _clean_token(s)


def norm_id_drop_x_if_numeric(s: str) -> str:
    s = norm_id_basic(s)
    if len(s) >= 2 and (s[0] in ("X", "x")) and s[1:].isdigit():
        return s[1:]
    return s


def norm_id_to_intstr_if_numeric(s: str) -> str:
    s = norm_id_basic(s)
    # 允许 "123.0" 这种
    try:
        if s.replace(".", "", 1).isdigit():
            return str(int(float(s)))
    except Exception:
        pass
    return s


def align_labels(cell_ids, label_obj, log_func):
    """
    返回：
    - y_raw: 与cell_ids对齐后的字符串标签数组
    - aligned: bool，是否成功按ID对齐（而非按顺序）
    - match_rate: 匹配率
    """
    cell_ids = np.array([str(x) for x in cell_ids], dtype=str)

    # 情况1：label是“单列向量”（没有索引） -> 只能按顺序
    if not isinstance(label_obj, pd.Series) or label_obj.index is None or label_obj.index.equals(pd.RangeIndex(start=0, stop=len(label_obj), step=1)):
        labels = np.asarray(label_obj).reshape(-1).astype(str)
        if len(labels) != len(cell_ids):
            log_func(f"[Warn] 单列label长度({len(labels)}) != 细胞数({len(cell_ids)}), 将截断到最小长度对齐。")
        m = min(len(labels), len(cell_ids))
        return labels[:m], False, 1.0

    # 情况2：label是Series(index=cell_id)
    label_series = label_obj.copy()
    label_series.index = label_series.index.astype(str)

    # 多策略：对cell_id与label_id分别做归一化，取匹配率最高的一种
    strategies = [
        ("basic-basic", norm_id_basic, norm_id_basic),
        ("dropXnum-basic", norm_id_drop_x_if_numeric, norm_id_basic),
        ("basic-dropXnum", norm_id_basic, norm_id_drop_x_if_numeric),
        ("dropXnum-dropXnum", norm_id_drop_x_if_numeric, norm_id_drop_x_if_numeric),
        ("int-basic", norm_id_to_intstr_if_numeric, norm_id_basic),
        ("basic-int", norm_id_basic, norm_id_to_intstr_if_numeric),
        ("int-int", norm_id_to_intstr_if_numeric, norm_id_to_intstr_if_numeric),
        ("int-dropXnum", norm_id_to_intstr_if_numeric, norm_id_drop_x_if_numeric),
        ("dropXnum-int", norm_id_drop_x_if_numeric, norm_id_to_intstr_if_numeric),
    ]

    best = None
    best_pack = None

    for name, f_cell, f_lab in strategies:
        cell_norm = np.array([f_cell(x) for x in cell_ids], dtype=str)
        lab_norm = np.array([f_lab(x) for x in label_series.index.values], dtype=str)

        # 建立映射：norm_id -> label（若重复，保留首次）
        mapping = {}
        for i, lab in zip(lab_norm, label_series.values.astype(str)):
            if i not in mapping:
                mapping[i] = str(lab)

        y = np.array([mapping.get(i, None) for i in cell_norm], dtype=object)
        match_rate = np.mean(y != None)

        if best is None or match_rate > best:
            best = match_rate
            best_pack = (name, y)

    name, y_best = best_pack
    match_rate = float(best)

    # 如果匹配率很低，但label长度刚好等于细胞数，则按顺序对齐（避免“随机对齐评价”）
    if match_rate < 0.2:
        labels = np.asarray(label_series.values).reshape(-1).astype(str)
        if len(labels) == len(cell_ids):
            log_func(f"[Warn] ID匹配率仅 {match_rate:.2%}，但label长度等于细胞数，改为按顺序对齐（更可能正确）。")
            return labels, False, 1.0
        else:
            log_func(f"[Warn] ID匹配率仅 {match_rate:.2%}，且label长度({len(labels)})!=细胞数({len(cell_ids)}). 将只保留可匹配细胞。")

    # 若存在缺失，则剔除缺失细胞（保证评估有效）
    mask = (y_best != None)
    if mask.mean() < 1.0:
        log_func(f"[Info] Strategy={name}, match_rate={mask.mean():.2%}. 将剔除未匹配到label的细胞。")
    else:
        log_func(f"[Info] Strategy={name}, match_rate=100%. 标签已按ID对齐。")

    return y_best[mask].astype(str), True, float(mask.mean())


# =========================
# 4) 预处理、降维、聚类、可视化
# =========================
def normalize_log1p(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    libsize = X.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    X_norm = X / libsize * target_sum
    return np.log1p(X_norm)


def select_hvg_by_variance(X: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or top_k >= X.shape[1]:
        return X
    var = X.var(axis=0)
    idx = np.argsort(var)[::-1][:top_k]
    return X[:, idx]


def run_pca(X: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    n_components = min(n_components, X.shape[0], X.shape[1])
    return PCA(n_components=n_components, random_state=random_state).fit_transform(X)


def cluster(Z: np.ndarray, n_clusters: int, method: str, random_state: int) -> np.ndarray:
    method = method.lower()
    if method == "kmeans":
        try:
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        except TypeError:
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        return model.fit_predict(Z)
    if method == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(Z)
    raise ValueError(f"未知聚类方法：{method}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
    }


def embed_2d(Z: np.ndarray, method: str, random_state: int) -> np.ndarray:
    method = method.lower()
    if method == "pca2":
        return PCA(n_components=2, random_state=random_state).fit_transform(Z)
    if method == "tsne":
        return TSNE(
            n_components=2, random_state=random_state, init="pca",
            learning_rate="auto", perplexity=30
        ).fit_transform(Z)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=random_state).fit_transform(Z)
    raise ValueError(f"未知embedding方法：{method}")


def plot_scatter(E2: np.ndarray, labels: np.ndarray, title: str, save_path: str):
    plt.figure(figsize=(7, 6))
    plt.scatter(E2[:, 0], E2[:, 1], c=labels, s=8)
    plt.title(title)
    plt.xlabel("Dim-1")
    plt.ylabel("Dim-2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _value_count_str(arr, topn=8):
    vc = pd.Series(arr).value_counts()
    head = vc.head(topn)
    return ", ".join([f"{k}:{v}" for k, v in head.items()])


# =========================
# 5) 单数据集流程
# =========================
def run_one_dataset(base_dir: str, dataset: str, n_clusters: int) -> dict:
    data_path = os.path.join(base_dir, f"{dataset}.rds")
    label_path = os.path.join(base_dir, f"{dataset}.rds_label")

    out_dir = os.path.join(base_dir, "outputs", dataset)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{dataset}_runlog.txt")

    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # 清空日志
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")

    log("=" * 78)
    log(f"Dataset: {dataset}")
    log(f"Target clusters: {n_clusters}")
    log(f"Params: RANDOM_STATE={RANDOM_STATE}, TOP_K_HVG={TOP_K_HVG}, PCA_DIM={PCA_DIM}, "
        f"CLUSTER_METHOD={CLUSTER_METHOD}, STANDARDIZE={DO_STANDARDIZE}, EMBEDS={EMBED_METHODS}")
    log("=" * 78)

    # 1) 读取表达矩阵
    log("[Step 1] Reading expression matrix and labels...")
    expr_df = read_table_auto(data_path, index_col=0)  # genes × cells
    log(f"Expression shape (genes × cells): {expr_df.shape}")

    X = expr_df.values.T.astype(np.float32)            # cells × genes
    cell_ids = expr_df.columns.astype(str)

    # 2) 读取label（鲁棒版）
    label_obj = read_label_file_robust(label_path)

    # 3) 对齐label（增强版）
    y_raw, aligned_by_id, match_rate = align_labels(cell_ids, label_obj, log)

    # 若剔除了未匹配细胞，需要同步裁剪X与cell_ids
    if aligned_by_id and match_rate < 1.0:
        # 重新用同样策略生成mask（为简洁：用basic+best策略太复杂，这里直接重新跑一遍对齐拿mask）
        # 这里用简单方式：再次调用align_labels但返回的是已经剔除后的y_raw，缺失的细胞已被剔除
        # 因此我们必须重建一个mask：按“可匹配”的细胞保留
        # 为保证一致，我们采取：在align_labels里剔除缺失后，cell_ids也剔除（在这里实现）
        # -> 重新做一次“映射匹配”，使用相同策略名的细节会很长，所以直接走“通用重建”：
        # 做一个“能匹配到label”的集合（对齐阶段已经给出最佳策略，但我们不暴露内部策略名）
        # 简化做法：用多策略里最高匹配的那组mask重建会更复杂。
        # 因此：我们采用更稳健的方式——如果存在index（cell_id），我们直接用“最大交集”的方式裁剪：
        # 先尝试dropXnum策略，因为最常见；若还不行再basic。
        lab_series = label_obj if isinstance(label_obj, pd.Series) and label_obj.index is not None else None
        if lab_series is not None and len(set(lab_series.index.astype(str))) > 1:
            expr_ids = np.array([norm_id_drop_x_if_numeric(x) for x in cell_ids], dtype=str)
            lab_ids = set([norm_id_drop_x_if_numeric(x) for x in lab_series.index.astype(str)])
            mask = np.array([x in lab_ids for x in expr_ids], dtype=bool)
            if mask.mean() < 0.2:
                expr_ids = np.array([norm_id_basic(x) for x in cell_ids], dtype=str)
                lab_ids = set([norm_id_basic(x) for x in lab_series.index.astype(str)])
                mask = np.array([x in lab_ids for x in expr_ids], dtype=bool)
            X = X[mask]
            cell_ids = cell_ids[mask]

    # 4) 编码label
    le = LabelEncoder()
    y_true = le.fit_transform(y_raw)

    log(f"[Check] Gold label unique={len(le.classes_)}, top_counts=({_value_count_str(y_raw)})")

    # 5) 归一化+log
    log("[Step 2] Normalize + log1p...")
    X_log = normalize_log1p(X, target_sum=1e4)

    # 6) HVG
    log("[Step 3] HVG selection...")
    X_hvg = select_hvg_by_variance(X_log, top_k=TOP_K_HVG)
    log(f"After HVG shape (cells × genes): {X_hvg.shape}")

    # 7) 标准化
    if DO_STANDARDIZE:
        log("[Step 4] Standardize features...")
        X_use = StandardScaler(with_mean=True, with_std=True).fit_transform(X_hvg)
    else:
        X_use = X_hvg

    # 8) PCA
    log("[Step 5] PCA...")
    Z = run_pca(X_use, n_components=PCA_DIM, random_state=RANDOM_STATE)
    log(f"PCA feature shape: {Z.shape}")

    # 9) 聚类
    log("[Step 6] Clustering...")
    y_pred = cluster(Z, n_clusters=n_clusters, method=CLUSTER_METHOD, random_state=RANDOM_STATE)
    log(f"[Check] Pred clusters unique={len(np.unique(y_pred))}, top_counts=({_value_count_str(y_pred)})")

    # 10) 指标
    log("[Step 7] Metrics (ARI/NMI)...")
    metrics = compute_metrics(y_true, y_pred)
    log(f"ARI = {metrics['ARI']:.6f}")
    log(f"NMI = {metrics['NMI']:.6f}")

    # 11) 保存结果
    log("[Step 8] Save CSV/JSON...")
    result_df = pd.DataFrame({
        "cell_id": cell_ids[:len(y_true)],  # 保守：确保长度一致
        "y_true_label": y_raw,
        "y_true_encoded": y_true,
        "y_pred_cluster": y_pred[:len(y_true)]
    })
    out_csv = os.path.join(out_dir, f"{dataset}_cluster_result.csv")
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_json = os.path.join(out_dir, f"{dataset}_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset,
            "n_clusters": int(n_clusters),
            "random_state": int(RANDOM_STATE),
            "top_k_hvg": int(TOP_K_HVG),
            "pca_dim": int(PCA_DIM),
            "cluster_method": CLUSTER_METHOD,
            "standardize": bool(DO_STANDARDIZE),
            "embed_methods": EMBED_METHODS,
            "label_aligned_by_id": bool(aligned_by_id),
            "label_match_rate": float(match_rate),
            "metrics": metrics
        }, f, ensure_ascii=False, indent=2)

    # 12) 可视化
    log("[Step 9] Visualization...")
    for em in EMBED_METHODS:
        E2 = embed_2d(Z, method=em, random_state=RANDOM_STATE)
        fig_pred = os.path.join(out_dir, f"{dataset}_{CLUSTER_METHOD}_{em}_pred.png")
        fig_true = os.path.join(out_dir, f"{dataset}_{em}_gold.png")
        plot_scatter(E2, y_pred, f"{dataset} Pred ({CLUSTER_METHOD}+{em})", fig_pred)
        plot_scatter(E2, y_true, f"{dataset} Gold ({em})", fig_true)
        log(f"Saved: {os.path.basename(fig_pred)} / {os.path.basename(fig_true)}")

    log("[Done] Outputs written to: " + out_dir)

    return {
        "dataset": dataset,
        "n_clusters": int(n_clusters),
        "ARI": metrics["ARI"],
        "NMI": metrics["NMI"],
        "out_dir": out_dir,
        "label_match_rate": float(match_rate),
        "label_aligned_by_id": bool(aligned_by_id),
    }


# =========================
# 6) 主程序：批量五数据集
# =========================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_root = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_root, exist_ok=True)

    summary = []
    global_log = os.path.join(outputs_root, "GLOBAL_RUNLOG.txt")
    with open(global_log, "w", encoding="utf-8") as f:
        f.write("")

    def glog(msg: str):
        print(msg)
        with open(global_log, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    glog("=" * 78)
    glog("Batch run: single-cell clustering (5 datasets)")
    glog(f"Base dir: {base_dir}")
    glog(f"Outputs root: {outputs_root}")
    glog("=" * 78)

    for ds, k in DATASET_INFO.items():
        data_path = os.path.join(base_dir, f"{ds}.rds")
        label_path = os.path.join(base_dir, f"{ds}.rds_label")

        if not (os.path.exists(data_path) and os.path.exists(label_path)):
            glog(f"[Skip] {ds}: missing files. Need {ds}.rds and {ds}.rds_label in base dir.")
            continue

        try:
            res = run_one_dataset(base_dir, ds, k)
            summary.append(res)
            glog(f"[OK] {ds}: ARI={res['ARI']:.6f}, NMI={res['NMI']:.6f}, "
                 f"label_by_id={res['label_aligned_by_id']}, match_rate={res['label_match_rate']:.2%}")
        except Exception as e:
            glog(f"[Fail] {ds}: {repr(e)}")
            glog(traceback.format_exc())

    if len(summary) > 0:
        summary_df = pd.DataFrame(summary)[
            ["dataset", "n_clusters", "ARI", "NMI", "label_aligned_by_id", "label_match_rate", "out_dir"]
        ]
        summary_csv = os.path.join(outputs_root, "summary_metrics.csv")
        summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        glog("-" * 78)
        glog("Summary saved: " + summary_csv)
        glog("-" * 78)
        glog("All done.")
    else:
        glog("No dataset was processed. Please check file names and locations.")


if __name__ == "__main__":
    main()
