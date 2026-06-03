# -*- coding: utf-8 -*-
"""
医学二分类预测完整示例：模拟数据 + 机器学习建模 + 评价 + 可视化 + 结果保存

直接运行：
    python medical_binary_prediction_simulation.py

输出目录：
    当前脚本所在目录 / medical_prediction_outputs

说明：
    1. 本脚本不依赖任何外部数据，会自动生成 1000 条医学二分类模拟数据。
    2. expose 为二分类标签：0=健康，1=患病。
    3. 模型包括：逻辑回归、随机森林、XGBoost。
    4. 如当前 Python 环境未安装 xgboost，脚本会自动使用 sklearn 的 GradientBoostingClassifier
       作为可运行兜底模型，并在结果中保留说明；若需真正 XGBoost，请安装 xgboost 后重新运行。
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# =========================
# 全局配置
# =========================
RANDOM_SEED = 42
N_SAMPLES = 1000
TEST_SIZE = 0.2
OUTPUT_DIR = Path(__file__).resolve().parent / "medical_prediction_outputs"


# =========================
# 中文显示设置，尽量避免图片中文乱码
# =========================
def setup_chinese_font() -> None:
    """设置 matplotlib 中文字体，兼容常见 Linux/Windows/macOS 环境。"""
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


# =========================
# 1. 生成医学模拟数据
# =========================
def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数，用于将线性风险转换为患病概率。"""
    return 1.0 / (1.0 + np.exp(-x))


def generate_simulated_medical_data(
    n_samples: int = N_SAMPLES,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    生成医学二分类预测任务的模拟 DataFrame。

    字段说明：
        expose: 0=健康，1=患病
        age: 年龄，数值型
        sex: 性别，分类变量
        height_cm: 身高，数值型
        weight_kg: 体重，数值型
        bmi: BMI，数值型
        smoking: 吸烟，分类变量
        drinking: 饮酒，分类变量
        其余医学特征为 0/1 one-hot 形式
    """
    rng = np.random.default_rng(random_seed)

    age = np.clip(rng.normal(loc=52, scale=13, size=n_samples), 18, 90).round(0).astype(int)
    sex = rng.choice(["男", "女"], size=n_samples, p=[0.52, 0.48])

    height_cm = np.where(
        sex == "男",
        rng.normal(loc=171, scale=6.5, size=n_samples),
        rng.normal(loc=160, scale=5.8, size=n_samples),
    )
    height_cm = np.clip(height_cm, 140, 195).round(1)

    base_bmi = rng.normal(loc=24.0, scale=3.4, size=n_samples)
    bmi = np.clip(base_bmi + (age - 50) * 0.025, 16.5, 38.0).round(1)
    weight_kg = (bmi * (height_cm / 100.0) ** 2).round(1)

    smoking = rng.choice(["从不", "既往", "当前"], size=n_samples, p=[0.58, 0.18, 0.24])
    drinking = rng.choice(["从不", "偶尔", "经常"], size=n_samples, p=[0.45, 0.38, 0.17])

    # 0/1 医学特征：概率随年龄、BMI 或生活方式轻微变化，使数据更接近真实医学风险结构
    hypertension_prob = sigmoid(-4.0 + 0.045 * age + 0.085 * (bmi - 24))
    diabetes_prob = sigmoid(-4.4 + 0.035 * age + 0.12 * (bmi - 24))
    dyslipidemia_prob = sigmoid(-3.0 + 0.02 * age + 0.09 * (bmi - 24))
    coronary_prob = sigmoid(-5.0 + 0.05 * age + 0.06 * (bmi - 24))
    stroke_prob = sigmoid(-5.6 + 0.055 * age + 0.35 * (hypertension_prob > 0.5))
    kidney_prob = sigmoid(-5.1 + 0.04 * age + 0.35 * (diabetes_prob > 0.4))
    fatty_liver_prob = sigmoid(-3.2 + 0.18 * (bmi - 24) + 0.35 * (drinking == "经常"))
    hyperuricemia_prob = sigmoid(-3.3 + 0.08 * (bmi - 24) + 0.25 * (sex == "男"))
    family_history_prob = np.full(n_samples, 0.24)
    regular_exercise_prob = sigmoid(1.1 - 0.015 * age - 0.08 * (bmi - 24))
    high_salt_diet_prob = np.full(n_samples, 0.30) + 0.08 * (drinking == "经常")
    medication_history_prob = sigmoid(-3.5 + 0.035 * age + 0.6 * (hypertension_prob > 0.5))

    hypertension = rng.binomial(1, hypertension_prob)
    diabetes = rng.binomial(1, diabetes_prob)
    dyslipidemia = rng.binomial(1, dyslipidemia_prob)
    coronary_heart_disease = rng.binomial(1, coronary_prob)
    stroke_history = rng.binomial(1, stroke_prob)
    chronic_kidney_disease = rng.binomial(1, kidney_prob)
    fatty_liver = rng.binomial(1, fatty_liver_prob)
    hyperuricemia = rng.binomial(1, hyperuricemia_prob)
    family_history = rng.binomial(1, family_history_prob)
    regular_exercise = rng.binomial(1, regular_exercise_prob)
    high_salt_diet = rng.binomial(1, np.clip(high_salt_diet_prob, 0.05, 0.85))
    medication_history = rng.binomial(1, medication_history_prob)

    # 根据临床上常见的风险方向构造 expose 概率，保证标签与特征存在可学习关系
    risk_score = (
        -5.2
        + 0.035 * age
        + 0.09 * (bmi - 24)
        + 0.30 * (sex == "男")
        + 0.42 * (smoking == "既往")
        + 0.72 * (smoking == "当前")
        + 0.22 * (drinking == "偶尔")
        + 0.52 * (drinking == "经常")
        + 0.85 * hypertension
        + 0.95 * diabetes
        + 0.65 * dyslipidemia
        + 0.75 * coronary_heart_disease
        + 0.70 * stroke_history
        + 0.58 * chronic_kidney_disease
        + 0.45 * fatty_liver
        + 0.38 * hyperuricemia
        + 0.62 * family_history
        - 0.50 * regular_exercise
        + 0.35 * high_salt_diet
        + 0.28 * medication_history
        + rng.normal(0, 0.45, n_samples)
    )
    expose_prob = sigmoid(risk_score)
    expose = rng.binomial(1, expose_prob)

    df = pd.DataFrame(
        {
            "expose": expose.astype(int),
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "smoking": smoking,
            "drinking": drinking,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "dyslipidemia": dyslipidemia,
            "coronary_heart_disease": coronary_heart_disease,
            "stroke_history": stroke_history,
            "chronic_kidney_disease": chronic_kidney_disease,
            "fatty_liver": fatty_liver,
            "hyperuricemia": hyperuricemia,
            "family_history": family_history,
            "regular_exercise": regular_exercise,
            "high_salt_diet": high_salt_diet,
            "medication_history": medication_history,
        }
    )
    return df


# =========================
# 2. 构建预处理与模型
# =========================
def make_one_hot_encoder() -> OneHotEncoder:
    """兼容不同 sklearn 版本的 OneHotEncoder 参数。"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    binary_features: List[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """构建特征预处理流程。"""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    binary_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
            ("binary", binary_transformer, binary_features),
        ],
        remainder="drop",
    )


def build_models(
    numeric_features: List[str],
    categorical_features: List[str],
    binary_features: List[str],
) -> Tuple[Dict[str, Pipeline], Dict[str, str]]:
    """构建逻辑回归、随机森林、XGBoost 三类模型。"""
    models: Dict[str, Pipeline] = {}
    notes: Dict[str, str] = {}

    models["LogisticRegression"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features, binary_features, scale_numeric=True),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
            ),
        ]
    )
    notes["LogisticRegression"] = "逻辑回归"

    models["RandomForest"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features, binary_features, scale_numeric=False),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    notes["RandomForest"] = "随机森林"

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        xgb_classifier = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        notes["XGBoost"] = "XGBoost"
    else:
        xgb_classifier = GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_SEED,
        )
        notes["XGBoost"] = "未检测到可用 xgboost，当前使用 GradientBoostingClassifier 兜底运行"

    models["XGBoost"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numeric_features, categorical_features, binary_features, scale_numeric=False),
            ),
            ("classifier", xgb_classifier),
        ]
    )

    return models, notes


# =========================
# 3. 评价指标与可视化
# =========================
def get_feature_names(pipeline: Pipeline) -> List[str]:
    """获取预处理后的特征名，用于特征重要性图。"""
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    feature_names: List[str] = []

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder":
            continue
        if transformer_name == "categorical":
            onehot = transformer.named_steps["onehot"]
            encoded_names = onehot.get_feature_names_out(columns).tolist()
            feature_names.extend(encoded_names)
        else:
            feature_names.extend(list(columns))

    return feature_names


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    note: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """计算模型评价指标。"""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        "model": model_name,
        "model_note": note,
        "AUC": roc_auc_score(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_Curve_AUC": auc(fpr, tpr),
    }
    return metrics, fpr, tpr, y_prob


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
) -> None:
    """绘制多模型 ROC 曲线。"""
    plt.figure(figsize=(8, 6), dpi=150)
    for model_name, (fpr, tpr, model_auc) in roc_data.items():
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={model_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="随机参考线")
    plt.xlabel("假阳性率 FPR")
    plt.ylabel("真阳性率 TPR")
    plt.title("医学二分类预测模型 ROC 曲线")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def extract_feature_importance(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    """提取模型特征重要性或逻辑回归系数绝对值。"""
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(pipeline)

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_[0])
        importance_type = "absolute_coefficient"
    else:
        importance = np.zeros(len(feature_names))
        importance_type = "not_available"

    return pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_names,
            "importance": importance,
            "importance_type": importance_type,
        }
    ).sort_values("importance", ascending=False)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """绘制单个模型 Top-N 特征重要性图。"""
    plot_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(9, 6), dpi=150)
    plt.barh(plot_df["feature"], plot_df["importance"], color="#2E86AB")
    plt.xlabel("重要性")
    plt.ylabel("特征")
    plt.title(f"{model_name} Top {top_n} 特征重要性")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# =========================
# 4. 主流程
# =========================
def main() -> None:
    setup_chinese_font()
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_simulated_medical_data(n_samples=N_SAMPLES, random_seed=RANDOM_SEED)
    data_path = OUTPUT_DIR / "simulated_medical_binary_data.csv"
    df.to_csv(data_path, index=False, encoding="utf-8-sig")

    target_col = "expose"
    numeric_features = ["age", "height_cm", "weight_kg", "bmi"]
    categorical_features = ["sex", "smoking", "drinking"]
    binary_features = [
        "hypertension",
        "diabetes",
        "dyslipidemia",
        "coronary_heart_disease",
        "stroke_history",
        "chronic_kidney_disease",
        "fatty_liver",
        "hyperuricemia",
        "family_history",
        "regular_exercise",
        "high_salt_diet",
        "medication_history",
    ]

    X = df[numeric_features + categorical_features + binary_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    models, model_notes = build_models(numeric_features, categorical_features, binary_features)

    metrics_records: List[Dict[str, float]] = []
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    all_importance_df: List[pd.DataFrame] = []
    prediction_output = pd.DataFrame({"true_expose": y_test.reset_index(drop=True)})

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        metrics, fpr, tpr, y_prob = evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
            note=model_notes[model_name],
        )
        metrics_records.append(metrics)
        roc_data[model_name] = (fpr, tpr, metrics["AUC"])
        prediction_output[f"{model_name}_prob"] = y_prob
        prediction_output[f"{model_name}_pred"] = (y_prob >= 0.5).astype(int)

        importance_df = extract_feature_importance(pipeline, model_name)
        all_importance_df.append(importance_df)
        importance_df.to_csv(
            OUTPUT_DIR / f"feature_importance_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_feature_importance(
            importance_df=importance_df,
            model_name=model_name,
            output_path=OUTPUT_DIR / f"feature_importance_{model_name}.png",
            top_n=15,
        )

    metrics_df = pd.DataFrame(metrics_records).sort_values("AUC", ascending=False)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    prediction_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    combined_importance_df = pd.concat(all_importance_df, ignore_index=True)
    combined_importance_df.to_csv(
        OUTPUT_DIR / "feature_importance_all_models.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_roc_curves(roc_data=roc_data, output_path=OUTPUT_DIR / "roc_curves.png")

    print("\n========== 医学二分类预测流程运行完成 ==========");
    print(f"样本量：{len(df)}")
    print(f"患病组 expose=1 数量：{int(df['expose'].sum())}")
    print(f"健康组 expose=0 数量：{int((1 - df['expose']).sum())}")
    print("\n模型评价指标：")
    print(metrics_df[["model", "AUC", "Accuracy", "Recall", "F1", "model_note"]].to_string(index=False))
    print(f"\n所有 CSV 和图片结果已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
