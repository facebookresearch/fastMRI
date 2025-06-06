import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_clinical_data(input_csv, output_csv, label_column="label"):
    print(f"📥 正在读取临床数据: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"🔍 原始维度: {df.shape}")

    # 1️⃣ 处理缺失值（数值字段）
    num_cols = df.select_dtypes(include=["number"]).columns
    imputer = SimpleImputer(strategy="mean")
    df[num_cols] = imputer.fit_transform(df[num_cols])
    print("✅ 缺失值已填补（均值）")

    # 2️⃣ 去除不合逻辑的值（例如年龄 < 0）
    if "age" in df.columns:
        df = df[df["age"] >= 0]
        print("✅ 异常值（负年龄）已移除")

    # 3️⃣ 标准化特征（排除 label）
    features = df.drop(columns=[label_column])
    labels = df[label_column]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print("✅ 已进行 z-score 标准化")

    # 4️⃣ SMOTE 过采样（仅当 label 是分类时）
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(features_scaled, labels)
    print(f"✅ SMOTE 过采样完成（类别平衡），新样本数: {len(y_resampled)}")

    # 5️⃣ 输出结果
    result_df = pd.DataFrame(X_resampled, columns=features.columns)
    result_df[label_column] = y_resampled
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    print(f"📤 已保存处理后数据到: {output_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="临床表型数据预处理")
    parser.add_argument("input_csv", help="原始 CSV 文件路径")
    parser.add_argument("output_csv", help="输出处理后 CSV 文件路径")
    parser.add_argument("--label_column", default="label", help="标签列名称，默认是 'label'")
    args = parser.parse_args()

    preprocess_clinical_data(args.input_csv, args.output_csv, args.label_column)
