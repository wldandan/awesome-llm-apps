import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def smart_data_analysis(dataset_path):
    """
    智能数据分析函数，能够自动适应不同的数据集结构
    """
    # 读取数据
    df = pd.read_csv(dataset_path)
    
    print("=== 数据集基本信息 ===")
    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n前5行数据:")
    print(df.head())
    print("\n数据类型:")
    print(df.dtypes)
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    # 智能分析：根据实际列名进行分析
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n数值列: {numeric_columns}")
    print(f"分类列: {categorical_columns}")
    
    # 创建可视化
    plt.figure(figsize=(15, 10))
    
    # 子图1：数值列的分布
    if numeric_columns:
        plt.subplot(2, 2, 1)
        if len(numeric_columns) >= 2:
            # 如果有多个数值列，创建散点图
            plt.scatter(df[numeric_columns[0]], df[numeric_columns[1]], alpha=0.6)
            plt.xlabel(numeric_columns[0])
            plt.ylabel(numeric_columns[1])
            plt.title(f'{numeric_columns[0]} vs {numeric_columns[1]}')
        else:
            # 如果只有一个数值列，创建直方图
            plt.hist(df[numeric_columns[0]], bins=20, alpha=0.7)
            plt.xlabel(numeric_columns[0])
            plt.ylabel('频次')
            plt.title(f'{numeric_columns[0]} 分布')
    
    # 子图2：分类列的分布
    if categorical_columns:
        plt.subplot(2, 2, 2)
        # 选择第一个分类列进行可视化
        cat_col = categorical_columns[0]
        value_counts = df[cat_col].value_counts().head(10)  # 只显示前10个
        plt.bar(range(len(value_counts)), value_counts.values)
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        plt.xlabel(cat_col)
        plt.ylabel('频次')
        plt.title(f'{cat_col} 分布 (前10个)')
    
    # 子图3：相关性热力图（如果有多个数值列）
    if len(numeric_columns) > 1:
        plt.subplot(2, 2, 3)
        correlation_matrix = df[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('数值列相关性热力图')
    
    # 子图4：箱线图（如果有数值列和分类列）
    if numeric_columns and categorical_columns:
        plt.subplot(2, 2, 4)
        # 选择第一个数值列和第一个分类列
        num_col = numeric_columns[0]
        cat_col = categorical_columns[0]
        
        # 如果分类列的唯一值太多，只选择前5个
        unique_cats = df[cat_col].value_counts().head(5).index
        filtered_df = df[df[cat_col].isin(unique_cats)]
        
        if len(filtered_df) > 0:
            sns.boxplot(data=filtered_df, x=cat_col, y=num_col)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'{num_col} 按 {cat_col} 分组')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计摘要
    print("\n=== 统计摘要 ===")
    if numeric_columns:
        print("\n数值列统计:")
        print(df[numeric_columns].describe())
    
    if categorical_columns:
        print("\n分类列统计:")
        for col in categorical_columns:
            print(f"\n{col} 的唯一值数量: {df[col].nunique()}")
            print(f"{col} 的前5个值:")
            print(df[col].value_counts().head())

# 使用示例
if __name__ == "__main__":
    # 替换为你的数据集路径
    dataset_path = './customers-1000.csv'
    smart_data_analysis(dataset_path) 