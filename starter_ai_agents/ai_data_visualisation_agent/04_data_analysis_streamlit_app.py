import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def safe_smart_analysis(dataset_path):
    """
    安全的智能数据分析函数，包含错误处理
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(dataset_path):
            st.error(f"文件不存在: {dataset_path}")
            return False
        
        # 读取数据
        st.info("正在读取数据...")
        df = pd.read_csv(dataset_path)
        st.success(f"成功读取数据！形状: {df.shape}")
        
        # 显示基本信息
        st.subheader("📊 数据集基本信息")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**行数:** {df.shape[0]}")
            st.write(f"**列数:** {df.shape[1]}")
            st.write(f"**列名:** {list(df.columns)}")
        
        with col2:
            st.write("**数据类型:**")
            st.write(df.dtypes)
        
        # 显示前几行数据
        st.subheader("📋 前5行数据")
        st.dataframe(df.head())
        
        # 缺失值统计
        st.subheader("🔍 缺失值统计")
        missing_data = df.isnull().sum()
        st.write(missing_data)
        
        # 智能分析：根据实际列名进行分析
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        st.subheader("📈 列类型分析")
        st.write(f"**数值列:** {numeric_columns}")
        st.write(f"**分类列:** {categorical_columns}")
        
        # 创建可视化
        st.subheader("📊 数据可视化")
        
        if numeric_columns or categorical_columns:
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('智能数据分析结果', fontsize=16)
            
            # 子图1：数值列的分布
            if numeric_columns:
                if len(numeric_columns) >= 2:
                    # 散点图
                    axes[0, 0].scatter(df[numeric_columns[0]], df[numeric_columns[1]], alpha=0.6)
                    axes[0, 0].set_xlabel(numeric_columns[0])
                    axes[0, 0].set_ylabel(numeric_columns[1])
                    axes[0, 0].set_title(f'{numeric_columns[0]} vs {numeric_columns[1]}')
                else:
                    # 直方图
                    axes[0, 0].hist(df[numeric_columns[0]], bins=20, alpha=0.7)
                    axes[0, 0].set_xlabel(numeric_columns[0])
                    axes[0, 0].set_ylabel('频次')
                    axes[0, 0].set_title(f'{numeric_columns[0]} 分布')
            else:
                axes[0, 0].text(0.5, 0.5, '无数值列', ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('无数值列')
            
            # 子图2：分类列的分布
            if categorical_columns:
                cat_col = categorical_columns[0]
                value_counts = df[cat_col].value_counts().head(10)
                axes[0, 1].bar(range(len(value_counts)), value_counts.values)
                axes[0, 1].set_xticks(range(len(value_counts)))
                axes[0, 1].set_xticklabels(value_counts.index, rotation=45, ha='right')
                axes[0, 1].set_xlabel(cat_col)
                axes[0, 1].set_ylabel('频次')
                axes[0, 1].set_title(f'{cat_col} 分布 (前10个)')
            else:
                axes[0, 1].text(0.5, 0.5, '无分类列', ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('无分类列')
            
            # 子图3：相关性热力图
            if len(numeric_columns) > 1:
                correlation_matrix = df[numeric_columns].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
                axes[1, 0].set_title('数值列相关性热力图')
            else:
                axes[1, 0].text(0.5, 0.5, '数值列不足', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('数值列不足')
            
            # 子图4：箱线图
            if numeric_columns and categorical_columns:
                num_col = numeric_columns[0]
                cat_col = categorical_columns[0]
                unique_cats = df[cat_col].value_counts().head(5).index
                filtered_df = df[df[cat_col].isin(unique_cats)]
                
                if len(filtered_df) > 0:
                    sns.boxplot(data=filtered_df, x=cat_col, y=num_col, ax=axes[1, 1])
                    axes[1, 1].tick_params(axis='x', rotation=45)
                    axes[1, 1].set_title(f'{num_col} 按 {cat_col} 分组')
                else:
                    axes[1, 1].text(0.5, 0.5, '数据不足', ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('数据不足')
            else:
                axes[1, 1].text(0.5, 0.5, '缺少数值或分类列', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('缺少数值或分类列')
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("没有找到数值列或分类列，无法创建可视化")
        
        # 统计摘要
        st.subheader("📋 统计摘要")
        if numeric_columns:
            st.write("**数值列统计:**")
            st.dataframe(df[numeric_columns].describe())
        
        if categorical_columns:
            st.write("**分类列统计:**")
            for col in categorical_columns:
                st.write(f"**{col}** 的唯一值数量: {df[col].nunique()}")
                st.write(f"**{col}** 的前5个值:")
                st.dataframe(df[col].value_counts().head())
        
        return True
        
    except Exception as e:
        st.error(f"分析过程中出现错误: {str(e)}")
        st.write("**错误详情:**")
        st.code(str(e), language="python")
        return False

def main():
    st.title("🔧 智能数据分析测试")
    st.write("这个工具可以智能分析任何CSV数据集")
    
    # 文件上传
    uploaded_file = st.file_uploader("选择CSV文件", type="csv")
    
    if uploaded_file is not None:
        # 保存上传的文件
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"文件 {uploaded_file.name} 上传成功！")
        
        # 运行分析
        if st.button("🚀 开始智能分析"):
            with st.spinner("正在进行智能分析..."):
                success = safe_smart_analysis(uploaded_file.name)
                
                if success:
                    st.success("🎉 分析完成！")
                    st.balloons()
                else:
                    st.error("❌ 分析失败，请检查文件格式")
    
    # 或者直接指定文件路径
    st.subheader("或者直接指定文件路径")
    file_path = st.text_input("输入CSV文件路径", value="./customers-1000.csv")
    
    if st.button("📊 分析指定文件"):
        if file_path:
            with st.spinner("正在分析指定文件..."):
                success = safe_smart_analysis(file_path)
                
                if success:
                    st.success("🎉 分析完成！")
                    st.balloons()
                else:
                    st.error("❌ 分析失败，请检查文件路径")
        else:
            st.error("请输入文件路径")

if __name__ == "__main__":
    main() 