# AI 数据可视化代理 - 测试文件说明

## 文件结构

### 主应用
- **`ai_data_visualisation_agent.py`** - 主应用，AI 数据可视化代理

### 测试文件（按使用顺序）

#### 1. E2B 连接测试
- **`01_e2b_connection_test.py`** - 简单的 E2B 连接测试
  - 测试 E2B API 密钥是否有效
  - 验证基本代码执行功能
  - 检查文件操作功能
  - 运行命令：`streamlit run 01_e2b_connection_test.py`

#### 2. E2B 综合测试
- **`02_e2b_comprehensive_test.py`** - 完整的 E2B 功能测试
  - 测试代码执行、文件操作、可视化库
  - 验证 pandas、matplotlib、seaborn 等库
  - 提供详细的调试信息
  - 运行命令：`streamlit run 02_e2b_comprehensive_test.py`

#### 3. 数据分析脚本
- **`03_data_analysis_script.py`** - 独立的数据分析脚本
  - 智能分析任何 CSV 数据集
  - 自动检测数值列和分类列
  - 生成多种类型的可视化图表
  - 运行命令：`python 03_data_analysis_script.py`

#### 4. 数据分析 Streamlit 应用
- **`04_data_analysis_streamlit_app.py`** - 数据分析 Web 应用
  - 用户友好的 Web 界面
  - 支持文件上传和路径指定
  - 完整的错误处理和调试信息
  - 运行命令：`streamlit run 04_data_analysis_streamlit_app.py`

## 使用流程

### 1. 环境准备
```bash
# 激活 conda 环境
conda activate agno

# 安装依赖（如果需要）
pip install seaborn plotly
```

### 2. 测试 E2B 连接
```bash
# 简单连接测试
streamlit run 01_e2b_connection_test.py

# 或完整功能测试
streamlit run 02_e2b_comprehensive_test.py
```

### 3. 测试数据分析功能
```bash
# 命令行脚本测试
python 03_data_analysis_script.py

# 或 Web 应用测试
streamlit run 04_data_analysis_streamlit_app.py
```

### 4. 运行主应用
```bash
streamlit run ai_data_visualisation_agent.py
```

## 测试数据
- **`customers-1000.csv`** - 示例数据集，包含 1000 条客户记录

## 功能特点

### E2B 测试
- ✅ API 密钥验证
- ✅ 代码执行测试
- ✅ 文件操作测试
- ✅ 可视化库测试
- ✅ 错误处理和调试

### 数据分析测试
- ✅ 智能列类型检测
- ✅ 自动可视化生成
- ✅ 多种图表类型
- ✅ 统计摘要
- ✅ 用户友好界面

## 故障排除

### 常见问题
1. **ModuleNotFoundError: No module named 'seaborn'**
   - 解决：`pip install seaborn plotly`

2. **E2B 连接失败**
   - 检查 API 密钥是否正确
   - 确认网络连接正常

3. **文件不存在错误**
   - 确保 CSV 文件在正确路径
   - 检查文件权限

### 调试技巧
- 使用 `01_e2b_connection_test.py` 验证 E2B 环境
- 使用 `03_data_analysis_script.py` 测试数据分析功能
- 查看详细的错误信息和调试输出 