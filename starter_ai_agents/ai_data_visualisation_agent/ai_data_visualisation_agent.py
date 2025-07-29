import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from openai import OpenAI
from e2b_code_interpreter import Sandbox

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

def code_interpret(e2b_code_interpreter: Sandbox, code: str) -> Optional[List[Any]]:
    with st.spinner('Executing code in E2B sandbox...'):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec = e2b_code_interpreter.run_code(code)

        # Display execution output for debugging
        if stderr_capture.getvalue():
            st.warning("Code execution warnings/errors:")
            st.text(stderr_capture.getvalue())

        if stdout_capture.getvalue():
            st.info("Code execution output:")
            st.text(stdout_capture.getvalue())

        if exec.error:
            st.error(f"Code execution error: {exec.error}")
            st.write("**Error Details:**")
            st.code(str(exec.error), language="python")
            st.write("**Troubleshooting Tips:**")
            st.write("1. Check if the column names in your dataset match what the code is trying to access")
            st.write("2. Make sure your dataset has the expected structure")
            st.write("3. Try asking a simpler question first to explore the data")
            return None
            
        # Debug: Show what results were returned
        if exec.results:
            st.success(f"Code executed successfully! Found {len(exec.results)} result(s)")
            for i, result in enumerate(exec.results):
                st.write(f"Result {i+1} type: {type(result)}")
                if hasattr(result, 'png'):
                    st.write(f"Result {i+1} has PNG data: {bool(result.png)}")
                if hasattr(result, 'figure'):
                    st.write(f"Result {i+1} has figure: {bool(result.figure)}")
        else:
            st.warning("Code executed but no results returned")
            
        return exec.results

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""

def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    # Update system prompt to include dataset path information
    system_prompt = f"""You're a Python data scientist and data visualization expert. You are given a dataset at path '{dataset_path}' and also the user's query.
You need to analyze the dataset and answer the user's query with a response and you run Python code to solve them.

IMPORTANT REQUIREMENTS:
1. Always use the dataset path variable '{dataset_path}' in your code when reading the CSV file.
2. ALWAYS start by exploring the dataset structure to understand what columns are available.
3. ALWAYS generate visualizations (charts, graphs, plots) to illustrate your analysis.
4. Use matplotlib, seaborn, or plotly for creating visualizations.
5. For matplotlib: ALWAYS call plt.show() after creating plots.
6. For seaborn: ALWAYS call plt.show() after creating plots.
7. For plotly: ALWAYS call fig.show() or display the figure.
8. Include both data analysis and visualization in your response.
9. Provide clear explanations of what the visualizations show.
10. Handle missing columns gracefully - check if columns exist before using them.

EXAMPLE CODE STRUCTURE:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read data
df = pd.read_csv('{dataset_path}')

# ALWAYS explore the dataset first
print("Dataset shape:", df.shape)
print("Dataset columns:", df.columns.tolist())
print("Dataset head:")
print(df.head())

# Identify column types
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print("Numeric columns:", numeric_columns)
print("Categorical columns:", categorical_columns)

# Create visualizations based on available data
plt.figure(figsize=(15, 10))

# Subplot 1: Distribution of first numeric column
if numeric_columns:
    plt.subplot(2, 2, 1)
    plt.hist(df[numeric_columns[0]], bins=20, alpha=0.7)
    plt.title(f'Distribution of {{numeric_columns[0]}}')
    plt.xlabel(numeric_columns[0])
    plt.ylabel('Frequency')

# Subplot 2: Bar plot of first categorical column
if categorical_columns:
    plt.subplot(2, 2, 2)
    value_counts = df[categorical_columns[0]].value_counts().head(10)
    plt.bar(range(len(value_counts)), value_counts.values)
    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
    plt.title(f'Top 10 {{categorical_columns[0]}}')
    plt.xlabel(categorical_columns[0])
    plt.ylabel('Count')

# Subplot 3: Correlation heatmap (if multiple numeric columns)
if len(numeric_columns) > 1:
    plt.subplot(2, 2, 3)
    correlation = df[numeric_columns].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')

# Subplot 4: Box plot (if both numeric and categorical columns)
if numeric_columns and categorical_columns:
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x=categorical_columns[0], y=numeric_columns[0])
    plt.xticks(rotation=45)
    plt.title(f'{{numeric_columns[0]}} by {{categorical_columns[0]}}')

plt.tight_layout()
plt.show()  # THIS IS CRUCIAL!

# Print summary statistics
print("\\nSummary Statistics:")
print(df.describe())
```"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner('Getting response from LLM model...'):
        # Determine base URL based on selected model
        if "deepseek" in st.session_state.model_name.lower():
            base_url = "https://api.deepseek.com/v1"
        elif "claude" in st.session_state.model_name.lower():
            base_url = "https://api.anthropic.com/v1"
        else:
            base_url = "https://api.openai.com/v1"
            
        client = OpenAI(
            api_key=st.session_state.openai_api_key,
            base_url=base_url
        )
        response = client.chat.completions.create(
            model=st.session_state.model_name,
            messages=messages,
        )

        response_message = response.choices[0].message
        python_code = match_code_blocks(response_message.content)
        
        if python_code:
            code_interpreter_results = code_interpret(e2b_code_interpreter, python_code)
            return code_interpreter_results, response_message.content
        else:
            st.warning(f"Failed to match any Python code in model's response")
            return None, response_message.content

def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error


def main():
    """Main Streamlit application."""
    st.title("📊 AI Data Visualization Agent")
    st.write("Upload your dataset and ask questions about it!")

    # Initialize session state variables
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''
    if 'e2b_api_key' not in st.session_state:
        st.session_state.e2b_api_key = ''
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''

    with st.sidebar:
        st.header("API Keys and Model Configuration")
        st.session_state.openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        st.sidebar.info("💡 Using OpenAI-compatible API with various models")
        st.sidebar.markdown("[Get OpenAI API Key](https://platform.openai.com/api-keys)")
        
        st.session_state.e2b_api_key = st.sidebar.text_input("Enter E2B API Key", type="password")
        st.sidebar.markdown("[Get E2B API Key](https://e2b.dev/docs/legacy/getting-started/api-key)")
        
        # Add model selection dropdown
        model_options = {
            "DeepSeek Chat": "deepseek-chat",
            "DeepSeek V3": "deepseek-v3",
            "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
            "GPT-4o": "gpt-4o"
        }
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0  # Default to DeepSeek Chat
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Display dataset with toggle
        df = pd.read_csv(uploaded_file)
        st.write("Dataset:")
        show_full = st.checkbox("Show full dataset")
        if show_full:
            st.dataframe(df)
        else:
            st.write("Preview (first 5 rows):")
            st.dataframe(df.head())
        # Query input
        query = st.text_area("What would you like to know about your data?",
                            "Can you show me the basic information about this dataset and create a simple visualization?")
        
        # Add a button to explore dataset structure
        if st.button("🔍 Explore Dataset Structure"):
            with st.spinner('Analyzing dataset structure...'):
                st.subheader("Dataset Information")
                st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                st.write(f"**Columns:** {list(df.columns)}")
                st.write("**Data Types:**")
                st.write(df.dtypes)
                st.write("**First 5 rows:**")
                st.dataframe(df.head())
                st.write("**Missing values:**")
                st.write(df.isnull().sum())
        
        if st.button("Analyze"):
            if not st.session_state.openai_api_key or not st.session_state.e2b_api_key:
                st.error("Please enter both API keys in the sidebar.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)
                    
                    # Pass dataset_path to chat_with_llm
                    code_results, llm_response = chat_with_llm(code_interpreter, query, dataset_path)
                    
                    # Extract and display the generated Python code
                    python_code = match_code_blocks(llm_response)
                    if python_code:
                        st.subheader("Generated Python Code:")
                        st.code(python_code, language="python")
                    
                    # Display LLM's text response
                    st.subheader("AI Response:")
                    st.write(llm_response)
                    
                    # Display results/visualizations
                    if code_results:
                        st.subheader("Generated Visualizations:")
                        st.write(f"Found {len(code_results)} result(s) from code execution")
                        
                        for i, result in enumerate(code_results):
                            st.write(f"**Result {i+1}:**")
                            st.write(f"Type: {type(result)}")
                            
                            # Handle different types of results
                            if hasattr(result, 'png') and result.png:  # Check if PNG data is available
                                try:
                                    # Decode the base64-encoded PNG data
                                    png_data = base64.b64decode(result.png)
                                    
                                    # Convert PNG data to an image and display it
                                    image = Image.open(BytesIO(png_data))
                                    st.image(image, caption=f"Visualization {i+1}", use_container_width=True)
                                    st.success(f"Successfully displayed PNG visualization {i+1}")
                                except Exception as e:
                                    st.error(f"Error displaying PNG image: {e}")
                                    
                            elif hasattr(result, 'figure'):  # For matplotlib figures
                                try:
                                    fig = result.figure  # Extract the matplotlib figure
                                    st.pyplot(fig)  # Display using st.pyplot
                                    st.success(f"Successfully displayed matplotlib figure {i+1}")
                                except Exception as e:
                                    st.error(f"Error displaying matplotlib figure: {e}")
                                    
                            elif hasattr(result, 'show'):  # For plotly figures
                                try:
                                    st.plotly_chart(result, use_container_width=True)
                                    st.success(f"Successfully displayed plotly chart {i+1}")
                                except Exception as e:
                                    st.error(f"Error displaying plotly chart: {e}")
                                    
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.write("DataFrame/Series result:")
                                st.dataframe(result)
                                
                            elif hasattr(result, 'to_html'):  # For other HTML-compatible objects
                                st.write("HTML-compatible result:")
                                st.write(result)
                                
                            else:
                                st.write(f"Unknown result type: {type(result)}")
                                st.write("Raw result:")
                                st.write(result)
                                
                            st.divider()
                    else:
                        st.error("No visualization results generated. This could be due to:")
                        st.write("1. The code didn't call plt.show() or fig.show()")
                        st.write("2. The visualization library wasn't properly imported")
                        st.write("3. The code had an error during execution")
                        st.write("4. The E2B sandbox didn't capture the visualization output")  

if __name__ == "__main__":
    main()