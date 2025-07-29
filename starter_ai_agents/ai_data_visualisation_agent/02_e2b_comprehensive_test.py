import streamlit as st
from e2b_code_interpreter import Sandbox
import io
import contextlib
import warnings

def test_e2b_connection(api_key: str):
    """测试 E2B 连接和基本功能"""
    try:
        with st.spinner('Testing E2B connection...'):
            with Sandbox(api_key=api_key) as sandbox:
                # 测试基本代码执行
                st.info("Testing basic code execution...")
                result = sandbox.run_code("print('Hello from E2B!')")
                
                if result.error:
                    st.error(f"Basic code execution failed: {result.error}")
                    return False
                else:
                    st.success("✅ Basic code execution successful!")
                
                # 测试文件操作
                st.info("Testing file operations...")
                test_content = "This is a test file content"
                sandbox.files.write("/test.txt", test_content)
                
                # 读取文件
                read_content = sandbox.files.read("/test.txt")
                if read_content == test_content:
                    st.success("✅ File operations successful!")
                else:
                    st.error("❌ File operations failed")
                    return False
                
                # 测试可视化库
                st.info("Testing visualization libraries...")
                viz_code = """
import matplotlib.pyplot as plt
import numpy as np

# Create a simple plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Test Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""
                viz_result = sandbox.run_code(viz_code)
                
                if viz_result.error:
                    st.error(f"Visualization test failed: {viz_result.error}")
                    return False
                
                if viz_result.results:
                    st.success("✅ Visualization libraries working!")
                    st.write(f"Found {len(viz_result.results)} visualization result(s)")
                    for i, result in enumerate(viz_result.results):
                        st.write(f"Result {i+1} type: {type(result)}")
                        if hasattr(result, 'png'):
                            st.write(f"Result {i+1} has PNG data: {bool(result.png)}")
                        if hasattr(result, 'figure'):
                            st.write(f"Result {i+1} has figure: {bool(result.figure)}")
                else:
                    st.warning("⚠️ Visualization test completed but no results returned")
                
                # 测试 pandas
                st.info("Testing pandas...")
                pandas_code = """
import pandas as pd
import numpy as np

# Create a test dataframe
df = pd.DataFrame({
    'A': np.random.randn(10),
    'B': np.random.randn(10),
    'C': ['test'] * 10
})
print("DataFrame created successfully:")
print(df.head())
"""
                pandas_result = sandbox.run_code(pandas_code)
                
                if pandas_result.error:
                    st.error(f"Pandas test failed: {pandas_result.error}")
                    return False
                else:
                    st.success("✅ Pandas working!")
                
                return True
                
    except Exception as e:
        st.error(f"❌ E2B connection failed: {str(e)}")
        return False

def main():
    st.title("🔧 E2B Connection Test")
    st.write("This tool tests if E2B is working correctly with your API key.")
    
    # API Key input
    api_key = st.text_input("Enter your E2B API Key", type="password")
    
    if st.button("Test E2B Connection"):
        if not api_key:
            st.error("Please enter your E2B API key")
            return
        
        # Run the test
        success = test_e2b_connection(api_key)
        
        if success:
            st.success("🎉 All tests passed! E2B is working correctly.")
            st.balloons()
        else:
            st.error("❌ Some tests failed. Please check your API key and try again.")
    
    # Instructions
    with st.expander("How to get E2B API Key"):
        st.write("""
        1. Go to [E2B Dashboard](https://e2b.dev/docs/legacy/getting-started/api-key)
        2. Sign up or log in to your account
        3. Navigate to the API Keys section
        4. Create a new API key
        5. Copy the key and paste it above
        """)

if __name__ == "__main__":
    main() 