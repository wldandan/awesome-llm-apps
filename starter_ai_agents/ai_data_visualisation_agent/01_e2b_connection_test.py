import os
import streamlit as st
from e2b_code_interpreter import Sandbox

def simple_e2b_test(api_key: str):
    """按照 E2B 官方文档的简单测试"""
    try:
        st.info("🚀 Starting E2B Sandbox...")
        
        # 创建沙箱
        with Sandbox(api_key=api_key) as sandbox:
            st.success("✅ Sandbox created successfully!")
            
            # 执行简单的 Python 代码
            st.info("📝 Running simple Python code...")
            execution = sandbox.run_code('print("hello world")')
            
            if execution.error:
                st.error(f"❌ Code execution failed: {execution.error}")
                return False
            else:
                st.success("✅ Code execution successful!")
                st.text("Output:")
                st.code(execution.logs)
            
            # 列出根目录文件
            st.info("📁 Listing files in root directory...")
            files = sandbox.files.list('/')
            st.success("✅ File listing successful!")
            st.write("Files in root directory:")
            for file in files:
                st.write(f"- {file.name}")
            
            return True
            
    except Exception as e:
        st.error(f"❌ E2B test failed: {str(e)}")
        return False

def main():
    st.title("🔧 E2B Quick Test (Official Guide)")
    st.write("This test follows the [E2B official quickstart guide](https://e2b.dev/docs/quickstart)")
    
    # API Key input
    api_key = st.text_input("Enter your E2B API Key", type="password", 
                           help="Get your API key from E2B Dashboard")
    
    if st.button("Run E2B Test"):
        if not api_key:
            st.error("Please enter your E2B API key")
            return
        
        # Run the test
        success = simple_e2b_test(api_key)
        
        if success:
            st.success("🎉 E2B is working correctly!")
            st.balloons()
        else:
            st.error("❌ E2B test failed. Please check your API key.")
    
    # Instructions from official docs
    with st.expander("📖 How to get E2B API Key (Official Guide)"):
        st.write("""
        ### 1. Create E2B account
        - Every new E2B account gets $100 in credits
        - Sign up at [E2B Dashboard](https://e2b.dev/docs/quickstart)
        
        ### 2. Get your API key
        1. Navigate to the E2B Dashboard
        2. Copy your API key
        3. Paste it in the input field above
        
        ### 3. Test your setup
        - Click "Run E2B Test" to verify everything works
        - This follows the official quickstart guide
        """)

if __name__ == "__main__":
    main() 