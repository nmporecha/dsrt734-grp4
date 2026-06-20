import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ==========================================
# 📊 MODULAR STATISTICAL PROCESSING FUNCTIONS
# ==========================================

def verify_normality(data_series, alpha=0.05):
    """
    Checks for Normality using the Shapiro-Wilk test.
    Returns a dict with key findings.
    """
    clean_data = data_series.dropna()
    if len(clean_data) < 3:
        return {
            "test_name": "Shapiro-Wilk Normality Test",
            "is_normal": False,
            "p_value": None,
            "statistic": None,
            "success": False,
            "message": "Insufficient data (minimum 3 samples required)."
        }
    
    if len(clean_data) > 5000:
        clean_data = clean_data.sample(5000, random_state=42)
        
    stat, p_val = stats.shapiro(clean_data)
    is_normal = p_val > alpha
    
    if is_normal:
        msg = f"Data appears normally distributed (Shapiro-Wilk p = {p_val:.4f} > {alpha})."
    else:
        msg = f"Data is NOT normally distributed (Shapiro-Wilk p = {p_val:.4f} <= {alpha})."
        
    return {
        "test_name": "Shapiro-Wilk Normality test",
        "is_normal": is_normal,
        "p_value": float(p_val),
        "statistic": float(stat),
        "success": True,
        "message": msg
    }

def calculate_correlation(df, col1, col2, alpha=0.05):
    """
    Takes two numeric column names, verifies normality of both, 
    and runs either Pearson Correlation (if both normal) or Spearman Correlation (if non-normal).
    """
    clean_df = df[[col1, col2]].dropna()
    if len(clean_df) < 5:
        return {
            "test_name": "Bivariate Correlation",
            "success": False,
            "message": "Insufficient data pairs (minimum 5 required)."
        }
        
    norm1 = verify_normality(clean_df[col1], alpha)
    norm2 = verify_normality(clean_df[col2], alpha)
    
    both_normal = norm1["is_normal"] and norm2["is_normal"]
    
    if both_normal:
        test_name = "Pearson product-moment correlation"
        corr_coef, p_val = stats.pearsonr(clean_df[col1], clean_df[col2])
    else:
        test_name = "Spearman rank correlation"
        corr_coef, p_val = stats.spearmanr(clean_df[col1], clean_df[col2])
        
    return {
        "test_name": test_name,
        "success": True,
        "correlation_coefficient": float(corr_coef),
        "p_value": float(p_val),
        "col1_normality": norm1,
        "col2_normality": norm2,
        "both_normal": both_normal,
        "message": f"Ran {test_name}: coefficient = {corr_coef:.4f}, p-value = {p_val:.4f}. Significant? {p_val < alpha}."
    }

def compare_groups(df, cat_col, num_col, alpha=0.05):
    """
    Compares a numerical variable across two categories.
    Runs Independent samples t-test or Mann-Whitney U test as fallback.
    """
    clean_df = df[[cat_col, num_col]].dropna()
    groups = clean_df[cat_col].unique()
    
    # Filter out empty or null groups
    groups = [g for g in groups if g is not None and str(g).strip() != ""]
    
    if len(groups) != 2:
        return {
            "test_name": "Group Comparison",
            "success": False,
            "message": f"Grouping variable must have exactly 2 unique groups. Found: {list(groups)}"
        }
        
    g1_data = clean_df[clean_df[cat_col] == groups[0]][num_col]
    g2_data = clean_df[clean_df[cat_col] == groups[1]][num_col]
    
    if len(g1_data) < 3 or len(g2_data) < 3:
        return {
            "test_name": "Group Comparison",
            "success": False,
            "message": "Each comparison group must have at least 3 samples."
        }
        
    norm1 = verify_normality(g1_data, alpha)
    norm2 = verify_normality(g2_data, alpha)
    
    both_normal = norm1["is_normal"] and norm2["is_normal"]
    
    if both_normal:
        # Check equal variance
        try:
            _, lev_p = stats.levene(g1_data, g2_data)
            equal_var = lev_p > alpha
        except Exception:
            equal_var = True
            
        test_name = "Independent Samples t-test"
        t_stat, p_val = stats.ttest_ind(g1_data, g2_data, equal_var=equal_var)
        result_msg = f"Ran {test_name} (equal_var={equal_var}): t = {t_stat:.4f}, p-value = {p_val:.4f}."
    else:
        test_name = "Mann-Whitney U test"
        u_stat, p_val = stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
        result_msg = f"Ran {test_name}: U = {u_stat:.1f}, p-value = {p_val:.4f}."
        
    return {
        "test_name": test_name,
        "success": True,
        "statistic": float(t_stat if both_normal else u_stat),
        "p_value": float(p_val),
        "group_1": str(groups[0]),
        "group_2": str(groups[1]),
        "group_1_size": len(g1_data),
        "group_2_size": len(g2_data),
        "group_1_mean": float(g1_data.mean()),
        "group_2_mean": float(g2_data.mean()),
        "both_normal": both_normal,
        "message": result_msg
    }

def run_linear_regression(df, iv_col, dv_col):
    """
    Fits an OLS linear regression model.
    Returns coefficients, p-values, and R-squared.
    """
    clean_df = df[[iv_col, dv_col]].dropna()
    if len(clean_df) < 5:
        return {
            "test_name": "OLS Linear Regression",
            "success": False,
            "message": "Insufficient data to fit regression line (minimum 5 points required)."
        }
        
    X = sm.add_constant(clean_df[iv_col])
    y = clean_df[dv_col]
    
    try:
        model = sm.OLS(y, X).fit()
        const_coef = model.params.get("const", 0.0)
        slope_coef = model.params.get(iv_col, 0.0)
        
        const_p = model.pvalues.get("const", 1.0)
        slope_p = model.pvalues.get(iv_col, 1.0)
        
        return {
            "test_name": "Ordinary Least Squares (OLS) Linear Regression",
            "success": True,
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "coefficient_intercept": float(const_coef),
            "coefficient_slope": float(slope_coef),
            "p_value_intercept": float(const_p),
            "p_value_slope": float(slope_p),
            "f_statistic": float(model.fvalue),
            "f_p_value": float(model.f_pvalue),
            "formula": f"{dv_col} = {const_coef:.4f} + ({slope_coef:.4f} * {iv_col})",
            "message": f"Successfully fit regression model (R² = {model.rsquared:.4f}). Significant? {slope_p < 0.05}."
        }
    except Exception as e:
        return {
            "test_name": "OLS Linear Regression",
            "success": False,
            "message": f"Failed to fit regression model: {str(e)}"
        }

def generate_analysis_plot(df, run_args):
    """
    Based on the run_args, returns a matplotlib figure object.
    """
    test_type = run_args.get("type")
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Custom look-and-feel of plots
    sns.set_theme(style="whitegrid")
    
    try:
        if test_type == "normality":
            col = run_args["col"]
            data = df[col].dropna()
            sns.histplot(data, kde=True, ax=ax, color="#4f46e5", alpha=0.7)
            ax.set_title(f"Normality Plot (Distribution with KDE curve) for {col}", fontsize=11, fontweight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            
        elif test_type == "correlation":
            col1 = run_args["col1"]
            col2 = run_args["col2"]
            clean_df = df[[col1, col2]].dropna()
            sns.scatterplot(data=clean_df, x=col1, y=col2, ax=ax, color="#059669", s=50, alpha=0.8)
            # Add regression line
            if len(clean_df) > 1:
                sns.regplot(data=clean_df, x=col1, y=col2, ax=ax, scatter=False, color="#ef4444", line_kws={"linewidth": 2})
            ax.set_title(f"Scatterplot: {col1} vs {col2} (with fitted trend line)", fontsize=11, fontweight="bold")
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            
        elif test_type == "group_comparison":
            cat_col = run_args["cat_col"]
            num_col = run_args["num_col"]
            clean_df = df[[cat_col, num_col]].dropna()
            # Boxplot with data points overlaid
            sns.boxplot(data=clean_df, x=cat_col, y=num_col, ax=ax, palette="Set2", showfliers=False, width=0.4)
            sns.stripplot(data=clean_df, x=cat_col, y=num_col, ax=ax, color="#1e293b", alpha=0.4, size=5, jitter=0.15)
            ax.set_title(f"Boxplot: {num_col} by {cat_col} (with data points overlaid)", fontsize=11, fontweight="bold")
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)
            
        elif test_type == "regression":
            iv_col = run_args["iv_col"]
            dv_col = run_args["dv_col"]
            clean_df = df[[iv_col, dv_col]].dropna()
            sns.scatterplot(data=clean_df, x=iv_col, y=dv_col, ax=ax, color="#3b82f6", s=50, alpha=0.8)
            if len(clean_df) > 1:
                sns.regplot(data=clean_df, x=iv_col, y=dv_col, ax=ax, scatter=False, color="#dc2626", line_kws={"linewidth": 2})
            ax.set_title(f"OLS Regression Line: Predictor = {iv_col}, Outcome = {dv_col}", fontsize=11, fontweight="bold")
            ax.set_xlabel(iv_col)
            ax.set_ylabel(dv_col)
    except Exception as plot_err:
        ax.text(0.5, 0.5, f"Plot Error: {str(plot_err)}", transform=ax.transAxes, ha="center", va="center", color="red")
        ax.set_title("Could not generate visualization")
        
    plt.tight_layout()
    return fig

# Set page configuration
st.set_page_config(
    page_title="StatMentor AI: Research & Decision Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title and Description
st.title("📊 StatMentor AI: Intelligent Research & Decision Support System")
st.markdown("""
Welcome to **StatMentor AI**! This intelligent workspace combines advanced Google Gemini generative AI 
with robust statistical computing to assist researchers, analysts, and decision-makers.
""")

# ==========================================
# 🔑 API KEY CONFIGURATION & SETUP
# ==========================================
# Retrieve the Google Gemini API Key securely from st.secrets or environment variables as default
env_api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

# Sidebar Configuration for API Key input field
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # User input field to let users paste their Google Gemini API Key
    api_key = st.text_input(
        "Google Gemini API Key:", 
        value=env_api_key,
        type="password", 
        help="Paste your Gemini API Key here to activate StatMentor chat. Get a key from Google AI Studio."
    )
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("🤖 Gemini API Connected!")
    else:
        st.warning("⚠️ Enter Google Gemini API Key to activate tutoring chatbot.")
        
    st.markdown("---")
    st.markdown("### System Info")
    st.info("Environment: Streamlit Sandbox\nModel: gemini-1.5-flash")

# ==========================================
# 🗄️ STATE INITIALIZATION (st.session_state)
# ==========================================
# Set up session_state variables for persistence across runs
if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None

if "file_name" not in st.session_state:
    st.session_state["file_name"] = ""

if "last_test_result" not in st.session_state:
    st.session_state["last_test_result"] = None

if "last_apa_report" not in st.session_state:
    st.session_state["last_apa_report"] = None

if "last_test_vars" not in st.session_state:
    st.session_state["last_test_vars"] = None

if "chat_history" not in st.session_state or len(st.session_state["chat_history"]) == 0:
    # Build welcoming message that reads the uploaded CSV's column names from state
    if st.session_state["uploaded_df"] is not None:
        cols_text = ", ".join([f"`{c}`" for c in st.session_state["uploaded_df"].columns])
        welcome_msg = (
            f"Hello there! I am **StatMentor AI**, your friendly and patient statistics tutor. 👋 "
            f"I see you have loaded the dataset **{st.session_state['file_name']}** with the following columns:\n"
            f"{cols_text}\n\n"
            f"Let's work together to structure your research study! To get started, I'd love to guide you through three important aspects:\n"
            f"1. **What is your primary research question?**\n"
            f"2. **Which of the columns above are your Independent Variables (IVs) and Dependent Variables (DVs)?**\n"
            f"3. **What are their measurement scales (Nominal, Ordinal, Interval, or Ratio)?**\n\n"
            f"If you're not sure what IVs/DVs or measurement scales mean, don't worry! Tell me what you're interested in, and I will explain them simply."
        )
    else:
        welcome_msg = (
            "Hello there! I am **StatMentor AI**, your friendly and patient statistics tutor. 👋 "
            "Let's work together to design and structure your research plan!\n\n"
            "I noticed you haven't uploaded a dataset yet. Please select or drag in a CSV file or load our sample "
            "Student Performance dataset in the right column. Once uploaded, I'll automatically read its column names, "
            "and we can build your research plan together!\n\n"
            "In the meantime, feel free to ask me general questions about statistical concepts or variable scales!"
        )
    st.session_state["chat_history"] = [("assistant", welcome_msg)]

# ==========================================
# 📐 TWO-COLUMN STRUCTURE
# ==========================================
# App layout: Two equal-width columns as specified
col1, col2 = st.columns(2)

# ==========================================
# COLUMN 1: RESEARCH & CHAT WORKSPACE
# ==========================================
with col1:
    st.header("🔍 Research & Chat Workspace")
    st.markdown("""
    Interact with StatMentor AI, a patient statistics tutor. Discuss research questions, 
    independent & dependent variables, and data measurement scales.
    """)
    
    # Check if a dataset has been uploaded and prepare data context for Gemini
    data_context = ""
    if st.session_state["uploaded_df"] is not None:
        df = st.session_state["uploaded_df"]
        num_rows, num_cols = df.shape
        columns_list = list(df.columns)
        head_data = df.head(3).to_string()
        
        data_context = f"\n\nCURRENT DATASET CONTEXT (User uploaded file: '{st.session_state['file_name']}'):"
        data_context += f"\n- Total Rows: {num_rows}, Columns Count: {num_cols}"
        data_context += f"\n- List of Columns: {', '.join(columns_list)}"
        data_context += f"\n- First few rows preview:\n{head_data}"
        data_context += "\n\nPlease ground your tutoring assistance based on the variables listed in this active dataset."

    if st.session_state.get("last_test_result") is not None:
        data_context += f"\n\n🚨 LATEST USER RUN TEST RESULTS:\n{st.session_state['last_test_result']}"
        data_context += "\nPlease help the user understand and interpret this statistical test result with warm pedagogical coaching (null hypothesis, effect size, statistical significance, and assumptions)."

    # Clear chat option (reinitialized with correct welcoming message based on active columns)
    if st.button("🔄 Reset Tutor Conversation", key="clear_chat"):
        if st.session_state["uploaded_df"] is not None:
            cols_text = ", ".join([f"`{c}`" for c in st.session_state["uploaded_df"].columns])
            welcome_msg = (
                f"Hello there! I am **StatMentor AI**, your friendly and patient statistics tutor. 👋 "
                f"I see you have loaded the dataset **{st.session_state['file_name']}** with the following columns:\n"
                f"{cols_text}\n\n"
                f"Let's work together to structure your research study! To get started, I'd love to guide you through three important aspects:\n"
                f"1. **What is your primary research question?**\n"
                f"2. **Which of the columns above are your Independent Variables (IVs) and Dependent Variables (DVs)?**\n"
                f"3. **What are their measurement scales (Nominal, Ordinal, Interval, or Ratio)?**\n\n"
                f"If you're not sure what IVs/DVs or measurement scales mean, don't worry! Tell me what you're interested in, and I will explain them simply."
            )
        else:
            welcome_msg = (
                "Hello there! I am **StatMentor AI**, your friendly and patient statistics tutor. 👋 "
                "Let's work together to design and structure your research plan!\n\n"
                "I noticed you haven't uploaded a dataset yet. Please select or drag in a CSV file or load our sample "
                "Student Performance dataset in the right column. Once uploaded, I'll automatically read its column names, "
                "and we can build your research plan together!"
            )
        st.session_state["chat_history"] = [("assistant", welcome_msg)]
        st.session_state["last_test_result"] = None
        st.session_state["last_apa_report"] = None
        st.session_state["last_test_vars"] = None
        st.rerun()

    st.markdown("---")
    
    # Display the classic Streamlit chat interface
    for role, text in st.session_state["chat_history"]:
        with st.chat_message(role, avatar="🤖" if role == "assistant" else None):
            st.markdown(text)

    # Chat input field
    user_query = st.chat_input("Ask StatMentor Tutor about your study plan or statistical scales...")
    
    if user_query:
        # Display user message instantly & append to chat history
        st.chat_message("user").markdown(user_query)
        st.session_state["chat_history"].append(("user", user_query))
        
        # Call Gemini API logic
        if not api_key:
            reply = (
                "⚠️ **Google Gemini API Key is missing**.\n\n"
                "Please add a valid key in the configuration input box in the sidebar to talk with me! "
                "I will then analyze your research question and dataset."
            )
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(reply)
            st.session_state["chat_history"].append(("assistant", reply))
        else:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("StatMentor formulating tutoring guidance..."):
                    try:
                        system_prompt = (
                            "You are StatMentor AI, a patient, warm, and highly encouraging statistics tutor for absolute beginners. "
                            "You guide users through structuring their research plan in a scaffolded, step-by-step dialogue.\n\n"
                            "Your core objective is to help the user identify:\n"
                            "1. What is their primary research question under study?\n"
                            "2. What are their Independent Variables (IVs) and Dependent Variables (DVs) based ONLY on the columns in their dataset?\n"
                            "3. What are the measurement scales (Nominal, Ordinal, Interval, or Ratio) for those variables?\n\n"
                            "Instruction Guidelines:\n"
                            "- Always explain concepts simply. If the user does not know what Nominal, Ordinal, Interval, or Ratio mean, explain them with fun, clear analogies (e.g., Nominal is like jerseys with team names, Ordinal is like military ranks, Scale/Interval/Ratio are like tape measures).\n"
                            "- Encourage the user. Validate their research interests and help them select appropriate columns from their file.\n"
                            "- Keep your responses organized, readable, and styled nicely with bold words and bullet points."
                        )
                        
                        # Build context
                        model = genai.GenerativeModel(
                            model_name="gemini-1.5-flash",
                            system_instruction=system_prompt
                        )
                        
                        full_prompt = "Conversation History:\n"
                        for role, text in st.session_state["chat_history"][:-1]:
                            full_prompt += f"{'User' if role=='user' else 'Assistant'}: {text}\n"
                        
                        full_prompt += data_context
                        full_prompt += f"\n\nUser: {user_query}\nTutor Answer:"
                        
                        response = model.generate_content(full_prompt)
                        reply = response.text
                        
                    except Exception as e:
                        reply = f"❌ **API Error**: StatMentor encountered a connection problem: {str(e)}"
                
                st.markdown(reply)
                st.session_state["chat_history"].append(("assistant", reply))
            
            # Request rerun to stabilize display
            st.rerun()

# ==========================================
# COLUMN 2: DATA DASHBOARD & VISUALIZATIONS
# ==========================================
with col2:
    st.header("📈 Data Dashboard & Visualizations")
    st.markdown("Upload a CSV file to generate smart diagnostic and descriptive reports.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    # If a new file is uploaded, update session state
    if uploaded_file is not None:
        try:
            # Prevent re-reading multiple times if it's already stored in session_state
            if st.session_state["file_name"] != uploaded_file.name:
                df = pd.read_csv(uploaded_file)
                st.session_state["uploaded_df"] = df
                st.session_state["file_name"] = uploaded_file.name
                st.toast(f"✅ Success! Loaded '{uploaded_file.name}'")
                st.rerun()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    # Display dataset details if available
    if st.session_state["uploaded_df"] is not None:
        df = st.session_state["uploaded_df"]
        st.markdown(f"### Active Dataset: **{st.session_state['file_name']}**")
        
        # 1. Total Rows & Columns (using st.metric cards)
        st.markdown("#### 📏 Dataset Dimensions")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Total Rows", value=f"{df.shape[0]:,}")
        with m_col2:
            st.metric(label="Total Columns", value=f"{df.shape[1]:,}")
            
        st.markdown("---")

        # 2. Detected Columns and their basic data types
        st.markdown("#### 🏷️ Detected Column Schema & Types")
        
        column_info = []
        for col in df.columns:
            # Simple data type classification
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            dtype_label = "Numeric (Float/Integer)" if is_numeric else "Text/Categorical (String)"
            column_info.append({
                "Column Name": col,
                "Datatype": str(df[col].dtype),
                "Classification": dtype_label
            })
            
        type_df = pd.DataFrame(column_info)
        st.dataframe(type_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")

        # 3. Automated Missing Data Report
        st.markdown("#### 🔍 Missing Data & Integrity Report")
        null_counts = df.isnull().sum()
        null_percent = (null_counts / len(df)) * 100
        
        missing_df = pd.DataFrame({
            "Column Name": df.columns,
            "Missing Values (Count)": null_counts,
            "Missingness (%)": null_percent.round(2)
        })
        
        # Filter for reporting all, but highlight columns with actual missingness
        has_missing = missing_df[missing_df["Missing Values (Count)"] > 0]
        
        if len(has_missing) == 0:
            st.success("🎉 Complete Dataset! No missing values detected in any columns.")
        else:
            st.warning(f"⚠️ Detected null values in {len(has_missing)} columns:")
            st.dataframe(has_missing, use_container_width=True, hide_index=True)
            
        st.markdown("---")

        # 4. Descriptive Statistics Table
        st.markdown("#### 📊 Descriptive Statistics (Numeric Columns Only)")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns detected in the dataset. Cannot compute mean, median, or standard deviation.")
        else:
            # Create a descriptive dataframe featuring Mean, Median (50%), and Standard Deviation (Std)
            desc_stats = df[numeric_cols].describe().T
            # Reorganize and polish the description columns
            desc_summary = pd.DataFrame({
                "Variable": desc_stats.index,
                "Mean": desc_stats["mean"].round(4),
                "Median (50%)": desc_stats["50%"].round(4),
                "Standard Deviation": desc_stats["std"].round(4),
                "Min": desc_stats["min"].round(4),
                "Max": desc_stats["max"].round(4)
            })
            
            st.dataframe(desc_summary, use_container_width=True, hide_index=True)
            
            # Interactive Visualizations (Extra high-quality feature for dashboards)
            st.markdown("---")
            st.markdown("#### 📈 Distribution Quick-Plot")
            selected_plot_col = st.selectbox("Select a numeric column to plot:", numeric_cols)
            
            if selected_plot_col:
                st.write(f"Distribution plot for `{selected_plot_col}`")
                # Plotting histogram natively in Streamlit
                st.bar_chart(df[selected_plot_col].value_counts().head(20))

            # Automated Statistical Testing Engine block
            st.markdown("---")
            st.markdown("#### 🔬 Native Statistical Testing Engine")
            st.markdown(
                "Run standard statistical tests natively on your active dataset. "
                "The actual math runs error-free, and results are fed directly into **StatMentor AI**'s "
                "active working memory to enable instant, guided interpretations."
            )
            
            test_type = st.selectbox(
                "Select Statistical Operation to Run:",
                [
                    "Verify Normality (Shapiro-Wilk)",
                    "Correlation Analysis (Pearson / Spearman)",
                    "Two-Group Comparison (t-test / Mann-Whitney)",
                    "OLS Linear Regression"
                ],
                key="stat_test_type"
            )
            
            all_cols = list(df.columns)
            numeric_cols_list = list(numeric_cols)
            cat_cols = [c for c in all_cols if c not in numeric_cols_list]
            
            can_run = False
            run_args = {}
            
            if test_type == "Verify Normality (Shapiro-Wilk)":
                if not all_cols:
                    st.error("No columns available.")
                else:
                    norm_col = st.selectbox("Select variable to check normality:", all_cols, key="norm_col_select")
                    can_run = True
                    run_args = {"type": "normality", "col": norm_col}
                        
            elif test_type == "Correlation Analysis (Pearson / Spearman)":
                if len(numeric_cols_list) < 2:
                    st.error("Requires at least 2 numeric columns for pairwise correlation analysis.")
                else:
                    c_col1 = st.selectbox("Select Variable 1 (Numeric):", numeric_cols_list, index=0, key="corr_v1")
                    c_col2 = st.selectbox("Select Variable 2 (Numeric):", numeric_cols_list, index=min(1, len(numeric_cols_list)-1), key="corr_v2")
                    if c_col1 == c_col2:
                        st.warning("Please select two different variables.")
                    else:
                        can_run = True
                        run_args = {"type": "correlation", "col1": c_col1, "col2": c_col2}
                        
            elif test_type == "Two-Group Comparison (t-test / Mann-Whitney)":
                two_class_cols = [c for c in all_cols if df[c].dropna().nunique() == 2]
                if not two_class_cols:
                    st.warning("No categorical variables with exactly 2 unique categories detected. Showing all categorical/text columns:")
                    group_cols_to_show = cat_cols if cat_cols else all_cols
                else:
                    group_cols_to_show = two_class_cols
                    
                if not group_cols_to_show:
                    st.error("No grouping column available.")
                elif not numeric_cols_list:
                    st.error("No numeric variable available to compare groups on.")
                else:
                    cat_col_selected = st.selectbox("Select Grouping Variable (e.g. Submission_Status):", group_cols_to_show, key="group_cat")
                    num_col_selected = st.selectbox("Select Test Variable (Numeric):", numeric_cols_list, key="group_num")
                    can_run = True
                    run_args = {"type": "group_comparison", "cat_col": cat_col_selected, "num_col": num_col_selected}
                        
            elif test_type == "OLS Linear Regression":
                if len(numeric_cols_list) < 2:
                    st.error("Requires at least 2 numeric columns for regression (1 predictor, 1 outcome).")
                else:
                    iv_col = st.selectbox("Select Predictor (IV - Numeric):", numeric_cols_list, index=0, key="reg_iv")
                    dv_col = st.selectbox("Select Outcome (DV - Numeric):", numeric_cols_list, index=min(1, len(numeric_cols_list)-1), key="reg_dv")
                    if iv_col == dv_col:
                        st.warning("Predictor (IV) and Outcome (DV) should be different variables.")
                    else:
                        can_run = True
                        run_args = {"type": "regression", "iv_col": iv_col, "dv_col": dv_col}
            
            # Draw unified Run Analysis button
            if can_run:
                if st.button("Run Analysis", type="primary", use_container_width=True):
                    with st.spinner("Processing Python statistical calculations..."):
                        if run_args["type"] == "normality":
                            res_val = verify_normality(df[run_args["col"]])
                            res_val["inputs"] = {"variable": run_args["col"]}
                        elif run_args["type"] == "correlation":
                            res_val = calculate_correlation(df, run_args["col1"], run_args["col2"])
                            res_val["inputs"] = {"variable_1": run_args["col1"], "variable_2": run_args["col2"]}
                        elif run_args["type"] == "group_comparison":
                            res_val = compare_groups(df, run_args["cat_col"], run_args["num_col"])
                            res_val["inputs"] = {"grouping_variable": run_args["cat_col"], "test_variable": run_args["num_col"]}
                        elif run_args["type"] == "regression":
                            res_val = run_linear_regression(df, run_args["iv_col"], run_args["dv_col"])
                            res_val["inputs"] = {"independent_variable": run_args["iv_col"], "dependent_variable": run_args["dv_col"]}
                        
                        st.session_state["last_test_result"] = res_val
                        st.session_state["last_test_vars"] = run_args
                        st.session_state["last_apa_report"] = None
                        
                    # Trigger APA Writer if successful using custom API logic
                    if res_val.get("success", False):
                        if api_key:
                            with st.spinner("Writing dissertation Chapter 4 narrative..."):
                                try:
                                    system_prompt_apa = (
                                        "You are an expert academic statistician writing a dissertation Chapter 4. "
                                        "Take these raw statistical results and generate: "
                                        "1. A plain-language interpretation of what this means for a layman. "
                                        "2. A formal, publication-ready dissertation narrative strictly adhering to current APA Style guidelines. "
                                        "3. A Markdown-formatted statistical table displaying the results cleanly."
                                    )
                                    model_apa = genai.GenerativeModel(
                                        model_name="gemini-1.5-flash",
                                        system_instruction=system_prompt_apa
                                    )
                                    prompt_apa = f"Here is the raw statistical output dictionary:\n{json.dumps(res_val, indent=2)}\n\nPlease write the final output strictly formatted withlayman explanation, APA narrative, and markdown table."
                                    response_apa = model_apa.generate_content(prompt_apa)
                                    st.session_state["last_apa_report"] = response_apa.text
                                except Exception as api_err:
                                    st.session_state["last_apa_report"] = f"❌ **Error generating APA report using Gemini:** {str(api_err)}"
                        else:
                            st.session_state["last_apa_report"] = "⚠️ **Cannot generate scholarly APA Narrative because Google Gemini API Key is missing.** Go to the sidebar and paste/enter a valid API Key to unlock automated academic writing!"
                    st.rerun()

            # Elegant interactive layout to display the outputs
            if st.session_state.get("last_test_result") is not None:
                res = st.session_state["last_test_result"]
                vars_info = st.session_state["last_test_vars"]
                
                st.markdown("---")
                st.subheader("📊 Latest Run Diagnostic Summary")
                
                if res.get("success", False):
                    st.success(f"📈 **Completed {res['test_name']} Successfully!**")
                    st.write(f"**Interpretation statement:** {res['message']}")
                    
                    tab1, tab2, tab3 = st.tabs(["🖼️ Automated Plot Visual", "🎓 APA Dissertation segment", "📋 Raw Dictionary"])
                    
                    with tab1:
                        st.markdown("##### 📈 Integrated Scientific Visualization")
                        # Plot the corresponding automated visualization graph natively
                        fig = generate_analysis_plot(df, vars_info)
                        st.pyplot(fig)
                        plt.close(fig) # Free matplotlib memory
                        
                    with tab2:
                        st.markdown("##### 📜 Guided Dissertation Narrations (APA 7th Edition)")
                        apa_text = st.session_state.get("last_apa_report")
                        if apa_text:
                            st.markdown(apa_text)
                        elif not api_key:
                            st.warning("⚠️ Enter Google Gemini API Key in the sidebar to activate dissertation report generator.")
                        else:
                            st.info("🔄 Processing academic summary. If it didn't generate, click 'Run Analysis' again.")
                            
                    with tab3:
                        st.markdown("##### 🗂️ Structured Data Outputs (JSON)")
                        st.json(res)
                        st.info("💡 **Tutor Sync Active**: You can now ask the chatbot on the left: *'Can you interpret the statistical results I just ran?'*")
                else:
                    st.error(f"❌ **Analysis Terminated**: {res.get('message')}")

    else:
        # Prompt to upload CSV if empty
        st.info("ℹ️ Please upload a CSV file to unlock the interactive data table, missing value analysis, and statistical summary tools!")
        
        # Provide sample data to make it easy to play around
        st.markdown("---")
        st.markdown("💡 **Don't have a CSV handy?** Click below to load a sample dataset:")
        
        if st.button("Load Sample Research Dataset"):
            # Create a mock Student Exam Score dataset
            np.random.seed(42)
            n_samples = 150
            sample_data = pd.DataFrame({
                "Student_ID": [f"STU_{i:04d}" for i in range(1, n_samples + 1)],
                "Study_Hours": np.random.normal(12, 3, n_samples).round(1).clip(2, 24),
                "Pre_Test_Score": np.random.normal(68, 10, n_samples).round(1).clip(30, 100),
                "Final_Grade": np.random.normal(74, 12, n_samples).round(1).clip(40, 100),
                "Attendance_Rate (%)": np.random.uniform(75, 100, n_samples).round(1),
                "Submission_Status": np.random.choice(["Submitted On Time", "Late Submission", "Grace Period"], n_samples, p=[0.8, 0.15, 0.05])
            })
            # Inject a few nan values for simulated missingness report
            sample_data.loc[np.random.choice(sample_data.index, 6), "Pre_Test_Score"] = np.nan
            sample_data.loc[np.random.choice(sample_data.index, 3), "Study_Hours"] = np.nan
            
            st.session_state["uploaded_df"] = sample_data
            st.session_state["file_name"] = "student_performance_sample.csv"
            st.toast("✅ Sample Student Exam Stats loaded!")
            st.rerun()
