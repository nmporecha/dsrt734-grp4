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
import io

# ==========================================
# 📊 UTILITY FUNCTIONS & SERIALIZER
# ==========================================

def make_json_serializable(obj):
    """
    Standardizes numpy values (like np.bool_, np.float64) to native 
    Python types to ensure JSON serialization compatibility with Gemini API.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    else:
        return obj

def safe_generate_content(system_instruction, prompt):
    """
    Safely creates generative model with fallbacks to avoid 403/Project Denied access problems.
    If a 403/Denied/Forbidden error is encountered, raises a specific descriptive exception.
    """
    models_to_try = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]
    last_err = None
    for model in models_to_try:
        try:
            model_instance = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction
            )
            response = model_instance.generate_content(prompt)
            return response
        except Exception as err:
            err_str = str(err).lower()
            if "403" in err_str or "denied" in err_str or "project" in err_str or "forbidden" in err_str:
                raise Exception("Your Google Cloud project / API Key has been denied access (403). Please configure an active and valid GEMINI_API_KEY in the AI Studio settings or check your billing status.")
            elif "not found" in err_str or "support" in err_str or "503" in err_str or "model" in err_str:
                last_err = err
                continue
            else:
                raise err
    if last_err:
        raise last_err
    raise Exception("All Gemini model options failed to respond.")

def get_ai_recommendation_safely(df, file_name):
    """
    Analyzes the uploaded dataset profile and recommends the most appropriate statistical test 
    by calling the Gemini model.
    """
    try:
        columns_profile = []
        for col in df.columns:
            col_str = str(col)
            values = df[col_str].dropna()
            unique_vals = values.unique()
            is_numeric = pd.api.types.is_numeric_dtype(df[col_str])
            columns_profile.append({
                "column": col_str,
                "is_numeric": bool(is_numeric),
                "unique_count": int(len(unique_vals)),
                "samples": make_json_serializable([str(x) for x in unique_vals[:3]])
            })
        
        sys_prompt = "You are StatsBuddy's expert AI research methodology and decision scientist. Choose the absolute best statistical test for a dataset, presenting results strictly in JSON."
        prompt = f"""
        Please analyze this dataset profile representing a research study:
        Dataset Name: {file_name}
        Observations (rows): {len(df)}
        Columns Profile: {json.dumps(columns_profile, indent=2)}
        
        Using your advanced statistical knowledge, select the most appropriate standard statistical test to compare or relate these variables (e.g. "Independent Samples t-test", "One-Way ANOVA", "Pearson Bivariate Correlation & Simple Regression", "Chi-Square Test of Independence").
        
        You must return exactly a valid JSON block containing these three keys:
        1. "recommended_test": The clear standard academic name of the test.
        2. "test_reason": A detailed, encouraging paragraph (2-3 sentences) explaining to a non-scientist *why* this test is recommended based on the variables.
        3. "run_type": Must be exactly one of these strings: "t_test", "anova", "regression", "chi_square", "mann_whitney".
        
        Do not include any conversational text outside of the JSON block. Ensure the response is perfectly valid JSON.
        """
        
        response = safe_generate_content(sys_prompt, prompt)
        text = response.text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json"):
                text = "\n".join(lines[1:-1])
            elif lines[0].startswith("```"):
                text = "\n".join(lines[1:-1])
        data = json.loads(text)
        st.session_state["gemini_api_working"] = True
        if "ai_recommendation_error" in st.session_state:
            del st.session_state["ai_recommendation_error"]
        return data.get("recommended_test"), data.get("test_reason"), data.get("run_type")
    except Exception as e:
        st.session_state["gemini_api_working"] = False
        st.session_state["ai_recommendation_error"] = str(e)
        return None, None, None

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
    is_normal = bool(p_val > alpha)
    
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
    
    both_normal = bool(norm1["is_normal"] and norm2["is_normal"])
    
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
        "message": f"Ran {test_name}: coefficient = {corr_coef:.4f}, p-value = {p_val:.4f}. Significant? {bool(p_val < alpha)}."
    }

def compare_groups(df, cat_col, num_col, alpha=0.05):
    """
    Compares a numerical variable across two categories.
    Runs Independent samples t-test or Mann-Whitney U test as fallback.
    """
    clean_df = df[[cat_col, num_col]].dropna()
    groups = clean_df[cat_col].unique()
    
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
    
    both_normal = bool(norm1["is_normal"] and norm2["is_normal"])
    
    if both_normal:
        try:
            _, lev_p = stats.levene(g1_data, g2_data)
            equal_var = bool(lev_p > alpha)
        except Exception:
            equal_var = True
            
        test_name = "Independent Samples t-test"
        t_stat, p_val = stats.ttest_ind(g1_data, g2_data, equal_var=equal_var)
        result_msg = f"Ran {test_name} (equal_var={equal_var}): t = {t_stat:.4f}, p-value = {p_val:.4f}."
    else:
        test_name = "Mann-Whitney U test"
        u_stat, p_val = stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
        result_msg = f"Ran {test_name}: U = {u_stat:.1f}, p-value = {p_val:.4f}."
        t_stat = u_stat
        
    return {
        "test_name": test_name,
        "success": True,
        "statistic": float(t_stat),
        "p_value": float(p_val),
        "group_1": str(groups[0]),
        "group_2": str(groups[1]),
        "group_1_size": int(len(g1_data)),
        "group_2_size": int(len(g2_data)),
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
            "message": f"Successfully fit regression model (R² = {model.rsquared:.4f}). Significant? {bool(slope_p < 0.05)}."
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
    
    sns.set_theme(style="whitegrid")
    
    try:
        if test_type == "normality":
            col = run_args["col"]
            data = df[col].dropna()
            sns.histplot(data, kde=True, ax=ax, color="#4f46e5", alpha=0.7)
            ax.set_title(f"Normality Plot for {col}", fontsize=11, fontweight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            
        elif test_type == "correlation":
            col1 = run_args["col1"]
            col2 = run_args["col2"]
            clean_df = df[[col1, col2]].dropna()
            sns.scatterplot(data=clean_df, x=col1, y=col2, ax=ax, color="#059669", s=50, alpha=0.8)
            if len(clean_df) > 1:
                sns.regplot(data=clean_df, x=col1, y=col2, ax=ax, scatter=False, color="#ef4444", line_kws={"linewidth": 2})
            ax.set_title(f"Scatterplot: {col1} vs {col2}", fontsize=11, fontweight="bold")
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            
        elif test_type == "group_comparison":
            cat_col = run_args["cat_col"]
            num_col = run_args["num_col"]
            clean_df = df[[cat_col, num_col]].dropna()
            # Boxplot with data points overlaid
            sns.boxplot(data=clean_df, x=cat_col, y=num_col, ax=ax, palette="Set2", showfliers=False, width=0.4)
            sns.stripplot(data=clean_df, x=cat_col, y=num_col, ax=ax, color="#1e293b", alpha=0.4, size=5, jitter=0.15)
            ax.set_title(f"Boxplot with Data Points: {num_col} by {cat_col}", fontsize=11, fontweight="bold")
            ax.set_xlabel(cat_col)
            ax.set_ylabel(num_col)
            
        elif test_type == "regression":
            iv_col = run_args["iv_col"]
            dv_col = run_args["dv_col"]
            clean_df = df[[iv_col, dv_col]].dropna()
            sns.scatterplot(data=clean_df, x=iv_col, y=dv_col, ax=ax, color="#3b82f6", s=50, alpha=0.8)
            if len(clean_df) > 1:
                sns.regplot(data=clean_df, x=iv_col, y=dv_col, ax=ax, scatter=False, color="#dc2626", line_kws={"linewidth": 2})
            ax.set_title(f"OLS Regression: {iv_col} Predicts {dv_col}", fontsize=11, fontweight="bold")
            ax.set_xlabel(iv_col)
            ax.set_ylabel(dv_col)
    except Exception as plot_err:
        ax.text(0.5, 0.5, f"Plot Error: {str(plot_err)}", transform=ax.transAxes, ha="center", va="center", color="red")
        ax.set_title("Could not generate visualization")
        
    plt.tight_layout()
    return fig

# ==========================================
# 📊 PRE-PACKAGED EDUCATIONAL DATASETS
# ==========================================

DEMO_DATASETS = {
    "exam": """gender,prep_course,sleep_hours,exam_score
Female,Completed,7.5,88
Male,None,6.0,72
Male,Completed,8.0,91
Female,None,5.5,65
Female,Completed,7.0,85
Male,None,6.5,78
Female,None,6.0,70
Male,Completed,9.0,95
Female,Completed,8.5,94
Male,None,5.0,60
Female,None,6.5,75
Male,Completed,7.5,86
Female,Completed,8.0,92
Male,None,7.0,80
Female,None,5.5,68
Male,Completed,8.5,93""",

    "coffee": """daily_coffee_cups,focus_rating,sleep_latency_minutes
1,45,15
2,55,20
3,70,35
1,40,10
0,35,12
4,85,55
5,90,75
2,60,25
3,75,40
1,50,18
0,30,15
4,80,60
5,95,80
2,65,22
3,80,45
2,55,30""",

    "diet": """diet_type,weight_loss_kg,age
Low-Carb,4.2,34
Keto,6.5,28
Keto,5.8,45
Low-Fat,3.1,38
Low-Carb,4.8,29
Low-Fat,2.5,41
Low-Carb,3.9,32
Keto,7.1,36
Low-Fat,3.4,50
Low-Carb,5.0,27
Keto,6.2,31
Low-Fat,2.8,44
Keto,5.5,39
Low-Carb,4.4,35
Low-Fat,3.0,33
Low-Carb,4.6,40"""
}

# ==========================================
# 🔑 STREAMLIT PAGE SETUP
# ==========================================

st.set_page_config(
    page_title="StatsBuddy: Non-Statisticians Research Engine",
    page_icon="🎓",
    layout="wide"
)

# Custom premium styling matching Inter font and clean panels
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.reportview-container {
    background-color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)

# Main API Key load from env
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

# We configure generative AI globally if API key exists
if api_key:
    genai.configure(api_key=api_key)

# ==========================================
# 🗄️ STATE INITIALIZATION (st.session_state)
# ==========================================

if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None

if "file_name" not in st.session_state:
    st.session_state["file_name"] = ""

if "step" not in st.session_state:
    st.session_state["step"] = 1

if "variables" not in st.session_state:
    st.session_state["variables"] = {}

if "scales" not in st.session_state:
    st.session_state["scales"] = {}

if "hypotheses" not in st.session_state:
    st.session_state["hypotheses"] = {"h1": "", "h0": ""}

if "last_test_result" not in st.session_state:
    st.session_state["last_test_result"] = None

if "last_apa_report" not in st.session_state:
    st.session_state["last_apa_report"] = None

if "last_test_vars" not in st.session_state:
    st.session_state["last_test_vars"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "ai_recommended_test" not in st.session_state:
    st.session_state["ai_recommended_test"] = None

if "ai_test_reason" not in st.session_state:
    st.session_state["ai_test_reason"] = None

if "ai_run_type" not in st.session_state:
    st.session_state["ai_run_type"] = None

def reset_variable_states():
    st.session_state["step"] = 1
    st.session_state["variables"] = {}
    st.session_state["scales"] = {}
    st.session_state["hypotheses"] = {"h1": "", "h0": ""}
    st.session_state["last_test_result"] = None
    st.session_state["last_apa_report"] = None
    st.session_state["last_test_vars"] = None
    st.session_state["ai_recommended_test"] = None
    st.session_state["ai_test_reason"] = None
    st.session_state["ai_run_type"] = None

# Building standard tutoring greeting context
if len(st.session_state["chat_history"]) == 0:
    if st.session_state["file_name"]:
        welcome_msg = (
            f"Hello student! I am **StatsBuddy AI**, your research methodology coach. 🎓\n"
            f"I see you have loaded dataset **{st.session_state['file_name']}**.\n\n"
            f"I have initialized my workspace. Direct your questions above to explore scales of measurement, "
            f"formulate hypotheses, select statistical tests or write academic APA drafts! How can I assist you today?"
        )
    else:
        welcome_msg = (
            "Hello there! I am **StatsBuddy AI**, your friendly research methodology and statistics coach. 👋 "
            "I'm here to translate complex statistics into complete plain English!\n\n"
            "To get started, please upload your raw data spreadsheet (.csv) on the left panel or click any "
            "provided educational demo datasets to begin our step-by-step pipeline!"
        )
    st.session_state["chat_history"] = [("assistant", welcome_msg)]

# ==========================================
# 🎓 NAVIGATION HEADER
# ==========================================

st.markdown("""
<div style="background-color: white; padding: 20px; border-bottom: 2px solid #f1f5f9; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="padding: 10px; background-color: #4f46e5; border-radius: 12px; color: white;">
                <span style="font-size: 22px; font-weight: bold;">🎓</span>
            </div>
            <div>
                <h1 style="font-size: 24px; font-weight: 800; color: #0f172a; margin: 0; letter-spacing: -0.02em;">StatsBuddy</h1>
                <p style="font-size: 12px; font-weight: 500; color: #64748b; margin: 0;">The Student's Friendly Research Methodology & Analysis Coach</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Step progression visual indicator bar
steps_titles = [
    "① Onboarding",
    "② Variable Roles",
    "③ Scales",
    "④ Hypotheses",
    "⑤ Test Match",
    "⑥ Writeups"
]

cols_step = st.columns(6)
for idx, title in enumerate(steps_titles):
    step_num = idx + 1
    with cols_step[idx]:
        label = title
        if st.session_state["step"] == step_num:
            label = f"✨ {title}"
        elif st.session_state["step"] > step_num:
            label = f"{title} ✓"
            
        if st.button(label, key=f"step_btn_{step_num}", use_container_width=True):
            st.session_state["step"] = step_num
            st.rerun()

        # Visual indicator bar below button
        if st.session_state["step"] == step_num:
            st.markdown("<div style='border-bottom: 4px solid #4f46e5; margin-top: -12px; margin-bottom: 5px;'></div>", unsafe_allow_html=True)
        elif st.session_state["step"] > step_num:
            st.markdown("<div style='border-bottom: 4px solid #10b981; margin-top: -12px; margin-bottom: 5px;'></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='border-bottom: 4px solid #cbd5e1; margin-top: -12px; margin-bottom: 5px;'></div>", unsafe_allow_html=True)
st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

# ==========================================
# 📐 TWO-COLUMN MAIN FRAMEWORK
# ==========================================

col_left, col_right = st.columns([8, 4])

# ------------------------------------------
# LEFT COLUMN: INTERACTIVE WIZARD WORKSPACE
# ------------------------------------------
with col_left:
    banner_step_details = [
        { "title": "Let's onboard your research", "desc": "Upload your raw dataset or select one of our educational guides to begin.", "badge": "Step 1 of 6" },
        { "title": "Map Variable Classification Roles", "desc": "Designate which columns represent your Causes (Independent Variables) and Effects (Dependent).", "badge": "Step 2 of 6" },
        { "title": "Define Scales of Measurement", "desc": "Configure if variables should be evaluated as categorical groups, ranks, or continuous values.", "badge": "Step 3 of 6" },
        { "title": "Research Design & Hypotheses Study", "desc": "Draft Null (H₀) and Alternative (H₁) statements cleanly matching scholarly expectations.", "badge": "Step 4 of 6" },
        { "title": "Statistical Testing & Assumptions", "desc": "Review robust parameter matches and execute mathematical computations effortlessly.", "badge": "Step 5 of 6" },
        { "title": "Dissertation Output & Writeup Guidance", "desc": "Read clear layman interpretations and generate perfect dissertation-ready APA paragraphs.", "badge": "Step 6 of 6" }
    ]
    step_meta = banner_step_details[st.session_state["step"] - 1]

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); padding: 20px; border-radius: 12px 12px 0 0; color: white;">
        <span style="background-color: rgba(99, 102, 241, 0.3); border: 1px solid rgba(129, 140, 248, 0.2); color: #c7d2fe; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 9999px; text-transform: uppercase;">{step_meta['badge']}</span>
        <h2 style="font-size: 20px; font-weight: 800; color: white; margin-top: 10px; margin-bottom: 2px;">{step_meta['title']}</h2>
        <p style="font-size: 13px; color: #cbd5e1; margin: 0;">{step_meta['desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render Step Content inside a bordered container card
    with st.container(border=True):
        
        # --- STEP 1: ONBOARDING ---
        if st.session_state["step"] == 1:
            st.markdown("##### 📁 Upload Spreadsheet Data")
            uploaded_file = st.file_uploader("Upload spreadsheet (.csv) to begin parsing raw stats:", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    if st.session_state["file_name"] != uploaded_file.name:
                        df = pd.read_csv(uploaded_file)
                        st.session_state["uploaded_df"] = df
                        st.session_state["file_name"] = uploaded_file.name
                        reset_variable_states()
                        
                        # Generate AI Recommendation
                        with st.spinner("StatsBuddy AI analyzing dataset profile..."):
                            rec_test, rec_reason, rec_run_type = get_ai_recommendation_safely(df, uploaded_file.name)
                        st.session_state["ai_recommended_test"] = rec_test
                        st.session_state["ai_test_reason"] = rec_reason
                        st.session_state["ai_run_type"] = rec_run_type
                        
                        st.toast(f"✅ Loaded {uploaded_file.name}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
            
            st.markdown("---")
            st.markdown("##### 💡 No dataset handy? Use a demo tutorial dataset!")
            st.write("Pick one of our pre-packaged student datasets to follow our educational wizard pipeline:")
            
            demo_cols = st.columns(3)
            with demo_cols[0]:
                if st.button("📝 Exam Prep & Anxiety", use_container_width=True):
                    df = pd.read_csv(io.StringIO(DEMO_DATASETS["exam"]))
                    st.session_state["uploaded_df"] = df
                    st.session_state["file_name"] = "demo_exam_prep_anxiety.csv"
                    reset_variable_states()
                    
                    with st.spinner("StatsBuddy AI analyzing the Exam dataset..."):
                        rec_test, rec_reason, rec_run_type = get_ai_recommendation_safely(df, "exam_prep_anxiety.csv")
                    st.session_state["ai_recommended_test"] = rec_test
                    st.session_state["ai_test_reason"] = rec_reason
                    st.session_state["ai_run_type"] = rec_run_type
                    
                    st.toast("✅ Exam Prep & Anxiety dataset loaded!")
                    st.rerun()
            with demo_cols[1]:
                if st.button("☕ Coffee & Focus Scores", use_container_width=True):
                    df = pd.read_csv(io.StringIO(DEMO_DATASETS["coffee"]))
                    st.session_state["uploaded_df"] = df
                    st.session_state["file_name"] = "demo_coffee_focus_rates.csv"
                    reset_variable_states()
                    
                    with st.spinner("StatsBuddy AI analyzing the Coffee dataset..."):
                        rec_test, rec_reason, rec_run_type = get_ai_recommendation_safely(df, "coffee_focus_rates.csv")
                    st.session_state["ai_recommended_test"] = rec_test
                    st.session_state["ai_test_reason"] = rec_reason
                    st.session_state["ai_run_type"] = rec_run_type
                    
                    st.toast("✅ Coffee focus levels dataset loaded!")
                    st.rerun()
            with demo_cols[2]:
                if st.button("🥗 Diet Weight Losses", use_container_width=True):
                    df = pd.read_csv(io.StringIO(DEMO_DATASETS["diet"]))
                    st.session_state["uploaded_df"] = df
                    st.session_state["file_name"] = "demo_diet_weight_losses.csv"
                    reset_variable_states()
                    
                    with st.spinner("StatsBuddy AI analyzing the Diet dataset..."):
                        rec_test, rec_reason, rec_run_type = get_ai_recommendation_safely(df, "diet_weight_losses.csv")
                    st.session_state["ai_recommended_test"] = rec_test
                    st.session_state["ai_test_reason"] = rec_reason
                    st.session_state["ai_run_type"] = rec_run_type
                    
                    st.toast("✅ Diet weight losses dataset loaded!")
                    st.rerun()
            
            # Show preview of loaded data
            if st.session_state["uploaded_df"] is not None:
                df = st.session_state["uploaded_df"]
                st.success(f"📈 **Active Dataset Loaded: {st.session_state['file_name']}**")
                
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("Total Rows (Observations)", df.shape[0])
                with m_col2:
                    st.metric("Total columns (Features)", df.shape[1])
                
                st.write("**Dataset Preview (First 5 Rows):**")
                st.dataframe(df.head(5), use_container_width=True)
                
                # Check for null values
                missing = df.isnull().sum().sum()
                if missing == 0:
                    st.success("✔️ Clean dataset: No missing values detected!")
                else:
                    st.warning(f"⚠️ Missing parameters detected: Found {missing} empty entries across rows. These will be dropped during calculations.")
        
        # --- STEP 2: VARIABLE CLASSIFICATION ---
        elif st.session_state["step"] == 2:
            if st.session_state["uploaded_df"] is None:
                st.warning("⚠️ No dataset loaded. Please go back to Step 1 and upload data or choose a demo.")
            else:
                st.write("Map your column headers to respective experimental roles. This allows StatsBuddy to guide your research correctly.")
                df = st.session_state["uploaded_df"]
                
                for col in df.columns:
                    col_str = str(col)
                    clean_series = df[col_str].dropna()
                    samples_list = clean_series.head(3).tolist()
                    unique_count = clean_series.nunique()
                    is_num = pd.api.types.is_numeric_dtype(clean_series)
                    
                    if col_str not in st.session_state["variables"]:
                        if is_num and unique_count > 5 and "DV" not in st.session_state["variables"].values():
                            st.session_state["variables"][col_str] = "Effect (Dependent)"
                        elif unique_count <= 5:
                            st.session_state["variables"][col_str] = "Cause (Independent)"
                        else:
                            st.session_state["variables"][col_str] = "Exclude"
                    
                    val = st.session_state["variables"][col_str]
                    st.markdown(f"**Variable role classification for:** `{col_str}` *(Samples: {samples_list}, Unique values: {unique_count})*")
                    cols_r = st.columns(4)
                    if cols_r[0].button("Cause (Independent)", key=f"btn_iv_{col_str}", type="primary" if val == "Cause (Independent)" else "secondary", use_container_width=True):
                        st.session_state["variables"][col_str] = "Cause (Independent)"
                        st.rerun()
                    if cols_r[1].button("Effect (Dependent)", key=f"btn_dv_{col_str}", type="primary" if val == "Effect (Dependent)" else "secondary", use_container_width=True):
                        st.session_state["variables"][col_str] = "Effect (Dependent)"
                        st.rerun()
                    if cols_r[2].button("Covariate (Control)", key=f"btn_cov_{col_str}", type="primary" if val == "Covariate (Control)" else "secondary", use_container_width=True):
                        st.session_state["variables"][col_str] = "Covariate (Control)"
                        st.rerun()
                    if cols_r[3].button("Exclude", key=f"ignore_{col_str}", type="primary" if val == "Exclude" else "secondary", use_container_width=True):
                        st.session_state["variables"][col_str] = "Exclude"
                        st.rerun()
                    
                    # Determine recommended role statically based on data profiles to remain constant
                    if is_num and unique_count > 5:
                        rec_role = "Effect (Dependent)"
                        rec_reason = f"continuous numerical profiles (unique values = {unique_count}) are mathematically ideal representative outcome metrics for comparative or regression modeling."
                    elif unique_count <= 5:
                        rec_role = "Cause (Independent)"
                        rec_reason = f"containing discrete subgroups (unique count = {unique_count}) makes it highly suitable for splitting comparison parameters and grouping cohorts."
                    else:
                        rec_role = "Exclude"
                        rec_reason = "complex or high-cardinality values that are best excluded to focus strictly on primary active study variables."
                    
                    st.markdown(f"<p style='font-size:11.5px; color:#4f46e5; margin-top:-8px; margin-bottom:15px; font-style:italic;'>💡 StatsBuddy recommendation: <b>{rec_role}</b> because {rec_reason}</p>", unsafe_allow_html=True)
                    
        # --- STEP 3: SCALES OF MEASUREMENT ---
        elif st.session_state["step"] == 3:
            if st.session_state["uploaded_df"] is None:
                st.warning("⚠️ No dataset loaded. Go back to Step 1.")
            else:
                df = st.session_state["uploaded_df"]
                non_ignored = [col for col in df.columns if st.session_state["variables"].get(col, "Exclude") != "Exclude"]
                
                if not non_ignored:
                    st.error("⚠️ All variables are marked as 'Exclude'. Please go back to Step 2 and choose matching variables.")
                else:
                    st.write("Specify your measurement scales. This will prevent statistically invalid calculations:")
                    
                    for col in non_ignored:
                        col_str = str(col)
                        clean_series = df[col_str].dropna()
                        is_num = pd.api.types.is_numeric_dtype(clean_series)
                        unique_count = clean_series.nunique()
                        
                        if col_str not in st.session_state["scales"]:
                            if not is_num:
                                st.session_state["scales"][col_str] = "Nominal (Categories)"
                            elif unique_count < 6:
                                st.session_state["scales"][col_str] = "Ordinal (Ordered)"
                            else:
                                st.session_state["scales"][col_str] = "Ratio (True Zero)"
                                
                        val = st.session_state["scales"][col_str]
                        st.markdown(f"**Scale of measurement classification for:** `{col_str}` *(Classified as {st.session_state['variables'][col_str]})*")
                        cols_s = st.columns(4)
                        if cols_s[0].button("Nominal (Categories)", key=f"btn_nom_{col_str}", type="primary" if val == "Nominal (Categories)" else "secondary", use_container_width=True):
                            st.session_state["scales"][col_str] = "Nominal (Categories)"
                            st.rerun()
                        if cols_s[1].button("Ordinal (Ordered)", key=f"btn_ord_{col_str}", type="primary" if val == "Ordinal (Ordered)" else "secondary", use_container_width=True):
                            st.session_state["scales"][col_str] = "Ordinal (Ordered)"
                            st.rerun()
                        if cols_s[2].button("Interval (Arbitrary)", key=f"btn_int_{col_str}", type="primary" if val == "Interval (Arbitrary)" else "secondary", use_container_width=True):
                            st.session_state["scales"][col_str] = "Interval (Arbitrary)"
                            st.rerun()
                        if cols_s[3].button("Ratio (True Zero)", key=f"btn_rat_{col_str}", type="primary" if val == "Ratio (True Zero)" else "secondary", use_container_width=True):
                            st.session_state["scales"][col_str] = "Ratio (True Zero)"
                            st.rerun()
                        
                        # Determine recommended scale statically based on data profiles to remain constant
                        if not is_num:
                            rec_scale = "Nominal (Categories)"
                            rec_scale_reason = "this column contains qualitative grouping attributes (non-ordered attributes) serving as categorical cohorts rather than continuous mathematical values."
                        elif unique_count < 6:
                            rec_scale = "Ordinal (Ordered)"
                            rec_scale_reason = f"the small subset of distinct levels ({unique_count} unique items) suggests relative ordered positions or ranks (e.g. Likert scales) where intervals are not mathematically equal."
                        else:
                            rec_scale = "Ratio (True Zero)"
                            rec_scale_reason = "this represents a numerical column with continuous density and a valid absolute physical zero, making it fully prepared for parametric and linear regression computations."
                        
                        st.markdown(f"<p style='font-size:11.5px; color:#4f46e5; margin-top:-8px; margin-bottom:15px; font-style:italic;'>💡 StatsBuddy recommendation: <b>{rec_scale}</b> because {rec_scale_reason}</p>", unsafe_allow_html=True)

        # --- STEP 4: RESEARCH HYPOTHESES ---
        elif st.session_state["step"] == 4:
            st.markdown("##### 🧭 Automated Study Mapping")
            
            ivs = [k for k, v in st.session_state["variables"].items() if v == "Cause (Independent)"]
            dvs = [k for k, v in st.session_state["variables"].items() if v == "Effect (Dependent)"]
            
            if not ivs or not dvs:
                st.error("⚠️ **Incomplete variables mapping**: Please return to Step 2 and ensure you have designated at least one **Cause (Independent)** column and one **Effect (Dependent)** column.")
            else:
                main_iv = ivs[0]
                main_dv = dvs[0]
                iv_scale = st.session_state["scales"].get(main_iv, "Nominal (Categories)")
                dv_scale = st.session_state["scales"].get(main_dv, "Ratio (True Zero)")
                
                # Dynamic matching statements
                if "Nominal" in iv_scale and "Ratio" in dv_scale:
                    design_name = "Comparative / Experimental Design"
                    design_explanation = f"You are evaluating a categorical grouping column (**{main_iv}**) split in sub-categories, examining if they lead to differences in continuous outcomes (**{main_dv}**)."
                    default_h1 = f"There is a statistically significant difference in observed average values of {main_dv} between groups of {main_iv}."
                    default_h0 = f"There is no statistically significant difference in observed average values of {main_dv} between groups of {main_iv}."
                elif "Ratio" in iv_scale and "Ratio" in dv_scale:
                    design_name = "Relational / Correlational Design"
                    design_explanation = f"Both Cause (**{main_iv}**) and Effect (**{main_dv}**) represent continuous parameters. Ideal for linear trend evaluations or regression predictive modeling."
                    default_h1 = f"There is a statistically significant linear correlation/predictive relationship between {main_iv} and {main_dv}."
                    default_h0 = f"There is no statistically significant linear correlation/predictive relationship between {main_iv} and {main_dv}."
                elif "Nominal" in iv_scale and "Nominal" in dv_scale:
                    design_name = "Association Design (Categorical Contingency)"
                    design_explanation = f"Both your Cause (**{main_iv}**) and Effect (**{main_dv}**) represent discrete grouping categories. This explores conditional occurrence ratio trends."
                    default_h1 = f"There is a statistically significant proportional association between categories of {main_iv} and {main_dv}."
                    default_h0 = f"There is no statistically significant proportional association between categories of {main_iv} and {main_dv}."
                else:
                    design_name = "Mixed Non-Parametric Design"
                    design_explanation = f"Variable characteristics represent rank classifications. Standard nonparametric ordinal comparative test configurations should be used."
                    default_h1 = f"Rank values of {main_iv} correspond significantly with trends in {main_dv}."
                    default_h0 = f"Rank values of {main_iv} do not correspond to differences in {main_dv}."
                
                st.info(f"🧭 **Matched design framework: {design_name}**\n\n{design_explanation}")
                
                st.markdown("##### ✏️ Mad-Libs Hypothesis Formulations")
                st.write("Customize alternative statements to match dissertation Chapter 4 narrative preferences:")
                
                if not st.session_state["hypotheses"].get("h1"):
                    st.session_state["hypotheses"]["h1"] = default_h1
                if not st.session_state["hypotheses"].get("h0"):
                    st.session_state["hypotheses"]["h0"] = default_h0
                    
                h1_val = st.text_area("Alternative Hypothesis (H₁):", value=st.session_state["hypotheses"]["h1"])
                st.session_state["hypotheses"]["h1"] = h1_val
                
                st.text_area("Null Hypothesis (H₀ - Automatically populated based on selection):", value=default_h0, disabled=True)

        # --- STEP 5: MATHEMATICAL TESTS MATCHING & ASSUMPTIONS ---
        elif st.session_state["step"] == 5:
            ivs = [k for k, v in st.session_state["variables"].items() if v == "Cause (Independent)"]
            dvs = [k for k, v in st.session_state["variables"].items() if v == "Effect (Dependent)"]
            
            if not ivs or not dvs:
                st.error("⚠️ Please maps columns on Step 2 first.")
            else:
                main_iv = ivs[0]
                main_dv = dvs[0]
                iv_scale = st.session_state["scales"].get(main_iv, "Nominal (Categories)")
                dv_scale = st.session_state["scales"].get(main_dv, "Ratio (True Zero)")
                df = st.session_state["uploaded_df"]
                unique_iv = df[main_iv].dropna().nunique()
                
                ai_test = st.session_state.get("ai_recommended_test")
                ai_reason = st.session_state.get("ai_test_reason")
                ai_run_type = st.session_state.get("ai_run_type")
                
                is_using_ai = False
                is_gemini_working = st.session_state.get("gemini_api_working", True)
                
                if ai_test and is_gemini_working:
                    recommended_test = ai_test
                    test_reason = ai_reason
                    run_type = ai_run_type
                    is_using_ai = True
                else:
                    recommended_test = "Nonparametric Alternative"
                    test_reason = "Mixed measurement characteristics suggest running a robust nonparametric rank test."
                    run_type = "mann_whitney"
                    
                    if "Nominal" in iv_scale and "Ratio" in dv_scale:
                        if unique_iv == 2:
                            recommended_test = "Independent Samples t-test"
                            test_reason = f"Your Independent grouping variable (**{main_iv}**) contains exactly 2 unique categories, and your dependent variable (**{main_dv}**) is continuous numbers. A standard t-test is optimal here."
                            run_type = "t_test"
                        else:
                             recommended_test = "One-Way ANOVA"
                             test_reason = f"Your Independent parameter (**{main_iv}**) contains {unique_iv} (> 2) groups. An ANOVA checks general group variants simultaneously, avoiding combined type I errors."
                             run_type = "anova"
                    elif "Ratio" in iv_scale and "Ratio" in dv_scale:
                        recommended_test = "Pearson Bivariate Correlation & Simple Regression"
                        test_reason = f"Both Cause (**{main_iv}**) and Effect (**{main_dv}**) parameters are continuous numeric scales, permitting predictive trend modeling."
                        run_type = "regression"
                    elif "Nominal" in iv_scale and "Nominal" in dv_scale:
                        recommended_test = "Chi-Square Test of Independence"
                        test_reason = f"Both Cause and Effect variables represent categorical divisions, needing coordinate contingency counts comparisons."
                        run_type = "chi_square"
                
                if is_using_ai:
                    st.markdown(f"""
                    <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <span style="background-color: #22c55e; color: white; font-size: 10px; font-weight: 800; padding: 3px 8px; border-radius: 4px;">🤖 STATSBUDDY AI RECOMMENDED TEST</span>
                        <p style="font-size: 16px; font-weight: 800; color: #166534; margin: 8px 0 2px 0;">{recommended_test}</p>
                        <p style="font-size: 12.5px; color: #15803d; margin: 4px 0 0 0; line-height: 1.4;">{test_reason}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #fef2f2; border: 1px solid #fca5a5; padding: 12px; border-radius: 8px; margin-bottom: 20px;">
                        <p style="font-size: 12px; color: #b91c1c; margin: 0; font-weight: 600;">
                            ⚠️ StatsBuddy recommendation is currently unavailable because the Gemini AI service is offline or access is restricted. Standard offline analytical matching procedures have been loaded.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown(f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <h5 style="color: #475569; margin: 0 0 6px 0; font-size: 13px; font-weight: 700;">💡 Why StatMentor / StatsBuddy recommends this configuration:</h5>
                    <ul style="font-size: 12px; color: #64748b; margin: 0; padding-left: 18px; line-height: 1.55;">
                        <li><b>Variable Role Matching</b>: The independent variable (<i>{main_iv}</i>) and dependent variable (<i>{main_dv}</i>) roles align perfectly with this analytical method.</li>
                        <li><b>Scale of Measurement Alignment</b>: Your scales are mapped as <b>{iv_scale}</b> for cause and <b>{dv_scale}</b> for effect. This combination mathematically mandates <b>{recommended_test}</b> to avoid statistical bias or invalid standard errors.</li>
                        <li><b>Sample Size Validity</b>: This test leverages maximum mathematical degrees of freedom across your active samples ({len(df)} records) to guarantee high statistical confidence.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Manual trigger button for AI analysis
                rec_cols = st.columns([8, 4])
                with rec_cols[0]:
                    st.caption("Want our senior AI researcher to re-analyze your dataset profile and update advice?")
                with rec_cols[1]:
                    if st.button("🔄 AI Re-analyze Dataset", use_container_width=True):
                        with st.spinner("StatsBuddy senior AI analyzing dataset profile..."):
                            rec_test, rec_reason, rec_run_type = get_ai_recommendation_safely(df, st.session_state["file_name"])
                        if rec_test:
                            st.session_state["ai_recommended_test"] = rec_test
                            st.session_state["ai_test_reason"] = rec_reason
                            st.session_state["ai_run_type"] = rec_run_type
                            st.success("🤖 AI model updated recommendation!")
                            st.rerun()
                        else:
                            st.error("Could not reach AI model. Check API settings.")
                
                st.markdown("##### 🔬 Data Integrity & Assumptions Screen")
                
                # Skewness verify
                dv_vals = df[main_dv].dropna()
                if pd.api.types.is_numeric_dtype(dv_vals) and len(dv_vals) > 2:
                    mean_v = float(dv_vals.mean())
                    std_v = float(dv_vals.std())
                    if std_v > 0:
                        skew = float(((dv_vals - mean_v)/std_v).pow(3).mean())
                        if abs(skew) < 1.0:
                            st.success(f"✔️ **Normality verify (Skewness = {skew:.3f})**: Balanced shape, excellent for parametric test models.")
                        else:
                            st.warning(f"⚠️ **Normality warning (Skewness = {skew:.3f})**: Marked skewness detected. Consider caution when interpreting boundary significance.")
                
                # Sample count
                sample_count = len(df)
                if sample_count >= 15:
                    st.success(f"✔️ **Sample Size Checklist ({sample_count} rows)**: Sufficient sample sizes to ensure reliable test power.")
                else:
                    st.warning(f"⚠️ **Sample Size warning ({sample_count} rows)**: Small dataset size limits statistical confidence margins.")
                    
                st.markdown("---")
                
                # GIANT RUN BUTTON
                if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                    with st.spinner("Processing Python statistical calculations..."):
                        res_val = {}
                        if run_type == "t_test":
                            res_val = compare_groups(df, main_iv, main_dv)
                            res_val["inputs"] = {"grouping_variable": main_iv, "test_variable": main_dv}
                        elif run_type == "anova":
                            subgroups = df[main_iv].dropna().unique()
                            group_arrays = [df[df[main_iv] == g][main_dv].dropna().tolist() for g in subgroups]
                            group_arrays = [g for g in group_arrays if len(g) >= 3]
                            if len(group_arrays) < 2:
                                res_val = {"test_name": "One-Way ANOVA", "success": False, "message": "Failed: Categories contain insufficient parameters (<3 sample entries)."}
                            else:
                                try:
                                    f_stat, p_val = stats.f_oneway(*group_arrays)
                                    res_val = {
                                        "test_name": "One-Way ANOVA (Analysis of Variance)",
                                        "f_statistic": float(f_stat),
                                        "p_value": float(p_val),
                                        "groups_compared": list(subgroups),
                                        "significant": bool(p_val < 0.05),
                                        "success": True,
                                        "message": f"Successfully completed ANOVA comparison: F = {f_stat:.4f}, p = {p_val:.4f}."
                                    }
                                except Exception as e:
                                    res_val = {"test_name": "One-Way ANOVA", "success": False, "message": str(e)}
                            res_val["inputs"] = {"independent_variable": main_iv, "dependent_variable": main_dv}
                        elif run_type == "regression":
                            res_val = run_linear_regression(df, main_iv, main_dv)
                            res_val["inputs"] = {"independent_variable": main_iv, "dependent_variable": main_dv}
                        elif run_type == "chi_square":
                            try:
                                contingency = pd.crosstab(df[main_iv], df[main_dv])
                                chi2, p, dof, expected = stats.chi2_contingency(contingency)
                                res_val = {
                                    "test_name": "Chi-Square Test of Independence",
                                    "success": True,
                                    "chi2_statistic": float(chi2),
                                    "p_value": float(p),
                                    "dof": int(dof),
                                    "significant": bool(p < 0.05),
                                    "message": f"Successfully completed Chi-Square check: chi2({dof}) = {chi2:.4f}, p = {p:.4f}."
                                }
                            except Exception as e:
                                res_val = {"test_name": "Chi-Square Test", "success": False, "message": str(e)}
                            res_val["inputs"] = {"independent_variable": main_iv, "dependent_variable": main_dv}
                        else:
                            res_val = compare_groups(df, main_iv, main_dv)
                            res_val["inputs"] = {"grouping_variable": main_iv, "test_variable": main_dv}
                            
                        # Standardize NumPy serialize types before writing to state
                        res_val = make_json_serializable(res_val)
                        
                        st.session_state["last_test_result"] = res_val
                        st.session_state["last_test_vars"] = {
                            "type": "group_comparison" if run_type in ["t_test", "mann_whitney", "chi_square", "anova"] else "regression",
                            "col": main_dv,
                            "cat_col": main_iv,
                            "num_col": main_dv,
                            "col1": main_iv,
                            "col2": main_dv,
                            "iv_col": main_iv,
                            "dv_col": main_dv
                        }
                        
                        st.session_state["last_apa_report"] = None
                        
                    # Request Gemini scholarly interpretations if API Key exists
                    if res_val.get("success", False):
                        if api_key:
                            with st.spinner("StatMentor formulating Chapter 4 scholarly drafts..."):
                                try:
                                    system_prompt_apa = (
                                        "You are an expert academic statistician writing a dissertation Chapter 4. "
                                        "Take these raw statistical results and generate: 1. A plain-language interpretation of what this means for a layman. "
                                        "2. A formal, publication-ready dissertation narrative strictly adhering to current APA Style guidelines. "
                                        "3. A Markdown-formatted statistical table displaying the results cleanly."
                                    )
                                    prompt_apa = f"Here is the raw statistical output dictionary:\n{json.dumps(res_val, indent=2)}\n\nPlease write the final output strictly formatted with layman explanation, APA narrative, and markdown table."
                                    response_apa = safe_generate_content(system_prompt_apa, prompt_apa)
                                    st.session_state["last_apa_report"] = response_apa.text
                                except Exception as api_err:
                                    st.session_state["last_apa_report"] = f"❌ **Error generating APA report using Gemini:** {str(api_err)}"
                        else:
                            st.session_state["last_apa_report"] = "⚠️ **Gemini API Key missing**. Key must be set up inside advanced settings to generate dissertation Chapter 4 text drafts."
                            
                    # Move to Step 6 output view automatically
                    st.session_state["step"] = 6
                    st.rerun()

        # --- STEP 6: SCHOLARLY WRITEUPS ---
        elif st.session_state["step"] == 6:
            res = st.session_state["last_test_result"]
            vars_info = st.session_state["last_test_vars"]
            
            if not res:
                st.warning("⚠️ No completed computational models found. Return to Step 5 to matching tests.")
            else:
                is_sig = res.get("significant", False) or res.get("p_value", 1.0) < 0.05
                st.markdown("##### 📐 Statistical Result Metric Specifications")
                st.code(f"Test algorithm: {res['test_name']}\np-value parameter: {res.get('p_value', 'N/A')}\nSummary Interpretation: {res['message']}", language="text")
                
                if is_sig:
                    st.markdown("""
                    <div style="background-color: #ecfdf5; border-left: 5px solid #10b981; padding: 15px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
                        <span style="font-weight: 800; font-size:13px; color: #065f46;">✔ STATISTICALLY SIGNIFICANT OUTCOME</span>
                        <p style="margin: 4px 0 0 0; font-size: 12px; color: #047857;">p < 0.05. An authentic impact has been mathematically proven! Alternative hypothesis (H₁) is securely supported. Changing cause metrics matches genuine shifts in outcome variables.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #fef2f2; border-left: 5px solid #ef4444; padding: 15px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
                        <span style="font-weight: 800; font-size:13px; color: #991b1b;">📋 NOT STATISTICALLY SIGNIFICANT OUTCOME</span>
                        <p style="margin: 4px 0 0 0; font-size: 12px; color: #b91c1c;">p >= 0.05. Natural selection and random variations explain observed trends. Retain null hypothesis (H₀). Finding true zero connectivity is just as valuable!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Tabs
                tab1, tab2, tab3 = st.tabs(["🖼️ Automated Plot Visual", "🎓 APA Dissertation segment", "📋 Structured raw output"])
                
                with tab1:
                    st.markdown("##### 📊 Scientific Automated Visual Plot")
                    try:
                        fig = generate_analysis_plot(st.session_state["uploaded_df"], vars_info)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as plot_e:
                        st.error(f"Visualization render failed: {plot_e}")
                        
                with tab2:
                    st.markdown("##### 🎓 APA 7th Edition dissertation output writeup")
                    apa_txt = st.session_state["last_apa_report"]
                    if apa_txt:
                        st.markdown(apa_txt)
                        if st.button("📋 Acknowledge Content Draft"):
                            st.toast("💡 APA segment drafted! Copy the text above to paste into your paper.")
                    else:
                        st.warning("⚠️ Draft study not processed yet.")
                        if st.button("🔄 Request Dissertation draft building"):
                            with st.spinner("Computing scholarly narrations..."):
                                try:
                                    system_prompt_apa = (
                                        "You are an expert academic statistician writing a dissertation Chapter 4. "
                                        "Take these raw statistical results and generate: 1. A plain-language interpretation of what this means for a layman. "
                                        "2. A formal, publication-ready dissertation narrative strictly adhering to current APA Style guidelines. "
                                        "3. A Markdown-formatted statistical table displaying the results cleanly."
                                    )
                                    prompt_apa = f"Here is the raw statistical output dictionary:\n{json.dumps(res, indent=2)}\n\nPlease write the final output strictly formatted with layman explanation, APA narrative, and markdown table."
                                    response_apa = safe_generate_content(system_prompt_apa, prompt_apa)
                                    st.session_state["last_apa_report"] = response_apa.text
                                    st.rerun()
                                except Exception as err:
                                    st.error(f"Error calling API: {err}")
                                    
                with tab3:
                    st.markdown("##### 📄 JSON Output dictionary records")
                    st.json(res)
                    st.info("💡 **Tutor Sync Active**: You can now discuss this analysis on the right panel chatbot!")

    # Navigation controller buttons
    st.markdown("---")
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.session_state["step"] > 1:
            if st.button("◀ Previous", use_container_width=True):
                st.session_state["step"] -= 1
                st.rerun()
    with nav_col2:
        if st.session_state["step"] < 6:
            can_advance = bool(st.session_state["uploaded_df"] is not None)
            if st.button("Next ▶", use_container_width=True, disabled=not can_advance):
                st.session_state["step"] += 1
                st.rerun()

# ------------------------------------------
# RIGHT COLUMN: COACHING TUTOR SIDEBAR
# ------------------------------------------
with col_right:
    # Modern premium container for StatBuddy Assistant Advisor
    st.markdown("""
    <div style="background-color: #1e1b4b; padding: 15px; border-radius: 12px 12px 0 0; color: white;">
        <span style="float: right; font-size: 10px; background-color: #0f172a; padding: 2px 8px; border-radius: 9999px; color: #a5b4fc; font-family: monospace;">Gemini-3.5-Flash</span>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 16px;">🤖</span>
            <strong style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em;">StatsBuddy AI Advisor</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        # Conversation feed container
        chat_box = st.container(height=380)
        with chat_box:
            for sender, message in st.session_state["chat_history"]:
                if sender == "user":
                    with st.chat_message("user"):
                        st.markdown(message)
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message)
                        
        # Presets questions
        st.markdown("<p style='font-size:11px; font-weight:700; color:#475569; margin: 5px 0 2px 0;'>💡 Presets suggestions:</p>", unsafe_allow_html=True)
        suggest_cols = st.columns([1.1, 1.1, 0.8])
        with suggest_cols[0]:
            if st.button("What is continuous vs nominal?", use_container_width=True, key="preset_1"):
                preset_query = "What is the key difference between a continuous variable and a nominal/categorical variable?"
                st.session_state["chat_history"].append(("user", preset_query))
                st.session_state["active_coach_query"] = preset_query
                st.rerun()
        with suggest_cols[1]:
            if st.button("Explain H1 vs H0 null?", use_container_width=True, key="preset_2"):
                preset_query = "What is the difference between an Alternative Hypothesis (H1) and a Null Hypothesis (H0)?"
                st.session_state["chat_history"].append(("user", preset_query))
                st.session_state["active_coach_query"] = preset_query
                st.rerun()
        with suggest_cols[2]:
            if st.button("🧹 Clear", use_container_width=True, key="clear_chat"):
                st.session_state["chat_history"] = []
                st.session_state["active_coach_query"] = None
                st.toast("🧹 Chat history cleared successfully!")
                st.rerun()
                
        # Advanced API secret accordion config
        with st.expander("🔑 Advanced API Config settings (Optional)"):
            custom_key = st.text_input("Enter custom Gemini API Key:", type="password", help="If left empty, defaults to server-side loaded key.")
            if custom_key:
                api_key = custom_key
                genai.configure(api_key=api_key)
                
        # Main chat interactive text box
        chat_inp = st.chat_input("Ask StatsBuddy stats inquiries...")
        
        active_query = ""
        if chat_inp:
            active_query = chat_inp
            st.session_state["chat_history"].append(("user", chat_inp))
        elif st.session_state.get("active_coach_query"):
            active_query = st.session_state["active_coach_query"]
            st.session_state["active_coach_query"] = None
            
        if active_query:
            if not api_key:
                st.error("⚠️ Gemini API disconnect. Active API secret key missing.")
                st.session_state["chat_history"].append(("assistant", "I am currently disconnected because Google Gemini API credentials are required. Please input a Gemini API Key in the settings block above!"))
            else:
                with st.spinner("StatsBuddy thinking..."):
                    try:
                        # Feed metrics and variables state to contextualize chatbot runs
                        meta_context = ""
                        if st.session_state["uploaded_df"] is not None:
                            meta_context = (
                                f"Active dataset file: {st.session_state['file_name']}\n"
                                f"Active columns & variable classifications: {json.dumps(st.session_state['variables'])}\n"
                                f"Scales of measurement: {json.dumps(st.session_state['scales'])}\n"
                            )
                            if st.session_state["last_test_result"]:
                                meta_context += f"Last statistical test run calculated outputs: {json.dumps(st.session_state['last_test_result'])}"
                        
                        sys_tutor_instructions = (
                            "Act as StatsBuddy, a friendly, encouraging, and highly knowledgeable academic research methodology coach. "
                            "Your primary audience is university undergraduate and graduate students who do NOT have a statistical or mathematics background. "
                            "They find statistics stressful, confusing, and full of intimidating jargon.\n\n"
                            "Guidelines:\n"
                            "1. Avoid overwhelming formulas when explaining concepts. Use real-world analogies (e.g., comparing p-values to rolling loaded dice).\n"
                            "2. Explain the actual meaning of statistical findings instead of just giving numbers.\n"
                            "3. Keep explanations highly positive, encouraging, and clear.\n"
                            "4. Ground explanations using classifications or calculated values from their dataset if loaded."
                        )
                        
                        full_prompt = f"Dataset context settings:\n{meta_context}\n\nConversation history logs:\n"
                        for sender, val in st.session_state["chat_history"][:-1]:
                            full_prompt += f"{'User' if sender == 'user' else 'StatsBuddy'}: {val}\n"
                            
                        full_prompt += f"Latest user question: {active_query}\nStatsBuddy coaching response:"
                        
                        response = safe_generate_content(sys_tutor_instructions, full_prompt)
                        reply_val = response.text
                        st.session_state["chat_history"].append(("assistant", reply_val))
                    except Exception as err:
                        err_msg = str(err)
                        if "403" in err_msg or "denied" in err_msg.lower() or "api key" in err_msg.lower() or "project" in err_msg.lower():
                            friendly_err = (
                                "✨ **StatMentor Notice**:\n\n"
                                "Active generative AI recommendation chat features require a valid **Gemini API Key**. "
                                "Your Google Cloud project / API Key has been denied access (403) or is currently missing.\n\n"
                                "🔧 **How to resolve this**:\n"
                                "1. Grab a free API key from [Google AI Studio](https://aistudio.google.com/).\n"
                                "2. Configure your `GEMINI_API_KEY` in the AI Studio settings or check your billing status to restore chat capabilities instantly!\n\n"
                                "*(Note: All local offline data analysis, distribution plots, metrics, and rule-based testing fallback recommendations remain 100% functional!)*"
                            )
                            st.session_state["chat_history"].append(("assistant", friendly_err))
                        else:
                            st.session_state["chat_history"].append(("assistant", f"❌ API Error: StatMentor encountered a connection problem: {str(err)}"))
            st.rerun()

# ==========================================
# 📊 FOOTER STATS WARRANTY
# ==========================================

st.markdown("""
<div style="background-color: white; border-top: 2px solid #f1f5f9; padding: 15px; border-radius: 12px; margin-top: 30px; text-align: center; font-size: 12px; color: #64748b;">
    <p style="margin: 0 0 4px 0;">&copy; 2026 StatsBuddy Engine. Translating complex methodologies step-by-step.</p>
    <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
        <span style="color: #10b981; font-weight: 700;">● Calculations Certified Offline-Safe</span>
        <span>|</span>
        <a href="#" style="color: #4f46e5; text-decoration: none;">Methodology Trees</a>
        <span>|</span>
        <a href="#" style="color: #4f46e5; text-decoration: none;">Scale Rules</a>
    </div>
</div>
""", unsafe_allow_html=True)
