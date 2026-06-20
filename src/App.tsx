import React, { useState, useEffect, useRef } from "react";
import Papa from "papaparse";
import { 
  Play, FileCode, Terminal, Copy, Trash2, Bot, User, Upload, ArrowRight,
  Sparkles, Check, BookOpen, AlertTriangle, Database, Columns, BarChart3,
  RefreshCw, Info, HelpCircle, FileSpreadsheet, ExternalLink, ChevronRight, Settings
} from "lucide-react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  LineChart, Line, ScatterChart, Scatter, Label
} from "recharts";

// ==========================================
// 💡 PRE-PACKAGED MOCK DATASET STRINGS (CSV Format)
// ==========================================
const STUDENT_PERFORMANCE_CSV = `Student_ID,Study_Hours,Pre_Test_Score,Final_Grade,Attendance_Rate_Pct,Submission_Status
STU_001,15.2,74.5,82.0,94.5,Submitted On Time
STU_002,10.1,58.0,70.5,88.2,Late Submission
STU_003,18.5,89.0,94.0,98.1,Submitted On Time
STU_004,8.4,,55.0,78.5,Grace Period
STU_005,14.6,71.2,79.5,,Submitted On Time
STU_006,12.0,68.0,76.5,92.0,Submitted On Time
STU_007,5.5,42.0,50.0,65.4,Late Submission
STU_008,19.2,95.0,99.2,99.5,Submitted On Time
STU_009,11.2,63.0,71.0,89.0,Submitted On Time
STU_010,,70.1,80.5,91.2,Submitted On Time
STU_011,16.4,85.2,91.0,96.4,Submitted On Time
STU_012,9.8,55.0,63.4,84.5,Late Submission
STU_013,13.8,77.5,84.0,93.2,Submitted On Time
STU_014,21.0,98.0,100.0,99.8,Submitted On Time
STU_015,14.0,,78.0,90.5,Submitted On Time
STU_016,11.5,62.4,72.2,87.6,Submitted On Time
STU_017,7.2,49.0,56.0,72.1,Late Submission
STU_018,17.8,88.0,93.5,97.5,Submitted On Time
STU_019,10.5,59.0,68.0,86.0,Grace Period
STU_020,13.1,72.0,,91.8,Submitted On Time
STU_021,15.0,78.2,85.0,94.0,Submitted On Time
STU_022,6.8,45.5,52.4,70.2,Late Submission
STU_023,19.8,94.0,98.5,99.0,Submitted On Time
STU_024,12.5,70.5,77.0,91.5,Submitted On Time
STU_025,8.9,52.0,60.5,81.4,Submitted On Time`;

const CLINICAL_TRIAL_CSV = `Patient_ID,Age,Dosage_mg,BP_Systolic,BP_Diastolic,Efficacy_Rating,Side_Effect_Severity
PAT_101,45,50,135,85,4.2,Mild
PAT_102,52,100,128,82,4.8,None
PAT_103,110,122,80,3.9,Moderate
PAT_104,28,50,,75,3.5,Mild
PAT_105,64,150,142,90,,Severe
PAT_106,37,100,121,78,4.5,None
PAT_107,49,50,130,84,4.1,None
PAT_108,71,150,148,93,4.9,Moderate
PAT_109,55,100,132,,4.4,Mild
PAT_110,42,50,126,81,3.8,None
PAT_111,60,150,139,88,4.6,Mild
PAT_112,31,100,124,79,4.0,None
PAT_113,47,50,129,83,3.7,None
PAT_114,68,150,145,91,4.7,Moderate
PAT_115,50,100,131,84,4.3,Mild`;

const RETAIL_SALES_CSV = `Transaction_ID,Price,Units_Sold,Store_Rating,Discount_Percentage,Category
TXN_501,19.99,145,4.2,10,Apparel
TXN_502,49.99,82,4.5,20,Electronics
TXN_503,12.50,230,3.9,0,Groceries
TXN_504,89.99,,4.1,15,Home Goods
TXN_505,15.99,112,3.8,5,Apparel
TXN_506,120.00,45,4.6,25,Electronics
TXN_507,8.50,340,4.3,,Groceries
TXN_508,35.00,95,4.0,10,Home Goods
TXN_509,24.99,150,4.1,0,Apparel
TXN_510,,68,3.7,5,Groceries
TXN_511,75.00,55,4.4,15,Home Goods
TXN_512,18.00,180,4.2,0,Apparel
TXN_513,110.00,38,4.5,20,Electronics
TXN_514,6.99,410,,10,Groceries
TXN_515,45.00,88,4.2,5,Home Goods`;

// ==========================================
// 🐍 python files
// ==========================================
const PYTHON_REQUIREMENTS_CODE = `# StatMentor AI - Streamlit Application Requirements
# Basic packages required for web development, data manipulation, statistical computing, and Google Gemini API

streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
google-generativeai>=0.8.3
`;

const PYTHON_APP_CODE = `import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os

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
# Retrieve the Google Gemini API Key securely from st.secrets or environment variables
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")

# Sidebar Configuration for API Key fallback if not set automatically
with st.sidebar:
    st.header("⚙️ Configuration")
    if not api_key:
        api_key_input = st.text_input(
            "Enter Google Gemini API Key:", 
            type="password", 
            help="Required for the Chat Workspace to function. Get a key from Google AI Studio."
        )
        if api_key_input:
            api_key = api_key_input
            genai.configure(api_key=api_key)
    else:
        st.success("🤖 Gemini API Connected (Loaded from Environment)")
        genai.configure(api_key=api_key)
        
    st.markdown("---")
    st.markdown("### System Info")
    st.info("Environment: Streamlit Sandbox\\nModel: gemini-3.5-flash")

# ==========================================
# 🗄️ STATE INITIALIZATION (st.session_state)
# ==========================================
# Set up session_state variables for persistence across runs
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None

if "file_name" not in st.session_state:
    st.session_state["file_name"] = ""

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
    Ask questions about statistical theory, choose appropriate tests, 
    or write custom queries to analyze your uploaded data.
    """)
    
    # Check if a dataset has been uploaded and prepare data context for Gemini
    data_context = ""
    if st.session_state["uploaded_df"] is not None:
        df = st.session_state["uploaded_df"]
        num_rows, num_cols = df.shape
        columns_list = list(df.columns)
        head_data = df.head(5).to_string()
        summary_stats = df.describe().to_string()
        
        data_context = f"\\\\n\\\\nCURRENT DATASET CONTEXT (User uploaded '{st.session_state['file_name']}'):"
        data_context += f"\\\\n- Row Count: {num_rows}, Column Count: {num_cols}"
        data_context += f"\\\\n- Columns: {', '.join(columns_list)}"
        data_context += f"\\\\n- Summary Descriptive Statistics:\\\\n{summary_stats}"
        data_context += f"\\\\n- First 5 rows of data:\\\\n{head_data}"
        data_context += "\\\\n\\\\nPlease refer to this dataset context if the user asks questions about analyzing or interpreting their dataset."

    # Clear chat option
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state["chat_history"] = []
        st.rerun()

    st.markdown("---")
    
    # Display historical chat messages
    chat_container = st.container(height=450)
    with chat_container:
        if len(st.session_state["chat_history"]) == 0:
            st.info("👋 Ask StatMentor AI a question! E.g., 'What is the difference between an ANOVA and a Chi-Square test?'")
        else:
            for role, text in st.session_state["chat_history"]:
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(text)
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(text)

    # Chat input field
    user_query = st.chat_input("Type your statistics or research question here...")
    
    if user_query:
        # Save user message to session memory
        st.session_state["chat_history"].append(("user", user_query))
        
        # Display instantly
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)
        
        # Call Gemini API logic
        if not api_key:
            reply = "⚠️ API Key is missing. Please provide your Google Gemini API Key in the sidebar config to activate StatMentor chat."
            st.session_state["chat_history"].append(("assistant", reply))
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(reply)
        else:
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Analyzing queries & drafting guidance..."):
                        try:
                            # Prompt constructing with core statistical prompt guidelines
                            system_prompt = (
                                "You are StatMentor AI, an expert research consultant, statistician, and decision scientist. "
                                "Your goal is to provide clear, helpful, mathematically accurate, and conversational guidance. "
                                "Explain complex statistical formulas simply, recommend appropriate statistical tests "
                                "based on the research questions, and help interpret data outcomes."
                            )
                            
                            # Construct model inputs
                            model = genai.GenerativeModel(
                                model_name="gemini-3.5-flash",
                                system_instruction=system_prompt
                            )
                            
                            # Build context with historical messages + current dataset context
                            full_prompt = "Conversation History:\\\\n"
                            for role, text in st.session_state["chat_history"][:-1]:
                                full_prompt += f"{'User' if role=='user' else 'Assistant'}: {text}\\\\n"
                            
                            full_prompt += data_context
                            full_prompt += f"\\\\n\\\\nLatest User Query: {user_query}\\\\nAnswer:"
                            
                            response = model.generate_content(full_prompt)
                            reply = response.text
                            
                        except Exception as e:
                            reply = f"❌ Error contacting Google Generative AI API: {str(e)}"
            
            # Save assistant reply to memory
            st.session_state["chat_history"].append(("assistant", reply))
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
            st.warning("No numeric columns detected in the dataset.")
        else:
            desc_stats = df[numeric_cols].describe().T
            desc_summary = pd.DataFrame({
                "Variable": desc_stats.index,
                "Mean": desc_stats["mean"].round(4),
                "Median (50%)": desc_stats["50%"].round(4),
                "Standard Deviation": desc_stats["std"].round(4),
                "Min": desc_stats["min"].round(4),
                "Max": desc_stats["max"].round(4)
            })
            st.dataframe(desc_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Please upload a CSV file to unlock the descriptive summary tools!")
`;

// ==========================================
// 📝 TYPES
// ==========================================
interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ColumnMetadata {
  name: string;
  type: "Numeric" | "Text/Categorical";
  originalType: string;
}

interface MissingReport {
  name: string;
  missingCount: number;
  percentage: number;
}

interface DescriptiveStat {
  variable: string;
  count: number;
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
}

export default function App() {
  // Navigation Tabs at the Project Level
  const [activeTab, setActiveTab] = useState<"sandbox" | "app_py" | "requirements" | "guide">("sandbox");
  const [copiedApp, setCopiedApp] = useState(false);
  const [copiedReqs, setCopiedReqs] = useState(false);

  // ==========================================
  // 🔬 SANDBOX WORKSPACE STATE (ST.SESSION_STATE equivalents)
  // ==========================================
  const [uploadedFileName, setUploadedFileName] = useState<string>("");
  const [datasetRows, setDatasetRows] = useState<number>(0);
  const [datasetCols, setDatasetCols] = useState<number>(0);
  const [columnsSchema, setColumnsSchema] = useState<ColumnMetadata[]>([]);
  const [missingDataReport, setMissingDataReport] = useState<MissingReport[]>([]);
  const [descriptiveStats, setDescriptiveStats] = useState<DescriptiveStat[]>([]);
  const [rawRows, setRawRows] = useState<any[]>([]);
  const [numericColumns, setNumericColumns] = useState<string[]>([]);
  const [selectedPlotCol, setSelectedPlotCol] = useState<string>("");

  // Chat State
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [userInput, setUserInput] = useState<string>("");
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll chat to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, isTyping]);

  // Load sample dataset on mount so the user doesn't see a blank app
  useEffect(() => {
    handleLoadSample("student");
  }, []);

  // Helper: Copy code to clipboard
  const copyToClipboard = (text: string, setCopied: React.Dispatch<React.SetStateAction<boolean>>) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // ==========================================
  // 📊 CSV STASTICAL COMPUTER (Javascript Pandas replica)
  // ==========================================
  const processCSVData = (csvString: string, filename: string) => {
    Papa.parse(csvString, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        const data = results.data as any[];
        const headers = results.meta.fields || [];

        if (data.length === 0 || headers.length === 0) {
          setErrorBanner("The CSV file seems to be empty or poorly formatted.");
          return;
        }

        setErrorBanner(null);
        setUploadedFileName(filename);
        setDatasetRows(data.length);
        setDatasetCols(headers.length);
        setRawRows(data);

        // 1. Calculate column schema & types (classify as numeric or categorical)
        const schema: ColumnMetadata[] = headers.map((colName) => {
          // Sample a few non-null rows to detect data type
          let isNum = true;
          let samplingCount = 0;
          let originalType = "string";

          for (let i = 0; i < Math.min(data.length, 30); i++) {
            const val = data[i][colName];
            if (val !== undefined && val !== null && val !== "") {
              samplingCount++;
              if (typeof val !== "number") {
                isNum = false;
                originalType = typeof val;
              } else {
                originalType = "number";
              }
            }
          }

          if (samplingCount === 0) {
            isNum = false; // fallback
          }

          return {
            name: colName,
            type: isNum ? "Numeric" : "Text/Categorical",
            originalType: originalType === "number" ? "float64" : "object"
          };
        });
        setColumnsSchema(schema);

        // Filter numeric columns
        const numCols = schema.filter(col => col.type === "Numeric").map(col => col.name);
        setNumericColumns(numCols);
        if (numCols.length > 0) {
          setSelectedPlotCol(numCols[0]);
        } else {
          setSelectedPlotCol("");
        }

        // 2. Compute Missing Data Report
        const report: MissingReport[] = headers.map((colName) => {
          let missingCount = 0;
          data.forEach((row) => {
            const val = row[colName];
            if (val === undefined || val === null || val === "" || (typeof val === "number" && isNaN(val))) {
              missingCount++;
            }
          });
          return {
            name: colName,
            missingCount,
            percentage: parseFloat(((missingCount / data.length) * 100).toFixed(2))
          };
        });
        setMissingDataReport(report);

        // 3. Descriptive statistics table
        const stats: DescriptiveStat[] = [];
        schema.forEach((col) => {
          if (col.type === "Numeric") {
            const numbers = data
              .map((row) => row[col.name])
              .filter((val) => typeof val === "number" && !isNaN(val)) as number[];

            if (numbers.length > 0) {
              const count = numbers.length;
              const sum = numbers.reduce((a, b) => a + b, 0);
              const mean = parseFloat((sum / count).toFixed(4));

              // Median
              const sorted = [...numbers].sort((a, b) => a - b);
              const mid = Math.floor(sorted.length / 2);
              const median = parseFloat(
                (sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2).toFixed(4)
              );

              // Standard deviation
              const variance = numbers.reduce((total, val) => total + Math.pow(val - mean, 2), 0) / (count > 1 ? count - 1 : 1);
              const std = parseFloat(Math.sqrt(variance).toFixed(4));
              const min = parseFloat(Math.min(...numbers).toFixed(4));
              const max = parseFloat(Math.max(...numbers).toFixed(4));

              stats.push({
                variable: col.name,
                count,
                mean,
                median,
                std,
                min,
                max
              });
            }
          }
        });
        setDescriptiveStats(stats);
      },
      error: (err) => {
        setErrorBanner(`CSV Parsing error: ${err.message}`);
      }
    });
  };

  // ==========================================
  // 📥 SAMPLE SELECTION HANDLER
  // ==========================================
  const handleLoadSample = (type: "student" | "clinical" | "sales") => {
    if (type === "student") {
      processCSVData(STUDENT_PERFORMANCE_CSV, "student_performance_sample.csv");
    } else if (type === "clinical") {
      processCSVData(CLINICAL_TRIAL_CSV, "clinical_trials_sample.csv");
    } else if (type === "sales") {
      processCSVData(RETAIL_SALES_CSV, "retail_sales_sample.csv");
    }
  };

  // ==========================================
  // 📁 USER FILE UPLOADER HANDLER
  // ==========================================
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      processCSVData(text, file.name);
    };
    reader.readAsText(file);
  };

  // ==========================================
  // 💬 CHAT HANDLERS & GEMINI DISPATCHER
  // ==========================================
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!userInput.trim() || isTyping) return;

    const currentInput = userInput;
    setUserInput("");
    setErrorBanner(null);

    // Save prompt to chat history state (session state equivalent)
    const userMsg: Message = { role: "user", content: currentInput };
    const updatedHistory = [...chatHistory, userMsg];
    setChatHistory(updatedHistory);
    setIsTyping(true);

    // Formulate dataset context string to help ground Gemini
    let datasetContextStr = "";
    if (uploadedFileName) {
      const statsSummary = descriptiveStats
        .map((s) => `${s.variable}: Mean=${s.mean}, Median=${s.median}, Std=${s.std}, Min=${s.min}, Max=${s.max}`)
        .join("\n");
      const columnsList = columnsSchema.map((c) => `${c.name} (${c.type})`).join(", ");
      
      datasetContextStr = `Uploaded File: ${uploadedFileName}
Dimensions: ${datasetRows} Rows, ${datasetCols} Columns
Variables: [${columnsList}]
Descriptive Analytics Profile:
${statsSummary}
Sample Rows (First 3 entries):
${JSON.stringify(rawRows.slice(0, 3), null, 2)}`;
    }

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: updatedHistory,
          datasetContext: datasetContextStr || null
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || data.error || "Server response failed");
      }

      setChatHistory([
        ...updatedHistory,
        { role: "assistant", content: data.reply }
      ]);
    } catch (err: any) {
      console.error("Chat error:", err);
      setChatHistory([
        ...updatedHistory,
        { 
          role: "assistant", 
          content: `⚠️ **Connection Error**: I could not process your query. ${err.message || 'Please check your Gemini key configurations.'}` 
        }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleClearHistory = () => {
    setChatHistory([]);
    setErrorBanner(null);
  };

  // ==========================================
  // 📈 CHART DATA PREPARATION
  // ==========================================
  const getChartData = () => {
    if (!selectedPlotCol || rawRows.length === 0) return [];

    // Group or aggregate counts for the quick-plot discrete frequency
    const freqMap: { [key: string]: number } = {};
    rawRows.forEach((row) => {
      const val = row[selectedPlotCol];
      if (val !== undefined && val !== null && val !== "") {
        let bucket = String(val);
        // If dense numeric list, let's round float columns to integer/single-decimal for clean visual distribution buckets
        if (typeof val === "number") {
          bucket = val % 1 === 0 ? String(val) : val.toFixed(1);
        }
        freqMap[bucket] = (freqMap[bucket] || 0) + 1;
      }
    });

    const chartArr = Object.entries(freqMap).map(([name, count]) => ({
      name,
      count
    }));

    // If numerical sorting possible, sort categories on X-axis for seamless distribution
    const isXNumeric = chartArr.every((item) => !isNaN(Number(item.name)));
    if (isXNumeric) {
      chartArr.sort((a, b) => Number(a.name) - Number(b.name));
    } else {
      chartArr.sort((a, b) => b.count - a.count); // sort by frequency
    }

    return chartArr.slice(0, 30); // clip to top 30 elements
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans selection:bg-teal-500 selection:text-white">
      
      {/* ==========================================
          HEADER BAR (Professional IDE Style)
         ========================================== */}
      <header id="app-header" className="bg-slate-950 border-b border-slate-800 px-6 py-4 flex flex-col md:flex-row md:items-center md:justify-between sticky top-0 z-50 shadow-lg">
        <div className="flex items-center space-x-3">
          <div className="bg-gradient-to-tr from-teal-500 to-indigo-600 p-2.5 rounded-xl shadow-inner border border-teal-400/20">
            <Sparkles className="w-6 h-6 text-teal-300 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-white font-sans">
                StatMentor AI
              </h1>
              <span className="text-[10px] bg-teal-500/15 border border-teal-500/30 text-teal-300 px-2.5 py-0.5 rounded-full font-mono font-medium uppercase tracking-wider">
                Developer IDE & Sandbox
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Intelligent Research & Decision Support System (Python + Streamlit Playground)
            </p>
          </div>
        </div>

        {/* Global Tab Selection */}
        <div className="flex flex-wrap items-center mt-4 md:mt-0 gap-1.5 bg-slate-900 p-1 rounded-xl border border-slate-800">
          <button
            id="tab-sandbox"
            onClick={() => setActiveTab("sandbox")}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all duration-200 ${
              activeTab === "sandbox"
                ? "bg-gradient-to-r from-teal-600 to-teal-500 text-white shadow-md shadow-teal-900/30 font-bold"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            <Play className="w-3.5 h-3.5" />
            <span>🌐 Streamlit Live App Sandbox</span>
          </button>
          
          <button
            id="tab-app-py"
            onClick={() => setActiveTab("app_py")}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all duration-200 ${
              activeTab === "app_py"
                ? "bg-slate-800 text-white border-b-2 border-teal-500 font-bold"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            <FileCode className="w-3.5 h-3.5 text-yellow-400" />
            <span>🐍 app.py</span>
          </button>

          <button
            id="tab-requirements"
            onClick={() => setActiveTab("requirements")}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all duration-200 ${
              activeTab === "requirements"
                ? "bg-slate-800 text-white border-b-2 border-teal-500 font-bold"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            <Terminal className="w-3.5 h-3.5 text-teal-400" />
            <span>📋 requirements.txt</span>
          </button>

          <button
            id="tab-guide"
            onClick={() => setActiveTab("guide")}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all duration-200 ${
              activeTab === "guide"
                ? "bg-slate-800 text-white border-b-2 border-teal-500 font-bold"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
            <span>📖 How to Deploy</span>
          </button>
        </div>
      </header>

      {/* ==========================================
          MAIN AREA CONTAINERS
         ========================================== */}
      <main className="flex-1 w-full max-w-[1600px] mx-auto p-4 md:p-6 flex flex-col min-h-0">
        
        {/* Error notification banner if any */}
        {errorBanner && (
          <div className="bg-red-950/80 border border-red-800/50 text-red-200 px-5 py-3 rounded-xl mb-4 text-xs font-medium flex items-center gap-3 backdrop-blur shadow-md">
            <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0" />
            <span>{errorBanner}</span>
          </div>
        )}

        {/* ==========================================
            TAB 1: STREAMLIT GRAPHIC PREVIEW
         ========================================== */}
        {activeTab === "sandbox" && (
          <div className="bg-slate-950 rounded-2xl border border-slate-800 flex flex-col flex-1 overflow-hidden shadow-2xl">
            
            {/* Streamlit Custom Browser Header */}
            <div className="bg-slate-900 border-b border-slate-800/80 px-4 py-2.5 flex items-center justify-between text-[11px] font-mono text-slate-400">
              <div className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-red-500/80 block"></span>
                <span className="w-3 h-3 rounded-full bg-yellow-500/80 block"></span>
                <span className="w-3 h-3 rounded-full bg-green-500/80 block"></span>
                <span className="ml-2 font-semibold text-slate-400 bg-slate-950 px-3 py-1 rounded-md border border-slate-850">
                  https://statmentor-ai.streamlit.app
                </span>
              </div>
              <div className="text-[10px] text-teal-400 flex items-center gap-1.5 font-bold">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse"></span>
                <span>Active Streamlit Sandbox Server running successfully (Port 8501 simulator)</span>
              </div>
            </div>

            {/* Sandbox Title Header */}
            <div className="bg-slate-950 p-6 border-b border-slate-800/40">
              <h2 className="text-2xl font-extrabold text-white flex items-center gap-3 tracking-tight">
                <span>📊</span> StatMentor AI: Intelligent Research & Decision Support System
              </h2>
              <p className="text-xs text-slate-400 mt-1 max-w-4xl">
                This is a high-fidelity interactive simulation of your Streamlit system. Column 1 simulates the 
                Google Gemini API chatbot workspace with automated context injection; Column 2 represents the Python pandas diagnostic table outputs, metric cards, missing statistics, and Recharts distributions.
              </p>
            </div>

            {/* Live Sandbox 2-Column layout */}
            <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-slate-850 overflow-y-auto lg:h-[620px] min-h-[500px]">
              
              {/* 🔍 COLUMN 1: RESEARCH & CHAT WORKSPACE */}
              <div className="p-6 flex flex-col h-full bg-slate-950/70 min-h-0">
                <div className="mb-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-base font-bold text-white flex items-center gap-2">
                      <span className="text-teal-400">🔍</span> Research & Chat Workspace
                    </h3>
                    <button
                      onClick={handleClearHistory}
                      className="text-[11px] font-medium text-slate-500 hover:text-slate-300 flex items-center gap-1.5 px-2.5 py-1 rounded-md hover:bg-slate-900 border border-transparent hover:border-slate-800 transition"
                      title="Simulates st.session_state = [] resetting"
                    >
                      <Trash2 className="w-3 h-3" />
                      Clear History
                    </button>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Ask questions about research methodologies, statistical equations, hypotheses or write questions about your active metadata stats.
                  </p>
                </div>

                {/* Simulated Chat Bubble Feed */}
                <div className="flex-1 bg-slate-900/60 border border-slate-800 rounded-xl p-4 overflow-y-auto space-y-4 mb-4 flex flex-col h-[350px]">
                  {chatHistory.length === 0 ? (
                    <div className="flex-1 flex flex-col items-center justify-center text-center p-6">
                      <div className="bg-slate-800 p-3.5 rounded-2xl mb-3 text-teal-400/80 border border-slate-700/60">
                        <Bot className="w-7 h-7" />
                      </div>
                      <h4 className="text-xs font-bold text-slate-300">StatMentor Chat Module Active</h4>
                      <p className="text-[11px] text-slate-500 max-w-xs mt-1">
                        👋 Hello! Put me to the test. Type issues like:
                      </p>
                      <div className="grid grid-cols-1 gap-1.5 mt-3 max-w-xs w-full text-[10px] text-slate-400">
                        <button 
                          onClick={() => setUserInput("When should I use a paired t-test vs an independent samples t-test?")}
                          className="bg-slate-950 hover:bg-slate-800 py-1.5 px-2.5 rounded border border-slate-850 hover:border-slate-800 text-left transition"
                        >
                          "Paired vs Independent t-Test?"
                        </button>
                        <button 
                          onClick={() => setUserInput("Analyze my loaded data. Tell me what stats match my columns.")}
                          className="bg-slate-950 hover:bg-slate-800 py-1.5 px-2.5 rounded border border-slate-850 hover:border-slate-800 text-left transition"
                        >
                          "Analyze my loaded dataset columns"
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {chatHistory.map((message, idx) => (
                        <div 
                          key={idx} 
                          className={`flex gap-3 max-w-[88%] ${message.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"}`}
                        >
                          <div className={`p-2 rounded-xl flex-shrink-0 flex items-center justify-center ${
                            message.role === "user" ? "bg-teal-500/10 border border-teal-500/20 text-teal-300" : "bg-indigo-500/10 border border-indigo-500/20 text-indigo-300"
                          } w-8 h-8`}>
                            {message.role === "user" ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                          </div>
                          
                          <div className={`p-3.5 rounded-2xl text-xs leading-relaxed ${
                            message.role === "user" 
                              ? "bg-teal-550 bg-teal-600 font-medium text-white rounded-tr-none" 
                              : "bg-slate-800/80 border border-slate-750 text-slate-200 rounded-tl-none"
                          }`}>
                            <div className="whitespace-pre-line font-sans prose prose-invert prose-xs">
                              {message.content}
                            </div>
                          </div>
                        </div>
                      ))}

                      {isTyping && (
                        <div className="flex gap-3 mr-auto max-w-[80%]">
                          <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 w-8 h-8 flex items-center justify-center flex-shrink-0">
                            <Bot className="w-4 h-4 animate-pulse" />
                          </div>
                          <div className="p-4 rounded-2xl bg-slate-800/60 border border-slate-800 text-slate-400 text-xs rounded-tl-none flex items-center space-x-2">
                            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></span>
                            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                            <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                            <span className="text-[10px] italic ml-1">StatMentor compiling statistical reply...</span>
                          </div>
                        </div>
                      )}
                      <div ref={messagesEndRef} />
                    </div>
                  )}
                </div>

                {/* Input Prompt Box */}
                <form onSubmit={handleSendMessage} className="flex gap-2">
                  <input
                    type="text"
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    placeholder={uploadedFileName ? `Ask about "${uploadedFileName}" or stat methods...` : "Type statistical queries or test questions..."}
                    className="flex-1 bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-teal-500 shadow-inner"
                  />
                  <button
                    type="submit"
                    disabled={isTyping || !userInput.trim()}
                    className="bg-indigo-600 hover:bg-indigo-500 text-white transition font-bold text-xs px-5 py-3 rounded-xl flex items-center gap-1.5 shadow disabled:opacity-50 disabled:hover:bg-indigo-600 cursor-pointer"
                  >
                    <span>Analyze</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </form>

                {/* Status indicator on active data coupling */}
                <div className="text-[10px] text-slate-500 mt-2 flex items-center justify-between">
                  <div className="flex items-center gap-1">
                    <Database className="w-3.5 h-3.5 text-indigo-400" />
                    <span>Active context binding:</span>
                    <strong className="text-slate-300">{uploadedFileName ? `ON (${uploadedFileName})` : "OFF (General Theory)"}</strong>
                  </div>
                  {uploadedFileName && (
                    <span className="text-teal-400 text-[9px] font-semibold bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">
                      Auto-Injecting Stats
                    </span>
                  )}
                </div>

              </div>

              {/* 📈 COLUMN 2: DATA DASHBOARD & VISUALIZATIONS */}
              <div className="p-6 flex flex-col h-full bg-slate-950/20 overflow-y-auto">
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <span className="text-indigo-400">📈</span> Data Dashboard & Visualizations
                </h3>
                <p className="text-xs text-slate-400 mt-1 mb-4">
                  Drag and drop a custom CSV file to view real-time shape dimensions, variable classifications, null percentage reports, and summary metrics.
                </p>

                {/* Drag-Drop / Active loader area */}
                <div className="bg-slate-900/60 border border-dashed border-slate-800 hover:border-slate-700 rounded-2xl p-5 mb-5 flex flex-col items-center justify-center text-center transition">
                  <Upload className="w-8 h-8 text-slate-500 mb-2" />
                  <p className="text-[11px] font-semibold text-slate-300">
                    {uploadedFileName ? `Active dataset Loaded: ${uploadedFileName}` : "Upload a statistical research CSV dataframe"}
                  </p>
                  <p className="text-[10px] text-slate-500 mt-0.5 mb-3">Accepts standard .csv files</p>
                  
                  <div className="flex gap-2">
                    <label className="bg-slate-800 hover:bg-slate-750 text-slate-200 transition text-[10px] font-bold px-3 py-1.5 rounded-lg border border-slate-700 cursor-pointer shadow-sm">
                      Choose Local CSV
                      <input 
                        type="file" 
                        accept=".csv" 
                        onChange={handleFileUpload} 
                        className="hidden" 
                      />
                    </label>
                  </div>

                  {/* Prebuilt Samples shortcut */}
                  <div className="mt-4 pt-4 border-t border-slate-800/60 w-full">
                    <p className="text-[9px] font-mono text-slate-500 tracking-wider uppercase mb-2">Or load a simulation-ready study dataset</p>
                    <div className="flex flex-wrap justify-center gap-1.5">
                      <button 
                        onClick={() => handleLoadSample("student")}
                        className={`text-[10px] px-2.5 py-1.5 rounded-lg font-medium border transition ${
                          uploadedFileName === "student_performance_sample.csv" 
                            ? "bg-teal-500/20 border-teal-500 text-teal-300"
                            : "bg-slate-950 border-slate-850 text-slate-400 hover:text-slate-200 hover:border-slate-750"
                        }`}
                      >
                        🎓 Student Exams
                      </button>
                      <button 
                        onClick={() => handleLoadSample("clinical")}
                        className={`text-[10px] px-2.5 py-1.5 rounded-lg font-medium border transition ${
                          uploadedFileName === "clinical_trials_sample.csv" 
                            ? "bg-teal-500/20 border-teal-500 text-teal-300"
                            : "bg-slate-950 border-slate-850 text-slate-400 hover:text-slate-200 hover:border-slate-750"
                        }`}
                      >
                        💊 Clinical Trial
                      </button>
                      <button 
                        onClick={() => handleLoadSample("sales")}
                        className={`text-[10px] px-2.5 py-1.5 rounded-lg font-medium border transition ${
                          uploadedFileName === "retail_sales_sample.csv" 
                            ? "bg-teal-500/20 border-teal-500 text-teal-300"
                            : "bg-slate-950 border-slate-850 text-slate-400 hover:text-slate-200 hover:border-slate-750"
                        }`}
                      >
                        🛍️ Retail Store Sales
                      </button>
                    </div>
                  </div>
                </div>

                {/* Dashboard Output (Simulated if CSV is parsed) */}
                {uploadedFileName ? (
                  <div className="space-y-6">

                    {/* Section title */}
                    <div className="flex items-center gap-2 text-slate-400 text-xs tracking-wider font-mono uppercase pb-1 border-b border-slate-800/40">
                      <FileSpreadsheet className="w-3.5 h-3.5 text-teal-400" />
                      <span>Diagnostics for: {uploadedFileName}</span>
                    </div>

                    {/* 1. Dimension st.metric representation */}
                    <div>
                      <h4 className="text-xs font-bold text-slate-300 mb-2 flex items-center gap-1">
                        <span>📏</span> Dataset Dimensions (Streamlit Metric equivalents)
                      </h4>
                      <div className="grid grid-cols-2 gap-4">
                        {/* Row metric card */}
                        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm relative overflow-hidden group">
                          <div className="absolute right-3 top-3 opacity-10 text-teal-300 group-hover:opacity-20 transition">
                            <Database className="w-10 h-10" />
                          </div>
                          <span className="text-[10px] text-slate-500 uppercase font-mono tracking-wider">Total Rows</span>
                          <div className="text-2xl font-black text-white mt-1 font-mono tracking-tight">
                            {datasetRows.toLocaleString()}
                          </div>
                          <div className="text-[9px] text-teal-400 font-medium mt-1">
                            st.metric(label="Total Rows")
                          </div>
                        </div>

                        {/* Col metric card */}
                        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-sm relative overflow-hidden group">
                          <div className="absolute right-3 top-3 opacity-10 text-teal-300 group-hover:opacity-20 transition">
                            <Columns className="w-10 h-10" />
                          </div>
                          <span className="text-[10px] text-slate-500 uppercase font-mono tracking-wider">Total Columns</span>
                          <div className="text-2xl font-black text-white mt-1 font-mono tracking-tight">
                            {datasetCols.toLocaleString()}
                          </div>
                          <div className="text-[9px] text-teal-400 font-medium mt-1">
                            st.metric(label="Total Columns")
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* 2. Detected Column types table */}
                    <div>
                      <h4 className="text-xs font-bold text-slate-300 mb-2 flex items-center gap-1">
                        <span>🏷️</span> Detected Column Schema & Types (Pandas equivalents)
                      </h4>
                      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
                        <div className="max-h-[160px] overflow-y-auto">
                          <table className="w-full text-left border-collapse text-[11px]">
                            <thead>
                              <tr className="bg-slate-950 font-mono text-[9px] text-slate-500 border-b border-slate-800">
                                <th className="p-2.5 pl-4 font-bold">Column Name</th>
                                <th className="p-2.5 font-bold">Pandas Type</th>
                                <th className="p-2.5 pr-4 font-bold">Classification Badge</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800/50">
                              {columnsSchema.map((col, idx) => (
                                <tr key={idx} className="hover:bg-slate-800/40 transition">
                                  <td className="p-2.5 pl-4 font-semibold text-slate-200">{col.name}</td>
                                  <td className="p-2.5 font-mono text-slate-400">{col.originalType}</td>
                                  <td className="p-2.5 pr-4">
                                    <span className={`inline-block px-2.5 py-0.5 rounded-full text-[9px] font-medium tracking-wide uppercase ${
                                      col.type === "Numeric" 
                                        ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-400" 
                                        : "bg-amber-500/10 border border-amber-500/20 text-amber-400"
                                    }`}>
                                      {col.type}
                                    </span>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                      <div className="text-[9px] font-mono text-slate-600 mt-1 pl-1">
                        Pandas auto-classified: np.number matches Numeric classification
                      </div>
                    </div>

                    {/* 3. Missing Data Analysis report */}
                    <div>
                      <h4 className="text-xs font-bold text-slate-300 mb-2 flex items-center gap-1">
                        <span>🔍</span> Missing Data Integrity Report
                      </h4>
                      
                      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
                        <table className="w-full text-left border-collapse text-[11px]">
                          <thead>
                            <tr className="bg-slate-950 font-mono text-[9px] text-slate-500 border-b border-slate-800">
                              <th className="p-2.5 pl-4 font-bold">Field Name</th>
                              <th className="p-2.5 font-bold">Null/Missing Count</th>
                              <th className="p-2.5 pr-4 font-bold">Missingness Percentage (%)</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/50">
                            {missingDataReport.map((m, idx) => (
                              <tr key={idx} className="hover:bg-slate-800/40">
                                <td className="p-2.5 pl-4 text-slate-300">{m.name}</td>
                                <td className="p-2.5 font-mono font-bold text-slate-400">
                                  {m.missingCount > 0 ? (
                                    <span className="text-amber-400 flex items-center gap-1 justify-start">
                                      <AlertTriangle className="w-3 h-3 text-amber-500" />
                                      {m.missingCount}
                                    </span>
                                  ) : (
                                    <span className="text-emerald-500">0 rows</span>
                                  )}
                                </td>
                                <td className="p-2.5 pr-4 font-mono font-semibold">
                                  {m.missingCount > 0 ? (
                                    <span className="text-amber-350 text-amber-400">{m.percentage}%</span>
                                  ) : (
                                    <span className="text-slate-600">0.0%</span>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      
                      {missingDataReport.reduce((a, b) => a + b.missingCount, 0) === 0 ? (
                        <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400/90 text-[10px] px-3.5 py-2 rounded-lg mt-2 flex items-center gap-2">
                          <Check className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                          <span>🎉 **Complete Integrity**: Dataset contains zero null values across all parameters!</span>
                        </div>
                      ) : (
                        <div className="bg-amber-500/10 border border-amber-550/20 text-amber-400/90 text-[10px] px-3.5 py-2 rounded-lg mt-2 flex items-center gap-2">
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
                          <span>⚠️ Missing parameters detected. StatMentor AI chatbot will accommodate these values.</span>
                        </div>
                      )}
                    </div>

                    {/* 4. Descriptive statistics table */}
                    <div>
                      <h4 className="text-xs font-bold text-slate-300 mb-2 flex items-center gap-1">
                        <span>📊</span> Descriptive Statistics Table (Numeric Columns Only)
                      </h4>
                      
                      {descriptiveStats.length === 0 ? (
                        <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl text-xs text-amber-400 flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4" />
                          <span>No Numeric Columns detected in this CSV; Descriptives omitted.</span>
                        </div>
                      ) : (
                        <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
                          <div className="overflow-x-auto">
                            <table className="w-full text-left border-collapse text-[10.5px]">
                              <thead>
                                <tr className="bg-slate-950 font-mono text-[9px] text-slate-500 border-b border-slate-800 whitespace-nowrap">
                                  <th className="p-2.5 pl-4 font-bold">Variable Name</th>
                                  <th className="p-2.5 font-bold">Valid N</th>
                                  <th className="p-2.5 font-bold text-teal-300">Mean (Avg)</th>
                                  <th className="p-2.5 font-bold text-indigo-300">Median (50%)</th>
                                  <th className="p-2.5 font-bold text-yellow-300">Std Dev</th>
                                  <th className="p-2.5 font-bold">Min</th>
                                  <th className="p-2.5 pr-4 font-bold">Max</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-800/50 font-mono whitespace-nowrap">
                                {descriptiveStats.map((stat, idx) => (
                                  <tr key={idx} className="hover:bg-slate-800/40">
                                    <td className="p-2.5 pl-4 font-sans font-semibold text-slate-250 text-slate-200">{stat.variable}</td>
                                    <td className="p-2.5 text-slate-500">{stat.count}</td>
                                    <td className="p-2.5 font-bold text-teal-350 text-teal-400">{stat.mean}</td>
                                    <td className="p-2.5 font-semibold text-indigo-350 text-indigo-300">{stat.median}</td>
                                    <td className="p-2.5 text-yellow-400">{stat.std}</td>
                                    <td className="p-2.5 text-slate-400">{stat.min}</td>
                                    <td className="p-2.5 pr-4 text-slate-300">{stat.max}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                      <div className="text-[9px] text-slate-600 font-mono mt-1.5 pl-1">
                        Computed in client-side matching Pandas math: descriptive_table = df.describe().T
                      </div>
                    </div>

                    {/* Interactive distribution quick plot */}
                    {numericColumns.length > 0 && (
                      <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl relative overflow-hidden shadow">
                        <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
                          <h4 className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                            <BarChart3 className="w-3.5 h-3.5 text-teal-400" />
                            <span>📊 Custom Quick-Plot Distribution</span>
                          </h4>
                          <div className="flex items-center gap-1.5 text-[10px]">
                            <span className="text-slate-500 font-mono">Column:</span>
                            <select
                              value={selectedPlotCol}
                              onChange={(e) => setSelectedPlotCol(e.target.value)}
                              className="bg-slate-950 border border-slate-800 text-teal-400 rounded px-2.5 py-1 text-[10.5px] font-semibold focus:outline-none focus:border-teal-500"
                            >
                              {numericColumns.map((colName) => (
                                <option key={colName} value={colName}>
                                  {colName}
                                </option>
                              ))}
                            </select>
                          </div>
                        </div>

                        {/* Visual Recharts Bar Representation */}
                        <div className="h-[180px] w-full mt-2">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={getChartData()} margin={{ top: 5, right: 5, left: -25, bottom: 5 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#2a3347" opacity={0.4} />
                              <XAxis 
                                dataKey="name" 
                                stroke="#64748b" 
                                fontSize={9} 
                                tickLine={false} 
                              />
                              <YAxis 
                                stroke="#64748b" 
                                fontSize={9} 
                                tickLine={false} 
                                allowDecimals={false}
                              />
                              <Tooltip
                                contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
                                itemStyle={{ color: "#2dd4bf", fontSize: 10, fontWeight: "bold" }}
                                labelStyle={{ color: "#94a3b8", fontSize: 10 }}
                              />
                              <Bar 
                                dataKey="count" 
                                fill="url(#colorStat)" 
                                radius={[4, 4, 0, 0]} 
                              />
                              <defs>
                                <linearGradient id="colorStat" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="#0d9488" stopOpacity={0.8}/>
                                  <stop offset="95%" stopColor="#4f46e5" stopOpacity={0.6}/>
                                </linearGradient>
                              </defs>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                        <div className="text-[9px] text-slate-500 font-mono text-center mt-1.5">
                          Frequency profile distributions for column `{selectedPlotCol}`
                        </div>
                      </div>
                    )}

                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center p-8 text-center border border-slate-800 rounded-xl bg-slate-900/40">
                    <AlertTriangle className="w-6 h-6 text-slate-500 mb-2" />
                    <p className="text-xs text-slate-400 font-semibold">No dataset active</p>
                    <p className="text-[11px] text-slate-500 max-w-xs mt-1">
                      Load a mock model sample dataset above or drag a CSV file in here to populate dimension counts, types classification, and diagnostics.
                    </p>
                  </div>
                )}

              </div>
            </div>

            {/* Simulated footer */}
            <div className="bg-slate-950 px-6 py-3 border-t border-slate-800 flex flex-col md:flex-row md:items-center md:justify-between text-[11px] text-slate-500 gap-2">
              <span className="font-mono">Streamlit App Runtime: v1.35.0 (Node Simulator backend)</span>
              <span className="flex items-center gap-1 font-mono text-[10px]">
                <Settings className="w-3.5 h-3.5 text-indigo-400" />
                Session State Preserved in st.session_state on active variables
              </span>
            </div>

          </div>
        )}

        {/* ==========================================
            TAB 2: app.py CODE CODE VIEW
         ========================================== */}
        {activeTab === "app_py" && (
          <div className="bg-slate-950 rounded-2xl border border-slate-800 flex flex-col flex-1 shadow-2xl overflow-hidden min-h-[500px]">
            
            {/* Tab header controller */}
            <div className="bg-slate-900 px-6 py-4 flex items-center justify-between border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <FileCode className="w-5 h-5 text-yellow-500" />
                <div>
                  <h3 className="text-xs font-bold text-white">app.py</h3>
                  <p className="text-[10px] text-slate-400 mt-0.5">Streamlit Application Script for Python Development</p>
                </div>
              </div>
              <button
                onClick={() => copyToClipboard(PYTHON_APP_CODE, setCopiedApp)}
                className="bg-slate-850 hover:bg-slate-800 border border-slate-750 hover:border-slate-700 text-slate-200 transition text-[11px] px-3.5 py-1.5 rounded-lg flex items-center gap-1.5 font-semibold cursor-pointer active:scale-95"
              >
                {copiedApp ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-teal-400" />
                    <span className="text-teal-400">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy app.py Code</span>
                  </>
                )}
              </button>
            </div>

            {/* Code Body */}
            <div id="code-content-app" className="flex-1 overflow-y-auto p-6 font-mono text-[11px] bg-slate-950 text-slate-350 leading-relaxed text-slate-300">
              <pre className="whitespace-pre overflow-x-auto">
                <code>
                  {PYTHON_APP_CODE}
                </code>
              </pre>
            </div>

          </div>
        )}

        {/* ==========================================
            TAB 3: requirements.txt CODE VIEW
         ========================================== */}
        {activeTab === "requirements" && (
          <div className="bg-slate-950 rounded-2xl border border-slate-800 flex flex-col shadow-2xl overflow-hidden min-h-[500px]">
            
            {/* Tab Header */}
            <div className="bg-slate-900 px-6 py-4 flex items-center justify-between border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <Terminal className="w-5 h-5 text-teal-400" />
                <div>
                  <h3 className="text-xs font-bold text-white">requirements.txt</h3>
                  <p className="text-[10px] text-slate-400 mt-0.5">Python package manifest representing basic statistics libraries required</p>
                </div>
              </div>
              <button
                onClick={() => copyToClipboard(PYTHON_REQUIREMENTS_CODE, setCopiedReqs)}
                className="bg-slate-850 hover:bg-slate-800 border border-slate-750 hover:border-slate-700 text-slate-200 transition text-[11px] px-3.5 py-1.5 rounded-lg flex items-center gap-1.5 font-semibold cursor-pointer active:scale-95"
              >
                {copiedReqs ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-teal-400" />
                    <span className="text-teal-400">Copied requirements!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy Pip Manifest</span>
                  </>
                )}
              </button>
            </div>

            {/* Code Body */}
            <div id="code-content-reqs" className="flex-1 overflow-y-auto p-6 font-mono text-xs bg-slate-950 text-teal-300 leading-relaxed">
              <pre className="whitespace-pre overflow-x-auto">
                <code>
                  {PYTHON_REQUIREMENTS_CODE}
                </code>
              </pre>
            </div>

          </div>
        )}

        {/* ==========================================
            TAB 4: HOW TO DEPLOY / INSTRUCTION MANUAL
         ========================================== */}
        {activeTab === "guide" && (
          <div className="bg-slate-950 rounded-2xl border border-slate-800 p-6 shadow-2xl flex-1 overflow-y-auto min-h-[500px]">
            
            <div className="flex items-center space-x-3 pb-4 border-b border-slate-800 mb-6">
              <div className="bg-indigo-500/10 p-2 rounded-xl text-indigo-400 border border-indigo-500/20">
                <BookOpen className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white">How to Run & Deploy Your Streamlit App Globally</h3>
                <p className="text-[11px] text-slate-400">Step-by-step developer setup protocol for local and cloud hosting</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-5xl">
              
              {/* Step 1 & 2 */}
              <div className="space-y-6">
                
                {/* Visual Step */}
                <div className="p-5 bg-slate-900 border border-slate-800 rounded-2xl relative shadow-sm">
                  <span className="absolute -left-3 -top-3 w-8 h-8 rounded-full bg-teal-500 text-slate-950 font-black text-xs flex items-center justify-center font-mono border-4 border-slate-950 shadow-md">
                    01
                  </span>
                  <h4 className="text-xs font-bold text-white tracking-tight flex items-center gap-2">
                    Initialize Local Environment
                  </h4>
                  <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">
                    Create a separate Python virtual environment directory to run isolated dependencies. Open your terminals and run:
                  </p>
                  <div className="bg-slate-950 rounded-lg p-3 mt-3 font-mono text-[10px] text-teal-400 border border-slate-850">
                    <div># Create virtual environment</div>
                    <div>python -m venv .venv</div>
                    <div className="mt-2"># Activate on Windows / macOS</div>
                    <div>.venv\Scripts\activate  <span className="text-slate-650 opacity-40"># Windows</span></div>
                    <div>source .venv/bin/activate  <span className="text-slate-650 opacity-40"># Mac/Linux</span></div>
                  </div>
                </div>

                {/* Visual Step */}
                <div className="p-5 bg-slate-900 border border-slate-800 rounded-2xl relative shadow-sm">
                  <span className="absolute -left-3 -top-3 w-8 h-8 rounded-full bg-teal-500 text-slate-950 font-black text-xs flex items-center justify-center font-mono border-4 border-slate-950 shadow-md">
                    02
                  </span>
                  <h4 className="text-xs font-bold text-white tracking-tight flex items-center gap-2">
                    Install Packed Manifest
                  </h4>
                  <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">
                    Run standard pip installations using the provided requirements.txt. Make sure you are in the directory containing `requirements.txt`:
                  </p>
                  <div className="bg-slate-950 rounded-lg p-3 mt-3 font-mono text-[10px] text-teal-400 border border-slate-850">
                    <div># Install requirements</div>
                    <div>pip install -r requirements.txt</div>
                  </div>
                  <div className="mt-3 text-[10px] text-slate-500 bg-slate-950/40 p-2 rounded border border-slate-850/60 flex items-center gap-1.5 leading-relaxed">
                    <Info className="w-3.5 h-3.5 text-indigo-400 flex-shrink-0" />
                    <span>Installs Pandas, standard Numpy matrices, Streamlit and Google's SDK.</span>
                  </div>
                </div>

              </div>

              {/* Step 3 & 4 */}
              <div className="space-y-6">
                
                {/* Visual Step */}
                <div className="p-5 bg-slate-900 border border-slate-800 rounded-2xl relative shadow-sm">
                  <span className="absolute -left-3 -top-3 w-8 h-8 rounded-full bg-indigo-500 text-slate-950 font-black text-xs flex items-center justify-center font-mono border-4 border-slate-950 shadow-md">
                    03
                  </span>
                  <h4 className="text-xs font-bold text-white tracking-tight flex items-center gap-2">
                    Configure Google Secrets
                  </h4>
                  <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">
                    Retrieve a Gemini API Key from Google AI Studio. Add it locally in stream secrets or environment variables:
                  </p>
                  <div className="bg-slate-950 rounded-lg p-3 mt-3 font-mono text-[10px] text-indigo-400 border border-slate-850">
                    <div># Option A: Save in Secrets configuration</div>
                    <div>mkdir .streamlit</div>
                    <div>echo "GEMINI_API_KEY = 'your_key_here'" &gt; .streamlit/secrets.toml</div>
                    <div className="mt-2.5"># Option B: Set Shell Environment Variable</div>
                    <div>export GEMINI_API_KEY="your_api_key_here"</div>
                  </div>
                </div>

                {/* Visual Step */}
                <div className="p-5 bg-slate-900 border border-slate-800 rounded-2xl relative shadow-sm">
                  <span className="absolute -left-3 -top-3 w-8 h-8 rounded-full bg-indigo-500 text-slate-950 font-black text-xs flex items-center justify-center font-mono border-4 border-slate-950 shadow-md">
                    04
                  </span>
                  <h4 className="text-xs font-bold text-white tracking-tight flex items-center gap-2">
                    Boot the Streamlit Server!
                  </h4>
                  <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">
                    Spin up your statistics dashboard local client server with simple Streamlit calls:
                  </p>
                  <div className="bg-slate-950 rounded-lg p-3 mt-3 font-mono text-[10px] text-indigo-400 border border-slate-850">
                    <div># Start application</div>
                    <div>streamlit run app.py</div>
                  </div>
                  <p className="text-[10px] text-teal-400 mt-2.5 flex items-center gap-1.5 font-bold">
                    <span className="w-1.5 h-1.5 rounded-full bg-teal-400 animate-pulse"></span>
                    <span>Loads port 8501 automatically! Ready for decision analysis.</span>
                  </p>
                </div>

              </div>

            </div>

            {/* Cloud Deploy Summary */}
            <div className="bg-gradient-to-r from-slate-900 to-indigo-950/45 p-6 rounded-2xl border border-slate-800 mt-6 max-w-5xl">
              <h4 className="text-xs font-bold text-white flex items-center gap-2">
                <ExternalLink className="w-4 h-4 text-teal-400" />
                Streamlit Community Cloud Deployment
              </h4>
              <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">
                Want to deploy **StatMentor AI** to the web absolutely free? Streamlit provides direct integration via GitHub:
              </p>
              <ol className="list-decimal list-inside text-[11px] text-slate-450 mt-3 space-y-2 text-slate-300">
                <li>Create a <strong>GitHub Repository</strong> and upload <code>app.py</code> and <code>requirements.txt</code>.</li>
                <li>Go to <a href="https://share.streamlit.io" target="_blank" rel="noreferrer" className="text-teal-400 font-semibold hover:underline">share.streamlit.io</a> and connect your GitHub account.</li>
                <li>Select your repository, branch, and entry point file path (<code>app.py</code>).</li>
                <li>Go to **Advanced Settings** in Streamlit Cloud Dashboard, and paste your <code>GEMINI_API_KEY="AI_Studio_value"</code> under Secrets.</li>
                <li>Click **Deploy**! Your model will compile and launch live across the web in 1-2 minutes.</li>
              </ol>
            </div>

          </div>
        )}

      </main>

      {/* ==========================================
          SYSTEM CONTROL PANEL (Anti-AI-Slop Compliant)
         ========================================== */}
      <footer className="bg-slate-950/80 border-t border-slate-800/80 py-4 px-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-400 mt-auto">
        <div className="flex items-center gap-2 text-[10px] text-slate-500 font-mono">
          <span>PROJECT METRIC:</span>
          <span className="bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-slate-400">Streamlit Sandbox Applet</span>
        </div>
        <div className="text-center sm:text-right text-[11px] text-slate-500 flex items-center gap-2">
          <span>Designed with high-contrast slate pairings matching custom code schemas.</span>
        </div>
      </footer>

    </div>
  );
}
