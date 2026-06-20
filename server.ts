import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

// Load environment variables
dotenv.config();

let aiClient: GoogleGenAI | null = null;

// Lazy initialization of GoogleGenAI SDK to prevent app crashing if GEMINI_API_KEY is omitted on boot
function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable is not defined");
  }
  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiClient;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  // JSON request parser
  app.use(express.json({ limit: '10mb' }));

  // API endpoint for Chat Workspace
  app.post("/api/chat", async (req, res) => {
    try {
      const { messages, datasetContext } = req.body;
      
      if (!messages || !Array.isArray(messages)) {
        return res.status(400).json({ error: "Messages array is required." });
      }

      // Check if API key exists
      if (!process.env.GEMINI_API_KEY) {
        return res.status(500).json({ 
          error: "Gemini API key is currently missing.",
          isConfigError: true,
          message: "⚠️ **API Key is missing**. Please click the **Settings > Secrets** panel in AI Studio to configure your `GEMINI_API_KEY`."
        });
      }

      const ai = getGeminiClient();
      
      // Inject guidelines as a system instruction
      const systemInstruction = 
        "You are StatMentor AI, an expert research consultant, statistician, and decision scientist. " +
        "Provide clear, conversational, helpful, and mathematically precise guidance on statistics, research, and data. " +
        "Suggest appropriate statistical tests (t-test, ANOVA, regression, Chi-Square, etc.) based on data types and hypotheses. " +
        "Format mathematical equations in clean inline markdown or code blocks. Always make user-facing outcomes friendly, clean, and professional.";

      // Reconstruct prompt. Use contents array structure
      const contentsPayload: any[] = [];
      
      // If dataset context is supplied, prepend it to help ground the chat
      if (datasetContext) {
        contentsPayload.push({
          role: 'user',
          parts: [{ text: `CONTEXT INFO FOR CURRENT RESEARCH DATASET:\n${datasetContext}\n\nPlease keep the above metadata of my dataset in mind for statistical context. Answer all following questions based on this or general theory.` }]
        });
        contentsPayload.push({
          role: 'model',
          parts: [{ text: "Understood. I have locked in your dataset metrics, variable categorizations, and descriptive statistics. I will ground my suggestions and analyses using this data profile!" }]
        });
      }

      // Add historical chat messages
      for (const msg of messages) {
        contentsPayload.push({
          role: msg.role === 'user' ? 'user' : 'model',
          parts: [{ text: msg.content }]
        });
      }

      // Query Gemini with automatic model fallback to prevent project denied (403) errors.
      let response;
      const modelsToTry = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"];
      let lastError: any = null;
      for (const model of modelsToTry) {
        try {
          response = await ai.models.generateContent({
            model: model,
            contents: contentsPayload,
            config: {
              systemInstruction,
              temperature: 0.7,
            }
          });
          break;
        } catch (error: any) {
          const errStr = (error.message || "").toLowerCase();
          if (errStr.includes("403") || errStr.includes("denied") || errStr.includes("project") || errStr.includes("forbidden") || error.status === 403) {
            throw new Error("Your Google Cloud project / API Key has been denied access (403). Please configure an active and valid GEMINI_API_KEY in the AI Studio settings or check your billing status.");
          } else if (errStr.includes("not found") || errStr.includes("model") || errStr.includes("503") || error.status === 404) {
            lastError = error;
            console.warn(`Model ${model} failed, trying next fallback...`);
            continue;
          } else {
            throw error;
          }
        }
      }
      if (!response) {
        throw lastError || new Error("All generative model fallbacks failed.");
      }

      const reply = response.text || "I was unable to formulate a response. Let me try again with more details.";
      return res.json({ reply });

    } catch (error: any) {
      const errStr = (error.message || "").toLowerCase();
      const is403 = errStr.includes("403") || errStr.includes("denied") || errStr.includes("project") || errStr.includes("forbidden") || error.status === 403;
      
      if (is403) {
        console.warn("Gemini API access notice (403 / restricted): Access is currently denied or key is offline.");
        const friendlyMarkdown = 
          "✨ **StatMentor Notice**:\n\n" +
          "Active generative AI recommendation chat features require a valid **Gemini API Key**. " +
          "Your Google Cloud project / API Key has been denied access (403) or is currently missing.\n\n" +
          "🔧 **How to resolve this**:\n" +
          "1. Grab a free API key from [Google AI Studio](https://aistudio.google.com/).\n" +
          "2. Configure your `GEMINI_API_KEY` in the AI Studio settings or check your billing status to restore chat capabilities instantly!\n\n" +
          "*(Note: All local offline data analysis, distribution plots, metrics, and rule-based testing fallback recommendations remain 100% functional!)*";
        return res.json({ reply: friendlyMarkdown });
      }

      console.error("Generic server processing error encountered:", error);
      return res.status(500).json({ 
        error: error.message || "An error occurred during generative processing.",
        message: `Oops! StatMentor AI encountered an issue: ${error.message || 'Unknown integration error'}.`
      });
    }
  });

  // Hot-serve API checks
  app.get("/api/health", (req, res) => {
    res.json({ 
      status: "online", 
      geminiConfigured: !!process.env.GEMINI_API_KEY 
    });
  });

  // Vite integration middleware
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    // SPA fallback
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`StatMentor AI server listening on http://localhost:${PORT} [ENV: ${process.env.NODE_ENV || 'dev'}]`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server:", err);
});
