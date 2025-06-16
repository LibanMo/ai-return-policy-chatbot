// backend/index.js

import dotenv from "dotenv"; // Ladda miljövariabler explicit
dotenv.config(); // Läs in variablerna från .env-filen

import express from "express"; // Webbramverk för Node.js
import cors from "cors"; // För att hantera CORS mellan frontend och backend
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"; // Google Gemini LLM
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"; // Google Embeddings
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase"; // Supabase vektorbutik
import { createClient } from "@supabase/supabase-js"; // Supabase klient
import { StringOutputParser } from "@langchain/core/output_parsers"; // För att parsa LLM-svaret till en sträng
import {
  RunnableSequence, // För att bygga upp kedjor av operationer
  RunnablePassthrough, // För att skicka input vidare i kedjan
} from "@langchain/core/runnables";
import { PromptTemplate } from "@langchain/core/prompts"; // För att skapa instruktionsprompts
import { BufferMemory } from "langchain/memory"; // För att hantera konversationsminne

const app = express(); // Skapa Express-applikationen
const PORT = process.env.PORT || 5000; // Ange port, antingen från .env eller standard 5000

// Middleware
app.use(cors()); // Tillåt Cross-Origin Resource Sharing (viktigt för frontend/backend-kom.)
app.use(express.json()); // Tolka JSON-request bodies

// Kontrollera att alla nödvändiga miljövariabler är laddade vid start
const googleApiKey = process.env.GOOGLE_API_KEY;
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

// Diagnostik: Logga status för miljövariablerna
console.log("--- Diagnostik: Miljövariabler (index.js) ---");
console.log("GOOGLE_API_KEY:", googleApiKey ? "Laddad" : "EJ LADAD");
console.log("SUPABASE_URL:", supabaseUrl ? "Laddad" : "EJ LADAD");
console.log(
  "SUPABASE_SERVICE_ROLE_KEY:",
  supabaseServiceRoleKey ? "Laddad" : "EJ LADAD",
);
console.log("----------------------------------------------");

// Avsluta servern om kritiska miljövariabler saknas
if (!googleApiKey || !supabaseUrl || !supabaseServiceRoleKey) {
  console.error(
    "FEL: Saknar en eller flera nödvändiga miljövariabler. Kontrollera din .env-fil.",
  );
  process.exit(1); // Avsluta programmet med felkod
}

// Globala instanser för AI-komponenter
let llm; // Vår stora språkmodell (Gemini)
let vectorStore; // Kopplingen till vår Supabase vektor-databas
let retriever; // Verktyget för att söka i vektor-databasen
let conversationChain; // Hela konversationskedjan med minnesstöd
let memory; // Vår minnesinstans för att lagra chatt-historik

// Asynkron funktion för att initialisera alla AI-komponenter
async function initializeAIComponents() {
  try {
    console.log("Initierar Google Gemini LLM...");
    llm = new ChatGoogleGenerativeAI({
      apiKey: googleApiKey,
      modelName: "gemini-1.5-flash", // Den specifika Gemini-modellen
      temperature: 0, // Lägre temperatur för faktabaserade, mindre "kreativa" svar
      maxRetries: 2, // Antal försök vid misslyckade anrop
    });
    console.log("Gemini LLM initierad.");

    console.log(
      "Initierar Supabase klient och vektorbutik med Google Embeddings...",
    );
    const supabase = createClient(supabaseUrl, supabaseServiceRoleKey);
    vectorStore = new SupabaseVectorStore(
      new GoogleGenerativeAIEmbeddings({
        apiKey: googleApiKey,
        modelName: "embedding-001", // Googles modell för att skapa embeddings (vektorer)
      }),
      {
        client: supabase,
        tableName: "documents", // Tabellen i Supabase där embeddings lagras
        queryName: "match_documents", // Den SQL-funktion vi skapade för sökning
      },
    );
    console.log("Supabase vektorbutik initierad.");

    // Skapa en retriever från vektorbutiken
    // k = 3 betyder att vi hämtar de 3 mest relevanta dokumentbitarna
    retriever = vectorStore.asRetriever({ k: 3 });
    console.log("Retriever för Supabase initierad.");

    console.log("Initierar konversationsminne...");
    memory = new BufferMemory({
      memoryKey: "chat_history", // Nyckeln som används i prompten för att injicera historik
      returnMessages: true, // Returnera historiken som meddelandeobjekt (HumanMessage, AIMessage)
    });
    console.log("Minne initierat.");

    // Prompt-mall för Gemini med plats för hämtad kontext OCH chatt-historik
    const chatPrompt = PromptTemplate.fromTemplate(`
        Använd enbart följande utdrag från returpolicyn för att svara på användarens fråga.
        Svara alltid neutralt och faktabaserat, baserat på den angivna informationen.

        Om du INTE kan hitta svaret i den tillhandahållna texten, vänligen svara artigt med något i stil med: "Jag kunde tyvärr inte hitta svar på din fråga i vår returpolicy. Vänligen kontakta oss via e-post på support@farskvaruhornan.se så hjälper vi dig vidare."

        ---------------------
        {context}  <-- Här injiceras relevanta textbitar från din policy
        ---------------------

        Konversationshistorik:
        {chat_history}  <-- Här injiceras tidigare meddelanden (frågor och svar)
        Användarens fråga: {question}
        Svar:`);

    // Bygg Retrieval-Augmented Generation (RAG) kedjan med stöd för konversation
    conversationChain = RunnableSequence.from([
      // Steg 1: Förbered input för prompten.
      // Här hanterar vi tre separata delar som alla går till prompten:
      {
        context: (input) => retriever.invoke(input.question), // Retrievern får ENDAST den nya frågan för att hitta kontext
        question: new RunnablePassthrough(), // Användarens nya fråga skickas direkt
        chat_history: async (input) => {
          // Hela konversationshistoriken hämtas och skickas till prompten
          const { chat_history } = await memory.loadMemoryVariables({});
          return chat_history;
        },
      },
      // Steg 2: Fyll prompt-mallen med den förberedda inputen
      chatPrompt,
      // Steg 3: Skicka den kompletta prompten till Gemini LLM
      llm,
      // Steg 4: Parsa svaret till en sträng
      new StringOutputParser(),
    ]);
    console.log("RAG-kedjan med minnesstöd är byggd och redo.");
  } catch (error) {
    console.error("FEL vid initialisering av AI-komponenter:", error);
    process.exit(1); // Avsluta om initiering misslyckas
  }
}

// Starta initialiseringen av AI-komponenterna innan servern börjar lyssna
initializeAIComponents().then(() => {
  // Enkel test-route för att verifiera att servern körs
  app.get("/", (req, res) => {
    res.send("Välkommen till Return Policy Chatbot Backend (AI redo)!");
  });

  // Huvud-endpoint för chattfrågor
  app.post("/query", async (req, res) => {
    const { question } = req.body; // Hämta användarens fråga från request body
    console.log("Mottagen fråga från frontend:", question);

    if (!question) {
      return res.status(400).json({ error: "Fråga saknas i förfrågan." });
    }

    try {
      // Kör hela RAG-kedjan med minnesstöd
      const result = await conversationChain.invoke({ question });

      // Spara den aktuella konversationsrundan till minnet
      await memory.saveContext(
        { input: question }, // Användarens input
        { output: result }, // AI:ns svar
      );

      console.log("AI-svar genererat och minne uppdaterat:", result);
      res.json({ answer: result }); // Skicka tillbaka svaret till frontend
    } catch (error) {
      console.error("FEL vid generering av AI-svar:", error);
      res
        .status(500)
        .json({ error: "Kunde inte generera svar. Försök igen senare." });
    }
  });

  // Starta Express-servern och börja lyssna på inkommande förfrågningar
  app.listen(PORT, () => {
    console.log(`Server körs på port ${PORT}`);
  });
});
