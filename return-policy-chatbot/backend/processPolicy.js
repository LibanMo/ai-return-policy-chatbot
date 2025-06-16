// backend/processPolicy.js

import "dotenv/config"; // Ladda miljövariabler

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";

// Importera path och fileURLToPath för att hantera sökvägar i ES Modules
import path from "path";
import { fileURLToPath } from "url";

// __dirname motsvarighet i ES Modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Testloggar för att verifiera att miljövariabler laddas
console.log("Scriptet har startats (processPolicy.js).");
console.log(
  "GOOGLE_API_KEY:",
  process.env.GOOGLE_API_KEY ? "Laddad" : "EJ LADAD",
);
console.log("SUPABASE_URL:", process.env.SUPABASE_URL ? "Laddad" : "EJ LADAD");
console.log(
  "SUPABASE_SERVICE_ROLE_KEY:",
  process.env.SUPABASE_SERVICE_ROLE_KEY ? "Laddad" : "EJ LADAD",
);

async function processAndEmbedPolicy() {
  console.log("Startar process för att ladda och bädda in returpolicy...");

  // 1. Kontrollera miljövariabler
  const googleApiKey = process.env.GOOGLE_API_KEY;
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!googleApiKey || !supabaseUrl || !supabaseServiceRoleKey) {
    console.error(
      "FEL: Saknar en eller flera nödvändiga miljövariabler (GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY). Kontrollera din .env-fil.",
    );
    process.exit(1);
  }

  // 2. Ladda PDF-dokumentet
  const pdfPath = path.join(__dirname, "returpolicy.pdf"); // Använd path.join för att säkerställa korrekt sökväg
  console.log(`Laddar PDF från: ${pdfPath}`);
  const loader = new PDFLoader(pdfPath);
  const docs = await loader.load();
  console.log(`PDF laddad. Antal sidor/dokument: ${docs.length}`);

  // 3. Dela upp texten i "chunks"
  console.log("Delar upp texten i mindre bitar...");
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const splittedDocs = await splitter.splitDocuments(docs);
  console.log(`Texten uppdelad i ${splittedDocs.length} bitar.`);

  // 4. Skapa embeddings (vektorer) - HÄR ÄR DEN ENDA EMBEDDING-MODELLEN VI BEHÖVER HÄR
  console.log(
    "Genererar embeddings för varje textbit med Google Gemini (embedding-001)...",
  );
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: googleApiKey,
    modelName: "embedding-001", // Denna modell är för embeddings
  });

  // 5. Initiera Supabase-klienten och ladda upp embeddings
  console.log("Laddar upp embeddings till Supabase...");
  const supabase = createClient(supabaseUrl, supabaseServiceRoleKey);

  try {
    await SupabaseVectorStore.fromDocuments(splittedDocs, embeddings, {
      client: supabase,
      tableName: "documents",
      queryName: "match_documents",
    });
    console.log("Embeddings har laddats upp till Supabase framgångsrikt!");
  } catch (error) {
    console.error("FEL vid uppladdning till Supabase:", error);
    if (error.message.includes("vector dimension mismatch")) {
      console.error(
        "Det verkar som att dimensionsstorleken för vektorn är felaktig i Supabase-tabellen.",
      );
      console.error(
        'Kontrollera dimensionen för "embedding-001" (Googles embedding-modell).',
      );
      console.error(
        "Du kan behöva ändra `VECTOR(1536)` i din `documents`-tabell SQL till rätt dimension (förmodligen 768).",
      );
      console.error(
        'Kolla Google Generative AI-dokumentationen för "embedding-001" för exakt dimension.',
      );
    }
  }

  console.log("Processen är klar.");
}

// Kör funktionen
processAndEmbedPolicy().catch(console.error);
