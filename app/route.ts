import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient } from "@libsql/client";
import { LibSQLVectorStore } from "./turso-vector-store";

const fileName = "example.pdf";

const loader = new PDFLoader(fileName, {});

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large",
  dimensions: 1024,
});

const db = createClient({
  url: process.env.TURSO_DATABASE_URL!,
  authToken: process.env.TURSO_DATABASE_AUTH_TOKEN,
});

const vectorstore = new LibSQLVectorStore(db, embeddings, {
  tableName: "documents",
  embeddingField: "embedding",
});

export async function GET() {
  const documents = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const chunks = await splitter.splitDocuments(documents);

  await vectorstore.addDocuments(chunks);

  return Response.json({ chunks });
}
