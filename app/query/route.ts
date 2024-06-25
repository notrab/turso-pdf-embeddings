import { OpenAIEmbeddings } from "@langchain/openai";
import { createClient } from "@libsql/client";
import { LibSQLVectorStore } from "../turso-vector-store";

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
  const queryEmbedding = await embeddings.embedQuery("measure");

  const result = await vectorstore.similaritySearchWithScore(
    `${queryEmbedding}`,
    2,
  );

  return Response.json(result);
}
