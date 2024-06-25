import type { Client } from "@libsql/client";
import { VectorStore } from "@langchain/core/vectorstores";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { Document } from "@langchain/core/documents";

export interface LibSQLVectorStoreArgs {
  tableName?: string;
  embeddingField?: string;
}

export class LibSQLVectorStore extends VectorStore {
  declare FilterType: (doc: Document) => boolean;

  private db;
  private tableName: string;
  private embeddingField: string;

  _vectorstoreType(): string {
    return "libsql";
  }

  constructor(
    db: Client,
    embeddings: EmbeddingsInterface,
    options: LibSQLVectorStoreArgs = {
      tableName: "vectors",
      embeddingField: "embedding",
    },
  ) {
    super(embeddings, options);

    this.db = db;
    this.tableName = options.tableName || "vectors";
    this.embeddingField = options.embeddingField || "embedding";

    this.initializeTable();
  }

  async initializeTable() {
    await this.db.batch(
      [
        `
            CREATE TABLE IF NOT EXISTS ${this.tableName} (
              content TEXT,
              metadata TEXT,
              ${this.embeddingField} FLOAT32(3)
            );
          `,
        `CREATE INDEX IF NOT EXISTS ${`${this.tableName}_idx`} USING diskann_cosine_ops ON ${this.tableName} (${this.embeddingField});`,
      ],
      "write",
    );
  }

  async addDocuments(documents: Document[]): Promise<void> {
    const texts = documents.map(({ pageContent }) => pageContent);
    const embeddings = await this.embeddings.embedDocuments(texts);

    return this.addVectors(embeddings, documents);
  }

  async addVectors(vectors: number[][], documents: Document[]): Promise<void> {
    const rows = vectors.map((embedding, idx) => ({
      content: documents[idx].pageContent,
      embedding,
      metadata: JSON.stringify(documents[idx].metadata),
    }));

    const batchSize = 100;

    for (let i = 0; i < rows.length; i += batchSize) {
      const chunk = rows.slice(i, i + batchSize).map((row) => {
        const sql = `INSERT INTO ${this.tableName} (content, metadata, ${this.embeddingField}) VALUES (?, ?, ?)`;

        return {
          sql,
          args: [
            row.content,
            row.metadata,
            `vector('[${row.embedding.join(",")}]')`,
            // new Float32Array(row.embedding).buffer as ArrayBuffer,
          ],
        };
      });

      await this.db.batch(chunk, "write");
    }
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this["FilterType"],
  ): Promise<[Document, number][]> {
    const searchQuery = `SELECT content FROM vector_top_k('${this.tableName}_idx', '[${query}]', ${k}) JOIN ${this.tableName} ON ${this.tableName}.rowid = id;`;

    const results = await this.db.execute(searchQuery);

    const documentsWithScores: [Document, number][] = results.rows.map(
      (row: any) => {
        // const metadata = JSON.parse(row.metadata);
        const doc = new Document({
          metadata: {},
          pageContent: row.content,
        });
        return [doc, row.similarity];
      },
    );

    return documentsWithScores;
  }

  static async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: EmbeddingsInterface,
    db: Client,
  ): Promise<LibSQLVectorStore> {
    const docs: Document[] = [];

    for (let i = 0; i < texts.length; i += 1) {
      const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas;
      const newDoc = new Document({
        pageContent: texts[i],
        metadata,
      });

      docs.push(newDoc);
    }

    return this.fromDocuments(docs, embeddings, db);
  }

  static async fromDocuments(
    docs: Document[],
    embeddings: EmbeddingsInterface,
    db: Client,
  ): Promise<LibSQLVectorStore> {
    const instance = new this(db, embeddings);
    await instance.addDocuments(docs);

    return instance;
  }
}
