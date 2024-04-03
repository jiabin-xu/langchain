import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";

export async function init() {
  const loader = new CheerioWebBaseLoader(
    "https://docs.smith.langchain.com/user_guide"
  );
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter();
  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings();
  const vectorstores = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  const retriever = vectorstores.asRetriever();
  return retriever;
}

export async function retrieval() {
  const retriever = await init();

  const prompt =
    ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:
<context>
{context}
</context>

Question: {input}`);
  const llm = new ChatOpenAI();
  const documentChain = await createStuffDocumentsChain({
    llm,
    prompt,
  });

  const retrievalChain = await createRetrievalChain({
    combineDocsChain: documentChain,
    retriever,
  });

  const result = await retrievalChain.invoke({
    input: "what is LangSmith?",
  });
  console.log("result :>> ", result);
}
