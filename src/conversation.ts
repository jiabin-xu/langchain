import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { init } from "./deeper";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";

export async function conversation() {
  const historyAwarePrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
    ],
  ]);
  const retriever = await init();
  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: new ChatOpenAI(),
    retriever,
    rephrasePrompt: historyAwarePrompt,
  });

  // const result = await historyAwareRetriever.invoke({
  //   chat_history: [
  //     new HumanMessage("Can LangSmith help test my LLM applications?"),
  //     new AIMessage("Yes "),
  //   ],
  //   input: "Tell me how",
  // });
  // console.log("result :>> ", result);

  const historyAwareRetrieverPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user's questions based on the below context:\n\n{context}",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);
  const historyAwareCombineDocsChain = await createStuffDocumentsChain({
    llm: new ChatOpenAI(),
    prompt: historyAwareRetrieverPrompt,
  });
  const conversationalRetrievalChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: historyAwareCombineDocsChain,
  });
  const result2 = await conversationalRetrievalChain.invoke({
    chat_history: [
      new HumanMessage("Can LangSmith help test my LLM applications?"),
      new AIMessage("Yes!"),
    ],
    input: "tell me how",
  });
  console.log("result2 :>> ", result2);
}
