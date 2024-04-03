import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

export async function simple() {
  const chatModel = new ChatOpenAI({});
  const res = await chatModel.invoke(" 在大语言模型中, Embedding怎么理解 ?");
  console.log("res :>> ", res);
}

export async function prompt() {
  const chatModel = new ChatOpenAI({});
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "你是AI大师, 能够深入浅出的解释AI相关概念,善于使用举例来解释问题",
    ],
    ["user", "{input}"],
  ]);
  const chain = prompt.pipe(chatModel);
  const res = await chain.invoke({
    input: " 在大语言模型中, Embedding怎么理解 ?",
  });
  console.log("res :>> ", res);
}
