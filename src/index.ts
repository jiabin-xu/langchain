import { ChatOpenAI } from "@langchain/openai";

async function main() {
  const llm = new ChatOpenAI({});
  const res = await llm.invoke("Hello, world!");

  console.log("res :>> ", res);
}

main();
