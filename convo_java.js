// import { apikey } from "./apikey.js"

import { LLMSingleActionAgent, AgentExecutor } from "langchain/agents";
import { AgentAction, AgentFinish } from "langchain/schema";
import { Document } from "langchain/dist/document.js";
import { FAISS } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { StructuredTool } from "langchain/tools";
import { OpenAI } from "langchain/llms/openai";
import { BraveSearch } from "langchain/tools";
// For OpenAI, you might need to check the specific path for import in the package
// It should be something similar to this
search = BraveSearch()
class SearchTool extends StructuredTool {
    constructor() {
        super({
            name: 'Search',
            func: async (input) => {
                // Implement search functionality here
            },
            description: 'Useful for when you need to present a conversation topic on current events'
        });
    }
}

// Define tools
const searchTool = new SearchTool();
const tools = [searchTool];

// Create documents for FAISS vector store
const docs = tools.map((tool, index) => new Document({ pageContent: tool.description, metadata: { index } }));
const vectorStore = FAISS.fromDocuments(docs, new OpenAIEmbeddings());
const retriever = vectorStore.asRetriever();

// Function to retrieve relevant tools based on a query
async function getTools(query) {
    const relevantDocs = await retriever.getRelevantDocuments(query);
    return relevantDocs.map(doc => tools[doc.metadata.index]);
}

// Custom prompt template
class ConvoPromptTemplate extends StringPromptTemplate {
    constructor(template, toolsGetter) {
        super(template);
        this.toolsGetter = toolsGetter;
    }

    async format(kwargs) {
        let thoughts = "";
        const intermediateSteps = kwargs.intermediateSteps || [];
        const user1 = kwargs.user1 || "";
        const user2 = kwargs.user2 || "";
        const tools = await this.toolsGetter(`${user1} ${user2}`);

        intermediateSteps.forEach(([action, observation]) => {
            thoughts += `Action: ${action}\nObservation: ${observation}\n`;
        });

        const toolDescriptions = tools.map(tool => `${tool.name}: ${tool.description}`).join("\n");
        const toolNames = tools.map(tool => tool.name).join(", ");

        return this.template.replace("{tools}", toolDescriptions)
                            .replace("{tool_names}", toolNames)
                            .replace("{user_1}", user1)
                            .replace("{user_2}", user2)
                            .replace("{agent_scratchpad}", thoughts);
    }
}

// Custom output parser
class CustomOutputParser {
    constructor() {
        this.lastAction = null;
        this.lastActionInput = null;
    }

    async parse(llmOutput) {
        // Implement the logic for parsing LLM output
        const observationMatch = llmOutput.match(/Observation:(.*)/s);
        if (observationMatch) {
            return new AgentFinish({
                returnValues: { output: observationMatch[1].trim() },
                log: llmOutput
            });
        }

        const finalConversationMatch = llmOutput.match(/Final Conversation:(.*)/s);
        if (finalConversationMatch) {
            return new AgentFinish({
                returnValues: { output: finalConversationMatch[1].trim() },
                log: llmOutput
            });
        }

        const actionMatch = llmOutput.match(/Action\s*:(.*?)\nAction\s*Input\s*:(.*)/s);
        if (actionMatch) {
            this.lastAction = actionMatch[1].trim();
            this.lastActionInput = actionMatch[2].trim();
            return new AgentAction({ tool: this.lastAction, toolInput: this.lastActionInput, log: llmOutput });
        }

        throw new Error("Could not parse LLM output");
    }
}

// Initialize the components
const llm = new OpenAI({ temperature: 0.9 });
const template = `Given the two user's data that includes interests and traits, find a current discussion topic and create a 16 sentence discussion between the two users using realistic emotional language. These are the tools available to you:
{tools}

Use the following format:

Topic: the current discussion topic based on user_1 and user_2 
Thought: You should always think about what to do 
Action: the action to take should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I know what the conversation will be about
Action: imitate User_1 to start discussion based on observation 
Thought: I have User_1's conversation starter
User_2 Response: imitate User_2 reply to User_1's Conversation starter
Thought: I have User_2's reply to User_1's response
User_1 Response: imitate User_1 reply to User_1's reply
Thought: I have User_1's reply to User_2's response
...(this User_2 Response/User_1 Response can repeat 8 times)

Thought: I now have the conversation 
Final Conversation: the final conversation based on user_1 and user_2

Begin! Remember to have the personalities of each user in mind when giving your final conversation.

User_1 = {user_1}
User_2 = {user_2}
{agent_scratchpad}`;
const promptTemplate = new ConvoPromptTemplate(template, getTools);
const outputParser = new CustomOutputParser();

// Create LLMChain
const llmChain = new LLMChain({ llm, prompt: promptTemplate, outputParser });

// Define the agent
const agent = new LLMSingleActionAgent({
    llmChain,
    stop: ["\nObservation:"],
    allowedTools: tools.map(tool => tool.name)
});

// Run the agent executor
async function runAgentExecutor(user1, user2) {
    const intermediateSteps = [];
    const agentExecutor = new AgentExecutor({
        agent,
        tools,
        handleParsingErrors: true,
        verbose: true
    });

    const conversation = await agentExecutor.run({ user1, user2, intermediateSteps });
    console.log(conversation);
}

// Example usage