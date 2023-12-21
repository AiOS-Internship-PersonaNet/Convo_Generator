import { LLMSingleActionAgent, AgentExecutor } from 'langchain/agents';
import { Document } from "langchain/document";
import { FaissStore } from 'langchain/vectorstores/faiss';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PromptTemplate } from 'langchain/prompts'; // Updated import for PromptTemplate
import { StructuredTool } from 'langchain/tools';
import { OpenAI } from 'langchain/llms/openai';
import { BraveSearch } from 'langchain/tools';

const apikey = "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"
class SearchTool extends StructuredTool {
    constructor() {
        super({
            name: 'Search',
            func: async (input) => {
                // Implement search functionality here, using SerpAPI for example
                return await search(input);
            },
            description: 'Useful for when you need to present a conversation topic on current events'
        });
    }
}

// Initialize SerpAPI tool
// const search = new SerpAPI();
const searchTool = new SearchTool();

// Define tools
const tools = [searchTool];

// Create documents for FAISS vector store
const docs = tools.map((tool, index) => new Document({ pageContent: tool.description, metadata: { index } }));



// Initialize FaissStore with documents and embeddings
const vectorStore = new FaissStore(new OpenAIEmbeddings({openAIApiKey: apikey}), {});
// await vectorStore.addDocuments(docs);

// Function to retrieve relevant tools based on a query
async function getTools(query) {
    const relevantDocs = await vectorStore.similaritySearch(query, tools.length);
    return relevantDocs.map(doc => tools[doc.metadata.index]);
}

class ConvoPromptTemplate extends PromptTemplate { // Updated base class
    constructor(template, inputVariables, toolsGetter) {
        super( inputVariables, template ); // Updated constructor call
        // super( inputVariables );
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

        const formattedTemplate = this.template.replace("{tools}", toolDescriptions)
                            .replace("{tool_names}", toolNames)
                            .replace("{user_1}", user1)
                            .replace("{user_2}", user2)
                            .replace("{agent_scratchpad}", thoughts);

        // Return formatted template
        return formattedTemplate;
    }
}

// Initialize the components
const llm = new OpenAI({ temperature: 0.9, openAIApiKey: apikey });
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
{agent_scratchpad}`; // Your template string here

const inputVariables = ["user_1", "user_2",];
const promptTemplate = new ConvoPromptTemplate(template, inputVariables, getTools);

// Define the agent
const agent = new LLMSingleActionAgent({
    llmChain: llm,
    promptTemplate: promptTemplate,
    tools: tools
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
runAgentExecutor("User1", "User2");