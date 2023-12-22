import { LLMSingleActionAgent, AgentExecutor } from 'langchain/agents';
import { LLMChain } from "langchain/chains";
import { Document } from "langchain/document";
import { FaissStore } from 'langchain/vectorstores/faiss';
import { formatLogToString } from "langchain/agents/format_scratchpad/log";
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PromptTemplate } from 'langchain/prompts'; // Updated import for PromptTemplate
import { RunnableSequence } from "langchain/schema/runnable";
import {
    HumanMessage,
} from "langchain/schema";
import { StructuredTool } from 'langchain/tools';
import {
    WikipediaQueryRun,
    RequestsGetTool,
    RequestsPostTool,
    AIPluginTool,
} from 'langchain/tools';

const apikey = "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"
const model = new ChatOpenAI({ temperature: 0.9, openAIApiKey: apikey }).bind({
    stop: ["\nObservation"],
});;

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
// const braveSearchTool = new BraveSearchParams()
const tool = new WikipediaQueryRun()
// Define tools
const tools = [new RequestsGetTool(),
new RequestsPostTool(),
await AIPluginTool.fromPluginUrl(
    "https://www.klarna.com/.well-known/ai-plugin.json"
),];

// Create documents for FAISS vector store
const docs = tools.map((tool, index) => new Document({ pageContent: tool.description, metadata: { index } }));



// Initialize FaissStore with documents and embeddings
const vectorStore = new FaissStore(new OpenAIEmbeddings({ openAIApiKey: apikey }), {});
// await vectorStore.addDocuments(docs);

// Function to retrieve relevant tools based on a query
async function getTools(query) {
    const relevantDocs = await vectorStore.similaritySearch(query, tools.length);
    return relevantDocs.map(doc => tools[doc.metadata.index]);
}

// Initialize the components

const PREFIX = `Given the two user's data that includes interests and traits, find a current discussion topic and create a 16 sentence discussion between the two users using realistic emotional language. These are the tools available to you:
{tools}`;

const TOOL_INSTRUCTIONS_TEMPLATE = `Use the following format:

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
`;

const SUFFIX = `Begin! Remember to have the personalities of each user in mind when giving your final conversation.

Users: {input}
Thought:`;
async function formatMessages(
    values
) {
    /** Check input and intermediate steps are both inside values */
    if (!("input" in values) || !("intermediate_steps" in values)) {
        throw new Error("Missing input or agent_scratchpad from values.");
    }
    /** Extract and case the intermediateSteps from values as Array<AgentStep> or an empty array if none are passed */
    const intermediateSteps = values.intermediate_steps
        ? (values.intermediate_steps)
        : [];
    /** Call the helper `formatLogToString` which returns the steps as a string  */
    const agentScratchpad = formatLogToString(intermediateSteps);
    /** Construct the tool strings */
    const toolStrings = tools
        .map((tool) => `${tool.name}: ${tool.description}`)
        .join("\n");
    const toolNames = tools.map((tool) => tool.name).join(",\n");
    /** Create templates and format the instructions and suffix prompts */
    const prefixTemplate = new PromptTemplate({
        template: PREFIX,
        inputVariables: ["tools"],
    });
    const instructionsTemplate = new PromptTemplate({
        template: TOOL_INSTRUCTIONS_TEMPLATE,
        inputVariables: ["tool_names"],
    });
    const suffixTemplate = new PromptTemplate({
        template: SUFFIX,
        inputVariables: ["input"],
    });
    /** Format both templates by passing in the input variables */
    const formattedPrefix = await prefixTemplate.format({
        tools: toolStrings,
    });
    const formattedInstructions = await instructionsTemplate.format({
        tool_names: toolNames,
    });
    const formattedSuffix = await suffixTemplate.format({
        input: values.input,
    });
    /** Construct the final prompt string */
    const formatted = [
        formattedPrefix,
        formattedInstructions,
        formattedSuffix,
        agentScratchpad,
    ].join("\n");
    /** Return the message as a HumanMessage. */
    return [new HumanMessage(formatted)];
}

// function customOutputParser(message) {
//     // Implement the logic for parsing LLM output
//     const llmOutput = message.content;
//     const observationMatch = llmOutput.match(/Observation:(.*)/s);
//     if (observationMatch) {
//         return new AgentStep({
//             returnValues: { output: observationMatch[1].trim() },
//             log: llmOutput
//         });
//     }

//     const finalConversationMatch = llmOutput.match(/Final Conversation:(.*)/s);
//     if (finalConversationMatch) {
//         return new AgentStep({
//             returnValues: { output: finalConversationMatch[1].trim() },
//             log: llmOutput
//         });
//     }

//     const actionMatch = llmOutput.match(/Action\s*:(.*?)\nAction\s*Input\s*:(.*)/s);
//     if (actionMatch) {
//         this.lastAction = actionMatch[1].trim();
//         this.lastActionInput = actionMatch[2].trim();
//         return new AgentStep({ tool: this.lastAction, toolInput: this.lastActionInput, log: llmOutput });
//     }

// }
function customOutputParser(message) {
    const text = message.content;
    if (typeof text !== "string") {
        throw new Error(
            `Message content is not a string. Received: ${JSON.stringify(
                text,
                null,
                2
            )}`
        );
    }
    const observationMatch = text.match(/Observation:(.*)/s);
    if (observationMatch) {
        return {
            returnValues: { output: observationMatch[1].trim() },
            log: text,
        };;
    }
    /** If the input includes "Final Answer" return as an instance of `AgentFinish` */
    if (text.includes("Final Conversation:")) {
        const parts = text.split("Final Conversation:");
        const input = parts[parts.length - 1].trim();
        const finalAnswers = { output: input };
        return { log: text, returnValues: finalAnswers };
    }
    /** Use RegEx to extract any actions and their values */
    const match = text.match(/Action\s*:(.*?)\nAction\s*Input\s*:(.*)/s);

    // if (!match) {
    //     throw new Error(`Could not parse LLM output: ${text}`);
    // }
    /** Return as an instance of `AgentAction` */
    if (match) {
        return {
            tool: match[1].trim(),
            toolInput: match[2].trim(),
            log: text
        };
    }
}


const runnable = RunnableSequence.from([
    {
        input: (values) => values.input,
        intermediate_steps: (values) => values.steps,
    },
    formatMessages,
    model,
    customOutputParser,
]);

const executor = new AgentExecutor({
    agent: runnable,
    tools,
    verbose: true,
});
const input = `User_1: Office worker that loves to play video games, on his days off he enjoys watching anime
User_2: Teacher that loves to do art in free time. Always up to date on politics`
const result = await executor.invoke({ input });

// Define the agent
// const llmChain = new LLMChain({ model, prompt: TOOL_INSTRUCTIONS_TEMPLATE, customOutputParser, formatMessages });

// const agent = new LLMSingleActionAgent({
//     llmChain: llmChain,
//     stop: ["\nObservation:"],
//     // promptTemplate: promptTemplate,
//     tools: tools
// });

// // Run the agent executor
// async function runAgentExecutor(user1, user2) {
//     const intermediateSteps = [];
//     const agentExecutor = new AgentExecutor({
//         agent: agent,
//         tools,
//         handleParsingErrors: true,
//         verbose: true
//     });

//     const conversation = await agentExecutor.invoke({ input });
//     console.log(conversation);
// }

// // Example usage
// runAgentExecutor("User1", "User2");