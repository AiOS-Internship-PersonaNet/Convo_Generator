import { LLMSingleActionAgent, AgentExecutor } from 'langchain/agents';
import { Document } from "langchain/document";
import { FaissStore } from 'langchain/vectorstores/faiss';
import { formatLogToString } from "langchain/agents/format_scratchpad/log";
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAIAssistantRunnable } from "langchain/experimental/openai_assistant";
import { PromptTemplate } from 'langchain/prompts'; // Updated import for PromptTemplate
import { RunnableSequence } from "langchain/schema/runnable";

import {
    HumanMessage,
} from "langchain/schema";
import {
    BraveSearch,
    StructuredTool,
    RequestsGetTool,
    RequestsPostTool,
    AIPluginTool,
    SerpAPI,
    DynamicTool,
} from 'langchain/tools';

import { apikey } from './apikeys.js';
import { instructionsTemplate } from './templates.js';

const model = new ChatOpenAI({ temperature: 0.9, openAIApiKey: apikey }).bind({
    stop: ["\nObservation"],
});;


// Define tools
const serpAPITool = new SerpAPI("af50c8c9e2d408b3256e8a1c316da119da0c4e16dcf05e07a560181de5f4f951")
const searchTool = new DynamicTool({
    name: "duckduckgo tool",
    description: "Tool for getting the latest information from duckduckgo api",
    func: async (query) => {
        const settings = {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
        };
        // const q = `https://api.duckduckgo.com/?q=${query}&format=json`;
        const q = "https://serpapi.com/search?engine=duckduckgo";
        const res = await fetch(q, settings)
            .then(function (response) {
                return response.json()
            }).catch(console.log)
        return res;
    },
});

export const tools = [
    searchTool,
    serpAPITool,
];

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

    if (match) {
        return {
            tool: match[1].trim(),
            toolInput: match[2].trim(),
            log: text
        };
    }
}

// const runnable = RunnableSequence.from([
//     {
//         input: (values) => values.input,
//         intermediate_steps: (values) => values.steps,
//     },
//     formatMessages,
//     model,
//     customOutputParser,
// ]);

// const agentExecutor = new AgentExecutor({
//     agent: runnable,
//     tools,
//     verbose: true,
// });

const input = `User_1: Office worker that loves sports, and watching them.
User_2: Teacher that loves art and painting.`

const instruction = instructionsTemplate.format({
    tools: tools,
    tool_names: tools.map((tool) => tool.name).join(",\n"),
    input: input,
});
const agent = await OpenAIAssistantRunnable.createAssistant({
    clientOptions: { apiKey: apikey },
    model: "gpt-3.5-turbo-1106",
    // instructions: instruction,
    instructions: "Given the two user's data that includes interests and traits, find a current discussion topic and create a 16 sentense discussion between the two users using realistic emotional language. These are the tools available to you",
    name: "Conversation Generator",
    tools,
    asAgent: true,
});
const agentExecutor = AgentExecutor.fromAgentAndTools({
    agent,
    tools,
});

const result = await agentExecutor.invoke({ content: input, });
console.log(result);

