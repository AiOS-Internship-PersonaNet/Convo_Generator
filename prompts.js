// Initialize the components

export const PREFIX = `Given the two user's data that includes interests and traits, find a current discussion topic and create a 16 sentence discussion between the two users using realistic emotional language. These are the tools available to you:
{tools}`;

export const TOOL_INSTRUCTIONS_TEMPLATE = `Use the following format:

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

export const SUFFIX = `Begin! Remember to have the personalities of each user in mind when giving your final conversation.

Users: {input}
Thought:`;