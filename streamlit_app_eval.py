# Import required libraries
from dotenv import load_dotenv
from itertools import zip_longest
import os, re
import streamlit as st
from streamlit_chat import message
from streamlit_extras.app_logo import add_logo
from langchain_community.chat_models import ChatOpenAI
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain_core.prompts import PromptTemplate
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# Load environment variables
#OPENAI_API_KEY='sk-proj-xwJ7bZHVKsQ8ItTp8i4ET3BlbkFJ2PxSITugQPkrUOwrAH8h'
OPENAI_API_KEY=st.secrets['openai']['OPENAI_KEY']

# Set streamlit page configuration
st.set_page_config(page_title="Royal Canin AI Assistant")
st.title("Royal Canin AI Assistant")
logo_url = 'royalcanin.jpg'
st.sidebar.image(logo_url)


# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

# Initialize the ChatOpenAI model
chat = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo",
    max_tokens=90,
    openai_api_key=OPENAI_API_KEY
)


eval_criterion = "Overall performance"
eval_criterion_comment = "Assess how well assistant performed with the respect to the input prompt given as reference."

Evaltemplate = """You are assessing a submitted answer on a given task or input based on a of criteria. Here is the data:
[BEGIN DATA]
***
[Input]: {input}
***
[Submission]: {output} 
***
[Reference]: {reference}
***
[Criteria]: {criteria}
***
[END DATA]

How well does the submission meet the Criteria? Think step by step and provide concise reasoning in 50 words max. Avoid simply stating the correct answers at the outset. Then print score in the range from 1 to 5 on its own line corresponding to to which extent the submission meets the criteria, where 1 means poorly meets criteria or doesnt meet at all, 2 means below average, 3 means average, 4 means above average and 5 means completely satisfying. If you are going with the score less than 5, then clearly mention what was lacking with respect to metric in the reasoning. At the end, repeat just the score again by itself on a new line."""

eval_prompt = PromptTemplate.from_template(Evaltemplate)

system_prompt = '''
# CONTEXT #         
You are a helpful AI assistant talking with a human. Consider yourself as a SEO expert who provides great insights for the metatitle and metadescription of the website. 
We have a website for the famous dog and breeds informations but the descriptions are missing for the products listed on the website. 
The brand is famous for the products for all the breeds of the cats and dogs and is very dedicated towards the service of the business users. 
Since there are descriptions missing,we need to write the best SEO metatitle and description possible for a given content 
page about cat or dog breeds. We'll provide the summary of the page's content, which will include detailed information about a 
specific breed, such as characteristics, temperament, care requirements, and other relevant details.

# OBJECTIVE #
The objective is to generate an SEO-optimized metatitle within a range of 50-60 characters and a metadescription within a range of  140-160 characters
that will attract search engines and improve search engine rankings. The generated metatitle and metadescription should be compelling, relevant to the content, 
and include appropriate keywords with high results in organic traffic to the page to enhance SEO performance.


# STYLE #
The style should be tailored for SEO, focusing on keyword optimization, clarity, and relevance to the content of the page. The metatitle and metadescription should be concise yet informative, providing a clear 
idea of the page content to both search engines and users.

# TONE #
The tone should be professional, reflecting the credibility and reliability of the content.

# AUDIENCE #
The primary audience is search engines, as they will use the SEO data to index and rank the content. 
However, the metatitle and metadescription should also be appealing to potential users
who come across the content in search results.

# RESPONSE FORMAT #
The response should be formatted as metatitle|||||metadescription.
Display the count of characters of the generated metatitle and metadescription in your response.
'''


def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        content=system_prompt)]

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg))  # Add user messages
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg))  # Add AI messages

    return zipped_messages


def generate_response():
    """
    Generate AI response using the ChatOpenAI model.
    """
    # Build the list of messages
    zipped_messages = build_message_list()

    # Generate response using the chat model
    ai_response = chat(zipped_messages)

    return ai_response.content

# custom eval definition
def get_custom_eval(criteria, eval_prompt):

  llm = ChatOpenAI(temperature=0, 
                        model_name='gpt-3.5-turbo',     
                        openai_api_key=OPENAI_API_KEY)
 
  return LabeledCriteriaEvalChain.from_llm(
      llm=llm,
      criteria=criteria,
      prompt=eval_prompt
  )

# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""


# Create a text input for user
st.text_input('YOU: ', key='prompt_input', on_change=submit)


if st.session_state.entered_prompt != "":
    # Get user query
    user_query = st.session_state.entered_prompt

    # Append user query to past queries
    st.session_state.past.append(user_query)

    # Generate response
    output = generate_response()

    # Append AI response to generated responses
    st.session_state.generated.append(output)

    chain = get_custom_eval(criteria={eval_criterion: eval_criterion_comment}, eval_prompt=eval_prompt)
    eval_result = chain.evaluate_strings(
        prediction=output,
        input=user_query,
        reference=system_prompt
    )

    sep = 'Score'
    score = re.findall(r'\d+', eval_result['value'])[-1] #Sometimes the reasoning is repeated in the "value" section. Here I just find the last number of the string and return it
    score_reasoning = eval_result["reasoning"].split(sep, 1)[0] #Remove "score" if mentioned in reasoning.
    
    st.markdown("## Score evaluation")
    st.write(f"Score is: {score}")
    st.write(f"Score reasoning is: {score_reasoning}")
    eval_result = []

    print("Score is :: ",score)
    print("Score reasoning is :: ",score_reasoning)
    #st.session_state.generated.append(score)
    #st.session_state.generated.append(score_reasoning)


# Display the chat history
#print("Message is ",message(st.session_state["generated"]))
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI response
        message(st.session_state["generated"][i], key=str(i))
        
        # Display user message
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')
        



# Add credit
st.markdown("""
---
Made with 🤖 by Vijayant Kumar(Valtech) :[LinkedIn](https://www.linkedin.com/in/vijayantkumarbansal/)
                                 [Github](https://github.com/vizzyno1)""")