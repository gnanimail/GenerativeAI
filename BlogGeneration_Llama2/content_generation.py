import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms.ctransformers import CTransformers


## generate response from Llama2 model
def generateLLMResponse(blog_content, blog_words, blog_audience):

    ### LLama2 model
    my_llm=CTransformers(model='model/llama-2-7b-chat.ggmlv3.q8_0.bin',
                         model_type='llama',
                         config={'max_new_tokens':256,
                                 'temperature':0.01})
    
    ## blog prompt template
    blog_template="""Write a blog for {blog_audience} job profile for a topic {blog_content}
                        within {blog_words} words."""
    
    # create blog prompt
    blog_prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=blog_template)
    
    ## Generate the ressponse from the LLama 2 model
    blog_response=my_llm(blog_prompt.format(blog_audience=blog_audience,
                                            blog_content=blog_content,
                                            blog_words=blog_words))
    
    return blog_response

#################################################################################################
## User Interface ##

def blog_generation():
    st.set_page_config(page_title="Generate Blogs",
                        page_icon='🤖',
                        layout='centered',
                        initial_sidebar_state='collapsed')

    st.header("Generate Blogs 🤖")

    blog_content=st.text_input("Enter the Blog Topic")

    ## creating to more columns for additonal 2 fields to split the screen 
    column1, column2=st.columns([5,5])

    # Number of words blog wants to generate 
    with column1:
        blog_words=st.text_input('No of Words')

    # To whom the blog was generate
    with column2:        
        blog_audience=st.selectbox('Writing the blog for',
                                    ('Researchers',
                                     'Data Scientist',
                                     'Common People'),index=0)
        
    # Request submit
    generate=st.button("Generate")

    # Response from LLM
    if generate:
        st.write(generateLLMResponse(blog_content, blog_words, blog_audience))



if __name__=="__main__":
    blog_generation()