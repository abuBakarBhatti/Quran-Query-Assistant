import streamlit as st
import rag

def chatbot_response(user_input):
    # Here you would implement your chatbot logic to generate a response
    # For simplicity, let's just echo the user's input
    # st.write(ans)
    return f"You said: '{user_input}'"

def main():
    st.title("Quran Inquiry Assistant")

    user_input = st.text_input("Enter your message here:", key="user_input")

    if st.button("Submit"):
        if user_input:
            # response = chatbot_response(user_input)
            response = rag.answerable(user_input)
            st.text_area("Response:", value=response, height=100, key="bot_response")

if __name__ == "__main__":
    main()



# import streamlit as st
# import rag

# def chatbot_response(user_input):
#     # Here you would implement your chatbot logic to generate a response
#     # For simplicity, let's just echo the user's input
#     #st.write(ans)
#     return f"You said: '{user_input}'"

# def main():
#     st.title("Quran GPT")

#     st.sidebar.header("User Input")
#     user_input = st.sidebar.text_input("Enter your message here:")

#     if st.sidebar.button("Send"):
#         if user_input:
#             #response = chatbot_response(user_input)
#             response = rag.answerable(user_input)
#             st.text_area("Bot Response:", value=response, height=100)

# if __name__ == "__main__":
#     main()