import streamlit as st

# Define the keyword-answer mapping
answer_map = {
    "contacts tab": """
        The contacts tab in the carrier profile allows you to add and manage the contacts for your carrier. 
        To configure this, go to the carrier profile and click on the Contacts tab. Here you can add details such as the contact's name, role (e.g., billing, pricing), email, and phone number. 
        This information helps you keep track of who is responsible for what within your carrier’s operations.
    """,
    "general settings": """
        To set up the general settings for a carrier, navigate to the 'General Settings' section of the carrier profile. 
        This will include the carrier’s name, address, contact information, and other relevant settings such as service preferences. 
        Review and confirm that all the information is accurate, then save the settings to finalize the configuration.
    """,
    "carrier profile": """
        The carrier profile requires several pieces of information to be fully configured. This includes basic details like the carrier's name, 
        contact information, service offerings, payment terms, certifications, and insurance details. This profile helps manage the carrier’s integration into your system.
    """,
    "assign contact": """
        To assign a contact to a carrier profile, go to the Contacts tab within the profile. Add the contact’s name, email address, role, and other required details. 
        Save the contact, and it will be linked to the carrier profile for communication purposes.
    """,
    "payment term": """
        A carrier's payment term specifies the duration within which payments are due after an invoice is issued. Common terms include 30, 60, or 90 days. 
        These terms are negotiable and depend on the carrier's policies and agreements with their clients.
    """,
    # More keywords and answers can be added here as needed
}

# Function to display answers based on the entered keyword
def display_answer(keyword):
    # Normalize the input (convert to lowercase for case-insensitive matching)
    keyword = keyword.lower()
    
    # Check if the entered keyword exists in the answer map
    if keyword in answer_map:
        st.write(f"### Answer for '{keyword.capitalize()}':")
        st.write(f"{answer_map[keyword]}")
    else:
        st.write(f"No detailed answer found for '{keyword}'. Please check the spelling or try a different keyword.")

# Streamlit app layout
st.title("Carrier Profile Management Assistant")
st.write("Enter a keyword to get a detailed answer.")

# Text input for custom keyword
user_keyword = st.text_input("Enter a Keyword")

# Display the answer for the entered keyword
if user_keyword:
    display_answer(user_keyword)
