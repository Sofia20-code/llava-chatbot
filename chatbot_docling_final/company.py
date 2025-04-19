import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os
import random
import matplotlib.pyplot as plt

# Model Initialization (ensure this is done outside of the main loop to avoid re-loading)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# FAQ Data File and Icon Directory
faq_data_file = "data.csv"
icon_dir = "icons_new"
table_input_file = "table_input.csv"

# Load FAQ data
def load_faq_data():
    if os.path.exists(faq_data_file):
        try:
            faq_data = pd.read_csv(faq_data_file, delimiter='|', on_bad_lines='skip')
            faq_data.columns = faq_data.columns.str.strip()
            if 'Question' not in faq_data.columns or 'Answer' not in faq_data.columns or 'Keyword' not in faq_data.columns:
                st.error("The FAQ data is missing necessary columns ('Question', 'Answer', 'Keyword').")
                return None
            elif faq_data.empty:
                st.error("The FAQ data file is empty.")
                return None
            
            faq_data['Question'] = faq_data['Question'].fillna('').astype(str)
            faq_data['Answer'] = faq_data['Answer'].fillna('').astype(str)
            faq_data['Keyword'] = faq_data['Keyword'].fillna('').apply(lambda x: [k.strip().lower() for k in str(x).split(',')])
            
            return faq_data
        except pd.errors.ParserError:
            st.error("CSV parsing issue. Ensure it's formatted correctly with '|' as delimiter.")
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    else:
        st.error("FAQ data file not found!")
    return None

# Function to find the best matching FAQ based on cosine similarity
def get_best_match(user_input, faq_data):
    if not user_input:
        return None, None
    
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    
    faq_embeddings = [model.encode(q, convert_to_tensor=True) for q in faq_data['Question']]
    similarities = [util.pytorch_cos_sim(query_embedding, emb).item() for emb in faq_embeddings]
    
    # Find the FAQ with the highest similarity score
    best_match_idx = similarities.index(max(similarities))
    best_faq = faq_data.iloc[best_match_idx]
    
    return best_faq, similarities[best_match_idx]

# Icon Map for Keyword to Icon Lookup
icon_map = {
    'add_button': 'add_button.PNG',
    'administrator': 'administrator.PNG',
    'attach': 'attach.PNG',
    'buisnes': 'buisnes.PNG',
    'buisness': 'buisness.PNG',
    'buisness_partner': 'Buisness_Partner.PNG',
    'carrier_account_type': 'carrier_account_type.PNG',
    'carrier_profile': 'carrier_profile.PNG',
    'clear_all': 'Clear_all.PNG',
    'contacts': 'Contacts.PNG',
    'deleted': 'deleted.PNG',
    'documents': 'Documents.PNG',
    'export': 'Export.PNG',
    'financial': 'Financial.PNG',
    'general_info': 'General_Info.PNG',
    'general_settings': 'general_settings.PNG',
    'logo': 'logo.PNG',
    'notes': 'notes.PNG',
    'notification': 'notification.PNG',
    'pres': 'pres.PNG',
    'press': 'press.PNG',
    'remove': 'remove.PNG',
    'search': 'search.PNG',
    'ticket_contacts': 'Ticket_Contacts.PNG'
}

# Streamlit UI - Page Navigation
st.sidebar.title("Navigation")
nav_option = st.sidebar.radio("Go to", ["Introduction", "ChatBot Q&A + Icon Displaying", "Submit a Question"])

# Page 1: Introduction
if nav_option == "Introduction":
    st.title("📘 Welcome Guide")
    try:
        with open("extracted1.md", "r", encoding='utf-8') as f:
            intro_text = f.read()
            st.markdown(intro_text, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error reading markdown file: {e}")

# Page 2: ChatBot Q&A Section
elif nav_option == "ChatBot Q&A + Icon Displaying":
    st.title("💬 ChatBot Q&A")
    user_input = st.text_input("Ask your question:")

    faq_data = load_faq_data()  # Load FAQ data

    if user_input:
        if faq_data is not None:
            best_faq, similarity_score = get_best_match(user_input, faq_data)
            
            if best_faq is not None:
                st.markdown(f"**Q:** {best_faq['Question']}")
                st.write(f"**A:** {best_faq['Answer'] if best_faq['Answer'] else 'No answer available for this question.'}")
                st.write(f"**Keyword(s):** {', '.join(best_faq['Keyword'])}")
                st.write(f"**Similarity Score:** {similarity_score:.4f}")
                
                # Plot the similarity score as a bar graph
                fig, ax = plt.subplots()
                ax.barh(['Similarity Score'], [similarity_score], color='skyblue')
                ax.set_xlim(0, 1)  # Cosine similarity ranges from 0 to 1
                ax.set_xlabel('Cosine Similarity')
                st.pyplot(fig)

                # Logic for displaying the icon related to the FAQ
                matched_icon_path = None
                for kw in best_faq['Keyword']:
                    icon_key = kw.strip().lower()
                    if icon_key in icon_map:
                        matched_icon_path = os.path.join(icon_dir, icon_map[icon_key])
                        break

                if matched_icon_path:
                    st.subheader("📂 Related Icon:")
                    st.image(matched_icon_path, width=200)
                else:
                    # Fallback: choose 1 random icon without displaying a message
                    random_key = random.choice(list(icon_map.keys()))
                    matched_icon_path = os.path.join(icon_dir, icon_map[random_key])
                    st.image(matched_icon_path, width=200)
                
            else:
                st.write("No matching answers found. Please try rephrasing your question.")
        else:
            st.write("FAQ data is missing or could not be loaded.")

# Page 3: Submit a Question
elif nav_option == "Submit a Question":
    st.title("📝 Submit a Question")
    
    # User's input for question
    user_question = st.text_input("Enter your question:")

    if st.button("Submit Question"):
        if user_question:
            # Prepare data to save to CSV
            new_data = {
                "Question": user_question
            }

            # Check if table_input.csv exists
            if os.path.exists(table_input_file):
                # Load the existing data
                existing_data = pd.read_csv(table_input_file)
                existing_data = existing_data.append(new_data, ignore_index=True)
                existing_data.to_csv(table_input_file, index=False)
            else:
                # Create a new file and write data
                new_data_df = pd.DataFrame([new_data])
                new_data_df.to_csv(table_input_file, index=False)

            st.success("Your question has been submitted successfully!")
        else:
            st.error("Please enter a question before submitting.")
