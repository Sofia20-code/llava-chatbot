import gradio as gr
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

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
                return "Error: The FAQ data is missing necessary columns ('Question', 'Answer', 'Keyword')."
            elif faq_data.empty:
                return "Error: The FAQ data file is empty."
            
            faq_data['Question'] = faq_data['Question'].fillna('').astype(str)
            faq_data['Answer'] = faq_data['Answer'].fillna('').astype(str)
            faq_data['Keyword'] = faq_data['Keyword'].fillna('').apply(lambda x: [k.strip().lower() for k in str(x).split(',')])
            
            return faq_data
        except pd.errors.ParserError:
            return "Error: CSV parsing issue. Ensure it's formatted correctly with '|' as delimiter."
        except Exception as e:
            return f"Error loading CSV: {e}"
    else:
        return "Error: FAQ data file not found!"

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

# Gradio interface functions
def chatbot_qna(user_input):
    faq_data = load_faq_data()  # Load FAQ data
    if isinstance(faq_data, pd.DataFrame):  # Check if data is valid
        best_faq, similarity_score = get_best_match(user_input, faq_data)
        
        if best_faq is not None:
            output = f"**Q:** {best_faq['Question']}\n"
            output += f"**A:** {best_faq['Answer'] if best_faq['Answer'] else 'No answer available for this question.'}\n"
            output += f"**Keyword(s):** {', '.join(best_faq['Keyword'])}\n"
            output += f"**Similarity Score:** {similarity_score:.4f}\n"
            
            # Plot the similarity score as a bar graph
            fig, ax = plt.subplots()
            ax.barh(['Similarity Score'], [similarity_score], color='skyblue')
            ax.set_xlim(0, 1)  # Cosine similarity ranges from 0 to 1
            ax.set_xlabel('Cosine Similarity')
            plt.close(fig)  # Close the figure to prevent duplicate rendering in Gradio
            
            matched_icon_path = None
            for kw in best_faq['Keyword']:
                icon_key = kw.strip().lower()
                if icon_key in icon_map:
                    matched_icon_path = os.path.join(icon_dir, icon_map[icon_key])
                    break

            icon_output = None
            if matched_icon_path and os.path.exists(matched_icon_path):
                icon_output = matched_icon_path
            else:
                # Fallback: choose a random icon
                random_key = random.choice(list(icon_map.keys()))
                matched_icon_path = os.path.join(icon_dir, icon_map[random_key])
                icon_output = matched_icon_path

            return output, fig, icon_output
        else:
            return "No matching answers found. Please try rephrasing your question.", None, None
    else:
        return faq_data, None, None

def submit_question(user_question):
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

    return "Your question has been submitted successfully!"

# Define Gradio Interface
with gr.Blocks() as demo:
    with gr.Tab("ChatBot Q&A + Icon Displaying"):
        user_input = gr.Textbox(label="Ask your question:")
        output = gr.Markdown(label="Answer")
        similarity_plot = gr.Plot(label="Similarity Score")
        icon_display = gr.Image(label="Related Icon", type="filepath")

        user_input.submit(chatbot_qna, inputs=user_input, outputs=[output, similarity_plot, icon_display])

    with gr.Tab("Submit a Question"):
        question_input = gr.Textbox(label="Enter your question:")
        submit_button = gr.Button("Submit Question")
        submit_message = gr.Textbox(label="Status", interactive=False)

        submit_button.click(submit_question, inputs=question_input, outputs=submit_message)

# Launch the app
demo.launch()
