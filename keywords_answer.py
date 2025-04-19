import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ----------------------- CONFIG -----------------------
st.set_page_config(page_title="Vanrise Solutions FAQ Assistant", layout="centered")

# Theme toggle
mode = st.selectbox("Choose Theme:", ["Light Mode", "Dark Mode"])
if mode == "Dark Mode":
    st.markdown("""
        <style>
        .stApp {
            background-color: #222;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# ----------------------- BERT Model -----------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.eval()

# ----------------------- Sidebar Navigation -----------------------
nav_option = st.sidebar.radio("Navigate to:", ["Introduction", "Search Bar", "Form Submission"])

# ----------------------- Data -----------------------
introduction_data = {
    "📘 Vanrise Solutions - FAQ Assistant": "Welcome to Vanrise Solutions! Our premier product, **T.One Wholesale BSS**, is designed to help telecommunications providers with streamlined billing and operational support.",
    "💼 About Vanrise Solutions": """
Welcome to **Vanrise Solutions**—a premier provider of wholesale billing and operational support systems for the telecommunications industry. 

Our flagship product, **T.One Wholesale BSS**, streamlines billing, invoicing, call data processing, and reporting for carriers and VoIP providers. Our software is tailored, secure, and driven by real-time data analytics. We partner with clients globally to ensure optimal efficiency and profitability.
"""
}

faq_data = pd.DataFrame([
    {"question": "What is the main purpose of the T.One Wholesale BSS?", "answer": "The T.One Wholesale BSS streamlines billing, invoicing, and operational processes for telecom carriers. It ensures accurate billing and offers data-driven insights."},
    {"question": "How does Vanrise Solutions differentiate itself?", "answer": "Through innovative tech, real-time processing, and exceptional support tailored for telecom clients."},
    {"question": "What industries are served?", "answer": "Primarily telecommunications, VoIP, and mobile service providers."},
    {"question": "Key features of T.One?", "answer": "Carrier profile management, automated invoicing, call data routing, financial tools, analytics."},
    {"question": "How is software secured?", "answer": "Through encryption, secure authentication, regular audits, and compliance."},
    {"question": "Customer support availability?", "answer": "Email, phone, helpdesk, documentation, and onboarding training."},
    {"question": "Is the platform customizable?", "answer": "Yes, workflows and reports can be tailored to client needs."},
    {"question": "Onboarding steps?", "answer": "Consultation, system setup, data migration, training, and ongoing support."},
    {"question": "How are updates handled?", "answer": "Periodic updates with minimal downtime, security patches, and new features."},
    {"question": "Customer feedback channels?", "answer": "Surveys, forums, and direct account manager contact."},
    {"question": "What’s in a Carrier Profile?", "answer": "Company details, billing info, contacts, and operational assignments."},
    {"question": "How to configure Carrier Profiles?", "answer": "Via the module interface with customizable tabs and fields."},
    {"question": "Why maintain accurate Carrier Profiles?", "answer": "Ensures correct billing, efficient communication, and compliance."},
    {"question": "Manage multiple Carrier Profiles?", "answer": "Central dashboard with bulk editing and sorting options."},
    {"question": "Store contact info?", "answer": "Multiple names, phones, and emails for smooth operations."},
    {"question": "Can users edit profiles?", "answer": "Yes, through the Edit button in the dashboard."},
    {"question": "Any profile creation limits?", "answer": "No limits—unlimited profiles supported."},
    {"question": "Role of carrier contacts?", "answer": "Billing clarifications, service agreements, and approvals."},
    {"question": "How does communication work?", "answer": "Integrated messaging and notification tagging."},
    {"question": "Review frequency?", "answer": "Quarterly or after major business changes."},
    {"question": "Invoice configuration?", "answer": "Template selection, billing cycles, auto-generation, etc."}
])

# ----------------------- Introduction Page -----------------------
if nav_option == "Introduction":
    for title, content in introduction_data.items():
        st.markdown(f"### {title}")
        st.markdown(content)

# ----------------------- Search Page -----------------------
elif nav_option == "Search Bar":
    st.title("🔍 Search Vanrise FAQ")
    user_query = st.text_input("Enter your question or keyword:")

    if user_query:
        inputs = tokenizer(user_query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            model_output = model(**inputs)

        # Using cosine similarity as a backup since our model is not trained
        faq_data['similarity'] = faq_data['question'].apply(
            lambda x: torch.nn.functional.cosine_similarity(
                tokenizer(x, return_tensors="pt", padding=True, truncation=True)["input_ids"].float(),
                inputs["input_ids"].float(), dim=-1
            ).mean().item()
        )

        top_faq = faq_data.sort_values("similarity", ascending=False).iloc[0]
        st.subheader("Most Relevant Answer:")
        st.markdown(f"**Q:** {top_faq['question']}")
        st.write(f"**A:** {top_faq['answer']}")

# ----------------------- Form Page (Optional Extension) -----------------------
elif nav_option == "Form Submission":
    st.title("📝 Submit Your Own Question")
    name = st.text_input("Your Name")
    question = st.text_area("Your Question")
    submitted = st.button("Submit")
    if submitted and name and question:
        st.success("Thanks for your submission! We'll get back to you shortly.")
