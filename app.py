# app.py
# Location: /home/yourusername/mysite/app.py   (PythonAnywhere)

import os
import sys
from flask import Flask, request, render_template, flash, redirect, url_for

# Add your project path (PythonAnywhere specific)
PROJECT_ROOT = '/home/mahajan1357/mysite'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ──────────────────────────────────────────────────────────────
#                LLM & LangChain imports
# ──────────────────────────────────────────────────────────────

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# ──────────────────────────────────────────────────────────────
#                Flask application
# ──────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.urandom(24)   # needed for flash messages

# ──────────────────────────────────────────────────────────────
#                Restaurant Knowledge (you can extend this a lot)
# ──────────────────────────────────────────────────────────────

RESTAURANT_CONTEXT = """\
Restaurant Name: Tasty Haveli
Location: Patel Nagar Jalandhar Road , kapurthala, Punjab
Cuisine: Authentic North Indian, Punjabi, Chinese, some Continental
Specialties:
  • Butter Chicken
  • Paneer Tikka Masala
  • Amritsari Kulcha
  • Dal Makhani
  • Chicken Tikka
  • Veg Biryani
  • Chilli Paneer (best in town!)
  • Lassi (sweet & salted)
Open Hours: 12:00 PM – 11:00 PM (all days)
Average cost for two: ₹950–₹1300 (approx)
Dining options: Dine-in, Takeaway, Home Delivery
Accepts: Cash, UPI, Cards, Paytm
Veg/Non-veg: Both available
Must try: Butter Chicken with Garlic Naan + Lassi
Ambiance: Family friendly, clean, good lighting, Punjabi music

Contact: +91 9878638743
#Instagram: @tastyhaveliludhiana
"""

# ──────────────────────────────────────────────────────────────
#                LLM Setup  (Groq example - very fast)
# ──────────────────────────────────────────────────────────────

# Put your real key here (better to use environment variable!)
GROQ_API_KEY = "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"

llm = ChatGroq(
    model="llama-3.3-70b-versatile",     # or "mixtral-8x7b-32768", "gemma2-9b-it" etc
    temperature=0.7,
    max_tokens=400,
    api_key=GROQ_API_KEY
)

# Simple but effective prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a warm, friendly, and knowledgeable waiter at Tasty Haveli restaurant.
Speak naturally in a polite Indian-English style.
Keep answers helpful, short to medium length (2-6 sentences).
Use the restaurant information provided below when answering.
Never make up prices, dishes or opening hours that aren't listed.

Restaurant information:
{context}

Current customer question: {question}"""),
    ("human", "{question}")
])

# Chain = prompt → LLM → string output
chain = prompt | llm | StrOutputParser()

# ──────────────────────────────────────────────────────────────
#                Routes
# ──────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def home():
    answer = None
    question = None

    if request.method == 'POST':
        question = request.form.get('question') or \
                  request.form.get('custom_question', '').strip()

        if not question:
            flash("Please select or type a question!", "warning")
            return redirect(url_for('home'))

        try:
            # Run the LangChain chain
            response = chain.invoke({
                "context": RESTAURANT_CONTEXT,
                "question": question
            })
            answer = response.strip()

        except Exception as e:
            flash(f"Sorry, something went wrong with AI :(\n({str(e)})", "danger")
            answer = None

    return render_template('index.html',
                         question=question,
                         answer=answer)


# For local development only (comment out on PythonAnywhere)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)